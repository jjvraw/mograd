# NVIDIA backward kernels: TF32 split dQ / dKdV and the FA2-shaped fused half kernel.
#
# Note: the simple TensorCore.load_a(tensor)/load_b(tensor) API does NOT emit
# ldmatrix on NVIDIA. It lowers to per-element distribute copies. Always use
# the shared-memory overloads with make_ldmatrix_swizzle, as these kernels do.

from std.gpu import WARP_SIZE, MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx, barrier, syncwarp
from std.gpu.compute.mma import mma, ld_matrix
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.memory import (
    CacheEviction,
    async_copy,
    external_memory,
    AddressSpace,
    async_copy_commit_group,
    async_copy_wait_all,
)
from std.gpu.memory import load as gpu_load
from std.gpu.primitives.warp import shuffle_idx as warp_shuffle_idx
from std.gpu.primitives.warp import shuffle_xor as warp_shuffle_xor
from std.gpu.primitives.warp import sum as warp_sum
from std.math import exp2, log, min
from std.sys import size_of
from std.utils import StaticTuple
from layout import Layout, LayoutTensor
from layout.int_tuple import IntTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core import TensorCore, shape_16x8x8, shape_16x8x16
from mograd.runtime.gpu.kernels.attention.config import (
    Br,
    Bc,
    BLOCK_SIZE,
    LOG2E,
    LN2,
    MMA_M,
    MMA_N,
    MMA_K,
    NUM_WARPS_MMA,
    Br_MMA,
    Bc_MMA,
    Bc_HMMA,
    KV_MMA_PAD,
    BLOCK_SIZE_MMA,
    BLOCK_SIZE_BWD_HALF,
    DQ_CONVERT_BLOCK,
    FWD_HALF_MAXNREG,
    QKMma,
    PVMma,
)
from std.sys._assembly import inlined_assembly


@always_inline
def _red_add_f32(ptr: UnsafePointer[Float32, MutAnyOrigin], val: Float32):
    """
    One-way PTX red.

    Atomic.fetch_add lowers to atom.global.add, whose discarded old-value round trip dominates
    accumulate-only traffic.

    Do not substitute fetch_add (measured far slower).
    """
    inlined_assembly["red.global.add.f32 [$0], $1;", NoneType, constraints="l,f"](ptr, val)


# ===-------------------------------------------------------------------===#
# Backward: dQ (MMA path)
#
# Grid=(B*H*ceil(S/Br_MMA),), Block=(BLOCK_SIZE_MMA,) = 4 warps.
# Warp warp_id owns query rows [tile_i*Br_MMA + warp_id*MMA_M, ...+MMA_M).
#
# Causal masking is applied per C-fragment element, and the dS C-layout
# fragments are converted to A-operand layout with warp shuffles for dS @ K.
# Every KV tile is treated uniformly (no separate "below diagonal" /
# "diagonal" fast paths). LSE and delta are staged in shared memory once per
# block.
#
# Per KV tile:
#   1. Load K → score = Q @ K.T → P = exp2(score*scale_log2e - lse), causal-masked
#   2. Load V → dp = dO @ V.T
#   3. dS = scale * P * (dp - delta)
#   4. Load K → dQ += dS @ K
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE_MMA)))
@__name(t"flash_attn_dq_mma_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_dq_kernel_mma[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    delta: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert D_BUCKET % MMA_K == 0, "D_BUCKET must be divisible by MMA_K=8"
    comptime assert Bc_MMA % MMA_N == 0, "Bc_MMA must be divisible by MMA_N=8"
    comptime D_TILES = D_BUCKET // MMA_K
    comptime N_TILES = Bc_MMA // MMA_N
    comptime O_TILES = D_BUCKET // MMA_N
    comptime BC_TILES = Bc_MMA // MMA_K
    comptime frag_size = QKMma.c_reg_type.size

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var bid = Int(block_idx.x)

    var num_tiles = (S + Br_MMA - 1) // Br_MMA
    var j_tiles = (S + Bc_MMA - 1) // Bc_MMA
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H

    var q_row_block = tile_i * Br_MMA
    # Q/K/V (and dQ) are BSHD. O and dO keep the fused node's BHSD layout.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var bh_base = b * H * S * D + h * S * D
    var lse_base = b * H * S + h * S
    var scale_log2e = scale * LOG2E

    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_dq_mma_smem"]()
    )
    comptime KV_MMA_STRIDE = D_BUCKET + KV_MMA_PAD
    var kv_smem = smem  # Bc_MMA * D_BUCKET, reused for K/V across phases
    var stage_smem = kv_smem + Bc_MMA * KV_MMA_STRIDE  # Br_MMA * MMA_K scratch for Q/dO staging
    var lse_smem = stage_smem + Br_MMA * MMA_K  # Br_MMA, pre-scaled by LOG2E
    var delta_smem = lse_smem + Br_MMA  # Br_MMA

    # Per-thread C-fragment coordinates (m16n8k8 TF32).
    var lane = tid % WARP_SIZE
    var frag_r0 = lane // 4
    var frag_r1 = frag_r0 + 8
    var frag_c0 = (lane % 4) * 2
    var frag_c1 = frag_c0 + 1

    # delta_i = dot(O_i, dO_i). One warp per row, lanes strided over D.
    # Written to global for the dK/dV kernels and kept in smem for this block.
    comptime for r in range(MMA_M):
        var row_local = warp_id * MMA_M + r
        var row = q_row_block + row_local
        var acc = Float32(0)
        if row < S:
            for d in range(lane, D, WARP_SIZE):
                var idx = bh_base + row * D + d
                acc += Float32(o[idx]) * Float32(dy[idx])
        var delta_row = warp_sum(acc)
        if lane == 0:
            delta_smem[row_local] = delta_row
            if row < S:
                delta[lse_base + row] = delta_row
    if tid < Br_MMA:
        var row = q_row_block + tid
        lse_smem[tid] = lse[lse_base + row] * LOG2E if row < S else Float32(0)
    barrier()

    var qk_op = QKMma()
    var pv_op = PVMma()

    comptime kv_layout = Layout(IntTuple(Bc_MMA, D_BUCKET), IntTuple(KV_MMA_STRIDE, 1))
    comptime stage_layout = Layout.row_major(Br_MMA, MMA_K)
    comptime a_reg_layout = Layout.row_major(1, PVMma.a_reg_type.size)

    # dQ accumulator: persists across all KV tiles.
    var dq_reg = (
        LayoutTensor[
            DType.float32, Layout.row_major(O_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
        ]
        .stack_allocation()
        .fill(0)
    )

    # CAUSAL: KV tiles strictly above this Q tile (all j > i) are fully masked
    # and contribute nothing, so stop after the diagonal tile.
    var j_tile_end = j_tiles
    comptime if CAUSAL:
        j_tile_end = min(j_tiles, (q_row_block + Br_MMA + Bc_MMA - 1) // Bc_MMA)

    for j_tile in range(j_tile_end):
        var jbase = j_tile * Bc_MMA

        # Load K into kv_smem (bounds-checked, zero-padded)
        comptime for idx in range((Bc_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
            var slot = tid + idx * BLOCK_SIZE_MMA
            if slot < Bc_MMA * D_BUCKET:
                var kv_row = jbase + slot // D_BUCKET
                var kv_col = slot % D_BUCKET
                kv_smem[(slot // D_BUCKET) * KV_MMA_STRIDE + kv_col] = Float32(
                    k[qkv_bh + kv_row * HD + kv_col]
                ) if kv_row < S and kv_col < D else Float32(0)
        barrier()

        # score = Q @ K^T
        var score_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(N_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime for d_tile in range(D_TILES):
            comptime for idx in range((Br_MMA * MMA_K + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Br_MMA * MMA_K:
                    var q_row = q_row_block + slot // MMA_K
                    var q_col = d_tile * MMA_K + slot % MMA_K
                    stage_smem[slot] = Float32(q[qkv_bh + q_row * HD + q_col]) if q_row < S and q_col < D else Float32(
                        0
                    )
            barrier()
            var q_tensor = LayoutTensor[DType.float32, stage_layout, address_space=AddressSpace.SHARED](stage_smem)
            var a_frag = qk_op.load_a(q_tensor.tile[MMA_M, MMA_K](warp_id, 0))
            var k_tensor = LayoutTensor[DType.float32, kv_layout, address_space=AddressSpace.SHARED](kv_smem)
            comptime for n_tile in range(N_TILES):
                var b_frag = qk_op.load_b(k_tensor.tile[MMA_N, MMA_K](n_tile, d_tile))
                var c_slice = score_reg.tile[1, frag_size](n_tile, 0)
                c_slice.copy_from(qk_op.mma_op(a_frag, b_frag, c_slice))
            barrier()

        # Convert score fragments to P fragments in registers.  The C fragment
        # layout is not a scalar row-major traversal, so apply causal coordinates
        # while each lane still owns its four logical C elements.
        comptime for n_tile in range(N_TILES):
            var row0 = q_row_block + warp_id * MMA_M + frag_r0
            var row1 = q_row_block + warp_id * MMA_M + frag_r1
            var col0 = jbase + n_tile * MMA_N + frag_c0
            var col1 = jbase + n_tile * MMA_N + frag_c1
            var lse0_l2 = lse_smem[warp_id * MMA_M + frag_r0]
            var lse1_l2 = lse_smem[warp_id * MMA_M + frag_r1]
            var score_vec = score_reg.tile[1, frag_size](n_tile, 0).vectorize[1, frag_size]()
            var p = rebind[SIMD[DType.float32, 4]](score_vec[0, 0])
            var ok00 = row0 < S and col0 < S
            var ok01 = row0 < S and col1 < S
            var ok10 = row1 < S and col0 < S
            var ok11 = row1 < S and col1 < S
            # Additive bias (mask) is part of the softmax argument in the
            # forward pass, so P must be reconstructed with it here too.
            var m00 = Float32(0)
            var m01 = Float32(0)
            var m10 = Float32(0)
            var m11 = Float32(0)
            comptime if CAUSAL:
                ok00 = ok00 and col0 <= row0
                ok01 = ok01 and col1 <= row0
                ok10 = ok10 and col0 <= row1
                ok11 = ok11 and col1 <= row1
            elif HAS_BIAS:
                var mask_bh = b * H * S * S + h * S * S
                m00 = Float32(mask[mask_bh + row0 * S + col0]) * LOG2E if ok00 else Float32(0)
                m01 = Float32(mask[mask_bh + row0 * S + col1]) * LOG2E if ok01 else Float32(0)
                m10 = Float32(mask[mask_bh + row1 * S + col0]) * LOG2E if ok10 else Float32(0)
                m11 = Float32(mask[mask_bh + row1 * S + col1]) * LOG2E if ok11 else Float32(0)
            p[0] = exp2(p[0] * scale_log2e + m00 - lse0_l2) if ok00 else Float32(0)
            p[1] = exp2(p[1] * scale_log2e + m01 - lse0_l2) if ok01 else Float32(0)
            p[2] = exp2(p[2] * scale_log2e + m10 - lse1_l2) if ok10 else Float32(0)
            p[3] = exp2(p[3] * scale_log2e + m11 - lse1_l2) if ok11 else Float32(0)
            score_vec[0, 0] = rebind[type_of(score_vec[0, 0])](p)

        # Load V into kv_smem (overwrite K)
        comptime for idx in range((Bc_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
            var slot = tid + idx * BLOCK_SIZE_MMA
            if slot < Bc_MMA * D_BUCKET:
                var kv_row = jbase + slot // D_BUCKET
                var kv_col = slot % D_BUCKET
                kv_smem[(slot // D_BUCKET) * KV_MMA_STRIDE + kv_col] = Float32(
                    v[qkv_bh + kv_row * HD + kv_col]
                ) if kv_row < S and kv_col < D else Float32(0)
        barrier()

        # dp = dO @ V^T
        var dp_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(N_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime for d_tile in range(D_TILES):
            comptime for idx in range((Br_MMA * MMA_K + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Br_MMA * MMA_K:
                    var q_row = q_row_block + slot // MMA_K
                    var q_col = d_tile * MMA_K + slot % MMA_K
                    stage_smem[slot] = Float32(dy[bh_base + q_row * D + q_col]) if q_row < S and q_col < D else Float32(
                        0
                    )
            barrier()
            var do_tensor = LayoutTensor[DType.float32, stage_layout, address_space=AddressSpace.SHARED](stage_smem)
            var a_frag = qk_op.load_a(do_tensor.tile[MMA_M, MMA_K](warp_id, 0))
            var v_tensor = LayoutTensor[DType.float32, kv_layout, address_space=AddressSpace.SHARED](kv_smem)
            comptime for n_tile in range(N_TILES):
                var b_frag = qk_op.load_b(v_tensor.tile[MMA_N, MMA_K](n_tile, d_tile))
                var c_slice = dp_reg.tile[1, frag_size](n_tile, 0)
                c_slice.copy_from(qk_op.mma_op(a_frag, b_frag, c_slice))
            barrier()

        # dS = scale * P * (dp - delta_i), kept in registers
        comptime for n_tile in range(N_TILES):
            var delta0 = delta_smem[warp_id * MMA_M + frag_r0]
            var delta1 = delta_smem[warp_id * MMA_M + frag_r1]
            var score_vec = score_reg.tile[1, frag_size](n_tile, 0).vectorize[1, frag_size]()
            var dp_vec = dp_reg.tile[1, frag_size](n_tile, 0).vectorize[1, frag_size]()
            var p = rebind[SIMD[DType.float32, 4]](score_vec[0, 0])
            var dp = rebind[SIMD[DType.float32, 4]](dp_vec[0, 0])
            p[0] = p[0] * (dp[0] - delta0) * scale
            p[1] = p[1] * (dp[1] - delta0) * scale
            p[2] = p[2] * (dp[2] - delta1) * scale
            p[3] = p[3] * (dp[3] - delta1) * scale
            score_vec[0, 0] = rebind[type_of(score_vec[0, 0])](p)

        # Reload K into kv_smem (overwrite V)
        comptime for idx in range((Bc_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
            var slot = tid + idx * BLOCK_SIZE_MMA
            if slot < Bc_MMA * D_BUCKET:
                var kv_row = jbase + slot // D_BUCKET
                var kv_col = slot % D_BUCKET
                kv_smem[(slot // D_BUCKET) * KV_MMA_STRIDE + kv_col] = Float32(
                    k[qkv_bh + kv_row * HD + kv_col]
                ) if kv_row < S and kv_col < D else Float32(0)
        barrier()

        # dQ += dS @ K
        # The dS C-tile for bc_tile is the same 16x8 matrix the A operand needs
        # (N_TILES == BC_TILES). Convert C-fragment layout to A-fragment layout
        # with warp shuffles instead of a shared-memory round trip.
        var k_tensor2 = LayoutTensor[DType.float32, kv_layout, address_space=AddressSpace.SHARED](kv_smem)
        comptime for bc_tile in range(BC_TILES):
            var score_vec = score_reg.tile[1, frag_size](bc_tile, 0).vectorize[1, frag_size]()
            var p = rebind[SIMD[DType.float32, 4]](score_vec[0, 0])
            var row_lane_base = lane - lane % 4
            var col_pair = lane % 2
            var low_src_lane = UInt32(row_lane_base + (lane % 4) // 2)
            var high_src_lane = UInt32(row_lane_base + 2 + (lane % 4) // 2)
            var low_r0_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[0]), low_src_lane)[0]
            var low_r0_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[1]), low_src_lane)[0]
            var low_r1_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[2]), low_src_lane)[0]
            var low_r1_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[3]), low_src_lane)[0]
            var high_r0_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[0]), high_src_lane)[0]
            var high_r0_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[1]), high_src_lane)[0]
            var high_r1_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[2]), high_src_lane)[0]
            var high_r1_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[3]), high_src_lane)[0]
            var a_p = LayoutTensor[
                DType.float32, a_reg_layout, MutAnyOrigin, address_space=AddressSpace.LOCAL
            ].stack_allocation()
            var a_vec = a_p.vectorize[1, PVMma.a_reg_type.size]()
            a_vec[0, 0] = rebind[type_of(a_vec[0, 0])](
                SIMD[DType.float32, 4](
                    low_r0_0 if col_pair == 0 else low_r0_1,
                    low_r1_0 if col_pair == 0 else low_r1_1,
                    high_r0_0 if col_pair == 0 else high_r0_1,
                    high_r1_0 if col_pair == 0 else high_r1_1,
                )
            )
            comptime for d_tile in range(O_TILES):
                var b_frag = pv_op.load_b(k_tensor2.tile[MMA_K, MMA_N](bc_tile, d_tile))
                var c_slice = dq_reg.tile[1, frag_size](d_tile, 0)
                c_slice.copy_from(pv_op.mma_op(a_p, b_frag, c_slice))
        barrier()

    # Write dQ to global memory via smem staging (reuses kv_smem)
    var dq_tensor = LayoutTensor[DType.float32, Layout.row_major(Br_MMA, D_BUCKET), address_space=AddressSpace.SHARED](
        kv_smem
    )
    comptime for d_tile in range(O_TILES):
        pv_op.store_d(dq_tensor.tile[MMA_M, MMA_N](warp_id, d_tile), dq_reg.tile[1, frag_size](d_tile, 0))
    barrier()
    comptime for idx in range((Br_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
        var slot = tid + idx * BLOCK_SIZE_MMA
        if slot < Br_MMA * D_BUCKET:
            var row = q_row_block + slot // D_BUCKET
            var col = slot % D_BUCKET
            if row < S and col < D:
                dq[qkv_bh + row * HD + col] = Scalar[dtype](kv_smem[slot])


# ===-------------------------------------------------------------------===#
# Backward: dK and dV (MMA path)
#
# Grid=(B*H*ceil(S/Bc_MMA),), Block=(BLOCK_SIZE_MMA,) = 4 warps.
# Warp warp_id owns KV rows [tile_j*Bc_MMA + warp_id*MMA_M, ...+MMA_M).
#
# P and dS never leave registers. Causal masking is applied per C-fragment
# element and the C-layout fragments convert to A-operand layout with warp
# shuffles.
# Uniform loop over every Q tile (no separate "diagonal" / "above diagonal"
# fast paths). See flash_attn_dq_kernel_mma for the mirror-image kernel (that
# one owns query rows and loops KV tiles).
#
# One kernel produces both dK and dV. The score/dp passes and staging are
# shared, halving global read traffic, at the cost of two live accumulators
# (255-register ceiling). The split kernels measured L2-bound rather than
# occupancy-bound, so trading occupancy for less traffic wins.
#
# Per Q tile:
#   score[Bc×Br] = K @ Q.T  → P via softmax (causal-masked), in registers
#   dp[Bc×Br] = V @ dO.T (== (dO @ V.T).T, same values at [kv,q])
#   dS = scale * P * (dp - delta). P and dS are shuffled to A-operand layout
#   dV += P @ dO and dK += dS @ Q, sharing one dO/Q staging pass per d_tile
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE_MMA)))
@__name(t"flash_attn_dkdv_mma_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_dkdv_kernel_mma[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    delta: UnsafePointer[Float32, ImmutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert D_BUCKET % MMA_K == 0, "D_BUCKET must be divisible by MMA_K=8"
    comptime assert Br_MMA % MMA_N == 0, "Br_MMA must be divisible by MMA_N=8"
    comptime D_TILES = D_BUCKET // MMA_K
    comptime N_TILES = Br_MMA // MMA_N
    comptime O_TILES = D_BUCKET // MMA_N
    comptime frag_size = QKMma.c_reg_type.size

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var bid = Int(block_idx.x)

    var num_tiles = (S + Bc_MMA - 1) // Bc_MMA
    var i_tiles = (S + Br_MMA - 1) // Br_MMA
    if bid >= B * H * num_tiles:
        return

    var tile_j = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H

    var kv_row_block = tile_j * Bc_MMA
    # Q/K/V (and dK/dV) are BSHD. dO keeps the fused node's BHSD layout.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var bh_base = b * H * S * D + h * S * D
    var lse_base = b * H * S + h * S
    var scale_log2e = scale * LOG2E

    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_dkdv_mma_smem"]()
    )
    var lse_smem = smem  # Br_MMA
    var delta_smem = lse_smem + Br_MMA  # Br_MMA
    var stage_smem = delta_smem + Br_MMA  # Bc_MMA * MMA_K scratch for K/V staging
    var qdo_stage_smem = stage_smem + Bc_MMA * MMA_K  # Br_MMA * MMA_K scratch for Q/dO staging
    var out_smem = qdo_stage_smem + Br_MMA * MMA_K  # Bc_MMA * MMA_N unpack tile

    var qk_op = QKMma()
    var pv_op = PVMma()

    comptime stage_layout = Layout.row_major(Bc_MMA, MMA_K)
    comptime qdo_stage_layout = Layout.row_major(Br_MMA, MMA_K)
    comptime out_layout = Layout.row_major(Bc_MMA, MMA_N)

    var dv_reg = (
        LayoutTensor[
            DType.float32, Layout.row_major(O_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
        ]
        .stack_allocation()
        .fill(0)
    )
    var dk_reg = (
        LayoutTensor[
            DType.float32, Layout.row_major(O_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
        ]
        .stack_allocation()
        .fill(0)
    )

    # CAUSAL: Q tiles strictly below this KV tile (all i < j) are fully masked
    # and contribute nothing, so skip them. Br_MMA == Bc_MMA so the first tile
    # with any i >= kv_row_block is exactly kv_row_block // Br_MMA.
    var i_tile_start = 0
    comptime if CAUSAL:
        i_tile_start = kv_row_block // Br_MMA

    for i_tile in range(i_tile_start, i_tiles):
        var ibase = i_tile * Br_MMA

        # Load LSE and delta for this Q tile
        if tid < Br_MMA:
            var q_row = ibase + tid
            lse_smem[tid] = lse[lse_base + q_row] * LOG2E if q_row < S else Float32(0)
            delta_smem[tid] = delta[lse_base + q_row] if q_row < S else Float32(0)
        barrier()

        # The logical score tile is [kv row, q row], so LSE/delta are indexed
        # by the fragment column's query row.
        var lane = tid % WARP_SIZE
        var frag_r0 = lane // 4
        var frag_r1 = frag_r0 + 8
        var frag_c0 = (lane % 4) * 2
        var frag_c1 = frag_c0 + 1

        # score = K @ Q^T
        var score_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(N_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime for d_tile in range(D_TILES):
            comptime for idx in range((Bc_MMA * MMA_K + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Bc_MMA * MMA_K:
                    var kv_row = kv_row_block + slot // MMA_K
                    var kv_col = d_tile * MMA_K + slot % MMA_K
                    stage_smem[slot] = Float32(
                        k[qkv_bh + kv_row * HD + kv_col]
                    ) if kv_row < S and kv_col < D else Float32(0)
            comptime for idx in range((Br_MMA * MMA_K + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Br_MMA * MMA_K:
                    var q_row = ibase + slot // MMA_K
                    var q_col = d_tile * MMA_K + slot % MMA_K
                    qdo_stage_smem[slot] = Float32(
                        q[qkv_bh + q_row * HD + q_col]
                    ) if q_row < S and q_col < D else Float32(0)
            barrier()
            var k_tensor = LayoutTensor[DType.float32, stage_layout, address_space=AddressSpace.SHARED](stage_smem)
            var a_frag = qk_op.load_a(k_tensor.tile[MMA_M, MMA_K](warp_id, 0))
            var q_tensor = LayoutTensor[DType.float32, qdo_stage_layout, address_space=AddressSpace.SHARED](
                qdo_stage_smem
            )
            comptime for n_tile in range(N_TILES):
                var b_frag = qk_op.load_b(q_tensor.tile[MMA_N, MMA_K](n_tile, 0))
                var c_slice = score_reg.tile[1, frag_size](n_tile, 0)
                c_slice.copy_from(qk_op.mma_op(a_frag, b_frag, c_slice))
            barrier()

        # P from score fragments (stays in registers, C layout)
        comptime for n_tile in range(N_TILES):
            var kv0 = kv_row_block + warp_id * MMA_M + frag_r0
            var kv1 = kv_row_block + warp_id * MMA_M + frag_r1
            var q0_local = n_tile * MMA_N + frag_c0
            var q1_local = n_tile * MMA_N + frag_c1
            var q0 = ibase + q0_local
            var q1 = ibase + q1_local
            var lse0_l2 = lse_smem[q0_local] if q0_local < Br_MMA else Float32(0)
            var lse1_l2 = lse_smem[q1_local] if q1_local < Br_MMA else Float32(0)
            var score_vec = score_reg.tile[1, frag_size](n_tile, 0).vectorize[1, frag_size]()
            var p = rebind[SIMD[DType.float32, 4]](score_vec[0, 0])
            var ok00 = kv0 < S and q0 < S
            var ok01 = kv0 < S and q1 < S
            var ok10 = kv1 < S and q0 < S
            var ok11 = kv1 < S and q1 < S
            var m00 = Float32(0)
            var m01 = Float32(0)
            var m10 = Float32(0)
            var m11 = Float32(0)
            comptime if CAUSAL:
                ok00 = ok00 and kv0 <= q0
                ok01 = ok01 and kv0 <= q1
                ok10 = ok10 and kv1 <= q0
                ok11 = ok11 and kv1 <= q1
            elif HAS_BIAS:
                var mask_bh = b * H * S * S + h * S * S
                m00 = Float32(mask[mask_bh + q0 * S + kv0]) * LOG2E if ok00 else Float32(0)
                m01 = Float32(mask[mask_bh + q1 * S + kv0]) * LOG2E if ok01 else Float32(0)
                m10 = Float32(mask[mask_bh + q0 * S + kv1]) * LOG2E if ok10 else Float32(0)
                m11 = Float32(mask[mask_bh + q1 * S + kv1]) * LOG2E if ok11 else Float32(0)
            p[0] = exp2(p[0] * scale_log2e + m00 - lse0_l2) if ok00 else Float32(0)
            p[1] = exp2(p[1] * scale_log2e + m01 - lse1_l2) if ok01 else Float32(0)
            p[2] = exp2(p[2] * scale_log2e + m10 - lse0_l2) if ok10 else Float32(0)
            p[3] = exp2(p[3] * scale_log2e + m11 - lse1_l2) if ok11 else Float32(0)
            score_vec[0, 0] = rebind[type_of(score_vec[0, 0])](p)

        # dp = V @ dO^T (dot(V_kv, dO_q), same shape/layout as score)
        var dp_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(N_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime for d_tile in range(D_TILES):
            comptime for idx in range((Bc_MMA * MMA_K + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Bc_MMA * MMA_K:
                    var kv_row = kv_row_block + slot // MMA_K
                    var kv_col = d_tile * MMA_K + slot % MMA_K
                    stage_smem[slot] = Float32(
                        v[qkv_bh + kv_row * HD + kv_col]
                    ) if kv_row < S and kv_col < D else Float32(0)
            comptime for idx in range((Br_MMA * MMA_K + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Br_MMA * MMA_K:
                    var q_row = ibase + slot // MMA_K
                    var q_col = d_tile * MMA_K + slot % MMA_K
                    qdo_stage_smem[slot] = Float32(
                        dy[bh_base + q_row * D + q_col]
                    ) if q_row < S and q_col < D else Float32(0)
            barrier()
            var v_tensor = LayoutTensor[DType.float32, stage_layout, address_space=AddressSpace.SHARED](stage_smem)
            var a_frag = qk_op.load_a(v_tensor.tile[MMA_M, MMA_K](warp_id, 0))
            var do_tensor2 = LayoutTensor[DType.float32, qdo_stage_layout, address_space=AddressSpace.SHARED](
                qdo_stage_smem
            )
            comptime for n_tile in range(N_TILES):
                var b_frag = qk_op.load_b(do_tensor2.tile[MMA_N, MMA_K](n_tile, 0))
                var c_slice = dp_reg.tile[1, frag_size](n_tile, 0)
                c_slice.copy_from(qk_op.mma_op(a_frag, b_frag, c_slice))
            barrier()

        # dS = scale * P * (dp - delta), then shuffle P and dS to A layout
        # The C-tile for n_tile is the same 16x8 matrix the A operand needs.
        # P (score_reg) and dS (dp_reg) are converted in place, so both output
        # matmuls below can run per d_tile without any re-conversion.
        comptime for n_tile in range(N_TILES):
            var q0_local = n_tile * MMA_N + frag_c0
            var q1_local = n_tile * MMA_N + frag_c1
            var delta0 = delta_smem[q0_local] if q0_local < Br_MMA else Float32(0)
            var delta1 = delta_smem[q1_local] if q1_local < Br_MMA else Float32(0)
            var score_vec = score_reg.tile[1, frag_size](n_tile, 0).vectorize[1, frag_size]()
            var dp_vec = dp_reg.tile[1, frag_size](n_tile, 0).vectorize[1, frag_size]()
            var p = rebind[SIMD[DType.float32, 4]](score_vec[0, 0])
            var dp = rebind[SIMD[DType.float32, 4]](dp_vec[0, 0])
            var ds = SIMD[DType.float32, 4](
                p[0] * (dp[0] - delta0) * scale,
                p[1] * (dp[1] - delta1) * scale,
                p[2] * (dp[2] - delta0) * scale,
                p[3] * (dp[3] - delta1) * scale,
            )
            var row_lane_base = lane - lane % 4
            var col_pair = lane % 2
            var low_src_lane = UInt32(row_lane_base + (lane % 4) // 2)
            var high_src_lane = UInt32(row_lane_base + 2 + (lane % 4) // 2)
            var p_low_r0_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[0]), low_src_lane)[0]
            var p_low_r0_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[1]), low_src_lane)[0]
            var p_low_r1_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[2]), low_src_lane)[0]
            var p_low_r1_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[3]), low_src_lane)[0]
            var p_high_r0_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[0]), high_src_lane)[0]
            var p_high_r0_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[1]), high_src_lane)[0]
            var p_high_r1_0 = warp_shuffle_idx(SIMD[DType.float32, 1](p[2]), high_src_lane)[0]
            var p_high_r1_1 = warp_shuffle_idx(SIMD[DType.float32, 1](p[3]), high_src_lane)[0]
            score_vec[0, 0] = rebind[type_of(score_vec[0, 0])](
                SIMD[DType.float32, 4](
                    p_low_r0_0 if col_pair == 0 else p_low_r0_1,
                    p_low_r1_0 if col_pair == 0 else p_low_r1_1,
                    p_high_r0_0 if col_pair == 0 else p_high_r0_1,
                    p_high_r1_0 if col_pair == 0 else p_high_r1_1,
                )
            )
            var ds_low_r0_0 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[0]), low_src_lane)[0]
            var ds_low_r0_1 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[1]), low_src_lane)[0]
            var ds_low_r1_0 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[2]), low_src_lane)[0]
            var ds_low_r1_1 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[3]), low_src_lane)[0]
            var ds_high_r0_0 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[0]), high_src_lane)[0]
            var ds_high_r0_1 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[1]), high_src_lane)[0]
            var ds_high_r1_0 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[2]), high_src_lane)[0]
            var ds_high_r1_1 = warp_shuffle_idx(SIMD[DType.float32, 1](ds[3]), high_src_lane)[0]
            dp_vec[0, 0] = rebind[type_of(dp_vec[0, 0])](
                SIMD[DType.float32, 4](
                    ds_low_r0_0 if col_pair == 0 else ds_low_r0_1,
                    ds_low_r1_0 if col_pair == 0 else ds_low_r1_1,
                    ds_high_r0_0 if col_pair == 0 else ds_high_r0_1,
                    ds_high_r1_0 if col_pair == 0 else ds_high_r1_1,
                )
            )

        # dV += P @ dO and dK += dS @ Q, sharing one staging pass
        comptime for d_tile in range(O_TILES):
            comptime for idx in range((Br_MMA * MMA_N + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Br_MMA * MMA_N:
                    var q_row = ibase + slot // MMA_N
                    var q_col = d_tile * MMA_N + slot % MMA_N
                    qdo_stage_smem[slot] = Float32(
                        dy[bh_base + q_row * D + q_col]
                    ) if q_row < S and q_col < D else Float32(0)
                    stage_smem[slot] = Float32(q[qkv_bh + q_row * HD + q_col]) if q_row < S and q_col < D else Float32(
                        0
                    )
            barrier()
            var do_tensor = LayoutTensor[DType.float32, qdo_stage_layout, address_space=AddressSpace.SHARED](
                qdo_stage_smem
            )
            var q_tensor2 = LayoutTensor[DType.float32, qdo_stage_layout, address_space=AddressSpace.SHARED](stage_smem)
            comptime for n_tile in range(N_TILES):
                var do_frag = pv_op.load_b(do_tensor.tile[MMA_K, MMA_N](n_tile, 0))
                var dv_slice = dv_reg.tile[1, frag_size](d_tile, 0)
                dv_slice.copy_from(pv_op.mma_op(score_reg.tile[1, frag_size](n_tile, 0), do_frag, dv_slice))
                var q_frag = pv_op.load_b(q_tensor2.tile[MMA_K, MMA_N](n_tile, 0))
                var dk_slice = dk_reg.tile[1, frag_size](d_tile, 0)
                dk_slice.copy_from(pv_op.mma_op(dp_reg.tile[1, frag_size](n_tile, 0), q_frag, dk_slice))
            barrier()

    # Write dK, dV to global memory via a compact per-d_tile unpack tile
    var out_tensor = LayoutTensor[DType.float32, out_layout, address_space=AddressSpace.SHARED](out_smem)
    comptime for d_tile in range(O_TILES):
        pv_op.store_d(out_tensor.tile[MMA_M, MMA_N](warp_id, 0), dv_reg.tile[1, frag_size](d_tile, 0))
        barrier()
        comptime for idx in range((Bc_MMA * MMA_N + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
            var slot = tid + idx * BLOCK_SIZE_MMA
            if slot < Bc_MMA * MMA_N:
                var row = kv_row_block + slot // MMA_N
                var col = d_tile * MMA_N + slot % MMA_N
                if row < S and col < D:
                    dv[qkv_bh + row * HD + col] = Scalar[dtype](out_smem[slot])
        barrier()
        pv_op.store_d(out_tensor.tile[MMA_M, MMA_N](warp_id, 0), dk_reg.tile[1, frag_size](d_tile, 0))
        barrier()
        comptime for idx in range((Bc_MMA * MMA_N + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
            var slot = tid + idx * BLOCK_SIZE_MMA
            if slot < Bc_MMA * MMA_N:
                var row = kv_row_block + slot // MMA_N
                var col = d_tile * MMA_N + slot % MMA_N
                if row < S and col < D:
                    dk[qkv_bh + row * HD + col] = Scalar[dtype](out_smem[slot])
        barrier()


# ===-------------------------------------------------------------------===#
# Backward: fused dK/dV/dQ, half precision (FA2-shaped)
#
# Grid=(B*H*ceil(S/Bc_MMA),), Block=(BLOCK_SIZE_BWD_HALF=256,) = 8 warps.
# Warp (warp_r, warp_c) = (warp_id % 4, warp_id // 4): warp_r picks the
# 16-row M group (kv rows for score/dp/dV/dK, q rows for dQ), warp_c picks
# the column half (q columns for score/dp, d columns for the output matmuls).
#
# Mirrors FA2's compute_dq_dk_dv_1colblock for its supported dtypes
# (fp16/bf16 only, FA2 has no f32 backward). K and V stay resident in
# shared memory for the whole block, Q and dO are loaded once per Q tile,
# and the score/dp matmuls run exactly once per (KV tile, Q tile) pair.
# P, dS, and dS^T are materialized in half-precision shared memory. They
# must be converted to half for the HMMA A-operands anyway, so the smem
# round trip is free and gives every warp access to every fragment (the
# 8-warp column split makes register-only P/dS impossible). dQ partials are
# accumulated into a float32 buffer with one-way red.global.add reductions
# (nondeterministic ordering, matching FA2's default) and converted to the
# output dtype by flash_attn_dq_convert_kernel.
#
# Half-precision tiles make residency affordable at D_BUCKET=128 (one
# block/SM). the block runs 8 warps because delivered L2 throughput scales
# with resident warps.
#
# Per Q tile (3 block-wide barriers total):
#   score[Bc×Br] = K @ Q.T and dp[Bc×Br] = V @ dO.T (one fused pass)
#   P = softmax reconstruction (causal-masked / +bias), dS = scale*P*(dp-delta)
#   P, dS → smem [kv, q] and dS^T → smem [q, kv]  (half)
#   dV += P @ dO, dK += dS @ Q, dQ_accum += dS^T @ K (atomic)
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE_MMA)))
@__name(t"flash_attn_bwd_delta_{dtype}")
def flash_attn_bwd_delta_kernel[
    dtype: DType
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    delta: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
):
    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var bid = Int(block_idx.x)

    var num_tiles = (S + Br_MMA - 1) // Br_MMA
    if bid >= B * H * num_tiles:
        return
    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H
    # O and dO keep the fused node's BHSD layout.
    var bh_base = b * H * S * D + h * S * D
    var delta_base = b * H * S + h * S

    # One warp per query row. The kernel is scalar-load latency bound, so
    # use 16B vectorized lanes when D allows and a scalar fallback for
    # ragged D.
    comptime for r in range(MMA_M):
        var row = tile_i * Br_MMA + warp_id * MMA_M + r
        var acc = Float32(0)
        if row < S:
            if D % 8 == 0:
                for dv8 in range(lane, D // 8, WARP_SIZE):
                    var idx = bh_base + row * D + dv8 * 8
                    var vo = o.load[width=8](idx).cast[DType.float32]()
                    var vdy = dy.load[width=8](idx).cast[DType.float32]()
                    acc += (vo * vdy).reduce_add()
            else:
                for d in range(lane, D, WARP_SIZE):
                    var idx = bh_base + row * D + d
                    acc += Float32(o[idx]) * Float32(dy[idx])
        var delta_row = warp_sum(acc)
        if lane == 0 and row < S:
            delta[delta_base + row] = delta_row


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(DQ_CONVERT_BLOCK)))
@__name(t"flash_attn_dq_convert_{dtype}")
def flash_attn_dq_convert_kernel[
    dtype: DType
](dq_accum: UnsafePointer[Float32, ImmutAnyOrigin], dq: UnsafePointer[Scalar[dtype], MutAnyOrigin], numel: Int,):
    var idx = Int(block_idx.x) * DQ_CONVERT_BLOCK + Int(thread_idx.x)
    if idx < numel:
        dq[idx] = Scalar[dtype](dq_accum[idx])


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE_BWD_HALF)))
@__name(t"flash_attn_dkdvdq_mma_half_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_bwd_kernel_mma_half[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    delta: UnsafePointer[Float32, ImmutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dq_accum: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert dtype.is_half_float(), "half MMA backward requires float16 or bfloat16 inputs"
    comptime HMMA_K = 16
    comptime assert D_BUCKET % HMMA_K == 0, "D_BUCKET must be divisible by HMMA_K=16"
    comptime assert D_BUCKET <= 128, "half MMA backward supports D_BUCKET <= 128 (resident tiles)"
    comptime D_TILES = D_BUCKET // HMMA_K  # K-depth tiles for score/dp
    comptime KQ_TILES = Br_MMA // HMMA_K  # K-depth tiles for dV/dK/dQ (= 4)
    comptime NW_TILES = 4  # q-column tiles per warp: 32 cols / MMA_N
    comptime DW_TILES = D_BUCKET // 2 // MMA_N  # d-column tiles per warp
    comptime QKHalf = TensorCore[DType.float32, dtype, shape_16x8x16, True]
    comptime PVHalf = TensorCore[DType.float32, dtype, shape_16x8x16, False]
    comptime frag_size = QKHalf.c_reg_type.size

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = tid % WARP_SIZE
    var warp_r = warp_id % 4
    var warp_c = warp_id // 4
    var bid = Int(block_idx.x)

    var num_tiles = (S + Bc_MMA - 1) // Bc_MMA
    var i_tiles = (S + Br_MMA - 1) // Br_MMA
    if bid >= B * H * num_tiles:
        return

    var tile_j = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H

    var kv_row_block = tile_j * Bc_MMA
    # Q/K/V (and dK/dV) are BSHD. dO keeps the fused node's BHSD layout.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var bh_base = b * H * S * D + h * S * D
    var lse_base = b * H * S + h * S
    var scale_log2e = scale * LOG2E

    # Shared memory holds a float32 lse/delta head, then half-precision
    # tiles stored UNPADDED with ldmatrix XOR-swizzled layouts. Every 16B
    # chunk lands at chunk index swizzle(row*CHUNKS_PER_ROW + col/8) and
    # the TensorCore shared-memory load_a/load_b overloads read them back
    # with single conflict-free ldmatrix.x4 instructions instead of the
    # simple API's per-element distribute copies.
    comptime COPY_VEC = 8  # halves per chunk
    comptime KV_CHUNKS = D_BUCKET // COPY_VEC
    comptime P_CHUNKS = Br_MMA // COPY_VEC
    comptime kv_swizzle = make_ldmatrix_swizzle[dtype, D_BUCKET]()
    comptime p_swizzle = make_ldmatrix_swizzle[dtype, Br_MMA]()
    comptime kv_swz: Optional[Swizzle] = kv_swizzle
    comptime p_swz: Optional[Swizzle] = p_swizzle
    comptime a_frag_size = QKHalf.a_reg_type.size
    comptime b_frag_size = QKHalf.b_reg_type.size
    var smem = rebind[UnsafePointer[Scalar[dtype], MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Scalar[dtype], address_space=AddressSpace.SHARED, alignment=16, name="attn_bwd_half_smem"]()
    )
    var lse_smem = smem.bitcast[Float32]()  # Br_MMA float32
    var delta_smem = lse_smem + Br_MMA  # Br_MMA float32
    var k_smem = smem + 4 * Br_MMA  # skip the f32 head (2*Br_MMA floats = 4*Br_MMA halves)
    var v_smem = k_smem + Bc_MMA * D_BUCKET
    var q_smem = v_smem + Bc_MMA * D_BUCKET
    var do_smem = q_smem + Br_MMA * D_BUCKET
    var p_smem = do_smem + Br_MMA * D_BUCKET  # P  [kv, q]
    var ds_smem = p_smem + Bc_MMA * Br_MMA  # dS [kv, q]
    var dst_smem = ds_smem + Bc_MMA * Br_MMA  # dS^T [q, kv]
    # Mask (additive bias) tile [q, kv], plain layout, only used non-causal.
    # It is staged per pair because the bias tensor is B*H*S*S and exceeds
    # L2 at large shapes, where scalar per-fragment reads stall the block.
    var mask_smem = dst_smem + Bc_MMA * Br_MMA

    var qk_op = QKHalf()
    var pv_op = PVHalf()

    comptime kv_layout = Layout.row_major(Bc_MMA, D_BUCKET)
    comptime p_layout = Layout.row_major(Bc_MMA, Br_MMA)

    # Per-thread C-fragment coordinates (m16n8k16, f32 C).
    var frag_r0 = lane // 4
    var frag_r1 = frag_r0 + 8
    var frag_c0 = (lane % 4) * 2
    var frag_c1 = frag_c0 + 1

    # dV/dK accumulators persist across all Q tiles. Each warp owns
    # [16 kv rows x D_BUCKET/2 columns].
    var dv_reg = (
        LayoutTensor[
            DType.float32, Layout.row_major(DW_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
        ]
        .stack_allocation()
        .fill(0)
    )
    var dk_reg = (
        LayoutTensor[
            DType.float32, Layout.row_major(DW_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
        ]
        .stack_allocation()
        .fill(0)
    )

    # Load this block's K and V tiles resident (half precision)
    # 16B vectorized loads (ldmatrix-swizzled destination). Per-element
    # fallback for ragged D.
    comptime KV_VECS = Bc_MMA * KV_CHUNKS
    comptime for idx in range((KV_VECS + BLOCK_SIZE_BWD_HALF - 1) // BLOCK_SIZE_BWD_HALF):
        var vslot = tid + idx * BLOCK_SIZE_BWD_HALF
        if vslot < KV_VECS:
            var row_local = vslot // KV_CHUNKS
            var kv_row = kv_row_block + row_local
            var kv_col = (vslot % KV_CHUNKS) * COPY_VEC
            var res_idx = kv_swizzle(vslot) * COPY_VEC
            if kv_row < S and kv_col + COPY_VEC <= D:
                k_smem.store(res_idx, k.load[width=COPY_VEC, alignment=16](qkv_bh + kv_row * HD + kv_col))
                v_smem.store(res_idx, v.load[width=COPY_VEC, alignment=16](qkv_bh + kv_row * HD + kv_col))
            else:
                comptime for e in range(COPY_VEC):
                    var kv_ok = kv_row < S and kv_col + e < D
                    k_smem[res_idx + e] = k[qkv_bh + kv_row * HD + kv_col + e] if kv_ok else Scalar[dtype](0)
                    v_smem[res_idx + e] = v[qkv_bh + kv_row * HD + kv_col + e] if kv_ok else Scalar[dtype](0)
    barrier()

    var k_tensor = LayoutTensor[dtype, kv_layout, address_space=AddressSpace.SHARED](k_smem)
    var v_tensor = LayoutTensor[dtype, kv_layout, address_space=AddressSpace.SHARED](v_smem)
    var q_tensor = LayoutTensor[dtype, kv_layout, address_space=AddressSpace.SHARED](q_smem)
    var do_tensor = LayoutTensor[dtype, kv_layout, address_space=AddressSpace.SHARED](do_smem)
    var p_tensor = LayoutTensor[dtype, p_layout, address_space=AddressSpace.SHARED](p_smem)
    var ds_tensor = LayoutTensor[dtype, p_layout, address_space=AddressSpace.SHARED](ds_smem)
    var dst_tensor = LayoutTensor[dtype, p_layout, address_space=AddressSpace.SHARED](dst_smem)

    # CAUSAL: Q tiles strictly below this KV tile contribute nothing.
    var i_tile_start = 0
    comptime if CAUSAL:
        i_tile_start = kv_row_block // Br_MMA

    for i_tile in range(i_tile_start, i_tiles):
        var ibase = i_tile * Br_MMA

        # Load LSE/delta and this Q tile's Q and dO (half)
        if tid < Br_MMA:
            var q_row = ibase + tid
            lse_smem[tid] = lse[lse_base + q_row] * LOG2E if q_row < S else Float32(0)
            delta_smem[tid] = delta[lse_base + q_row] if q_row < S else Float32(0)

        # Hot per-tile Q/dO staging via cp.async
        comptime Q_VECS = Br_MMA * KV_CHUNKS
        if D % COPY_VEC == 0:
            comptime for idx in range((Q_VECS + BLOCK_SIZE_BWD_HALF - 1) // BLOCK_SIZE_BWD_HALF):
                var vslot = tid + idx * BLOCK_SIZE_BWD_HALF
                if vslot < Q_VECS:
                    var row_local = vslot // KV_CHUNKS
                    var q_row = ibase + row_local
                    var q_col = (vslot % KV_CHUNKS) * COPY_VEC
                    var res_idx = kv_swizzle(vslot) * COPY_VEC
                    var src_bytes = Int32(2 * COPY_VEC) if q_row < S else Int32(0)
                    async_copy[16, fill=Scalar[dtype](0)](
                        (q + qkv_bh + q_row * HD + q_col).address_space_cast[AddressSpace.GLOBAL](),
                        q_smem + res_idx,
                        src_bytes,
                    )
                    async_copy[16, fill=Scalar[dtype](0)](
                        (dy + bh_base + q_row * D + q_col).address_space_cast[AddressSpace.GLOBAL](),
                        do_smem + res_idx,
                        src_bytes,
                    )
            async_copy_commit_group()
            async_copy_wait_all()
        else:
            comptime for idx in range((Q_VECS + BLOCK_SIZE_BWD_HALF - 1) // BLOCK_SIZE_BWD_HALF):
                var vslot = tid + idx * BLOCK_SIZE_BWD_HALF
                if vslot < Q_VECS:
                    var row_local = vslot // KV_CHUNKS
                    var q_row = ibase + row_local
                    var q_col = (vslot % KV_CHUNKS) * COPY_VEC
                    var res_idx = kv_swizzle(vslot) * COPY_VEC
                    comptime for e in range(COPY_VEC):
                        var q_ok = q_row < S and q_col + e < D
                        q_smem[res_idx + e] = q[qkv_bh + q_row * HD + q_col + e] if q_ok else Scalar[dtype](0)
                        do_smem[res_idx + e] = dy[bh_base + q_row * D + q_col + e] if q_ok else Scalar[dtype](0)
        comptime if HAS_BIAS:
            # Bias tile via cp.async. Evict-first when the bias exceeds L2
            # (an oversize mask must not evict the reused K/Q/dO tiles).
            # Ragged S tails zero-fill, S % 8 != 0 uses element staging.
            var mask_oversize = B * H * S * S * size_of[Scalar[dtype]]() > 32 * 1024 * 1024
            var mask_bh_s = b * H * S * S + h * S * S
            comptime MASK_VECS = Br_MMA * (Bc_MMA // COPY_VEC)
            if S % COPY_VEC == 0:
                comptime for idx in range((MASK_VECS + BLOCK_SIZE_BWD_HALF - 1) // BLOCK_SIZE_BWD_HALF):
                    var vslot = tid + idx * BLOCK_SIZE_BWD_HALF
                    if vslot < MASK_VECS:
                        var mq = vslot // (Bc_MMA // COPY_VEC)
                        var mk = (vslot % (Bc_MMA // COPY_VEC)) * COPY_VEC
                        var g_q = ibase + mq
                        var g_k = kv_row_block + mk
                        var res_idx = mq * Bc_MMA + mk
                        var valid = S - g_k
                        valid = COPY_VEC if valid > COPY_VEC else (0 if valid < 0 else valid)
                        var src_bytes = Int32(2 * valid) if g_q < S else Int32(0)
                        if mask_oversize:
                            async_copy[16, fill=Scalar[dtype](0), eviction_policy=CacheEviction.EVICT_FIRST](
                                (mask + mask_bh_s + g_q * S + g_k).address_space_cast[AddressSpace.GLOBAL](),
                                mask_smem + res_idx,
                                src_bytes,
                            )
                        else:
                            async_copy[16, fill=Scalar[dtype](0)](
                                (mask + mask_bh_s + g_q * S + g_k).address_space_cast[AddressSpace.GLOBAL](),
                                mask_smem + res_idx,
                                src_bytes,
                            )
                async_copy_commit_group()
                async_copy_wait_all()
            else:
                comptime for idx in range((MASK_VECS + BLOCK_SIZE_BWD_HALF - 1) // BLOCK_SIZE_BWD_HALF):
                    var vslot = tid + idx * BLOCK_SIZE_BWD_HALF
                    if vslot < MASK_VECS:
                        var mq = vslot // (Bc_MMA // COPY_VEC)
                        var mk = (vslot % (Bc_MMA // COPY_VEC)) * COPY_VEC
                        var g_q = ibase + mq
                        var g_k = kv_row_block + mk
                        var res_idx = mq * Bc_MMA + mk
                        comptime for e in range(COPY_VEC):
                            var m_ok = g_q < S and g_k + e < S
                            mask_smem[res_idx + e] = mask[mask_bh_s + g_q * S + g_k + e] if m_ok else Scalar[dtype](0)
        barrier()

        # score = K @ Q^T and dp = V @ dO^T, one fused K-depth pass
        # Warp owns score/dp rows [warp_r*16, +16), q columns [warp_c*32, +32).
        var score_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(NW_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        var dp_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(NW_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime for dk_t in range(D_TILES):
            var a_k = (
                LayoutTensor[dtype, Layout.row_major(1, a_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL]
                .stack_allocation()
                .vectorize[1, a_frag_size]()
            )
            var a_v = (
                LayoutTensor[dtype, Layout.row_major(1, a_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL]
                .stack_allocation()
                .vectorize[1, a_frag_size]()
            )
            qk_op.load_a[kv_swz](k_tensor.tile[MMA_M, D_BUCKET](warp_r, 0), a_k, dk_t)
            qk_op.load_a[kv_swz](v_tensor.tile[MMA_M, D_BUCKET](warp_r, 0), a_v, dk_t)
            var b_q = (
                LayoutTensor[
                    dtype, Layout.row_major(NW_TILES, b_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
                ]
                .stack_allocation()
                .vectorize[1, b_frag_size]()
            )
            var b_do = (
                LayoutTensor[
                    dtype, Layout.row_major(NW_TILES, b_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
                ]
                .stack_allocation()
                .vectorize[1, b_frag_size]()
            )
            qk_op.load_b(q_tensor.tile[NW_TILES * MMA_N, D_BUCKET](warp_c, 0), b_q, dk_t, 0)
            qk_op.load_b(do_tensor.tile[NW_TILES * MMA_N, D_BUCKET](warp_c, 0), b_do, dk_t, 0)
            qk_op.mma(a_k, b_q, score_reg.vectorize[1, frag_size]())
            qk_op.mma(a_v, b_do, dp_reg.vectorize[1, frag_size]())

        # P and dS per fragment, stored as P, dS [kv, q] and dS^T [q, kv]
        comptime for n_t in range(NW_TILES):
            var n_g = warp_c * NW_TILES + n_t
            var kvr0 = warp_r * MMA_M + frag_r0
            var kvr1 = kvr0 + 8
            var kv0 = kv_row_block + kvr0
            var kv1 = kv_row_block + kvr1
            var q0_local = n_g * MMA_N + frag_c0
            var q1_local = q0_local + 1
            var q0 = ibase + q0_local
            var q1 = ibase + q1_local
            var lse0_l2 = lse_smem[q0_local]
            var lse1_l2 = lse_smem[q1_local]
            var delta0 = delta_smem[q0_local]
            var delta1 = delta_smem[q1_local]
            var score_vec = score_reg.tile[1, frag_size](n_t, 0).vectorize[1, frag_size]()
            var dp_vec = dp_reg.tile[1, frag_size](n_t, 0).vectorize[1, frag_size]()
            var p = rebind[SIMD[DType.float32, 4]](score_vec[0, 0])
            var dp = rebind[SIMD[DType.float32, 4]](dp_vec[0, 0])
            var ok00 = kv0 < S and q0 < S
            var ok01 = kv0 < S and q1 < S
            var ok10 = kv1 < S and q0 < S
            var ok11 = kv1 < S and q1 < S
            var m00 = Float32(0)
            var m01 = Float32(0)
            var m10 = Float32(0)
            var m11 = Float32(0)
            comptime if CAUSAL:
                ok00 = ok00 and kv0 <= q0
                ok01 = ok01 and kv0 <= q1
                ok10 = ok10 and kv1 <= q0
                ok11 = ok11 and kv1 <= q1
            elif HAS_BIAS:
                m00 = Float32(mask_smem[q0_local * Bc_MMA + kvr0]) * LOG2E if ok00 else Float32(0)
                m01 = Float32(mask_smem[q1_local * Bc_MMA + kvr0]) * LOG2E if ok01 else Float32(0)
                m10 = Float32(mask_smem[q0_local * Bc_MMA + kvr1]) * LOG2E if ok10 else Float32(0)
                m11 = Float32(mask_smem[q1_local * Bc_MMA + kvr1]) * LOG2E if ok11 else Float32(0)
            p[0] = exp2(p[0] * scale_log2e + m00 - lse0_l2) if ok00 else Float32(0)
            p[1] = exp2(p[1] * scale_log2e + m01 - lse1_l2) if ok01 else Float32(0)
            p[2] = exp2(p[2] * scale_log2e + m10 - lse0_l2) if ok10 else Float32(0)
            p[3] = exp2(p[3] * scale_log2e + m11 - lse1_l2) if ok11 else Float32(0)
            var ds = SIMD[DType.float32, 4](
                p[0] * (dp[0] - delta0) * scale,
                p[1] * (dp[1] - delta1) * scale,
                p[2] * (dp[2] - delta0) * scale,
                p[3] * (dp[3] - delta1) * scale,
            )
            # Swizzled element stores: (row, col) → swz(row*P_CHUNKS + col/8)*8
            # + col%8. q0/q1 are column-adjacent (same chunk) while kvr0/kvr1 are
            # 8 rows apart (distinct chunks).
            var p_base0 = p_swizzle(kvr0 * P_CHUNKS + q0_local // COPY_VEC) * COPY_VEC
            var p_base1 = p_swizzle(kvr1 * P_CHUNKS + q0_local // COPY_VEC) * COPY_VEC
            var qe0 = q0_local % COPY_VEC
            p_smem.store(p_base0 + qe0, SIMD[dtype, 2](Scalar[dtype](p[0]), Scalar[dtype](p[1])))
            p_smem.store(p_base1 + qe0, SIMD[dtype, 2](Scalar[dtype](p[2]), Scalar[dtype](p[3])))
            ds_smem.store(p_base0 + qe0, SIMD[dtype, 2](Scalar[dtype](ds[0]), Scalar[dtype](ds[1])))
            ds_smem.store(p_base1 + qe0, SIMD[dtype, 2](Scalar[dtype](ds[2]), Scalar[dtype](ds[3])))
            # dS^T is not materialized. The dQ A-operand reads ds_smem
            # (dS[kv,q]) transposed via ldmatrix.trans below, mimicking
            # FA2's sdSt view.
        barrier()

        # dV += P @ dO, dK += dS @ Q, dQ += dS^T @ K
        # Warp owns M rows [warp_r*16, +16) (kv rows for dV/dK, q rows for
        # dQ) and d columns [warp_c*D_BUCKET/2, +D_BUCKET/2).
        var dq_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(DW_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime for kq in range(KQ_TILES):
            var a_p = (
                LayoutTensor[dtype, Layout.row_major(1, a_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL]
                .stack_allocation()
                .vectorize[1, a_frag_size]()
            )
            var a_ds = (
                LayoutTensor[dtype, Layout.row_major(1, a_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL]
                .stack_allocation()
                .vectorize[1, a_frag_size]()
            )
            var a_dst = (
                LayoutTensor[dtype, Layout.row_major(1, a_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL]
                .stack_allocation()
                .vectorize[1, a_frag_size]()
            )
            pv_op.load_a[p_swz](p_tensor.tile[MMA_M, Br_MMA](warp_r, 0), a_p, kq)
            pv_op.load_a[p_swz](ds_tensor.tile[MMA_M, Br_MMA](warp_r, 0), a_ds, kq)
            # a_dst = dS^T[q, kv] read transposed from ds_smem (dS[kv, q])
            # via ldmatrix.trans x4. Lane maps to matrix m = lane//8 and row
            # r = lane%8, giving the A-tile quad (q_half = m%2, kv_half =
            # m//2). The source 8-chunk is contiguous under p_swizzle since
            # q_col is a multiple of 8, so each lane needs one swizzled base.
            var m_idx = lane // 8
            var r_idx = lane % 8
            var t_kv_row = kq * MMA_M + (m_idx // 2) * 8 + r_idx
            var t_q_col = warp_r * MMA_M + (m_idx % 2) * 8
            var t_addr = p_swizzle(t_kv_row * P_CHUNKS + t_q_col // COPY_VEC) * COPY_VEC
            a_dst[0, 0] = rebind[type_of(a_dst[0, 0])](
                ld_matrix[a_frag_size, transpose=True](ds_smem.as_immutable() + t_addr)
            )
            comptime if DW_TILES >= 8:
                # Split the B-fragment batches in half to bound register
                # liveness. A full batch keeps 3 x DW_TILES fragments live at
                # once and spills to local memory at D=128. The 32-column
                # quarter tiles rely on the load_b WN==32 swizzle correction
                # via warp_tile_coord_n.
                comptime DW_HALF = DW_TILES // 2
                comptime for hh in range(2):
                    var b_do = (
                        LayoutTensor[
                            dtype,
                            Layout.row_major(DW_HALF, b_frag_size),
                            MutAnyOrigin,
                            address_space=AddressSpace.LOCAL,
                        ]
                        .stack_allocation()
                        .vectorize[1, b_frag_size]()
                    )
                    var b_q = (
                        LayoutTensor[
                            dtype,
                            Layout.row_major(DW_HALF, b_frag_size),
                            MutAnyOrigin,
                            address_space=AddressSpace.LOCAL,
                        ]
                        .stack_allocation()
                        .vectorize[1, b_frag_size]()
                    )
                    var b_k = (
                        LayoutTensor[
                            dtype,
                            Layout.row_major(DW_HALF, b_frag_size),
                            MutAnyOrigin,
                            address_space=AddressSpace.LOCAL,
                        ]
                        .stack_allocation()
                        .vectorize[1, b_frag_size]()
                    )
                    var n_coord = warp_c * 2 + hh
                    pv_op.load_b(do_tensor.tile[Br_MMA, D_BUCKET // 4](0, n_coord), b_do, kq, n_coord)
                    pv_op.load_b(q_tensor.tile[Br_MMA, D_BUCKET // 4](0, n_coord), b_q, kq, n_coord)
                    pv_op.load_b(k_tensor.tile[Bc_MMA, D_BUCKET // 4](0, n_coord), b_k, kq, n_coord)
                    pv_op.mma(a_p, b_do, dv_reg.tile[DW_HALF, frag_size](hh, 0).vectorize[1, frag_size]())
                    pv_op.mma(a_ds, b_q, dk_reg.tile[DW_HALF, frag_size](hh, 0).vectorize[1, frag_size]())
                    pv_op.mma(a_dst, b_k, dq_reg.tile[DW_HALF, frag_size](hh, 0).vectorize[1, frag_size]())
            else:
                var b_do = (
                    LayoutTensor[
                        dtype, Layout.row_major(DW_TILES, b_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
                    ]
                    .stack_allocation()
                    .vectorize[1, b_frag_size]()
                )
                var b_q = (
                    LayoutTensor[
                        dtype, Layout.row_major(DW_TILES, b_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
                    ]
                    .stack_allocation()
                    .vectorize[1, b_frag_size]()
                )
                var b_k = (
                    LayoutTensor[
                        dtype, Layout.row_major(DW_TILES, b_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
                    ]
                    .stack_allocation()
                    .vectorize[1, b_frag_size]()
                )
                pv_op.load_b(do_tensor.tile[Br_MMA, D_BUCKET // 2](0, warp_c), b_do, kq, warp_c)
                pv_op.load_b(q_tensor.tile[Br_MMA, D_BUCKET // 2](0, warp_c), b_q, kq, warp_c)
                pv_op.load_b(k_tensor.tile[Bc_MMA, D_BUCKET // 2](0, warp_c), b_k, kq, warp_c)
                pv_op.mma(a_p, b_do, dv_reg.vectorize[1, frag_size]())
                pv_op.mma(a_ds, b_q, dk_reg.vectorize[1, frag_size]())
                pv_op.mma(a_dst, b_k, dq_reg.vectorize[1, frag_size]())

        # dQ rows are shared across KV-tile blocks → accumulate into a f32
        # buffer with one-way red.global.add.
        #
        # The reds are coalesced in-register. A raw C-fragment would scatter
        # each warp's red across 8 D-strided rows, so warp shuffles
        # redistribute them. Red k (0..3) gives lane L the target row
        # 4k + L//8 and col L%8, which is 4 rows of 8 contiguous columns.
        # Each value is reduced immediately after its two shuffles because
        # holding a redistributed array live spills (measured). The source
        # lane is (R%8)*4 + C//2 with the slot picked by k and lane parity.
        comptime for dn in range(DW_TILES):
            var dn_g = warp_c * DW_TILES + dn
            var dq_vals = rebind[SIMD[DType.float32, 4]](
                dq_reg.tile[1, frag_size](dn, 0).vectorize[1, frag_size]()[0, 0]
            )
            comptime for k in range(4):
                comptime base = 0 if k < 2 else 2
                var R = 4 * k + (lane // 8)
                var C = lane % 8
                var src = (R % 8) * 4 + (C // 2)
                var s_even = warp_shuffle_idx(dq_vals[base], UInt32(src))
                var s_odd = warp_shuffle_idx(dq_vals[base + 1], UInt32(src))
                var val = s_odd if lane % 2 == 1 else s_even
                var g_row = ibase + warp_r * MMA_M + R
                var g_col = dn_g * MMA_N + C
                if g_row < S and g_col < D:
                    _red_add_f32(dq_accum + qkv_bh + g_row * HD + g_col, val)
        barrier()

    # Write dV and dK (per-lane guarded stores from C fragments)
    comptime for dn in range(DW_TILES):
        var dn_g = warp_c * DW_TILES + dn
        var row0 = kv_row_block + warp_r * MMA_M + frag_r0
        var row1 = row0 + 8
        var col0 = dn_g * MMA_N + frag_c0
        var col1 = col0 + 1
        var dv_vals = rebind[SIMD[DType.float32, 4]](dv_reg.tile[1, frag_size](dn, 0).vectorize[1, frag_size]()[0, 0])
        var dk_vals = rebind[SIMD[DType.float32, 4]](dk_reg.tile[1, frag_size](dn, 0).vectorize[1, frag_size]()[0, 0])
        # C-fragment cols col0 and col1=col0+1 are contiguous, so store the
        # pair as one 4B (2xhalf) write instead of two scalar STG.U16. Rows
        # are 8 apart and stay separate. The ragged-D tail (col1 >= D) falls
        # back to a single scalar store.
        if row0 < S:
            if col1 < D:
                dv.store[width=2, alignment=4](
                    qkv_bh + row0 * HD + col0, SIMD[dtype, 2](Scalar[dtype](dv_vals[0]), Scalar[dtype](dv_vals[1]))
                )
                dk.store[width=2, alignment=4](
                    qkv_bh + row0 * HD + col0, SIMD[dtype, 2](Scalar[dtype](dk_vals[0]), Scalar[dtype](dk_vals[1]))
                )
            elif col0 < D:
                dv[qkv_bh + row0 * HD + col0] = Scalar[dtype](dv_vals[0])
                dk[qkv_bh + row0 * HD + col0] = Scalar[dtype](dk_vals[0])
        if row1 < S:
            if col1 < D:
                dv.store[width=2, alignment=4](
                    qkv_bh + row1 * HD + col0, SIMD[dtype, 2](Scalar[dtype](dv_vals[2]), Scalar[dtype](dv_vals[3]))
                )
                dk.store[width=2, alignment=4](
                    qkv_bh + row1 * HD + col0, SIMD[dtype, 2](Scalar[dtype](dk_vals[2]), Scalar[dtype](dk_vals[3]))
                )
            elif col0 < D:
                dv[qkv_bh + row1 * HD + col0] = Scalar[dtype](dv_vals[2])
                dk[qkv_bh + row1 * HD + col0] = Scalar[dtype](dk_vals[2])


def _flash_attn_bwd_launch_mma_half[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    # delta preprocess -> fused dK/dV/dQ (atomic dQ accum) -> dQ convert.
    var delta_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S)
    var delta_ptr = delta_buf.unsafe_ptr().as_unsafe_any_origin()
    var numel = B * S * H * D
    var dq_accum_buf = ctx.enqueue_create_buffer[DType.float32](numel)
    dq_accum_buf.enqueue_fill(Float32(0))
    var dq_accum_ptr = dq_accum_buf.unsafe_ptr().as_unsafe_any_origin()

    # f32 lse/delta head + K/V/Q/dO half tiles + P/dS/dS^T half tiles +
    # mask tile (non-causal only). Unpadded because the ldmatrix swizzle
    # replaces padding. Fits the shared-memory opt-in at D_BUCKET=128.
    comptime smem_half = (
        4 * Br_MMA
        + 2 * Bc_MMA * D_BUCKET
        + 2 * Br_MMA * D_BUCKET
        + 3 * Bc_MMA * Br_MMA
        + (Br_MMA * Bc_MMA if HAS_BIAS else 0)
    ) * size_of[Scalar[dtype]]()

    var num_tiles_q = (S + Br_MMA - 1) // Br_MMA
    var num_tiles_kv = (S + Bc_MMA - 1) // Bc_MMA

    comptime delta_fn = flash_attn_bwd_delta_kernel[dtype]
    ctx.enqueue_function[delta_fn](
        dy,
        o,
        delta_ptr,
        B,
        S,
        H,
        D,
        grid_dim=(B * H * num_tiles_q,),
        block_dim=(BLOCK_SIZE_MMA,),
    )

    comptime bwd_fn = flash_attn_bwd_kernel_mma_half[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
    ctx.enqueue_function[bwd_fn](
        dy,
        q,
        k,
        v,
        mask,
        lse,
        delta_ptr,
        dk,
        dv,
        dq_accum_ptr,
        B,
        S,
        H,
        D,
        scale,
        grid_dim=(B * H * num_tiles_kv,),
        block_dim=(BLOCK_SIZE_BWD_HALF,),
        shared_mem_bytes=smem_half,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_half)),
    )

    comptime convert_fn = flash_attn_dq_convert_kernel[dtype]
    ctx.enqueue_function[convert_fn](
        dq_accum_ptr,
        dq,
        numel,
        grid_dim=((numel + DQ_CONVERT_BLOCK - 1) // DQ_CONVERT_BLOCK,),
        block_dim=(DQ_CONVERT_BLOCK,),
    )

    _ = delta_buf^
    _ = dq_accum_buf^


def _flash_attn_bwd_launch_mma[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    var delta_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S)
    var delta_ptr = delta_buf.unsafe_ptr().as_unsafe_any_origin()

    # dQ smem: padded KV tile (reused K/V) + Q/dO staging scratch + lse/delta rows.
    comptime smem_dq = (Bc_MMA * (D_BUCKET + KV_MMA_PAD) + Br_MMA * MMA_K + 2 * Br_MMA) * size_of[Float32]()
    # dK/dV smem: lse/delta + K/V stage + Q/dO stage + output unpack tile.
    comptime smem_dkv = (2 * Br_MMA + Bc_MMA * MMA_K + Br_MMA * MMA_K + Bc_MMA * MMA_N) * size_of[Float32]()

    var num_tiles_q = (S + Br_MMA - 1) // Br_MMA
    var num_tiles_kv = (S + Bc_MMA - 1) // Bc_MMA

    comptime dq_fn = flash_attn_dq_kernel_mma[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
    ctx.enqueue_function[dq_fn](
        dy,
        o,
        q,
        k,
        v,
        mask,
        lse,
        dq,
        delta_ptr,
        B,
        S,
        H,
        D,
        scale,
        grid_dim=(B * H * num_tiles_q,),
        block_dim=(BLOCK_SIZE_MMA,),
        shared_mem_bytes=smem_dq,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_dq)),
    )

    comptime dkdv_fn = flash_attn_dkdv_kernel_mma[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
    ctx.enqueue_function[dkdv_fn](
        dy,
        q,
        k,
        v,
        mask,
        lse,
        delta_ptr,
        dk,
        dv,
        B,
        S,
        H,
        D,
        scale,
        grid_dim=(B * H * num_tiles_kv,),
        block_dim=(BLOCK_SIZE_MMA,),
        shared_mem_bytes=smem_dkv,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_dkv)),
    )

    # An over-L2 bias is read evict-first inside both kernels (the split
    # path reads the full mask twice, once per kernel, so pollution control
    # matters double here). A resident bias stays cache-warm instead.

    _ = delta_buf^
