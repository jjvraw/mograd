# NVIDIA forward kernels: TF32 m16n8k8 MMA and sm89-tuned m16n8k16 half.
#
# Note: the simple TensorCore.load_a(tensor)/load_b(tensor) API does NOT emit
# ldmatrix on NVIDIA. It lowers to per-element distribute copies. Always use
# the shared-memory overloads with make_ldmatrix_swizzle, as these kernels do.

from std.gpu import WARP_SIZE, MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx, barrier, syncwarp
from std.gpu.compute.mma import mma
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
from std.memory import stack_allocation
from std.sys import size_of
from std.utils import StaticTuple
from layout import Layout, LayoutTensor
from layout.int_tuple import IntTuple
from layout.swizzle import Swizzle, make_ldmatrix_swizzle
from layout.tensor_core import TensorCore, shape_16x8x8, shape_16x8x16
from mograd.runtime.gpu.kernels.attention.staging import COPY_VEC, chunk_valid, stage_chunk_async, stage_chunk_elems
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


# ===-------------------------------------------------------------------===#
# Forward: MMA path
#
# Grid=(B*H*ceil(S/Br_MMA),), Block=(NUM_WARPS_MMA*WARP_SIZE=128,)
# Each warp owns MMA_M=16 query rows: tile_i*Br_MMA + warp_id*MMA_M .. +15
#
# MMA shape: m16n8k8 TF32 (NVIDIA f32→f32)
#
# Fragment layout per thread `lane` (0..31), NVIDIA TF32 m16n8k8:
#   A (Q/P):  rows {lane//4, lane//4+8}, k-cols {lane%4, lane%4+4}
#             SIMD[f32, 4] = [A[r0,k0], A[r1,k0], A[r0,k0+4], A[r1,k0+4]]
#   B (K^T/V): n-col {lane//4}, k-rows {lane%4, lane%4+4}
#             SIMD[f32, 2] = [B[k0,n], B[k0+4,n]]
#   C/D:      rows {lane//4, lane//4+8}, n-cols {(lane%4)*2, (lane%4)*2+1}
#             SIMD[f32, 4] = [C[r0,n0], C[r0,n1], C[r1,n0], C[r1,n1]]
#
# Online softmax groups: threads {4k, 4k+1, 4k+2, 4k+3} share rows {k, k+8}.
# Row max/sum is reduced across 4 threads using shuffle_xor(1) then xor(2).
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE_MMA)))
@__name(t"flash_attn_fwd_mma_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_fwd_kernel_mma[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert D_BUCKET % MMA_K == 0, "D_BUCKET must be divisible by MMA_K=8"
    comptime assert Bc_MMA % MMA_N == 0, "Bc_MMA must be divisible by MMA_N=8"
    comptime D_TILES = D_BUCKET // MMA_K  # K-depth MMA tiles for QK^T
    comptime N_TILES = Bc_MMA // MMA_N  # N-width MMA tiles per KV tile
    comptime O_TILES = D_BUCKET // MMA_N  # N-width MMA tiles for output
    comptime BC_TILES = Bc_MMA // MMA_K  # K-depth MMA tiles for P×V

    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)

    var num_tiles = (S + Br_MMA - 1) // Br_MMA
    var j_tiles = (S + Bc_MMA - 1) // Bc_MMA
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H

    # This warp's Q row range: [q_row_base, q_row_base + MMA_M)
    var q_row_base = tile_i * Br_MMA + warp_id * MMA_M

    # Per-thread offsets within one MMA tile. TF32 A/B operands use k lanes
    # lane%4 and lane%4+4. C/D output uses adjacent N columns.
    var frag_r0 = lane // 4  # rows 0..7 of the 16-row MMA tile
    var frag_c0 = (lane % 4) * 2
    var frag_k0 = lane % 4

    # Global pointers. Q/K/V are BSHD, so row s of head h lives at qkv_bh + s*HD.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D

    # Load Q A-fragments into registers: [r0,k0], [r1,k0], [r0,k0+4], [r1,k0+4].
    var q_frag = stack_allocation[D_TILES * 4, Float32]()
    comptime for dk in range(D_TILES):
        var qr0 = q_row_base + frag_r0
        var qr1 = qr0 + 8
        var qc0 = dk * MMA_K + frag_k0
        var qc1 = qc0 + 4
        q_frag[dk * 4 + 0] = Float32(q[qkv_bh + qr0 * HD + qc0]) if qr0 < S and qc0 < D else Float32(0)
        q_frag[dk * 4 + 1] = Float32(q[qkv_bh + qr1 * HD + qc0]) if qr1 < S and qc0 < D else Float32(0)
        q_frag[dk * 4 + 2] = Float32(q[qkv_bh + qr0 * HD + qc1]) if qr0 < S and qc1 < D else Float32(0)
        q_frag[dk * 4 + 3] = Float32(q[qkv_bh + qr1 * HD + qc1]) if qr1 < S and qc1 < D else Float32(0)

    # Output O accumulator: O_TILES × SIMD[f32, 4], zero-initialized
    var o_frag = stack_allocation[O_TILES * 4, Float32]()
    comptime for i in range(O_TILES * 4):
        o_frag[i] = Float32(0)

    # Online softmax state: one max/sum per logical row (frag_r0 and frag_r0+8)
    var rowmax0 = Float32(-1e38)
    var rowmax1 = Float32(-1e38)
    var rowsum0 = Float32(0)
    var rowsum1 = Float32(0)

    var scale_l2 = scale * LOG2E

    # Shared memory: K and V alternate using the same buffer (Bc_MMA × D_BUCKET).
    # The tail is per-warp scratch for materializing one 16x8 P tile in C layout
    # before reloading it as a TF32 A operand for P @ V.
    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_fwd_mma_smem"]()
    )
    comptime KV_MMA_STRIDE = D_BUCKET + KV_MMA_PAD
    var p_smem = smem + Bc_MMA * KV_MMA_STRIDE
    var p_tile_smem = p_smem + warp_id * MMA_M * MMA_K
    var pv_op = PVMma()
    comptime kv_layout = Layout(IntTuple(Bc_MMA, D_BUCKET), IntTuple(KV_MMA_STRIDE, 1))
    comptime p_tile_layout = Layout.row_major(MMA_M, MMA_K)

    comptime if CAUSAL:
        # One causal KV tile runs load K, QK^T, scale/mask, the online
        # softmax update, load V (same smem buffer), then O += P@V.
        # Below-diagonal tiles skip masking and causal_diag enables the
        # predicate on the straddling tile. The non-causal (bounds+bias)
        # variant stays INLINE below because routing it through this closure
        # measurably regressed the bias rows.
        @always_inline
        def process_tile(
            j_tile: Int, causal_diag: Bool
        ) {
            imm tid,
            imm S,
            imm D,
            imm HD,
            imm k,
            imm v,
            imm qkv_bh,
            imm smem,
            imm p_tile_smem,
            imm pv_op,
            imm q_frag,
            imm o_frag,
            imm q_row_base,
            imm frag_r0,
            imm frag_c0,
            imm frag_k0,
            imm scale_l2,
            mut rowmax0,
            mut rowmax1,
            mut rowsum0,
            mut rowsum1,
        }:
            # Load K into smem
            comptime for idx in range((Bc_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Bc_MMA * D_BUCKET:
                    var kv_row = j_tile * Bc_MMA + slot // D_BUCKET
                    var kv_col = slot % D_BUCKET
                    smem[(slot // D_BUCKET) * KV_MMA_STRIDE + kv_col] = Float32(
                        k[qkv_bh + kv_row * HD + kv_col]
                    ) if kv_row < S and kv_col < D else Float32(0)
            barrier()

            # QK^T via MMA
            var score = stack_allocation[N_TILES * 4, Float32]()
            comptime for i in range(N_TILES * 4):
                score[i] = Float32(0)

            comptime for n_tile in range(N_TILES):
                comptime for dk in range(D_TILES):
                    # B fragment: B[k,n] = K[n, k] (transposed), col-major
                    var b_n = n_tile * MMA_N + frag_r0  # K row = n-col
                    var b_k0 = dk * MMA_K + frag_k0  # K col = k-row
                    var b_frag = SIMD[DType.float32, 2](
                        smem[b_n * KV_MMA_STRIDE + b_k0],
                        smem[b_n * KV_MMA_STRIDE + b_k0 + 4],
                    )
                    var a_frag = SIMD[DType.float32, 4](
                        q_frag[dk * 4 + 0],
                        q_frag[dk * 4 + 1],
                        q_frag[dk * 4 + 2],
                        q_frag[dk * 4 + 3],
                    )
                    var c_frag = SIMD[DType.float32, 4](
                        score[n_tile * 4 + 0],
                        score[n_tile * 4 + 1],
                        score[n_tile * 4 + 2],
                        score[n_tile * 4 + 3],
                    )
                    mma(c_frag, a_frag, b_frag, c_frag)
                    score[n_tile * 4 + 0] = c_frag[0]
                    score[n_tile * 4 + 1] = c_frag[1]
                    score[n_tile * 4 + 2] = c_frag[2]
                    score[n_tile * 4 + 3] = c_frag[3]

            # Scale, mask, find local row max
            var new_max0 = Float32(-1e38)
            var new_max1 = Float32(-1e38)
            comptime for n_tile in range(N_TILES):
                var s0 = score[n_tile * 4 + 0] * scale_l2
                var s1 = score[n_tile * 4 + 1] * scale_l2
                var s2 = score[n_tile * 4 + 2] * scale_l2
                var s3 = score[n_tile * 4 + 3] * scale_l2
                comptime if CAUSAL:
                    # Mask future positions and out-of-bounds (diagonal tile only)
                    if causal_diag:
                        var i0 = q_row_base + frag_r0
                        var i1 = i0 + 8
                        var j0 = j_tile * Bc_MMA + n_tile * MMA_N + frag_c0
                        var j1 = j0 + 1
                        s0 = s0 if i0 < S and j0 < S and j0 <= i0 else Float32(-1e38)
                        s1 = s1 if i0 < S and j1 < S and j1 <= i0 else Float32(-1e38)
                        s2 = s2 if i1 < S and j0 < S and j0 <= i1 else Float32(-1e38)
                        s3 = s3 if i1 < S and j1 < S and j1 <= i1 else Float32(-1e38)
                score[n_tile * 4 + 0] = s0
                score[n_tile * 4 + 1] = s1
                score[n_tile * 4 + 2] = s2
                score[n_tile * 4 + 3] = s3
                new_max0 = new_max0 if new_max0 > s0 else s0
                new_max0 = new_max0 if new_max0 > s1 else s1
                new_max1 = new_max1 if new_max1 > s2 else s2
                new_max1 = new_max1 if new_max1 > s3 else s3

            # Reduce across 4 threads sharing the same row (xor 1 then 2)
            var peer0 = warp_shuffle_xor(new_max0, UInt32(1))
            new_max0 = new_max0 if new_max0 > peer0 else peer0
            peer0 = warp_shuffle_xor(new_max0, UInt32(2))
            new_max0 = new_max0 if new_max0 > peer0 else peer0
            var peer1 = warp_shuffle_xor(new_max1, UInt32(1))
            new_max1 = new_max1 if new_max1 > peer1 else peer1
            peer1 = warp_shuffle_xor(new_max1, UInt32(2))
            new_max1 = new_max1 if new_max1 > peer1 else peer1

            # Update running max and compute O-rescale correction
            var m_new0 = rowmax0 if rowmax0 > new_max0 else new_max0
            var m_new1 = rowmax1 if rowmax1 > new_max1 else new_max1
            var corr0 = exp2(rowmax0 - m_new0)
            var corr1 = exp2(rowmax1 - m_new1)
            rowmax0 = m_new0
            rowmax1 = m_new1

            # Exponentiate scores, accumulate local sums
            var local_sum0 = Float32(0)
            var local_sum1 = Float32(0)
            comptime for n_tile in range(N_TILES):
                var e0 = exp2(score[n_tile * 4 + 0] - rowmax0)
                var e1 = exp2(score[n_tile * 4 + 1] - rowmax0)
                var e2 = exp2(score[n_tile * 4 + 2] - rowmax1)
                var e3 = exp2(score[n_tile * 4 + 3] - rowmax1)
                score[n_tile * 4 + 0] = e0
                score[n_tile * 4 + 1] = e1
                score[n_tile * 4 + 2] = e2
                score[n_tile * 4 + 3] = e3
                local_sum0 += e0 + e1
                local_sum1 += e2 + e3

            # Reduce rowsum across same 4-thread row group
            var s_peer0 = warp_shuffle_xor(local_sum0, UInt32(1))
            local_sum0 += s_peer0
            s_peer0 = warp_shuffle_xor(local_sum0, UInt32(2))
            local_sum0 += s_peer0
            var s_peer1 = warp_shuffle_xor(local_sum1, UInt32(1))
            local_sum1 += s_peer1
            s_peer1 = warp_shuffle_xor(local_sum1, UInt32(2))
            local_sum1 += s_peer1

            # Rescale O and update running sum
            comptime for ot in range(O_TILES):
                o_frag[ot * 4 + 0] *= corr0
                o_frag[ot * 4 + 1] *= corr0
                o_frag[ot * 4 + 2] *= corr1
                o_frag[ot * 4 + 3] *= corr1
            rowsum0 = rowsum0 * corr0 + local_sum0
            rowsum1 = rowsum1 * corr1 + local_sum1

            # Load V into smem (overwrite K)
            barrier()
            comptime for idx in range((Bc_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = tid + idx * BLOCK_SIZE_MMA
                if slot < Bc_MMA * D_BUCKET:
                    var kv_row = j_tile * Bc_MMA + slot // D_BUCKET
                    var kv_col = slot % D_BUCKET
                    smem[(slot // D_BUCKET) * KV_MMA_STRIDE + kv_col] = Float32(
                        v[qkv_bh + kv_row * HD + kv_col]
                    ) if kv_row < S and kv_col < D else Float32(0)
            barrier()

            # O += softmax(score) × V via MMA
            # score is now P (attention weights), used as A-fragments
            var p_tensor = LayoutTensor[DType.float32, p_tile_layout, address_space=AddressSpace.SHARED](p_tile_smem)
            var v_tensor = LayoutTensor[DType.float32, kv_layout, address_space=AddressSpace.SHARED](smem)
            comptime for bc_tile in range(BC_TILES):
                # A: P[16, 8] slice at bc_tile. score is C-layout, so materialize
                # it as a 16x8 tile once, then reuse it for every output d tile.
                var p_col = frag_c0
                p_tile_smem[frag_r0 * MMA_K + p_col] = score[bc_tile * 4 + 0]
                p_tile_smem[frag_r0 * MMA_K + p_col + 1] = score[bc_tile * 4 + 1]
                p_tile_smem[(frag_r0 + 8) * MMA_K + p_col] = score[bc_tile * 4 + 2]
                p_tile_smem[(frag_r0 + 8) * MMA_K + p_col + 1] = score[bc_tile * 4 + 3]
                syncwarp()
                var a_p = pv_op.load_a(p_tensor.tile[MMA_M, MMA_K](0, 0))
                comptime for d_tile in range(O_TILES):
                    var c_tile = LayoutTensor[
                        DType.float32,
                        Layout.row_major(1, PVMma.c_reg_type.size),
                        MutAnyOrigin,
                        address_space=AddressSpace.LOCAL,
                    ].stack_allocation()
                    var c_vec = c_tile.vectorize[1, PVMma.c_reg_type.size]()
                    c_vec[0, 0] = rebind[type_of(c_vec[0, 0])](
                        SIMD[DType.float32, 4](
                            o_frag[d_tile * 4 + 0],
                            o_frag[d_tile * 4 + 1],
                            o_frag[d_tile * 4 + 2],
                            o_frag[d_tile * 4 + 3],
                        )
                    )
                    var b_v = pv_op.load_b(v_tensor.tile[MMA_K, MMA_N](bc_tile, d_tile))
                    var c_out = pv_op.mma_op(a_p, b_v, c_tile)
                    var o = rebind[SIMD[DType.float32, 4]](c_out.vectorize[1, PVMma.c_reg_type.size]()[0, 0])
                    o_frag[d_tile * 4 + 0] = o[0]
                    o_frag[d_tile * 4 + 1] = o[1]
                    o_frag[d_tile * 4 + 2] = o[2]
                    o_frag[d_tile * 4 + 3] = o[3]
            barrier()  # done with V smem before next iteration overwrites K smem

        # diag_tile is the last KV tile that can contribute to this Q tile.
        # Tiles fully below it need no masking. Tiles above it contribute
        # nothing and are skipped.
        var diag_tile = (tile_i * Br_MMA + Br_MMA - 1) // Bc_MMA
        for j_tile in range(diag_tile):
            process_tile(j_tile, False)
        if diag_tile < j_tiles:
            process_tile(diag_tile, True)
    else:
        # Non-causal: iterate all KV tiles
        for j_tile in range(j_tiles):
            # Load K
            comptime for idx in range((Bc_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = Int(thread_idx.x) + idx * BLOCK_SIZE_MMA
                if slot < Bc_MMA * D_BUCKET:
                    var kv_row = j_tile * Bc_MMA + slot // D_BUCKET
                    var kv_col = slot % D_BUCKET
                    smem[(slot // D_BUCKET) * KV_MMA_STRIDE + kv_col] = Float32(
                        k[qkv_bh + kv_row * HD + kv_col]
                    ) if kv_row < S and kv_col < D else Float32(0)
            barrier()

            var score = stack_allocation[N_TILES * 4, Float32]()
            comptime for i in range(N_TILES * 4):
                score[i] = Float32(0)

            comptime for n_tile in range(N_TILES):
                comptime for dk in range(D_TILES):
                    var b_n = n_tile * MMA_N + frag_r0
                    var b_k0 = dk * MMA_K + frag_k0
                    var b_frag = SIMD[DType.float32, 2](
                        smem[b_n * KV_MMA_STRIDE + b_k0],
                        smem[b_n * KV_MMA_STRIDE + b_k0 + 4],
                    )
                    var a_frag = SIMD[DType.float32, 4](
                        q_frag[dk * 4 + 0],
                        q_frag[dk * 4 + 1],
                        q_frag[dk * 4 + 2],
                        q_frag[dk * 4 + 3],
                    )
                    var c_frag = SIMD[DType.float32, 4](
                        score[n_tile * 4 + 0],
                        score[n_tile * 4 + 1],
                        score[n_tile * 4 + 2],
                        score[n_tile * 4 + 3],
                    )
                    mma(c_frag, a_frag, b_frag, c_frag)
                    score[n_tile * 4 + 0] = c_frag[0]
                    score[n_tile * 4 + 1] = c_frag[1]
                    score[n_tile * 4 + 2] = c_frag[2]
                    score[n_tile * 4 + 3] = c_frag[3]

            # Scale, add additive bias (mask), find local max
            var new_max0 = Float32(-1e38)
            var new_max1 = Float32(-1e38)
            comptime for n_tile in range(N_TILES):
                var i0 = q_row_base + frag_r0
                var i1 = i0 + 8
                var j0 = j_tile * Bc_MMA + n_tile * MMA_N + frag_c0
                var j1 = j0 + 1
                var s0 = score[n_tile * 4 + 0] * scale_l2
                var s1 = score[n_tile * 4 + 1] * scale_l2
                var s2 = score[n_tile * 4 + 2] * scale_l2
                var s3 = score[n_tile * 4 + 3] * scale_l2
                # Mask out-of-bounds positions
                s0 = s0 if j0 < S and i0 < S else Float32(-1e38)
                s1 = s1 if j1 < S and i0 < S else Float32(-1e38)
                s2 = s2 if j0 < S and i1 < S else Float32(-1e38)
                s3 = s3 if j1 < S and i1 < S else Float32(-1e38)
                comptime if HAS_BIAS:
                    # Add attention bias from mask tensor (additive bias, e.g. ALiBi)
                    var mask_bh = b * H * S * S + h * S * S
                    s0 += Float32(mask[mask_bh + i0 * S + j0]) * LOG2E if j0 < S and i0 < S else Float32(0)
                    s1 += Float32(mask[mask_bh + i0 * S + j1]) * LOG2E if j1 < S and i0 < S else Float32(0)
                    s2 += Float32(mask[mask_bh + i1 * S + j0]) * LOG2E if j0 < S and i1 < S else Float32(0)
                    s3 += Float32(mask[mask_bh + i1 * S + j1]) * LOG2E if j1 < S and i1 < S else Float32(0)
                score[n_tile * 4 + 0] = s0
                score[n_tile * 4 + 1] = s1
                score[n_tile * 4 + 2] = s2
                score[n_tile * 4 + 3] = s3
                new_max0 = new_max0 if new_max0 > s0 else s0
                new_max0 = new_max0 if new_max0 > s1 else s1
                new_max1 = new_max1 if new_max1 > s2 else s2
                new_max1 = new_max1 if new_max1 > s3 else s3

            var peer0 = warp_shuffle_xor(new_max0, UInt32(1))
            new_max0 = new_max0 if new_max0 > peer0 else peer0
            peer0 = warp_shuffle_xor(new_max0, UInt32(2))
            new_max0 = new_max0 if new_max0 > peer0 else peer0
            var peer1 = warp_shuffle_xor(new_max1, UInt32(1))
            new_max1 = new_max1 if new_max1 > peer1 else peer1
            peer1 = warp_shuffle_xor(new_max1, UInt32(2))
            new_max1 = new_max1 if new_max1 > peer1 else peer1

            var m_new0 = rowmax0 if rowmax0 > new_max0 else new_max0
            var m_new1 = rowmax1 if rowmax1 > new_max1 else new_max1
            var corr0 = exp2(rowmax0 - m_new0)
            var corr1 = exp2(rowmax1 - m_new1)
            rowmax0 = m_new0
            rowmax1 = m_new1

            var local_sum0 = Float32(0)
            var local_sum1 = Float32(0)
            comptime for n_tile in range(N_TILES):
                var e0 = exp2(score[n_tile * 4 + 0] - rowmax0)
                var e1 = exp2(score[n_tile * 4 + 1] - rowmax0)
                var e2 = exp2(score[n_tile * 4 + 2] - rowmax1)
                var e3 = exp2(score[n_tile * 4 + 3] - rowmax1)
                score[n_tile * 4 + 0] = e0
                score[n_tile * 4 + 1] = e1
                score[n_tile * 4 + 2] = e2
                score[n_tile * 4 + 3] = e3
                local_sum0 += e0 + e1
                local_sum1 += e2 + e3

            var s_peer0 = warp_shuffle_xor(local_sum0, UInt32(1))
            local_sum0 += s_peer0
            s_peer0 = warp_shuffle_xor(local_sum0, UInt32(2))
            local_sum0 += s_peer0
            var s_peer1 = warp_shuffle_xor(local_sum1, UInt32(1))
            local_sum1 += s_peer1
            s_peer1 = warp_shuffle_xor(local_sum1, UInt32(2))
            local_sum1 += s_peer1

            comptime for ot in range(O_TILES):
                o_frag[ot * 4 + 0] *= corr0
                o_frag[ot * 4 + 1] *= corr0
                o_frag[ot * 4 + 2] *= corr1
                o_frag[ot * 4 + 3] *= corr1
            rowsum0 = rowsum0 * corr0 + local_sum0
            rowsum1 = rowsum1 * corr1 + local_sum1

            # Load V
            barrier()
            comptime for idx in range((Bc_MMA * D_BUCKET + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var slot = Int(thread_idx.x) + idx * BLOCK_SIZE_MMA
                if slot < Bc_MMA * D_BUCKET:
                    var kv_row = j_tile * Bc_MMA + slot // D_BUCKET
                    var kv_col = slot % D_BUCKET
                    smem[(slot // D_BUCKET) * KV_MMA_STRIDE + kv_col] = Float32(
                        v[qkv_bh + kv_row * HD + kv_col]
                    ) if kv_row < S and kv_col < D else Float32(0)
            barrier()

            var p_tensor = LayoutTensor[DType.float32, p_tile_layout, address_space=AddressSpace.SHARED](p_tile_smem)
            var v_tensor = LayoutTensor[DType.float32, kv_layout, address_space=AddressSpace.SHARED](smem)
            comptime for bc_tile in range(BC_TILES):
                var p_col = frag_c0
                p_tile_smem[frag_r0 * MMA_K + p_col] = score[bc_tile * 4 + 0]
                p_tile_smem[frag_r0 * MMA_K + p_col + 1] = score[bc_tile * 4 + 1]
                p_tile_smem[(frag_r0 + 8) * MMA_K + p_col] = score[bc_tile * 4 + 2]
                p_tile_smem[(frag_r0 + 8) * MMA_K + p_col + 1] = score[bc_tile * 4 + 3]
                syncwarp()
                var a_p = pv_op.load_a(p_tensor.tile[MMA_M, MMA_K](0, 0))
                comptime for d_tile in range(O_TILES):
                    var c_tile = LayoutTensor[
                        DType.float32,
                        Layout.row_major(1, PVMma.c_reg_type.size),
                        MutAnyOrigin,
                        address_space=AddressSpace.LOCAL,
                    ].stack_allocation()
                    var c_vec = c_tile.vectorize[1, PVMma.c_reg_type.size]()
                    c_vec[0, 0] = rebind[type_of(c_vec[0, 0])](
                        SIMD[DType.float32, 4](
                            o_frag[d_tile * 4 + 0],
                            o_frag[d_tile * 4 + 1],
                            o_frag[d_tile * 4 + 2],
                            o_frag[d_tile * 4 + 3],
                        )
                    )
                    var b_v = pv_op.load_b(v_tensor.tile[MMA_K, MMA_N](bc_tile, d_tile))
                    var c_out = pv_op.mma_op(a_p, b_v, c_tile)
                    var o = rebind[SIMD[DType.float32, 4]](c_out.vectorize[1, PVMma.c_reg_type.size]()[0, 0])
                    o_frag[d_tile * 4 + 0] = o[0]
                    o_frag[d_tile * 4 + 1] = o[1]
                    o_frag[d_tile * 4 + 2] = o[2]
                    o_frag[d_tile * 4 + 3] = o[3]
            barrier()

    # Write normalized output
    # Thread lane owns:
    #   row0 = q_row_base + frag_r0 = q_row_base + lane//4
    #   row1 = row0 + 8
    # For each d_tile: cols n0=(lane%4)*2, n1=n0+1
    var o_r0 = q_row_base + frag_r0
    var o_r1 = o_r0 + 8
    var o_base = b * H * S * D + h * S * D
    var inv0 = Float32(1) / rowsum0
    var inv1 = Float32(1) / rowsum1
    comptime for d_tile in range(O_TILES):
        var dc0 = d_tile * MMA_N + frag_c0
        var dc1 = dc0 + 1
        if o_r0 < S and dc0 < D:
            dst[o_base + o_r0 * D + dc0] = Scalar[dtype](o_frag[d_tile * 4 + 0] * inv0)
        if o_r0 < S and dc1 < D:
            dst[o_base + o_r0 * D + dc1] = Scalar[dtype](o_frag[d_tile * 4 + 1] * inv0)
        if o_r1 < S and dc0 < D:
            dst[o_base + o_r1 * D + dc0] = Scalar[dtype](o_frag[d_tile * 4 + 2] * inv1)
        if o_r1 < S and dc1 < D:
            dst[o_base + o_r1 * D + dc1] = Scalar[dtype](o_frag[d_tile * 4 + 3] * inv1)

    # Write LSE (one thread per row pair, lane%4==0 is the canonical writer)
    if lane % 4 == 0:
        var lse_base = b * H * S + h * S
        if o_r0 < S:
            lse[lse_base + o_r0] = rowmax0 * LN2 + log(rowsum0)
        if o_r1 < S:
            lse[lse_base + o_r1] = rowmax1 * LN2 + log(rowsum1)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE_MMA)))
# Cap registers so a third 128-thread block fits per SM: measured faster
# despite the spills it forces (occupancy beats spills on this latency-bound kernel).
@__llvm_metadata(`nvvm.maxnreg`=SIMDLength(FWD_HALF_MAXNREG))
@__name(
    t"flash_attn_fwd_mma_half_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}_ragged_{RAGGED_D}_ovs_{MASK_OVERSIZE}"
)
def flash_attn_fwd_kernel_mma_half[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool, RAGGED_D: Bool, MASK_OVERSIZE: Bool
](
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert dtype.is_half_float(), "half MMA path requires float16 or bfloat16 inputs"
    comptime HMMA_K = 16
    comptime D_TILES = D_BUCKET // HMMA_K
    comptime N_TILES = Bc_HMMA // MMA_N
    comptime O_TILES = D_BUCKET // MMA_N
    comptime BC_TILES = Bc_HMMA // HMMA_K
    comptime QKHalf = TensorCore[DType.float32, dtype, shape_16x8x16, True]
    comptime PVHalf = TensorCore[DType.float32, dtype, shape_16x8x16, False]
    comptime frag_size = QKHalf.c_reg_type.size

    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)

    var num_tiles = (S + Br_MMA - 1) // Br_MMA
    var j_tiles = (S + Bc_HMMA - 1) // Bc_HMMA
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H
    var q_row_base = tile_i * Br_MMA + warp_id * MMA_M
    var frag_r0 = lane // 4
    var frag_c0 = (lane % 4) * 2

    # Q/K/V are BSHD, so row s of head h lives at qkv_bh + s*HD.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D

    var o_reg = (
        LayoutTensor[
            DType.float32,
            Layout.row_major(O_TILES, QKHalf.c_reg_type.size),
            MutAnyOrigin,
            address_space=AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    var rowmax0 = Float32(-1e38)
    var rowmax1 = Float32(-1e38)
    var rowsum0 = Float32(0)
    var rowsum1 = Float32(0)
    var scale_l2 = scale * LOG2E

    # Unpadded, ldmatrix-swizzled smem tiles. The write/read contract
    # matches the fused backward in nvidia_bwd.mojo. Ragged S rows and the
    # ragged-D column tail are zero-filled, so D <= D_BUCKET is supported.
    var smem = rebind[UnsafePointer[Scalar[dtype], MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Scalar[dtype], address_space=AddressSpace.SHARED, alignment=16, name="attn_fwd_mma_half_smem"]()
    )
    comptime D_CHUNKS = D_BUCKET // COPY_VEC
    comptime fwd_swizzle = make_ldmatrix_swizzle[dtype, D_BUCKET]()
    comptime fwd_swz: Optional[Swizzle] = fwd_swizzle
    comptime a_frag_size = QKHalf.a_reg_type.size
    comptime b_frag_size = QKHalf.b_reg_type.size
    var q_smem = smem
    var k_smem = q_smem + Br_MMA * D_BUCKET
    var v_smem = k_smem + Bc_HMMA * D_BUCKET
    # Additive-bias tile [q, kv], plain layout, staged per KV tile with
    # coalesced evict-first vector loads (the per-fragment scalar global
    # reads it replaces ran at half the achievable mask bandwidth).
    var mask_smem = v_smem + Bc_HMMA * D_BUCKET

    comptime q_full_layout = Layout.row_major(Br_MMA, D_BUCKET)
    comptime kv_layout = Layout.row_major(Bc_HMMA, D_BUCKET)
    comptime p_reg_layout = Layout.col_major(1, PVHalf.a_reg_type.size)
    var qk_op = QKHalf()
    var pv_op = PVHalf()

    # cp.async staging bypasses the register file (no scoreboard entries or
    # LSU issue slots). OOB rows and ragged-D tails zero-fill via src_size.
    # Rows not 16B aligned (D % 8 != 0) fall back to element staging.
    comptime Q_VECS_F = Br_MMA * D_CHUNKS
    comptime if RAGGED_D:
        if D % COPY_VEC == 0:
            comptime for idx in range((Q_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var vslot = tid + idx * BLOCK_SIZE_MMA
                if vslot < Q_VECS_F:
                    var qr = tile_i * Br_MMA + vslot // D_CHUNKS
                    var qc = (vslot % D_CHUNKS) * COPY_VEC
                    var res_idx = fwd_swizzle(vslot) * COPY_VEC
                    stage_chunk_async(q + qkv_bh + qr * HD + qc, q_smem + res_idx, chunk_valid(qr < S, qc, D))
            async_copy_commit_group()
            async_copy_wait_all()
        else:
            comptime for idx in range((Q_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var vslot = tid + idx * BLOCK_SIZE_MMA
                if vslot < Q_VECS_F:
                    var qr = tile_i * Br_MMA + vslot // D_CHUNKS
                    var qc = (vslot % D_CHUNKS) * COPY_VEC
                    var res_idx = fwd_swizzle(vslot) * COPY_VEC
                    stage_chunk_elems(q + qkv_bh + qr * HD + qc, q_smem + res_idx, qr < S, qc, D)
    else:
        comptime for idx in range((Q_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
            var vslot = tid + idx * BLOCK_SIZE_MMA
            if vslot < Q_VECS_F:
                var qr = tile_i * Br_MMA + vslot // D_CHUNKS
                var qc = (vslot % D_CHUNKS) * COPY_VEC
                var res_idx = fwd_swizzle(vslot) * COPY_VEC
                stage_chunk_async(q + qkv_bh + qr * HD + qc, q_smem + res_idx, COPY_VEC if qr < S else 0)
        async_copy_commit_group()
        async_copy_wait_all()
    barrier()

    @always_inline
    def qk_tile(
        j_tile: Int, causal_diag: Bool, score: UnsafePointer[Float32, MutAnyOrigin]
    ) {
        imm tid,
        imm D,
        imm HD,
        imm S,
        imm k,
        imm qkv_bh,
        imm q_smem,
        imm k_smem,
        imm mask_smem,
        imm tile_i,
        imm warp_id,
        imm qk_op,
        imm q_row_base,
        imm frag_r0,
        imm frag_c0,
        imm scale_l2,
        imm mask,
        imm b,
        imm H,
        imm h,
    }:
        comptime KV_VECS_F = Bc_HMMA * D_CHUNKS
        comptime if RAGGED_D:
            if D % COPY_VEC == 0:
                comptime for idx in range((KV_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                    var vslot = tid + idx * BLOCK_SIZE_MMA
                    if vslot < KV_VECS_F:
                        var kv_row = j_tile * Bc_HMMA + vslot // D_CHUNKS
                        var kv_col = (vslot % D_CHUNKS) * COPY_VEC
                        var res_idx = fwd_swizzle(vslot) * COPY_VEC
                        stage_chunk_async(
                            k + qkv_bh + kv_row * HD + kv_col, k_smem + res_idx, chunk_valid(kv_row < S, kv_col, D)
                        )
            else:
                comptime for idx in range((KV_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                    var vslot = tid + idx * BLOCK_SIZE_MMA
                    if vslot < KV_VECS_F:
                        var kv_row = j_tile * Bc_HMMA + vslot // D_CHUNKS
                        var kv_col = (vslot % D_CHUNKS) * COPY_VEC
                        var res_idx = fwd_swizzle(vslot) * COPY_VEC
                        stage_chunk_elems(k + qkv_bh + kv_row * HD + kv_col, k_smem + res_idx, kv_row < S, kv_col, D)
        else:
            comptime for idx in range((KV_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var vslot = tid + idx * BLOCK_SIZE_MMA
                if vslot < KV_VECS_F:
                    var kv_row = j_tile * Bc_HMMA + vslot // D_CHUNKS
                    var kv_col = (vslot % D_CHUNKS) * COPY_VEC
                    var res_idx = fwd_swizzle(vslot) * COPY_VEC
                    stage_chunk_async(
                        k + qkv_bh + kv_row * HD + kv_col, k_smem + res_idx, COPY_VEC if kv_row < S else 0
                    )
        # With MASK_OVERSIZE, an over-L2 bias is staged via evict-first smem so it
        # cannot evict the reused Q/K/V tiles, a resident bias reads directly.
        comptime if HAS_BIAS and MASK_OVERSIZE:
            var mask_bh_s = b * H * S * S + h * S * S
            comptime MASK_VECS = Br_MMA * (Bc_HMMA // COPY_VEC)
            comptime for idx in range((MASK_VECS + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var vslot = tid + idx * BLOCK_SIZE_MMA
                if vslot < MASK_VECS:
                    var mq = vslot // (Bc_HMMA // COPY_VEC)
                    var mk = (vslot % (Bc_HMMA // COPY_VEC)) * COPY_VEC
                    var g_q = tile_i * Br_MMA + mq
                    var g_k = j_tile * Bc_HMMA + mk
                    var res_idx = mq * Bc_HMMA + mk
                    if g_q < S and g_k + COPY_VEC <= S:
                        mask_smem.store(
                            res_idx,
                            gpu_load[width=COPY_VEC, eviction_policy=CacheEviction.EVICT_FIRST](
                                mask + mask_bh_s + g_q * S + g_k
                            ),
                        )
                    else:
                        stage_chunk_elems(mask + mask_bh_s + g_q * S + g_k, mask_smem + res_idx, g_q < S, g_k, S)
        async_copy_commit_group()
        async_copy_wait_all()
        barrier()

        var q_tensor = LayoutTensor[dtype, q_full_layout, address_space=AddressSpace.SHARED](q_smem)
        var k_tensor = LayoutTensor[dtype, kv_layout, address_space=AddressSpace.SHARED](k_smem)
        var score_reg = (
            LayoutTensor[
                DType.float32, Layout.row_major(N_TILES, frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
            ]
            .stack_allocation()
            .fill(0)
        )
        comptime for dk in range(D_TILES):
            var a_q = (
                LayoutTensor[dtype, Layout.row_major(1, a_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL]
                .stack_allocation()
                .vectorize[1, a_frag_size]()
            )
            qk_op.load_a[fwd_swz](q_tensor.tile[MMA_M, D_BUCKET](warp_id, 0), a_q, dk)
            var b_k = (
                LayoutTensor[
                    dtype, Layout.row_major(N_TILES, b_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
                ]
                .stack_allocation()
                .vectorize[1, b_frag_size]()
            )
            qk_op.load_b(k_tensor, b_k, dk, 0)
            qk_op.mma(a_q, b_k, score_reg.vectorize[1, frag_size]())
        comptime for n_tile in range(N_TILES):
            var c_vec = score_reg.tile[1, frag_size](n_tile, 0).vectorize[1, frag_size]()
            var s = rebind[SIMD[DType.float32, 4]](c_vec[0, 0])
            var i0 = q_row_base + frag_r0
            var i1 = i0 + 8
            var j0 = j_tile * Bc_HMMA + n_tile * MMA_N + frag_c0
            var j1 = j0 + 1
            var s0 = s[0] * scale_l2
            var s1 = s[1] * scale_l2
            var s2 = s[2] * scale_l2
            var s3 = s[3] * scale_l2
            comptime if CAUSAL:
                if causal_diag:
                    s0 = s0 if i0 < S and j0 < S and j0 <= i0 else Float32(-1e38)
                    s1 = s1 if i0 < S and j1 < S and j1 <= i0 else Float32(-1e38)
                    s2 = s2 if i1 < S and j0 < S and j0 <= i1 else Float32(-1e38)
                    s3 = s3 if i1 < S and j1 < S and j1 <= i1 else Float32(-1e38)
            else:
                s0 = s0 if i0 < S and j0 < S else Float32(-1e38)
                s1 = s1 if i0 < S and j1 < S else Float32(-1e38)
                s2 = s2 if i1 < S and j0 < S else Float32(-1e38)
                s3 = s3 if i1 < S and j1 < S else Float32(-1e38)
                comptime if HAS_BIAS and MASK_OVERSIZE:
                    var q0_l = warp_id * MMA_M + frag_r0
                    var q1_l = q0_l + 8
                    var k0_l = n_tile * MMA_N + frag_c0
                    var k1_l = k0_l + 1
                    s0 += Float32(mask_smem[q0_l * Bc_HMMA + k0_l]) * LOG2E if i0 < S and j0 < S else Float32(0)
                    s1 += Float32(mask_smem[q0_l * Bc_HMMA + k1_l]) * LOG2E if i0 < S and j1 < S else Float32(0)
                    s2 += Float32(mask_smem[q1_l * Bc_HMMA + k0_l]) * LOG2E if i1 < S and j0 < S else Float32(0)
                    s3 += Float32(mask_smem[q1_l * Bc_HMMA + k1_l]) * LOG2E if i1 < S and j1 < S else Float32(0)
                elif HAS_BIAS:
                    var mask_bh = b * H * S * S + h * S * S
                    s0 += Float32(mask[mask_bh + i0 * S + j0]) * LOG2E if i0 < S and j0 < S else Float32(0)
                    s1 += Float32(mask[mask_bh + i0 * S + j1]) * LOG2E if i0 < S and j1 < S else Float32(0)
                    s2 += Float32(mask[mask_bh + i1 * S + j0]) * LOG2E if i1 < S and j0 < S else Float32(0)
                    s3 += Float32(mask[mask_bh + i1 * S + j1]) * LOG2E if i1 < S and j1 < S else Float32(0)
            score[n_tile * frag_size + 0] = s0
            score[n_tile * frag_size + 1] = s1
            score[n_tile * frag_size + 2] = s2
            score[n_tile * frag_size + 3] = s3
        barrier()

    @always_inline
    def softmax_pv(
        j_tile: Int, score: UnsafePointer[Float32, MutAnyOrigin]
    ) {
        imm tid,
        imm D,
        imm HD,
        imm H,
        imm S,
        imm v,
        imm qkv_bh,
        imm v_smem,
        imm frag_r0,
        imm frag_c0,
        imm q_row_base,
        imm pv_op,
        mut rowmax0,
        mut rowmax1,
        mut rowsum0,
        mut rowsum1,
        mut o_reg,
    }:
        # V is issued via cp.async at entry and waited only after the
        # softmax statistics, so the reduction work hides the copy latency,
        # an overlap register-path staging could not express.
        comptime KV_VECS_F = Bc_HMMA * D_CHUNKS
        comptime if RAGGED_D:
            if D % COPY_VEC == 0:
                comptime for idx in range((KV_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                    var vslot = tid + idx * BLOCK_SIZE_MMA
                    if vslot < KV_VECS_F:
                        var kv_row = j_tile * Bc_HMMA + vslot // D_CHUNKS
                        var kv_col = (vslot % D_CHUNKS) * COPY_VEC
                        var res_idx = fwd_swizzle(vslot) * COPY_VEC
                        stage_chunk_async(
                            v + qkv_bh + kv_row * HD + kv_col, v_smem + res_idx, chunk_valid(kv_row < S, kv_col, D)
                        )
            else:
                comptime for idx in range((KV_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                    var vslot = tid + idx * BLOCK_SIZE_MMA
                    if vslot < KV_VECS_F:
                        var kv_row = j_tile * Bc_HMMA + vslot // D_CHUNKS
                        var kv_col = (vslot % D_CHUNKS) * COPY_VEC
                        var res_idx = fwd_swizzle(vslot) * COPY_VEC
                        stage_chunk_elems(v + qkv_bh + kv_row * HD + kv_col, v_smem + res_idx, kv_row < S, kv_col, D)
        else:
            comptime for idx in range((KV_VECS_F + BLOCK_SIZE_MMA - 1) // BLOCK_SIZE_MMA):
                var vslot = tid + idx * BLOCK_SIZE_MMA
                if vslot < KV_VECS_F:
                    var kv_row = j_tile * Bc_HMMA + vslot // D_CHUNKS
                    var kv_col = (vslot % D_CHUNKS) * COPY_VEC
                    var res_idx = fwd_swizzle(vslot) * COPY_VEC
                    stage_chunk_async(
                        v + qkv_bh + kv_row * HD + kv_col, v_smem + res_idx, COPY_VEC if kv_row < S else 0
                    )
        async_copy_commit_group()

        var new_max0 = Float32(-1e38)
        var new_max1 = Float32(-1e38)
        comptime for n_tile in range(N_TILES):
            var s0 = score[n_tile * frag_size + 0]
            var s1 = score[n_tile * frag_size + 1]
            var s2 = score[n_tile * frag_size + 2]
            var s3 = score[n_tile * frag_size + 3]
            new_max0 = new_max0 if new_max0 > s0 else s0
            new_max0 = new_max0 if new_max0 > s1 else s1
            new_max1 = new_max1 if new_max1 > s2 else s2
            new_max1 = new_max1 if new_max1 > s3 else s3

        var peer0 = warp_shuffle_xor(new_max0, UInt32(1))
        new_max0 = new_max0 if new_max0 > peer0 else peer0
        peer0 = warp_shuffle_xor(new_max0, UInt32(2))
        new_max0 = new_max0 if new_max0 > peer0 else peer0
        var peer1 = warp_shuffle_xor(new_max1, UInt32(1))
        new_max1 = new_max1 if new_max1 > peer1 else peer1
        peer1 = warp_shuffle_xor(new_max1, UInt32(2))
        new_max1 = new_max1 if new_max1 > peer1 else peer1

        var m_new0 = rowmax0 if rowmax0 > new_max0 else new_max0
        var m_new1 = rowmax1 if rowmax1 > new_max1 else new_max1
        var corr0 = exp2(rowmax0 - m_new0)
        var corr1 = exp2(rowmax1 - m_new1)
        rowmax0 = m_new0
        rowmax1 = m_new1

        var local_sum0 = Float32(0)
        var local_sum1 = Float32(0)
        comptime for n_tile in range(N_TILES):
            var i0 = q_row_base + frag_r0
            var i1 = i0 + 8
            var j0 = j_tile * Bc_HMMA + n_tile * MMA_N + frag_c0
            var j1 = j0 + 1
            var e0 = exp2(score[n_tile * frag_size + 0] - rowmax0)
            var e1 = exp2(score[n_tile * frag_size + 1] - rowmax0)
            var e2 = exp2(score[n_tile * frag_size + 2] - rowmax1)
            var e3 = exp2(score[n_tile * frag_size + 3] - rowmax1)
            comptime if CAUSAL:
                e0 = e0 if i0 < S and j0 < S and j0 <= i0 else Float32(0)
                e1 = e1 if i0 < S and j1 < S and j1 <= i0 else Float32(0)
                e2 = e2 if i1 < S and j0 < S and j0 <= i1 else Float32(0)
                e3 = e3 if i1 < S and j1 < S and j1 <= i1 else Float32(0)
            else:
                e0 = e0 if i0 < S and j0 < S else Float32(0)
                e1 = e1 if i0 < S and j1 < S else Float32(0)
                e2 = e2 if i1 < S and j0 < S else Float32(0)
                e3 = e3 if i1 < S and j1 < S else Float32(0)
            score[n_tile * frag_size + 0] = e0
            score[n_tile * frag_size + 1] = e1
            score[n_tile * frag_size + 2] = e2
            score[n_tile * frag_size + 3] = e3
            local_sum0 += e0 + e1
            local_sum1 += e2 + e3

        var s_peer0 = warp_shuffle_xor(local_sum0, UInt32(1))
        local_sum0 += s_peer0
        s_peer0 = warp_shuffle_xor(local_sum0, UInt32(2))
        local_sum0 += s_peer0
        var s_peer1 = warp_shuffle_xor(local_sum1, UInt32(1))
        local_sum1 += s_peer1
        s_peer1 = warp_shuffle_xor(local_sum1, UInt32(2))
        local_sum1 += s_peer1

        comptime for ot in range(O_TILES):
            var o_vec = o_reg.tile[1, frag_size](ot, 0).vectorize[1, frag_size]()
            var o = rebind[SIMD[DType.float32, 4]](o_vec[0, 0])
            o[0] *= corr0
            o[1] *= corr0
            o[2] *= corr1
            o[3] *= corr1
            o_vec[0, 0] = rebind[type_of(o_vec[0, 0])](o)
        rowsum0 = rowsum0 * corr0 + local_sum0
        rowsum1 = rowsum1 * corr1 + local_sum1

        async_copy_wait_all()
        barrier()

        var v_tensor = LayoutTensor[dtype, kv_layout, address_space=AddressSpace.SHARED](v_smem)
        comptime for bc_tile in range(BC_TILES):
            var n0 = bc_tile * 2
            var n1 = n0 + 1
            var a_p = LayoutTensor[
                dtype,
                p_reg_layout,
                MutAnyOrigin,
                address_space=AddressSpace.LOCAL,
            ].stack_allocation()
            # PTX m16n8k16 A-fragment register order: {a0,a1}=(r0,k0..1),
            # {a2,a3}=(r0+8,k0..1), {a4,a5}=(r0,k8..9), {a6,a7}=(r0+8,k8..9).
            # score C fragments hold (r0,c0),(r0,c1) at +0,+1 and the r0+8
            # pair at +2,+3. n0/n1 are the k0/k8 column halves. The packing is
            # value-agnostic, so it serves the causal path too (masking was
            # already applied to score[] above).
            var a_vec = a_p.vectorize[1, 2]()
            a_vec[0, 0] = rebind[type_of(a_vec[0, 0])](
                SIMD[dtype, 2](Scalar[dtype](score[n0 * frag_size + 0]), Scalar[dtype](score[n0 * frag_size + 1]))
            )
            a_vec[0, 1] = rebind[type_of(a_vec[0, 1])](
                SIMD[dtype, 2](Scalar[dtype](score[n0 * frag_size + 2]), Scalar[dtype](score[n0 * frag_size + 3]))
            )
            a_vec[0, 2] = rebind[type_of(a_vec[0, 2])](
                SIMD[dtype, 2](Scalar[dtype](score[n1 * frag_size + 0]), Scalar[dtype](score[n1 * frag_size + 1]))
            )
            a_vec[0, 3] = rebind[type_of(a_vec[0, 3])](
                SIMD[dtype, 2](Scalar[dtype](score[n1 * frag_size + 2]), Scalar[dtype](score[n1 * frag_size + 3]))
            )
            var b_v = (
                LayoutTensor[
                    dtype, Layout.row_major(O_TILES, b_frag_size), MutAnyOrigin, address_space=AddressSpace.LOCAL
                ]
                .stack_allocation()
                .vectorize[1, b_frag_size]()
            )
            pv_op.load_b(v_tensor, b_v, bc_tile, 0)
            pv_op.mma(a_p.vectorize[1, PVHalf.a_reg_type.size](), b_v, o_reg.vectorize[1, frag_size]())
        barrier()

    comptime if CAUSAL:
        # Below-diagonal KV tiles need no masking. Straddling tiles get the
        # per-element predicate via causal_diag and tiles above are skipped.
        var diag_start = (tile_i * Br_MMA) // Bc_HMMA
        var j_tile_end = min(j_tiles, (tile_i * Br_MMA + Br_MMA + Bc_HMMA - 1) // Bc_HMMA)
        for j_tile in range(j_tile_end):
            var score = stack_allocation[N_TILES * frag_size, Float32]()
            qk_tile(j_tile, j_tile >= diag_start, score.as_unsafe_any_origin())
            softmax_pv(j_tile, score.as_unsafe_any_origin())
    else:
        for j_tile in range(j_tiles):
            var score = stack_allocation[N_TILES * frag_size, Float32]()
            qk_tile(j_tile, False, score.as_unsafe_any_origin())
            softmax_pv(j_tile, score.as_unsafe_any_origin())

    var o_r0 = q_row_base + frag_r0
    var o_r1 = o_r0 + 8
    var o_base = b * H * S * D + h * S * D
    var inv0 = Float32(1) / rowsum0
    var inv1 = Float32(1) / rowsum1
    comptime for d_tile in range(O_TILES):
        var dc0 = d_tile * MMA_N + frag_c0
        var dc1 = dc0 + 1
        var o_vec = o_reg.tile[1, frag_size](d_tile, 0).vectorize[1, frag_size]()
        var o = rebind[SIMD[DType.float32, 4]](o_vec[0, 0])
        if o_r0 < S and dc0 < D:
            dst[o_base + o_r0 * D + dc0] = Scalar[dtype](o[0] * inv0)
        if o_r0 < S and dc1 < D:
            dst[o_base + o_r0 * D + dc1] = Scalar[dtype](o[1] * inv0)
        if o_r1 < S and dc0 < D:
            dst[o_base + o_r1 * D + dc0] = Scalar[dtype](o[2] * inv1)
        if o_r1 < S and dc1 < D:
            dst[o_base + o_r1 * D + dc1] = Scalar[dtype](o[3] * inv1)

    if lane % 4 == 0:
        var lse_base = b * H * S + h * S
        if o_r0 < S:
            lse[lse_base + o_r0] = rowmax0 * LN2 + log(rowsum0)
        if o_r1 < S:
            lse[lse_base + o_r1] = rowmax1 * LN2 + log(rowsum1)


def _flash_attn_fwd_launch_mma[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    var num_tiles = (S + Br_MMA - 1) // Br_MMA
    # Half dtypes with a bucket the ldmatrix swizzle supports run the HMMA
    # kernel, including ragged D (the kernel zero-fills the padded tail).
    # D_BUCKET=256 would blow the register budget, so it stays on TF32.
    comptime if dtype.is_half_float() and D_BUCKET <= 128:

        @parameter
        @always_inline
        def launch_half[RAGGED: Bool, OVERSIZE: Bool]() raises:
            comptime smem_bytes = (
                Br_MMA * D_BUCKET + 2 * Bc_HMMA * D_BUCKET + (Br_MMA * Bc_HMMA if HAS_BIAS and OVERSIZE else 0)
            ) * size_of[dtype]()
            comptime kernel_fn = flash_attn_fwd_kernel_mma_half[dtype, D_BUCKET, CAUSAL, HAS_BIAS, RAGGED, OVERSIZE]
            ctx.enqueue_function[kernel_fn](
                q,
                k,
                v,
                mask,
                dst,
                lse,
                B,
                S,
                H,
                D,
                scale,
                grid_dim=(B * H * num_tiles,),
                block_dim=(BLOCK_SIZE_MMA,),
                shared_mem_bytes=smem_bytes,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
            )

        comptime if HAS_BIAS:
            # An over-L2 bias runs the staged evict-first variant, while a
            # resident bias reads directly and stays warm. Comptime-specialized
            # so neither variant pays a hot-loop branch.
            var mask_oversize = B * H * S * S * size_of[Scalar[dtype]]() > 32 * 1024 * 1024
            if D == D_BUCKET:
                if mask_oversize:
                    launch_half[False, True]()
                else:
                    launch_half[False, False]()
            else:
                if mask_oversize:
                    launch_half[True, True]()
                else:
                    launch_half[True, False]()
        else:
            if D == D_BUCKET:
                launch_half[False, False]()
            else:
                launch_half[True, False]()
    else:
        comptime smem_bytes = (Bc_MMA * (D_BUCKET + KV_MMA_PAD) + NUM_WARPS_MMA * MMA_M * MMA_K) * size_of[Float32]()
        comptime kernel_fn = flash_attn_fwd_kernel_mma[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
        ctx.enqueue_function[kernel_fn](
            q,
            k,
            v,
            mask,
            dst,
            lse,
            B,
            S,
            H,
            D,
            scale,
            grid_dim=(B * H * num_tiles,),
            block_dim=(BLOCK_SIZE_MMA,),
            shared_mem_bytes=smem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
        )
