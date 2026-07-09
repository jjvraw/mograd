# Apple-GPU flash-attention kernels

# Metal has no external_memory lowering and the smem-tiled SIMT kernels
# miscompile on the Metal backend even with static threadgroup memory.
# Modular's Apple attention work (max/kernels/src/nn/attention/gpu/apple/)
# also measured SMEM staging as a net loss on this hardware. Reuse comes
# from wide threadgroup geometry plus L2, not threadgroup memory. These
# kernels follow their naive_fa_decode / fa_prefill shape:
#
#   - One query (or key) row per simdgroup. APPLE_QROWS simdgroups per
#     threadgroup stream the same KV (or Q/dO) range, so a row is pulled
#     from DRAM once and served to the other co-resident rows from L2.
#   - Row operands and accumulators live in registers, split across lanes
#     (D_PT = D_BUCKET / WARP_SIZE elements per lane).
#   - Dot products are per-row GEMVs. Lane partials reduce with warp sum
#     (air.simd_sum on Metal). The result is lane-uniform, so per-row
#     softmax state runs identically in every lane's registers.
#
# The backward mirrors the generic rowwarp kernels' math with the smem
# staging removed. The dq kernel also writes delta = dot(O, dO) per query
# row and the dkdv kernel reads it (same stream, ordering guaranteed).
#
# Layouts match the generic kernels: Q/K/V and dQ/dK/dV are BSHD, O and
# dO are BHSD, LSE is (B, H, S) float32 in natural log.

from std.gpu import WARP_SIZE, MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.gpu.primitives.warp import sum as warp_sum
from std.math import exp2, log
from std.memory import stack_allocation
from std.utils import StaticTuple
from mograd.runtime.gpu.kernels.attention.config import LOG2E, LN2

# Simdgroups per threadgroup. Wide geometry is the perf lever: 16 warps
# of 32 lanes give 512 co-resident threads walking one KV range per block.
comptime APPLE_QROWS = 16


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(APPLE_QROWS * WARP_SIZE)))
@__name(t"flash_attn_fwd_apple_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_fwd_kernel_apple[
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
    comptime assert D_BUCKET % WARP_SIZE == 0, "D_BUCKET must be divisible by WARP_SIZE"
    comptime D_PT = D_BUCKET // WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE

    var num_tiles = (S + APPLE_QROWS - 1) // APPLE_QROWS
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H
    var i = tile_i * APPLE_QROWS + warp_id
    # No smem and no barriers anywhere, so an idle row can leave early.
    # The branch is warp-uniform (i is identical across the simdgroup).
    if i >= S:
        return

    var HD = H * D
    var qkv_bh = b * S * H * D + h * D

    var q_reg = stack_allocation[D_PT, Float32]()
    var o_acc = stack_allocation[D_PT, Float32]()
    comptime for di in range(D_PT):
        var d = lane + di * WARP_SIZE
        q_reg[di] = Float32(q[qkv_bh + i * HD + d]) if d < D else Float32(0)
        o_acc[di] = Float32(0)

    var m = Float32(-1e38)  # running max in log2 space
    var l = Float32(0)
    var scale_log2e = scale * LOG2E

    var j_end: Int
    comptime if CAUSAL:
        j_end = i + 1
    else:
        j_end = S

    for j in range(j_end):
        var kv_base = qkv_bh + j * HD
        var partial = Float32(0)
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            partial += q_reg[di] * (Float32(k[kv_base + d]) if d < D else Float32(0))
        var s_l2 = warp_sum(partial) * scale_log2e
        comptime if HAS_BIAS and not CAUSAL:
            # mask is additive bias in natural-log space, converted to log2
            s_l2 += Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E

        # Online softmax in registers. s_l2 is lane-uniform after warp_sum,
        # so every lane tracks the same m and l without any exchange.
        var m_new = m if m > s_l2 else s_l2
        var corr = exp2(m - m_new)
        var p = exp2(s_l2 - m_new)
        l = l * corr + p
        m = m_new
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            o_acc[di] = o_acc[di] * corr + p * (Float32(v[kv_base + d]) if d < D else Float32(0))

    # O is BHSD. LSE converts from log2 back to natural log.
    var o_base = b * H * S * D + h * S * D + i * D
    comptime for di in range(D_PT):
        var d = lane + di * WARP_SIZE
        if d < D:
            dst[o_base + d] = Scalar[dtype](o_acc[di] / l)
    if lane == 0:
        lse[b * H * S + h * S + i] = m * LN2 + log(l)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(APPLE_QROWS * WARP_SIZE)))
@__name(t"flash_attn_dq_apple_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_dq_kernel_apple[
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
    comptime assert D_BUCKET % WARP_SIZE == 0, "D_BUCKET must be divisible by WARP_SIZE"
    comptime D_PT = D_BUCKET // WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE

    var num_tiles = (S + APPLE_QROWS - 1) // APPLE_QROWS
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H
    var i = tile_i * APPLE_QROWS + warp_id
    if i >= S:
        return

    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var o_bh = b * H * S * D + h * S * D

    var q_reg = stack_allocation[D_PT, Float32]()
    var do_reg = stack_allocation[D_PT, Float32]()
    var dq_reg = stack_allocation[D_PT, Float32]()
    var delta_partial = Float32(0)
    comptime for di in range(D_PT):
        dq_reg[di] = Float32(0)
        var d = lane + di * WARP_SIZE
        if d < D:
            var o_idx = o_bh + i * D + d
            q_reg[di] = Float32(q[qkv_bh + i * HD + d])
            var do_val = Float32(dy[o_idx])
            do_reg[di] = do_val
            delta_partial += Float32(o[o_idx]) * do_val
        else:
            q_reg[di] = Float32(0)
            do_reg[di] = Float32(0)
    var delta_i = warp_sum(delta_partial)
    if lane == 0:
        delta[b * H * S + h * S + i] = delta_i

    var lse_i_l2 = lse[b * H * S + h * S + i] * LOG2E
    var scale_log2e = scale * LOG2E

    var j_end: Int
    comptime if CAUSAL:
        j_end = i + 1
    else:
        j_end = S

    for j in range(j_end):
        var kv_base = qkv_bh + j * HD
        var k_frag = stack_allocation[D_PT, Float32]()
        var pqk = Float32(0)
        var pdp = Float32(0)
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            k_frag[di] = Float32(k[kv_base + d]) if d < D else Float32(0)
            pqk += q_reg[di] * k_frag[di]
            pdp += do_reg[di] * (Float32(v[kv_base + d]) if d < D else Float32(0))
        var s_l2 = warp_sum(pqk) * scale_log2e
        comptime if HAS_BIAS and not CAUSAL:
            s_l2 += Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E
        var p_ij = exp2(s_l2 - lse_i_l2)
        var ds_ij = p_ij * (warp_sum(pdp) - delta_i)
        comptime for di in range(D_PT):
            dq_reg[di] += ds_ij * k_frag[di]

    # The scale factor is applied once at the epilogue.
    var out_base = qkv_bh + i * HD
    comptime for di in range(D_PT):
        var d = lane + di * WARP_SIZE
        if d < D:
            dq[out_base + d] = Scalar[dtype](dq_reg[di] * scale)


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(APPLE_QROWS * WARP_SIZE)))
@__name(t"flash_attn_dkdv_apple_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_dkdv_kernel_apple[
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
    comptime assert D_BUCKET % WARP_SIZE == 0, "D_BUCKET must be divisible by WARP_SIZE"
    comptime D_PT = D_BUCKET // WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE

    var num_tiles = (S + APPLE_QROWS - 1) // APPLE_QROWS
    if bid >= B * H * num_tiles:
        return

    var tile_j = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H
    var j = tile_j * APPLE_QROWS + warp_id
    if j >= S:
        return

    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var o_bh = b * H * S * D + h * S * D

    var k_reg = stack_allocation[D_PT, Float32]()
    var v_reg = stack_allocation[D_PT, Float32]()
    var dk_reg = stack_allocation[D_PT, Float32]()
    var dv_reg = stack_allocation[D_PT, Float32]()
    comptime for di in range(D_PT):
        dk_reg[di] = Float32(0)
        dv_reg[di] = Float32(0)
        var d = lane + di * WARP_SIZE
        var kv_base = qkv_bh + j * HD
        k_reg[di] = Float32(k[kv_base + d]) if d < D else Float32(0)
        v_reg[di] = Float32(v[kv_base + d]) if d < D else Float32(0)

    var scale_log2e = scale * LOG2E

    # Causal rows below the key row contribute nothing and are skipped.
    var i_start: Int
    comptime if CAUSAL:
        i_start = j
    else:
        i_start = 0

    for i in range(i_start, S):
        var q_base = qkv_bh + i * HD
        var do_base = o_bh + i * D
        var do_frag = stack_allocation[D_PT, Float32]()
        var q_frag = stack_allocation[D_PT, Float32]()
        var pqk = Float32(0)
        var pdp = Float32(0)
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            q_frag[di] = Float32(q[q_base + d]) if d < D else Float32(0)
            do_frag[di] = Float32(dy[do_base + d]) if d < D else Float32(0)
            pqk += q_frag[di] * k_reg[di]
            pdp += do_frag[di] * v_reg[di]
        var s_l2 = warp_sum(pqk) * scale_log2e
        comptime if HAS_BIAS and not CAUSAL:
            s_l2 += Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E
        var p_ij = exp2(s_l2 - lse[b * H * S + h * S + i] * LOG2E)
        var ds_ij = p_ij * (warp_sum(pdp) - delta[b * H * S + h * S + i])
        comptime for di in range(D_PT):
            dv_reg[di] += p_ij * do_frag[di]
            dk_reg[di] += ds_ij * q_frag[di]

    # The scale factor is applied once at the epilogue.
    var out_base = qkv_bh + j * HD
    comptime for di in range(D_PT):
        var d = lane + di * WARP_SIZE
        if d < D:
            dk[out_base + d] = Scalar[dtype](dk_reg[di] * scale)
            dv[out_base + d] = Scalar[dtype](dv_reg[di])


def _flash_attn_fwd_launch_apple[
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
    var num_tiles = (S + APPLE_QROWS - 1) // APPLE_QROWS
    comptime kernel_fn = flash_attn_fwd_kernel_apple[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
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
        block_dim=(APPLE_QROWS * WARP_SIZE,),
    )


def _flash_attn_bwd_launch_apple[
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
    # dq writes delta and dkdv reads it (same stream).
    var delta_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S)
    var delta_ptr = delta_buf.unsafe_ptr().as_unsafe_any_origin()

    var num_tiles = (S + APPLE_QROWS - 1) // APPLE_QROWS
    comptime dq_fn = flash_attn_dq_kernel_apple[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
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
        grid_dim=(B * H * num_tiles,),
        block_dim=(APPLE_QROWS * WARP_SIZE,),
    )
    comptime dkdv_fn = flash_attn_dkdv_kernel_apple[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
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
        grid_dim=(B * H * num_tiles,),
        block_dim=(APPLE_QROWS * WARP_SIZE,),
    )

    _ = delta_buf^
