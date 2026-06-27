from std.gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, get_gpu_target, DeviceAttribute
from std.gpu.memory import AddressSpace
from std.gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from std.math import ceildiv, rsqrt
from std.memory import stack_allocation
from std.sys.info import simd_width_of
from std.utils.index import IndexList
from std.utils.numerics import get_accum_type


from layout import Coord, TileTensor, row_major

from nn.normalization import block_reduce_dual_sum, layer_norm_gpu

# ===-------------------------------------------------------------------===#
# LayerNorm forward
# ===-------------------------------------------------------------------===#


def layer_norm_fwd[
    dtype: DType
](
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    beta: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    var beta_tile = TileTensor[mut=False, dtype](beta.as_unsafe_any_origin(), row_major(Coord(cols)))

    @__copy_capture(x, cols)
    @always_inline
    @parameter
    def input_fn[width: Int, rank: Int, alignment: Int](coords: IndexList[rank]) -> SIMD[dtype, width]:
        return x.load[width=width](Int(coords[0]) * cols + Int(coords[rank - 1]))

    @__copy_capture(gamma)
    @always_inline
    @parameter
    def gamma_fn[width: Int, rank: Int, alignment: Int](coords: IndexList[rank]) -> SIMD[dtype, width]:
        return gamma.load[width=width](Int(coords[rank - 1]))

    @__copy_capture(dst, cols)
    @always_inline
    @parameter
    def output_fn[width: SIMDSize, rank: Int, alignment: Int](coords: IndexList[rank], val: SIMD[dtype, width]):
        dst.store[width=width](Int(coords[0]) * cols + Int(coords[rank - 1]), val)

    layer_norm_gpu[input_fn, gamma_fn, output_fn](IndexList[2](rows, cols), beta_tile, eps, ctx=ctx)


# ===-------------------------------------------------------------------===#
# LayerNorm backward
#
# Kernel 1: one block per group of rows (num_sm groups total). Each block owns its
#   rows exclusively and loops through them one at a time. For each row it
#   makes three passes over the columns: first to compute the mean and
#   inverse std, second to compute the correction terms needed for dx, third
#   to write dx and accumulate the dgamma/dbeta contributions.
#
#   The dgamma/dbeta accumulation stays off global memory until the end:
#     - For cols up to 4096 (fp32): accumulated into shared memory, which
#       fits comfortably under 48 KB.
#     - For wider cols: accumulated into a small register array instead,
#       avoiding shmem limit entirely.
#   At the end of the group, each block writes its accumulated partials to
#   its own exclusive row in a small tmp buffer.
#
# Kernel 2: reduces the tmp buffer from [num_sm, cols] to [cols] by summing
#   across groups. Each thread owns a slice of columns, loops over the
#   num_sm rows, and writes the final dgamma/dbeta with a plain store.
# ===-------------------------------------------------------------------===#


def layer_norm_bwd_dx_grouped_kernel[
    dtype: DType,
    simd_width: Int,
    max_warps_per_block: Int,
    max_cols: Int,
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dx: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dg_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    db_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rows: Int,
    cols: Int,
    group_size: Int,
    eps: Scalar[dtype],
):
    comptime accum = get_accum_type[dtype]()
    var g = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row_start = g * group_size
    var row_end = min(row_start + group_size, rows)
    var n = Scalar[accum](cols)

    # shmem partial accumulators
    # each thread owns exclusive col ranges
    var shmem_g = stack_allocation[max_cols, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var shmem_b = stack_allocation[max_cols, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var col = tid * simd_width
    while col < cols:
        shmem_g.store[width=simd_width](col, SIMD[dtype, simd_width](0))
        shmem_b.store[width=simd_width](col, SIMD[dtype, simd_width](0))
        col += Int(block_dim.x) * simd_width

    with PDL():
        for row in range(row_start, row_end):
            var base = row * cols

            # Pass 1: mu, rstd
            var sx = Scalar[accum](0)
            var sx2 = Scalar[accum](0)
            col = tid * simd_width
            while col < cols:
                var v = x.load[width=simd_width](base + col).cast[accum]()
                sx += v.reduce_add()
                sx2 += (v * v).reduce_add()
                col += Int(block_dim.x) * simd_width
            var r1 = block_reduce_dual_sum[max_warps_per_block=max_warps_per_block](sx, sx2)
            var mu = r1[0] / n
            var rstd = rsqrt(max(r1[1] / n - mu * mu, Scalar[accum](0)) + eps.cast[accum]())

            # Pass 2: c1, c2
            var tc1 = Scalar[accum](0)
            var tc2 = Scalar[accum](0)
            col = tid * simd_width
            while col < cols:
                var dy_v = dy.load[width=simd_width](base + col).cast[accum]()
                var x_v = x.load[width=simd_width](base + col).cast[accum]()
                var g_v = gamma.load[width=simd_width](col).cast[accum]()
                var xhat = (x_v - mu) * rstd
                var dy_g = dy_v * g_v
                tc1 += dy_g.reduce_add()
                tc2 += (dy_g * xhat).reduce_add()
                col += Int(block_dim.x) * simd_width
            var r2 = block_reduce_dual_sum[max_warps_per_block=max_warps_per_block](tc1, tc2)
            var c1 = r2[0] / n
            var c2 = r2[1] / n

            # Pass 3: dx + accumulate into shmem
            col = tid * simd_width
            while col < cols:
                var dy_v = dy.load[width=simd_width](base + col)
                var x_v = x.load[width=simd_width](base + col).cast[accum]()
                var g_v = gamma.load[width=simd_width](col).cast[accum]()
                var xhat = (x_v - mu) * rstd
                dx.store[width=simd_width](
                    base + col, (rstd * (dy_v.cast[accum]() * g_v - c1 - xhat * c2)).cast[dtype]()
                )
                shmem_g.store[width=simd_width](
                    col, shmem_g.load[width=simd_width](col) + (dy_v.cast[accum]() * xhat).cast[dtype]()
                )
                shmem_b.store[width=simd_width](col, shmem_b.load[width=simd_width](col) + dy_v)
                col += Int(block_dim.x) * simd_width

        # Flush shmem → exclusive group slot in tmp.
        var tmp_base = g * cols
        col = tid * simd_width
        while col < cols:
            dg_tmp.store[width=simd_width](tmp_base + col, shmem_g.load[width=simd_width](col))
            db_tmp.store[width=simd_width](tmp_base + col, shmem_b.load[width=simd_width](col))
            col += Int(block_dim.x) * simd_width


def layer_norm_bwd_dx_reg_kernel[
    dtype: DType,
    simd_width: Int,
    max_warps_per_block: Int,
    max_cols: Int,
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dx: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dg_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    db_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rows: Int,
    cols: Int,
    group_size: Int,
    eps: Scalar[dtype],
):
    comptime accum = get_accum_type[dtype]()
    comptime num_chunks = ceildiv(max_cols, max_warps_per_block * WARP_SIZE * simd_width)
    var g = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var row_start = g * group_size
    var row_end = min(row_start + group_size, rows)
    var n = Scalar[accum](cols)

    # local array (2 × simd_width floats per thread at max_cols=8192).
    var acc_g = stack_allocation[num_chunks * simd_width, Scalar[dtype]]()
    var acc_b = stack_allocation[num_chunks * simd_width, Scalar[dtype]]()
    for i in range(num_chunks):
        acc_g.store[width=simd_width](i * simd_width, SIMD[dtype, simd_width](0))
        acc_b.store[width=simd_width](i * simd_width, SIMD[dtype, simd_width](0))

    with PDL():
        for row in range(row_start, row_end):
            var base = row * cols

            # Pass 1: mu, rstd
            var sx = Scalar[accum](0)
            var sx2 = Scalar[accum](0)
            var col = tid * simd_width
            while col < cols:
                var v = x.load[width=simd_width](base + col).cast[accum]()
                sx += v.reduce_add()
                sx2 += (v * v).reduce_add()
                col += Int(block_dim.x) * simd_width
            var r1 = block_reduce_dual_sum[max_warps_per_block=max_warps_per_block](sx, sx2)
            var mu = r1[0] / n
            var rstd = rsqrt(max(r1[1] / n - mu * mu, Scalar[accum](0)) + eps.cast[accum]())

            # Pass 2: c1, c2
            var tc1 = Scalar[accum](0)
            var tc2 = Scalar[accum](0)
            col = tid * simd_width
            while col < cols:
                var dy_v = dy.load[width=simd_width](base + col).cast[accum]()
                var x_v = x.load[width=simd_width](base + col).cast[accum]()
                var g_v = gamma.load[width=simd_width](col).cast[accum]()
                var xhat = (x_v - mu) * rstd
                var dy_g = dy_v * g_v
                tc1 += dy_g.reduce_add()
                tc2 += (dy_g * xhat).reduce_add()
                col += Int(block_dim.x) * simd_width
            var r2 = block_reduce_dual_sum[max_warps_per_block=max_warps_per_block](tc1, tc2)
            var c1 = r2[0] / n
            var c2 = r2[1] / n

            # Pass 3: dx + accumulate into register array.
            # chunk resets each row so the index stays in [0, num_chunks).
            var chunk = 0
            col = tid * simd_width
            while col < cols:
                var dy_v = dy.load[width=simd_width](base + col)
                var x_v = x.load[width=simd_width](base + col).cast[accum]()
                var g_v = gamma.load[width=simd_width](col).cast[accum]()
                var xhat = (x_v - mu) * rstd
                dx.store[width=simd_width](
                    base + col, (rstd * (dy_v.cast[accum]() * g_v - c1 - xhat * c2)).cast[dtype]()
                )
                var slot = chunk * simd_width
                acc_g.store[width=simd_width](
                    slot, acc_g.load[width=simd_width](slot) + (dy_v.cast[accum]() * xhat).cast[dtype]()
                )
                acc_b.store[width=simd_width](slot, acc_b.load[width=simd_width](slot) + dy_v)
                col += Int(block_dim.x) * simd_width
                chunk += 1

        # Flush register accumulators
        var chunk = 0
        var col = tid * simd_width
        while col < cols:
            var slot = chunk * simd_width
            dg_tmp.store[width=simd_width](g * cols + col, acc_g.load[width=simd_width](slot))
            db_tmp.store[width=simd_width](g * cols + col, acc_b.load[width=simd_width](slot))
            col += Int(block_dim.x) * simd_width
            chunk += 1


def layer_norm_reduce_params_kernel[
    dtype: DType,
    simd_width: Int,
](
    dg_tmp: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    db_tmp: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dgamma: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dbeta: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rows: Int,
    cols: Int,
):
    var col = (Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)) * simd_width
    if col >= cols:
        return
    var acc_g = SIMD[dtype, simd_width](0)
    var acc_b = SIMD[dtype, simd_width](0)
    for r in range(rows):
        acc_g += dg_tmp.load[width=simd_width](r * cols + col)
        acc_b += db_tmp.load[width=simd_width](r * cols + col)
    dgamma.store[width=simd_width](col, acc_g)
    dbeta.store[width=simd_width](col, acc_b)


def layer_norm_bwd[
    dtype: DType
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dx: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dgamma: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dbeta: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    var num_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var num_groups = min(rows, num_sm)
    var group_size = ceildiv(rows, num_groups)

    var k1_block = min(
        max(
            ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
            4 * WARP_SIZE,
        ),
        WARP_SIZE * max_warps,
    )

    var dg_tmp = ctx.enqueue_create_buffer[dtype](num_groups * cols)
    var db_tmp = ctx.enqueue_create_buffer[dtype](num_groups * cols)

    @parameter
    def launch_k1g[max_cols: Int]() raises:
        comptime k1 = layer_norm_bwd_dx_grouped_kernel[dtype, simd_width, max_warps, max_cols]
        ctx.enqueue_function[k1](
            dy,
            x,
            gamma,
            dx,
            dg_tmp.unsafe_ptr(),
            db_tmp.unsafe_ptr(),
            rows,
            cols,
            group_size,
            eps,
            grid_dim=num_groups,
            block_dim=k1_block,
            attributes=pdl_launch_attributes(),
        )

    @parameter
    def launch_k1r[max_cols: Int]() raises:
        comptime k1 = layer_norm_bwd_dx_reg_kernel[dtype, simd_width, max_warps, max_cols]
        ctx.enqueue_function[k1](
            dy,
            x,
            gamma,
            dx,
            dg_tmp.unsafe_ptr(),
            db_tmp.unsafe_ptr(),
            rows,
            cols,
            group_size,
            eps,
            grid_dim=num_groups,
            block_dim=k1_block,
            attributes=pdl_launch_attributes(),
        )

    if cols <= 1024:
        launch_k1g[1024]()
    elif cols <= 2048:
        launch_k1g[2048]()
    elif cols <= 4096:
        launch_k1g[4096]()
    elif cols <= 8192:
        launch_k1r[8192]()
    else:
        launch_k1r[16384]()

    comptime k2 = layer_norm_reduce_params_kernel[dtype, simd_width]
    var k2_block = min(256, ceildiv(cols, simd_width))
    var k2_grid = ceildiv(ceildiv(cols, simd_width), k2_block)
    ctx.enqueue_function[k2](
        dg_tmp.unsafe_ptr().as_immutable(),
        db_tmp.unsafe_ptr().as_immutable(),
        dgamma,
        dbeta,
        num_groups,
        cols,
        grid_dim=k2_grid,
        block_dim=k2_block,
        attributes=pdl_launch_attributes(),
    )
