from std.gpu import WARP_SIZE, barrier, block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.primitives.grid_controls import PDL, pdl_launch_attributes
from std.math import ceildiv, rsqrt
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
# K1: one block per row, 3 passes.
#   pass 1: mu, rstd via dual_sum over x
#   pass 2: c1=mean(dy*g), c2=mean(dy*g*xhat) via dual_sum
#   pass 3: write dx, scatter xhat*dy and dy into dg_tmp/db_tmp[row, :]
#
# K2: column-wise sum of dg_tmp/db_tmp [rows, cols] → dgamma/dbeta [cols].
#   1D grid: one block per col-tile, exclusive ownership → plain store.
# ===-------------------------------------------------------------------===#


def layer_norm_bwd_dx_kernel[
    dtype: DType,
    simd_width: Int,
    max_warps_per_block: Int,
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    x: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dx: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dg_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    db_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    cols: Int,
    eps: Scalar[dtype],
):
    comptime accum = get_accum_type[dtype]()
    var tid = Int(thread_idx.x)
    var row = Int(block_idx.x)
    var base = row * cols
    var n = Scalar[accum](cols)

    with PDL():
        # Pass 1: mu, rstd
        var thread_sx = Scalar[accum](0)
        var thread_sx2 = Scalar[accum](0)
        var col = tid * simd_width
        while col < cols:
            var v = x.load[width=simd_width](base + col).cast[accum]()
            thread_sx += v.reduce_add()
            thread_sx2 += (v * v).reduce_add()
            col += Int(block_dim.x) * simd_width

        var r1 = block_reduce_dual_sum[max_warps_per_block=max_warps_per_block](thread_sx, thread_sx2)
        var mu = r1[0] / n
        var rstd = rsqrt(max(r1[1] / n - mu * mu, Scalar[accum](0)) + eps.cast[accum]())

        # Pass 2: c1, c2
        var thread_c1 = Scalar[accum](0)
        var thread_c2 = Scalar[accum](0)
        col = tid * simd_width
        while col < cols:
            var dy_v = dy.load[width=simd_width](base + col).cast[accum]()
            var x_v = x.load[width=simd_width](base + col).cast[accum]()
            var g_v = gamma.load[width=simd_width](col).cast[accum]()
            var xhat = (x_v - mu) * rstd
            var dy_g = dy_v * g_v
            thread_c1 += dy_g.reduce_add()
            thread_c2 += (dy_g * xhat).reduce_add()
            col += Int(block_dim.x) * simd_width

        var r2 = block_reduce_dual_sum[max_warps_per_block=max_warps_per_block](thread_c1, thread_c2)
        var c1 = r2[0] / n
        var c2 = r2[1] / n

        # Pass 3: dx + dg/db partials
        col = tid * simd_width
        while col < cols:
            var dy_v = dy.load[width=simd_width](base + col)
            var x_v = x.load[width=simd_width](base + col).cast[accum]()
            var g_v = gamma.load[width=simd_width](col).cast[accum]()
            var xhat = (x_v - mu) * rstd
            var dx_v = rstd * (dy_v.cast[accum]() * g_v - c1 - xhat * c2)
            dx.store[width=simd_width](base + col, dx_v.cast[dtype]())
            dg_tmp.store[width=simd_width](base + col, (dy_v.cast[accum]() * xhat).cast[dtype]())
            db_tmp.store[width=simd_width](base + col, dy_v)
            col += Int(block_dim.x) * simd_width


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
    dg_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    db_tmp: UnsafePointer[Scalar[dtype], MutAnyOrigin],
) raises:
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime max_warps = ctx.default_device_info.max_thread_block_size // WARP_SIZE
    var k1_block = min(
        ceildiv(ceildiv(cols, simd_width), WARP_SIZE) * WARP_SIZE,
        WARP_SIZE * max_warps,
    )

    comptime k1 = layer_norm_bwd_dx_kernel[dtype, simd_width, max_warps]
    ctx.enqueue_function[k1](
        dy,
        x,
        gamma,
        dx,
        dg_tmp,
        db_tmp,
        cols,
        eps,
        grid_dim=rows,
        block_dim=k1_block,
        attributes=pdl_launch_attributes(),
    )

    comptime k2 = layer_norm_reduce_params_kernel[dtype, simd_width]
    var k2_block = min(256, ceildiv(cols, simd_width))
    var k2_grid = ceildiv(ceildiv(cols, simd_width), k2_block)
    ctx.enqueue_function[k2](
        dg_tmp.as_immutable(),
        db_tmp.as_immutable(),
        dgamma,
        dbeta,
        rows,
        cols,
        grid_dim=k2_grid,
        block_dim=k2_block,
        attributes=pdl_launch_attributes(),
    )
