from std.gpu.primitives.warp import sum as warp_sum
from std.gpu import thread_idx, block_idx
from std.gpu.host import DeviceContext, get_gpu_target
from std.sys.info import simd_width_of

from layout import Coord, TileTensor, row_major
from nn.softmax import softmax as nn_softmax

from mograd.runtime.gpu.kernels.utils import strided_offset

# ===-------------------------------------------------------------------===#
# Softmax
# ===-------------------------------------------------------------------===#


def softmax[
    dtype: DType
](
    a: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var out = TileTensor(dst.as_unsafe_any_origin(), row_major(Coord(rows, cols)))

    def input_fn[width: Int](coords: Coord) capturing -> SIMD[dtype, width]:
        return a.load[width=width](Int(coords[0].value()) * cols + Int(coords[1].value()))

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    nn_softmax[dtype, simd_width, 2, input_fn, "gpu"](Coord(rows, cols), out, axis=1, context=ctx)
    ctx.synchronize()


def softmax_strided[
    dtype: DType
](
    a: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    rows: Int,
    cols: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises:
    # Input addresses aren't a simple flat offset once the source isn't
    # contiguous, so this reads scalar-at-a-time (simd_width=1) via
    # strided_offset, same as unary_strided_map's strided fallback.
    var out = TileTensor(dst.as_unsafe_any_origin(), row_major(Coord(rows, cols)))

    def input_fn[width: Int](coords: Coord) capturing -> SIMD[dtype, width]:
        var flat = Int(coords[0].value()) * cols + Int(coords[1].value())
        return a.load[width=width](strided_offset(flat, rank, inner, sa))

    nn_softmax[dtype, 1, 2, input_fn, "gpu"](Coord(rows, cols), out, axis=1, context=ctx)
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Softmax grad
# ===-------------------------------------------------------------------===#


def softmax_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    y: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    upstream: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    size: Int,
):
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * size
    var dot = Scalar[dtype](0.0)
    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dot += y[row_offset + i] * upstream[row_offset + i]
    dot = warp_sum(dot)
    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dst[row_offset + i] = y[row_offset + i] * (upstream[row_offset + i] - dot)
