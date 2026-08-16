from std.gpu.primitives.warp import sum as warp_sum
from std.gpu import thread_idx, block_idx
from max.gpu.host import DeviceContext
from std.gpu.host import get_gpu_target
from std.sys.info import simd_width_of

from layout import Coord, TileTensor, row_major
from nn.softmax import softmax as nn_softmax

from std.utils.index import IndexList

from mograd.runtime.gpu.kernels.strided import strided_offset

# ===-------------------------------------------------------------------===#
# Softmax
# ===-------------------------------------------------------------------===#


def softmax[
    dtype: DType
](
    a: Pointer[mut=False, Scalar[dtype], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var out = TileTensor(dst.as_unsafe_any_origin(), row_major(Coord(rows, cols)))

    @always_inline
    def input_fn[
        width: Int, alignment: Int, _rank: Int
    ](coords: IndexList[_rank]) {imm a, imm cols} -> SIMD[dtype, width]:
        return a.unsafe_load[width=width](Int(coords[0]) * cols + Int(coords[_rank - 1]))

    nn_softmax[dtype, 2, target="gpu"](input_fn, Coord(rows, cols), Scalar[DType.int](cols), out, axis=1, context=ctx)
    ctx.synchronize()


def softmax_strided[
    dtype: DType
](
    a: Pointer[mut=False, Scalar[dtype], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    rows: Int,
    cols: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises:
    # Input addresses aren't a simple flat offset once the source isn't
    # contiguous, so this reads scalar-at-a-time (simd_width=1) via
    # strided_offset, same as unary_strided_map's strided fallback.
    var out = TileTensor(dst.as_unsafe_any_origin(), row_major(Coord(rows, cols)))

    @always_inline
    def input_fn[
        width: Int, alignment: Int, _rank: Int
    ](coords: IndexList[_rank]) {imm a, imm cols, imm rank, imm inner, imm sa} -> SIMD[dtype, width]:
        var v = SIMD[dtype, width]()
        for lane in range(width):
            var flat = Int(coords[0]) * cols + Int(coords[_rank - 1]) + lane
            v[lane] = a.unsafe_load(strided_offset(flat, rank, inner, sa))
        return v

    nn_softmax[dtype, 2, target="gpu"](input_fn, Coord(rows, cols), Scalar[DType.int](cols), out, axis=1, context=ctx)
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Softmax grad
# ===-------------------------------------------------------------------===#


def softmax_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    y: Pointer[Scalar[dtype], ImmutAnyOrigin],
    upstream: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    N_arg: Int64,
    size_arg: Int64,
):
    var N = Int(N_arg)
    var size = Int(size_arg)
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * size
    var dot = Scalar[dtype](0.0)
    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dot += y[unsafe_offset=row_offset + i] * upstream[unsafe_offset=row_offset + i]
    dot = warp_sum(dot)
    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dst[unsafe_offset=row_offset + i] = y[unsafe_offset=row_offset + i] * (
            upstream[unsafe_offset=row_offset + i] - dot
        )
