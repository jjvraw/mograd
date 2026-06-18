from std.utils import StaticTuple, IndexList
from std.algorithm.backend.gpu.reduction import reduce_kernel
from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext

from layout import Coord

from mograd.runtime.gpu.kernels.utils import strided_offset

# ===-------------------------------------------------------------------===#
# Sum
# ===-------------------------------------------------------------------===#


def sum[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, ImmutAnyOrigin],
    sa: UnsafePointer[Int64, ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    @always_inline
    def input_fn[_d: DType, w: Int, r: Int](coords: IndexList[r]) capturing -> SIMD[_d, w]:
        return a.load(strided_offset(coords[0], rank, inner, sa))._refine[_d]()

    @always_inline
    def output_fn[_d: DType, w: SIMDSize, r: Int](coords: IndexList[r], val: StaticTuple[SIMD[_d, w], 1]) capturing:
        dst.store[width=w](coords[0], val[0]._refine[dtype]())

    @always_inline
    def reduce_fn[ty: DType, w: SIMDSize, ri: Int](v1: SIMD[ty, w], v2: SIMD[ty, w]) capturing -> SIMD[ty, w]:
        return v1 + v2

    comptime kernel = reduce_kernel[1, 0, 1, CE_BLOCK, input_fn, output_fn, reduce_fn, dtype, 1]
    ctx.enqueue_function[kernel](
        IndexList[1](n),
        StaticTuple[Scalar[dtype], 1](0),
        grid_dim=(1,),
        block_dim=(CE_BLOCK,),
    )
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Sum axis
# ===-------------------------------------------------------------------===#


def sum_axis[
    dtype: DType
](
    a: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    outer: Int,
    reduce_size: Int,
    inner: Int,
    rank: Int,
    inner_sizes: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises:
    @always_inline
    def input_fn[_d: DType, w: Int, r: Int](coords: IndexList[r]) capturing -> SIMD[_d, w]:
        # coords: [outer_idx, reduce_idx, inner_idx]
        var flat = coords[0] * reduce_size * inner + coords[1] * inner + coords[2]
        return a.load(strided_offset(flat, rank, inner_sizes, sa))._refine[_d]()

    @always_inline
    def output_fn[_d: DType, w: SIMDSize, r: Int](coords: IndexList[r], val: StaticTuple[SIMD[_d, w], 1]) capturing:
        # coords: [outer_idx, 0, inner_idx]
        var flat = coords[0] * inner + coords[2]
        dst.store[width=w](flat, val[0]._refine[dtype]())

    @always_inline
    def reduce_fn[ty: DType, w: SIMDSize, ri: Int](v1: SIMD[ty, w], v2: SIMD[ty, w]) capturing -> SIMD[ty, w]:
        return v1 + v2

    comptime kernel = reduce_kernel[3, 1, 1, CE_BLOCK, input_fn, output_fn, reduce_fn, dtype, 1]
    ctx.enqueue_function[kernel](
        IndexList[3](outer, reduce_size, inner),
        StaticTuple[Scalar[dtype], 1](0),
        grid_dim=(outer * inner,),
        block_dim=(CE_BLOCK,),
    )
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Argmax
# ===-------------------------------------------------------------------===#


def argmax[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, ImmutAnyOrigin],
    sa: UnsafePointer[Int64, ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def kernel[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var best_val = Scalar[dtype].MIN
        var best_idx = Scalar[dtype](0)
        for i in range(n):
            var v = a.load(strided_offset(i, rank, inner, sa))
            if v > best_val:
                best_val = v
                best_idx = Scalar[dtype](i)
        dst.store(0, best_idx)

    elementwise[simd_width=1, target="gpu"](kernel, Coord(1), ctx)
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Argmax axis (stride-aware)
# ===-------------------------------------------------------------------===#


def argmax_axis[
    dtype: DType
](
    a: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    outer: Int,
    reduce_size: Int,
    inner: Int,
    rank: Int,
    inner_sizes: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises:
    def kernel[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat_out = Int(coord[0].value())
        var o = flat_out // inner
        var i = flat_out % inner
        var best_val = Scalar[dtype].MIN
        var best_idx = Scalar[dtype](0)
        for r in range(reduce_size):
            var flat = o * reduce_size * inner + r * inner + i
            var v = a.load(strided_offset(flat, rank, inner_sizes, sa))
            if v > best_val:
                best_val = v
                best_idx = Scalar[dtype](r)
        dst.store(flat_out, best_idx)

    elementwise[simd_width=1, target="gpu"](kernel, Coord(outer * inner), ctx)
    ctx.synchronize()
