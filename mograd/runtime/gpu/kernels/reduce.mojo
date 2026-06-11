from mograd.runtime.gpu.kernels.utils import strided_offset

# ===-------------------------------------------------------------------===#
# Sum
# ===-------------------------------------------------------------------===#


def sum[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
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
