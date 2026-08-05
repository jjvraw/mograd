from std.algorithm.functional import elementwise

from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform

from mograd.runtime.gpu.kernels.utils import strided_offset

# ===-------------------------------------------------------------------===#
# Tensor.randn
# ===-------------------------------------------------------------------===#


def randn[
    dtype: DType
](
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    n: Int,
    mean: Float32,
    std: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises:
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(seed)

    def store[width: SIMDSize, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        dst.store(idx[0], val)

    random_normal[output_fn=store, target="gpu"](
        IndexList[1](n),
        mean,
        std,
        seed_buf.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
    )
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Tensor.uniform
# ===-------------------------------------------------------------------===#


def uniform[
    dtype: DType
](
    params: UnsafePointer[mut=False, Float32, _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    n: Int,
    seed: UInt64,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    var low = Scalar[dtype](params[0])
    var high = Scalar[dtype](params[1])
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(seed)

    def store[width: SIMDSize, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        dst.store(idx[0], val)

    random_uniform[output_fn=store, target="gpu"](
        IndexList[1](n), low, high, seed_buf.unsafe_ptr().as_unsafe_any_origin(), ctx
    )
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Tensor.full / Tensor.zeros / Tensor.ones
# ===-------------------------------------------------------------------===#


def full[
    dtype: DType
](val: Scalar[dtype], dst: UnsafePointer[mut=True, Scalar[dtype], _], n: Int, ctx: DeviceContext) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        dst.store[simd_width](Int(coord[0].value()), val)

    comptime width = simd_width_of[dtype, target=get_gpu_target()]()
    elementwise[simd_width=width, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Tensor.one_hot
# ===-------------------------------------------------------------------===#


def one_hot[
    in_dtype: DType, out_dtype: DType
](
    a: UnsafePointer[mut=False, Scalar[in_dtype], _],
    dst: UnsafePointer[mut=True, Scalar[out_dtype], _],
    n: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sd: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises where out_dtype.is_integral():
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var C = Int(inner[0])
        var rem = flat
        var dst_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            dst_off += idx * Int(sd[i])
        var row = flat // C
        var col = flat % C
        var class_idx = Int(a.load(row * Int(sa[0])))
        dst.store(dst_off, Scalar[out_dtype](1) if class_idx == col else Scalar[out_dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)
