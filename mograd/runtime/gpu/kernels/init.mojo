from std.algorithm.functional import elementwise

from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform

# ===-------------------------------------------------------------------===#
# Tensor.randn
# ===-------------------------------------------------------------------===#


def randn[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    mean: Float32,
    std: Float32,
    seed: Float32,
    ctx: DeviceContext,
) raises:
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(seed)))

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        dst.store(idx[0], val)

    random_normal[output_fn=store, target="gpu"](
        IndexList[1](n),
        mean,
        std,
        seed_buf.unsafe_ptr(),
        ctx,
    )
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Tensor.uniform
# ===-------------------------------------------------------------------===#


def uniform[
    dtype: DType
](
    params: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    var low = Scalar[dtype](params[0])
    var high = Scalar[dtype](params[1])
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(params[2])))

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        dst.store(idx[0], val)

    random_uniform[output_fn=store, target="gpu"](IndexList[1](n), low, high, seed_buf.unsafe_ptr(), ctx)
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Tensor.full / Tensor.zeros / Tensor.ones
# ===-------------------------------------------------------------------===#


def full[
    dtype: DType
](val: Scalar[dtype], dst: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int, ctx: DeviceContext) raises:
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
    a: UnsafePointer[Scalar[in_dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[out_dtype], MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sd: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    ctx: DeviceContext,
) raises where out_dtype.is_integral():
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        # inner[0] == C; read on GPU side since inner is a device pointer
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
