from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from std.gpu.host import get_gpu_target
from std.sys.info import simd_width_of
from std.utils.index import IndexList
from layout import Coord
from mograd.memory import scratch_take

from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform

from mograd.runtime.gpu.kernels.strided import strided_offset

# ===-------------------------------------------------------------------===#
# Tensor.randn
# ===-------------------------------------------------------------------===#


def randn[
    dtype: DType
](
    dst: Pointer[mut=True, Scalar[dtype], _],
    n: Int,
    mean: Float32,
    std: Float32,
    seed: UInt64,
    ctx: DeviceContext,
) raises:
    var seed_buf = scratch_take[DType.uint64](ctx, 1)
    seed_buf.enqueue_fill(seed)

    @always_inline
    def store[width: SIMDLength, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) {imm dst}:
        dst.unsafe_store(idx[0], val)

    random_normal[target="gpu"](
        IndexList[1](n),
        mean,
        std,
        seed_buf.unsafe_ptr().as_unsafe_any_origin(),
        ctx,
        store,
    )
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Tensor.uniform
# ===-------------------------------------------------------------------===#


def uniform[
    dtype: DType
](
    params: Pointer[mut=False, Float32, _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    n: Int,
    seed: UInt64,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    var low = Scalar[dtype](params[unsafe_offset=0])
    var high = Scalar[dtype](params[unsafe_offset=1])
    var seed_buf = scratch_take[DType.uint64](ctx, 1)
    seed_buf.enqueue_fill(seed)

    @always_inline
    def store[width: SIMDLength, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) {imm dst}:
        dst.unsafe_store(idx[0], val)

    random_uniform[target="gpu"](IndexList[1](n), low, high, seed_buf.unsafe_ptr().as_unsafe_any_origin(), ctx, store)
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Tensor.full / Tensor.zeros / Tensor.ones
# ===-------------------------------------------------------------------===#


def full[dtype: DType](val: Scalar[dtype], dst: Pointer[mut=True, Scalar[dtype], _], n: Int, ctx: DeviceContext) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        dst.unsafe_store[simd_width](Int(coord[0].value()), val)

    comptime width = simd_width_of[dtype, target=get_gpu_target()]()
    elementwise[simd_width=width, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Tensor.one_hot
# ===-------------------------------------------------------------------===#


def one_hot[
    in_dtype: DType, out_dtype: DType
](
    a: Pointer[mut=False, Scalar[in_dtype], _],
    dst: Pointer[mut=True, Scalar[out_dtype], _],
    n: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sd: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises where out_dtype.is_integral():
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var C = Int(inner[unsafe_offset=0])
        var rem = flat
        var dst_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[unsafe_offset=i])
            rem %= Int(inner[unsafe_offset=i])
            dst_off += idx * Int(sd[unsafe_offset=i])
        var row = flat // C
        var col = flat % C
        var class_idx = Int(a.unsafe_load(row * Int(sa[unsafe_offset=0])))
        dst.unsafe_store(dst_off, Scalar[out_dtype](1) if class_idx == col else Scalar[out_dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)
