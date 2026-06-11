from std.math import log as math_log, exp as math_exp
from std.algorithm.functional import elementwise

from mograd.runtime.gpu.kernels.utils import strided_offset, unary_strided_map

# ===-------------------------------------------------------------------===#
# Scalar ops (used with dispatch_unary_map / dispatch_binary_map)
# ===-------------------------------------------------------------------===#


# TODO: Remove once all ops respect layout strides (broadcast can then stay a zero-copy view)
@always_inline
def identity_op[d: DType](x: Scalar[d]) -> Scalar[d]:
    return x


@always_inline
def neg_op[d: DType](x: Scalar[d]) -> Scalar[d]:
    return -x


@always_inline
def log_op[d: DType](x: Scalar[d]) -> Scalar[d] where d.is_floating_point():
    return math_log(x)


@always_inline
def exp_op[d: DType](x: Scalar[d]) -> Scalar[d] where d.is_floating_point():
    return math_exp(x)


@always_inline
def relu_op[d: DType](x: Scalar[d]) -> Scalar[d]:
    return x if x > Scalar[d](0) else Scalar[d](0)


@always_inline
def add_op[d: DType](x: Scalar[d], y: Scalar[d]) -> Scalar[d]:
    return x + y


@always_inline
def mul_op[d: DType](x: Scalar[d], y: Scalar[d]) -> Scalar[d]:
    return x * y


@always_inline
def div_op[d: DType](x: Scalar[d], y: Scalar[d]) -> Scalar[d]:
    return x / y


@always_inline
def eq_op[d: DType](x: Scalar[d], y: Scalar[d]) -> Scalar[d]:
    return Scalar[d](1) if x == y else Scalar[d](0)


@always_inline
def relu_grad_op[d: DType](x: Scalar[d], y: Scalar[d]) -> Scalar[d]:
    return y if x > Scalar[d](0) else Scalar[d](0)


# ===-------------------------------------------------------------------===#
# Cast
# ===-------------------------------------------------------------------===#


def cast[
    src_dtype: DType, dst_dtype: DType
](
    a: UnsafePointer[Scalar[src_dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store(flat, a.load(strided_offset(flat, rank, inner, sa)).cast[dst_dtype]())

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Slice grad
# ===-------------------------------------------------------------------===#


def slice_grad[
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
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store(strided_offset(flat, rank, inner, sa), a.load(flat))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Add (contiguous fast path)
# ===-------------------------------------------------------------------===#


def add[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime width = simd_width_of[dtype, target=get_gpu_target()]()

    def apply_fast[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store[simd_width](flat, a.load[simd_width](flat) + b.load[simd_width](flat))

    elementwise[simd_width=width, target="gpu"](apply_fast, Coord(n), ctx)
