from std.math import log as math_log, exp as math_exp
from std.algorithm.functional import elementwise

from mograd.runtime.gpu.kernels.utils import strided_offset, unary_strided_map

# ===-------------------------------------------------------------------===#
# Scalar ops (used with dispatch_unary_map / dispatch_binary_map)
# ===-------------------------------------------------------------------===#


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
    a: UnsafePointer[mut=False, Scalar[src_dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dst_dtype], _],
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
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
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, ImmutAnyOrigin],
    sa: UnsafePointer[Int64, ImmutAnyOrigin],
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
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime width = simd_width_of[dtype, target=get_gpu_target()]()

    def apply_fast[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store[simd_width](flat, a.load[simd_width](flat) + b.load[simd_width](flat))

    elementwise[simd_width=width, target="gpu"](apply_fast, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Triu
# ===-------------------------------------------------------------------===#


def triu_impl[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    numel: Int,
    rows: Int,
    cols: Int,
    rank: Int,
    inner: UnsafePointer[Int64, ImmutAnyOrigin],
    sa: UnsafePointer[Int64, ImmutAnyOrigin],
    diagonal: Int64,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var diag = Int(diagonal)
        var flat = Int(coord[0].value())
        var idx_in_last2d = flat % (rows * cols)
        var row = idx_in_last2d // cols
        var col = idx_in_last2d % cols
        if col >= row + diag:
            dst.store(flat, a.load(strided_offset(flat, rank, inner, sa)))
        else:
            dst.store(flat, Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(numel), ctx)
