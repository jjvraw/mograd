from std.math import log as math_log, exp as math_exp
from std.algorithm.functional import elementwise

# ===-------------------------------------------------------------------===#
# Neg
# ===-------------------------------------------------------------------===#


def neg[
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
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
        dst.store(flat, -a.load(a_off))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Log
# ===-------------------------------------------------------------------===#


def log[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
        dst.store(flat, math_log(a.load(a_off)))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Exp
# ===-------------------------------------------------------------------===#


def exp[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
        dst.store(flat, math_exp(a.load(a_off)))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Relu
# ===-------------------------------------------------------------------===#


def relu[
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
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
        x = a.load(a_off)
        dst.store(flat, x if x > Scalar[dtype](0) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


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
        var rem = flat
        var a_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
        dst.store(flat, a.load(a_off).cast[dst_dtype]())

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Slice grad
# ===-------------------------------------------------------------------===#


def slice_grad[
    dtype: DType
](
    upstream: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
        dst.store(a_off, upstream.load(flat))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Add
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


def add_strided[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        var b_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
            b_off += idx * Int(sb[i])
        dst.store(flat, a.load(a_off) + b.load(b_off))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Multiply
# ===-------------------------------------------------------------------===#


def mul[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        var b_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
            b_off += idx * Int(sb[i])
        dst.store(flat, a.load(a_off) * b.load(b_off))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Divide
# ===-------------------------------------------------------------------===#


def div[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        var b_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
            b_off += idx * Int(sb[i])
        dst.store(flat, a.load(a_off) / b.load(b_off))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Equals
# ===-------------------------------------------------------------------===#


def eq[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        var b_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
            b_off += idx * Int(sb[i])
        dst.store(flat, Scalar[dtype](1) if a.load(a_off) == b.load(b_off) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Relu grad
# ===-------------------------------------------------------------------===#


def relu_grad[
    dtype: DType
](
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    upstream: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var x_off = 0
        var up_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            x_off += idx * Int(sa[i])
            up_off += idx * Int(sb[i])
        x_val = x.load(x_off)
        up_val = upstream.load(up_off)
        dst.store(flat, up_val if x_val > Scalar[dtype](0) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Scale
# ===-------------------------------------------------------------------===#


def scale[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    var scalar_val = b.load(0)

    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
        dst.store(flat, a.load(a_off) * scalar_val)

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)
