from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext

from layout import Coord

from mograd.buffer import AnyBuffer, BufferArm
from mograd.layout import Layout

# ===-------------------------------------------------------------------===#
# Strided offset helpers
# ===-------------------------------------------------------------------===#


@always_inline
def strided_offset(
    flat: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    strides: UnsafePointer[mut=False, Int64, _],
) -> Int:
    var rem = flat
    var off = 0
    for i in range(rank):
        var idx = rem // Int(inner[i])
        rem %= Int(inner[i])
        off += idx * Int(strides[i])
    return off


@always_inline
def strided_offsets(
    flat: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    sb: UnsafePointer[mut=False, Int64, _],
) -> Tuple[Int, Int]:
    var rem = flat
    var a_off = 0
    var b_off = 0
    for i in range(rank):
        var idx = rem // Int(inner[i])
        rem %= Int(inner[i])
        a_off += idx * Int(sa[i])
        b_off += idx * Int(sb[i])
    return (a_off, b_off)


# ===-------------------------------------------------------------------===#
# Generic strided GPU kernels
# ===-------------------------------------------------------------------===#


def unary_strided_map[
    dtype: DType,
    op: def(Scalar[dtype]) thin -> Scalar[dtype],
](
    a: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store(flat, op(a.load(strided_offset(flat, rank, inner, sa))))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


def binary_strided_scalar_map[
    dtype: DType,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](
    a: UnsafePointer[mut=False, Scalar[dtype], _],
    b: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    n: Int,
    ctx: DeviceContext,
) raises:
    scalar_val = b.load(0)

    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store(flat, op(a.load(strided_offset(flat, rank, inner, sa)), scalar_val))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


def binary_strided_map[
    dtype: DType,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](
    a: UnsafePointer[mut=False, Scalar[dtype], _],
    b: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    sb: UnsafePointer[mut=False, Int64, _],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var offs = strided_offsets(flat, rank, inner, sa, sb)
        dst.store(flat, op(a.load(offs[0]), b.load(offs[1])))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Dtype dispatch
# ===-------------------------------------------------------------------===#


def dispatch_dtype[
    body: def[d: DType]() capturing raises -> None,
    float_only: Bool = False,
](dtype: DType) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if (not float_only) or d.is_floating_point():
            if dtype == d:
                body[d]()
                return
    raise Error("unsupported dtype")


# ===-------------------------------------------------------------------===#
# SO-internal dispatch helpers
# Used inside the GPU shared library by exported functions.
# ===-------------------------------------------------------------------===#

comptime UnaryStridedKernel = def[dtype: DType](
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, ImmutAnyOrigin],
    sa: UnsafePointer[Int64, ImmutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises thin -> None

comptime BinaryContigKernel = def[dtype: DType](
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises thin -> None


def dispatch_unary[
    kernel: UnaryStridedKernel,
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() raises capturing:
        kernel[d](
            a.bitcast[Scalar[d]](),
            dst.bitcast[Scalar[d]](),
            rank,
            inner.as_unsafe_any_origin(),
            sa.as_unsafe_any_origin(),
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_contiguous[
    kernel: BinaryContigKernel,
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    b: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        kernel[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), numel, ctx)

    dispatch_dtype[body, float_only](dtype)


def dispatch_unary_map[
    op: def[d: DType](x: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        unary_strided_map[d, op[d]](
            a.bitcast[Scalar[d]](),
            dst.bitcast[Scalar[d]](),
            rank,
            inner,
            sa,
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_map[
    op: def[d: DType](x: Scalar[d], y: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    b: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    sb: UnsafePointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        binary_strided_map[d, op[d]](
            a.bitcast[Scalar[d]](),
            b.bitcast[Scalar[d]](),
            dst.bitcast[Scalar[d]](),
            rank,
            inner,
            sa,
            sb,
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_scalar_map[
    op: def[d: DType](x: Scalar[d], y: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    b: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[mut=False, Int64, _],
    sa: UnsafePointer[mut=False, Int64, _],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        binary_strided_scalar_map[d, op[d]](
            a.bitcast[Scalar[d]](),
            b.bitcast[Scalar[d]](),
            dst.bitcast[Scalar[d]](),
            rank,
            inner,
            sa,
            numel,
            ctx,
        )

    dispatch_dtype[body, float_only](dtype)


# ===-------------------------------------------------------------------===#
# CPU-side runtime helpers
# Load SO functions by name and dispatch from the graph scheduler.
# ===-------------------------------------------------------------------===#

comptime FactoryKernel = def(
    params: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None

comptime UnaryStrided = def(
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    read layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def unary_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[UnaryStrided](name)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        layout,
        node.dtype(),
        device.ctx,
    )
    return out^


comptime AxisReduceKernel = def(
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    read layout: Layout,
    axis: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def axis_reduce_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var axis = node.attr_int("axis")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[AxisReduceKernel](name)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        layout,
        axis,
        node.dtype(),
        device.ctx,
    )
    return out^


comptime MatmulStrided = def(
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    b: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    read la: Layout,
    read lb: Layout,
    N: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def matmul_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[MatmulStrided](name)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.shape(node.layout().rank() - 1),
        node.dtype(),
        device.ctx,
    )
    return out^


comptime MatmulBiasStrided = def(
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    b: UnsafePointer[NoneType, ImmutAnyOrigin],
    bias: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    read la: Layout,
    read lb: Layout,
    N: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def matmul_bias_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[MatmulBiasStrided](name)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.shape(node.layout().rank() - 1),
        node.dtype(),
        device.ctx,
    )
    return out^


comptime BinaryStrided = def(
    a: UnsafePointer[NoneType, ImmutAnyOrigin],
    b: UnsafePointer[NoneType, ImmutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    read la: Layout,
    read lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def binary_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[BinaryStrided](name)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.dtype(),
        device.ctx,
    )
    return out^
