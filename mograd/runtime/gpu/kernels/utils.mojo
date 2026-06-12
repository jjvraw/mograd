from layout import Coord
from std.algorithm.functional import elementwise
from std.gpu.host import DeviceContext

from mograd.buffer import AnyBuffer, BufferArm

# ===-------------------------------------------------------------------===#
# Strided offset helpers
# ===-------------------------------------------------------------------===#


@always_inline
def strided_offset(
    flat: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    strides: UnsafePointer[Int64, MutAnyOrigin],
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
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
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
        dst.store(flat, op(a.load(strided_offset(flat, rank, inner, sa))))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


def binary_strided_scalar_map[
    dtype: DType,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
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
    scalar_val = b.load(0)

    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store(flat, op(a.load(strided_offset(flat, rank, inner, sa)), scalar_val))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


def binary_strided_map[
    dtype: DType,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
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
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises thin -> None

comptime BinaryContigKernel = def[dtype: DType](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises thin -> None


def dispatch_unary[
    kernel: UnaryStridedKernel,
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() raises capturing:
        kernel[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), rank, inner, sa, numel, ctx)

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_contiguous[
    kernel: BinaryContigKernel,
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
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
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        unary_strided_map[d, op[d]](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), rank, inner, sa, numel, ctx)

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_map[
    op: def[d: DType](x: Scalar[d], y: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        binary_strided_map[d, op[d]](
            a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), rank, inner, sa, sb, numel, ctx
        )

    dispatch_dtype[body, float_only](dtype)


def dispatch_binary_scalar_map[
    op: def[d: DType](x: Scalar[d], y: Scalar[d]) thin -> Scalar[d],
    float_only: Bool = False,
](
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) raises:
    @always_inline
    def body[d: DType]() capturing raises:
        binary_strided_scalar_map[d, op[d]](
            a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), rank, inner, sa, numel, ctx
        )

    dispatch_dtype[body, float_only](dtype)


# ===-------------------------------------------------------------------===#
# CPU-side runtime helpers
# Load SO functions by name and dispatch from the graph scheduler.
# ===-------------------------------------------------------------------===#

comptime FactoryKernel = def(
    params: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None

comptime UnaryStrided = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    strides_a: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def unary_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.layout().numel())
    device.handle[].get_function[UnaryStrided](name)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        layout.numel(),
        layout.rank(),
        layout.inner_sizes_buffer(device.ctx).unsafe_ptr(),
        layout.strides_buffer(device.ctx).unsafe_ptr(),
        node.dtype(),
        device.ctx,
    )
    return out^


comptime SumAxisKernel = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    outer: Int,
    reduce_size: Int,
    inner: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def axis_reduce_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var axis = node.attr_int("axis")
    var outer, reduce_size, inner = layout.reduce_dims(axis)
    var out = AnyBuffer.create(node.dtype(), device, node.layout().numel())
    device.handle[].get_function[SumAxisKernel](name)(
        inputs[0].data_ptr(), out.data_ptr(), outer, reduce_size, inner, node.dtype(), device.ctx
    )
    return out^


comptime ArgmaxAxisKernel = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    outer: Int,
    reduce_size: Int,
    inner: Int,
    rank: Int,
    inner_sizes: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    outer_stride: Int,
    reduce_stride: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def argmax_axis_strided(read node: OpRef, read inputs: List[AnyBuffer], read device: Device) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var axis = node.attr_int("axis")
    var outer, reduce_size, inner = layout.reduce_dims(axis)
    var out = AnyBuffer.create(node.dtype(), device, node.layout().numel())
    device.handle[].get_function[ArgmaxAxisKernel]("mograd_argmax_axis")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        outer,
        reduce_size,
        inner,
        layout.rank(),
        layout.inner_sizes_buffer(device.ctx).unsafe_ptr(),
        layout.strides_buffer(device.ctx).unsafe_ptr(),
        layout.stride(0),
        layout.stride(axis),
        node.dtype(),
        device.ctx,
    )
    return out^


comptime MatmulStrided = def(
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    Int,
    DType,
    DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def matmul_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.layout().numel())
    device.handle[].get_function[MatmulStrided](name)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        la.shape(0),
        la.shape(1),
        node.layout().shape(1),
        la.stride(0),
        la.stride(1),
        lb.stride(0),
        lb.stride(1),
        node.dtype(),
        device.ctx,
    )
    return out^


comptime BinaryStrided = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    strides_a: UnsafePointer[Int64, MutAnyOrigin],
    strides_b: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None


@always_inline
def binary_strided(
    read name: String, read node: OpRef, read inputs: List[AnyBuffer], read device: Device
) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.layout().numel())
    device.handle[].get_function[BinaryStrided](name)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        node.layout().numel(),
        la.rank(),
        la.inner_sizes_buffer(device.ctx).unsafe_ptr(),
        la.strides_buffer(device.ctx).unsafe_ptr(),
        lb.strides_buffer(device.ctx).unsafe_ptr(),
        node.dtype(),
        device.ctx,
    )
    return out^
