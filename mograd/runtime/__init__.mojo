from std.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle
from std.pathlib.path import Path
from std.os.env import getenv

from mograd import Device
from mograd.layout import Layout
from mograd.op import Op, OpRef, OpType
from mograd.buffer import AnyBuffer, Buffer, BufferArm
from mograd.pattern_matcher import Rule, Pat
from mograd.scheduler import Scheduler, BoundExecFn, SchedulerRules
from mograd.simplify import Simplifier
from mograd.runtime.gpu.rewrites import MATMUL_T, GPU_REWRITES
from mograd.runtime.gpu.kernels.utils import (
    FactoryKernel,
    UnaryStrided,
    unary_strided,
    binary_strided,
    axis_reduce_strided,
    matmul_strided,
)

# ===-------------------------------------------------------------------===#
# Runtime
# ===-------------------------------------------------------------------===#


trait Runtime:
    @staticmethod
    def run(
        root: OpRef, device: Optional[Device], var extern_rules: Optional[SchedulerRules] = None
    ) raises -> AnyBuffer:
        ...


# TODO: Move to GPURuntime, have seperate CPURuntime. Hence why we have a trait.
struct NativeRuntime(Runtime):
    @staticmethod
    def run(
        root: OpRef, device: Optional[Device], var extern_rules: Optional[SchedulerRules] = None
    ) raises -> AnyBuffer:
        if not device:
            raise Error("NativeRuntime requires a Device")

        var extra_sched = SchedulerRules()
        var compound = SchedulerRules()

        if extern_rules:
            for j in range(len(extern_rules.value())):
                if extern_rules.value()[j].pat.is_compound():
                    compound.append(extern_rules.value()[j].copy())
                else:
                    extra_sched.append(extern_rules.value()[j].copy())

        var simplified = Simplifier(GPU_REWRITES()).run(root, compound^, extra_sched)

        var scheduler_rules: List[Rule[BoundExecFn]] = [
            # Factory
            Rule(Pat(OpType.RANDN), randn),
            Rule(Pat(OpType.UNIFORM), uniform),
            Rule(Pat(OpType.FULL), full),
            Rule(Pat(OpType.DISK), disk),
            # Unary Elementwise
            Rule(Pat(OpType.NEG), neg),
            Rule(Pat(OpType.LOG), log),
            Rule(Pat(OpType.EXP), exp),
            Rule(Pat(OpType.RELU), relu),
            Rule(Pat(OpType.CAST), cast),
            # Binary Elementwise
            Rule(Pat(OpType.ADD), add),
            Rule(Pat(OpType.MUL), mul),
            Rule(Pat(OpType.DIV), div),
            Rule(Pat(OpType.EQ), eq),
            Rule(Pat(OpType.RELU_GRAD), relu_grad),
            Rule(Pat(OpType.SCALE), scale),
            Rule(Pat(OpType.SLICE_GRAD), slice_grad),
            # Reduce
            Rule(Pat(OpType.SUM), sum),
            Rule(Pat(OpType.SOFTMAX), softmax),
            Rule(Pat(OpType.ARGMAX), argmax),
            # Linalg
            Rule(Pat(OpType.MATMUL), matmul),
            Rule(Pat(MATMUL_T), matmul_t),
            # Indexing & Encoding
            Rule(Pat(OpType.ONE_HOT), one_hot),
            # Layout
            Rule(Pat(OpType.CONTIGUOUS), contiguous),
            Rule(Pat(OpType.RESHAPE), reshape),
            Rule(Pat(OpType.EXPAND), expand),
            Rule(Pat(OpType.VIEW), view),
            Rule(Pat(OpType.SLICE), slice),
            # TODO:
            Rule(Pat(OpType.CROSS_ENTROPY), cross_entropy),
            Rule(Pat(OpType.TRANSPOSE), transpose),
            Rule(Pat(OpType.SOFTMAX_GRAD), softmax_grad),
            Rule(Pat(OpType.CROSS_ENTROPY_GRAD), cross_entropy_grad),
        ]

        extra_sched += scheduler_rules^
        return Scheduler(extra_sched^).run(simplified, device.value())


# ===-------------------------------------------------------------------===#
# Signatures
# ===-------------------------------------------------------------------===#

comptime OneHotOp = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    in_dtype: DType,
    out_dtype: DType,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sd: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None

comptime BinaryElementWise = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None

comptime BinaryScalarElementWiseStrided = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    strides_a: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None

comptime CastOp = def(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    read layout: Layout,
    in_dtype: DType,
    out_dtype: DType,
    ctx: DeviceContext,
) thin abi("Mojo") raises -> None

# ===-------------------------------------------------------------------===#
# Factory
# ===-------------------------------------------------------------------===#


def randn(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var params = alloc[Float32](3)
    params[0] = node.attr("mean")
    params[1] = node.attr("std")
    params[2] = node.attr("seed")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[FactoryKernel]("mograd_randn")(
        params.bitcast[NoneType](),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        device.ctx,
    )
    params.free()
    return out^


def uniform(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var params = alloc[Float32](3)
    params[0] = node.attr("low")
    params[1] = node.attr("high")
    params[2] = node.attr("seed")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[FactoryKernel]("mograd_uniform")(
        params.bitcast[NoneType](), out.data_ptr(), node.numel(), node.dtype(), device.ctx
    )
    params.free()
    return out^


def full(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var v = alloc[Float32](1)
    v[0] = node.attr("value")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[FactoryKernel]("mograd_full")(
        v.bitcast[NoneType](),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        device.ctx,
    )
    v.free()
    return out^


def one_hot(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var ld = node.layout()
    var out = AnyBuffer.create(node.dtype(), device, ld.numel())
    device.handle[].get_function[OneHotOp]("mograd_one_hot")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        ld.numel(),
        node.src(0).dtype(),
        node.dtype(),
        ld.rank(),
        ld.inner_sizes_buffer(device.ctx).unsafe_ptr(),
        ld.strides_buffer(device.ctx).unsafe_ptr(),
        la.strides_buffer(device.ctx).unsafe_ptr(),
        device.ctx,
    )
    return out^


# ===-------------------------------------------------------------------===#
# Unary Elementwise
# ===-------------------------------------------------------------------===#


def neg(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return unary_strided("mograd_neg", node, inputs, device)


def log(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return unary_strided("mograd_log", node, inputs, device)


def exp(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return unary_strided("mograd_exp", node, inputs, device)


def relu(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return unary_strided("mograd_relu", node, inputs, device)


def cast(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var layout = node.src(0).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[CastOp]("mograd_cast")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        layout,
        node.src(0).dtype(),
        node.dtype(),
        device.ctx,
    )
    return out^


# ===-------------------------------------------------------------------===#
# Binary Elementwise
# ===-------------------------------------------------------------------===#


def add(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    if la.is_contiguous() and lb.is_contiguous():
        null = alloc[Int64](1)
        device.handle[].get_function[BinaryElementWise]("mograd_add")(
            inputs[0].data_ptr(),
            inputs[1].data_ptr(),
            out.data_ptr(),
            node.numel(),
            node.dtype(),
            device.ctx,
        )
        null.free()
        return out^

    return binary_strided("mograd_add_strided", node, inputs, device)


def mul(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return binary_strided("mograd_mul", node, inputs, device)


def div(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return binary_strided("mograd_div", node, inputs, device)


def eq(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return binary_strided("mograd_eq", node, inputs, device)


def relu_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return binary_strided("mograd_relu_grad", node, inputs, device)


def scale(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var la = node.src(0).layout()
    var s = alloc[Float32](1)
    s[0] = node.attrs()["scalar"][Float32]
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[BinaryScalarElementWiseStrided]("mograd_scale")(
        inputs[0].data_ptr(),
        s.bitcast[NoneType](),
        out.data_ptr(),
        node.numel(),
        la.rank(),
        la.inner_sizes_buffer(device.ctx).unsafe_ptr(),
        la.strides_buffer(device.ctx).unsafe_ptr(),
        node.dtype(),
        device.ctx,
    )
    return out^


def slice_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var slice_layout = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel(), fill=0.0)
    var out_view = out.view(slice_layout)
    device.handle[].get_function[UnaryStrided]("mograd_slice_grad")(
        inputs[0].data_ptr(),
        out_view.data_ptr(),
        slice_layout.numel(),
        slice_layout.rank(),
        slice_layout.inner_sizes_buffer(device.ctx).unsafe_ptr(),
        slice_layout.strides_buffer(device.ctx).unsafe_ptr(),
        node.dtype(),
        device.ctx,
    )
    return out^


# ===-------------------------------------------------------------------===#
# Binary Reduce
# ===-------------------------------------------------------------------===#


def sum(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    if "axis" in node.attrs():
        return axis_reduce_strided("mograd_sum_axis", node, inputs, device)
    return unary_strided("mograd_sum", node, inputs, device)


def argmax(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    if "axis" in node.attrs():
        return axis_reduce_strided("mograd_argmax_axis", node, inputs, device)
    return unary_strided("mograd_argmax", node, inputs, device)


# ===-------------------------------------------------------------------===#
# Matmul
# ===-------------------------------------------------------------------===#


def matmul(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return matmul_strided("mograd_matmul", node, inputs, device)


def matmul_t(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return matmul_strided("mograd_matmul_t", node, inputs, device)


# ===-------------------------------------------------------------------===#
# Shape / metadata ops
# ===-------------------------------------------------------------------===#


def contiguous(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return unary_strided("mograd_contiguous", node, inputs, device)


def expand(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return inputs[0].view(node.layout())


def reshape(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return inputs[0].view(node.layout())


def view(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return inputs[0].view(node.layout())


# _-_-_-_-_-
# TODO: All the below
# _-_-_-_-_-


def softmax(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var layout = node.layout()
    var rank = layout.rank()
    var params = alloc[Float32](2)
    params[0] = Float32(layout.shape(0) if rank > 1 else 1)
    params[1] = Float32(layout.shape(rank - 1))
    var out = AnyBuffer.create(node.dtype(), device, layout.numel())
    device.handle[].get_function[BinaryOp]("mograd_softmax")(
        inputs[0].data_ptr(),
        params.bitcast[NoneType](),
        out.data_ptr(),
        layout.numel(),
        node.dtype(),
        device.ctx,
    )
    params.free()
    return out^


# ===-------------------------------------------------------------------===#
# BinaryOps
# ===-------------------------------------------------------------------===#

comptime BinaryOp = def(
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    Int,
    DType,
    DeviceContext,
) thin abi("Mojo") raises -> None


def transpose(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var p = alloc[Float32](2)
    p[0] = Float32(node.src(0).layout().shape(0))
    p[1] = Float32(node.src(0).layout().shape(1))
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[BinaryOp]("mograd_transpose")(
        inputs[0].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        device.ctx,
    )
    p.free()
    return out^


# ===-------------------------------------------------------------------===#
# BinaryOps
# ===-------------------------------------------------------------------===#

comptime TernaryOp = def(
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    Int,
    DType,
    DeviceContext,
) thin abi("Mojo") raises -> None


def cross_entropy(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var p = alloc[Float32](2)
    p[0] = Float32(node.src(0).layout().shape(0))
    p[1] = Float32(node.src(0).layout().shape(1))
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[TernaryOp]("mograd_cross_entropy")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        device.ctx,
    )
    p.free()
    return out^


# TODO: Revisit this, this is slop and can probably be cleaned up.
def disk(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var size = node.numel()
    var bytes = Path(node.attrs()["path"][String]).read_bytes()
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if node.dtype() == d:
            var ptr = bytes.unsafe_ptr().bitcast[Scalar[d]]()
            var data = List[Scalar[d]]()
            data.reserve(size)
            for i in range(size):
                data.append(ptr[i])
            return AnyBuffer(Buffer[d].from_data(device, data))
    raise Error("unsupported dtype")


def slice(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    return inputs[0].view(node.layout())


# ===-------------------------------------------------------------------===#
# Grad ops
# ===-------------------------------------------------------------------===#


def softmax_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var layout = node.layout()
    var rank = layout.rank()
    var N = 1 if rank == 1 else layout.shape(0)
    var size = layout.shape(rank - 1)
    var p = alloc[Float32](2)
    p[0] = Float32(N)
    p[1] = Float32(size)
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[TernaryOp]("mograd_softmax_grad")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        layout.numel(),
        node.dtype(),
        device.ctx,
    )
    p.free()
    return out^


comptime QuaternaryOp = def(
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    Int,
    DType,
    DeviceContext,
) thin abi("Mojo") raises -> None


def cross_entropy_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> AnyBuffer:
    var p = alloc[Float32](2)
    p[0] = Float32(node.src(0).layout().shape(0))
    p[1] = Float32(node.src(0).layout().shape(1))
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    device.handle[].get_function[QuaternaryOp]("mograd_cross_entropy_grad")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        device.ctx,
    )
    p.free()
    return out^
