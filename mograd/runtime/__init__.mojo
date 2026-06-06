from std.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle
from std.pathlib.path import Path
from std.os.env import getenv

from mograd import Device
from mograd.op import OpRef, OpType
from mograd.buffer import AnyBuffer, Buffer, BufferArm
from mograd.pattern_matcher import Rule, Pat
from mograd.scheduler import Scheduler, BoundExecFn
from mograd.simplify import Simplifier
from mograd.runtime.native.gpu.rewrites import MATMUL_T, native_gpu_rewrites


def path() raises -> String:
    var p = getenv("MOGRAD_SO")
    if not p:
        raise Error("MOGRAD_SO not set — run `pixi run build-gpu` first")
    return p


# ===-------------------------------------------------------------------===#
# Runtime
# ===-------------------------------------------------------------------===#


trait Runtime:
    @staticmethod
    def run(root: OpRef, ctx: Optional[Device]) raises -> AnyBuffer:
        ...


struct NativeRuntime(Runtime):
    @staticmethod
    def run(root: OpRef, ctx: Optional[Device]) raises -> AnyBuffer:
        if not ctx:
            raise Error("NativeRuntime requires a Device")
        var optimized = Simplifier(native_gpu_rewrites()).run(root)
        return Scheduler(
            [
                # Unary
                Rule(Pat(OpType.RANDN), randn),
                Rule(Pat(OpType.FULL), full),
                Rule(Pat(OpType.ONE_HOT), one_hot),
                Rule(Pat(OpType.ADD), add),
                Rule(Pat(OpType.EQ), eq),
                Rule(Pat(OpType.MUL), mul),
                Rule(Pat(OpType.DIV), div),
                # Binary
                Rule(Pat(OpType.LOG), log),
                Rule(Pat(OpType.RELU), relu),
                Rule(Pat(OpType.EXP), exp),
                Rule(Pat(OpType.NEG), neg),
                Rule(Pat(OpType.SOFTMAX), softmax),
                Rule(Pat(OpType.ARGMAX), argmax),
                Rule(Pat(OpType.SUM), sum),
                Rule(Pat(OpType.CROSS_ENTROPY), cross_entropy),
                Rule(Pat(OpType.MATMUL), matmul),
                Rule(Pat(MATMUL_T), matmul_t),
                Rule(Pat(OpType.TRANSPOSE), transpose),
                Rule(Pat(OpType.SCALE), scale),
                Rule(Pat(OpType.DISK), disk),
                Rule(Pat(OpType.SLICE), slice),
                Rule(Pat(OpType.CAST), cast),
                Rule(Pat(OpType.BROADCAST), broadcast),
                Rule(Pat(OpType.UNIFORM), uniform),
                Rule(Pat(OpType.RESHAPE), reshape),
                Rule(Pat(OpType.RELU_GRAD), relu_grad),
                Rule(Pat(OpType.SOFTMAX_GRAD), softmax_grad),
                Rule(Pat(OpType.CROSS_ENTROPY_GRAD), cross_entropy_grad),
            ]
        ).run(optimized, ctx.value())


# ===-------------------------------------------------------------------===#
# UnaryOps
# ===-------------------------------------------------------------------===#

comptime UnaryOp = def(
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    Int,
    DType,
    DeviceContext,
) thin abi("C") raises -> None


comptime OneHotOp = def(
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    Int,
    DType,
    DType,
    DeviceContext,
) thin abi("C") raises -> None


def one_hot(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var c_ptr = alloc[Float32](1)
    c_ptr[0] = node.attrs()["num_classes"][Float32]
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[OneHotOp]("mograd_one_hot")(
        inputs[0].data_ptr(),
        c_ptr.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        inputs[0].dtype(),
        node.dtype(),
        ctx.ctx,
    )
    c_ptr.free()
    return out^


def randn(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var params = alloc[Float32](3)
    params[0] = node.attrs()["mean"][Float32]
    params[1] = node.attrs()["std"][Float32]
    params[2] = node.attrs()["seed"][Float32]
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_randn")(
        params.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    params.free()
    return out^


def full(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var v = alloc[Float32](1)
    v[0] = node.attrs()["value"][Float32]
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_full")(
        v.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    v.free()
    return out^


def argmax(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var p = alloc[Float32](2)
    p[0] = Float32(inputs[0].shape()[0])
    p[1] = Float32(inputs[0].shape()[1])
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_argmax")(
        inputs[0].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    p.free()
    return out^


def sum(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), (1,), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_sum")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        node.src(0).shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    return out^


def softmax(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var shape = node.shape()
    var rank = len(shape)
    var params = alloc[Float32](2)
    params[0] = Float32(shape[0] if rank > 1 else 1)
    params[1] = Float32(shape[rank - 1])
    var out = AnyBuffer.empty(node.dtype(), shape, ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_softmax")(
        inputs[0].data_ptr(),
        params.bitcast[NoneType](),
        out.data_ptr(),
        shape.numel(),
        node.dtype(),
        ctx.ctx,
    )
    params.free()
    return out^


def log(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_log")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    return out^


def relu(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_relu")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    return out^


def exp(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_exp")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    return out^


def neg(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_neg")(
        inputs[0].data_ptr(),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
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
) thin abi("C") raises -> None


def eq(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_eq")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    return out^


def mul(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_mul")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    return out^


def div(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_div")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    return out^


def add(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var func = ctx.handle[].get_function[BinaryOp]("mograd_add")
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    func(inputs[0].data_ptr(), inputs[1].data_ptr(), out.data_ptr(), node.shape().numel(), node.dtype(), ctx.ctx)
    return out^


def transpose(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var p = alloc[Float32](2)
    p[0] = Float32(inputs[0].shape()[0])
    p[1] = Float32(inputs[0].shape()[1])
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_transpose")(
        inputs[0].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    p.free()
    return out^


def scale(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var func = ctx.handle[].get_function[BinaryOp]("mograd_scale")
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    var s = alloc[Float32](1)
    s[0] = node.attrs()["scalar"][Float32]
    func(inputs[0].data_ptr(), s.bitcast[NoneType](), out.data_ptr(), node.shape().numel(), node.dtype(), ctx.ctx)
    s.free()
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
) thin abi("C") raises -> None


def cross_entropy(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var p = alloc[Float32](2)
    p[0] = Float32(inputs[0].shape()[0])
    p[1] = Float32(inputs[0].shape()[1])
    var out = AnyBuffer.empty(node.dtype(), (1,), ctx)
    ctx.handle[].get_function[TernaryOp]("mograd_cross_entropy")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    p.free()
    return out^


def matmul(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var p = alloc[Float32](3)
    p[0] = Float32(inputs[0].shape()[0])
    p[1] = Float32(inputs[0].shape()[1])
    p[2] = Float32(inputs[1].shape()[1])
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[TernaryOp]("mograd_matmul")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    p.free()
    return out^


def disk(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var size = node.shape().numel()
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
            return AnyBuffer(Buffer[d].from_data(ctx, data, node.shape()))
    raise Error("unsupported dtype")


def slice(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var start = Int(node.attrs()["start"][Float32])
    var cols = node.src(0).shape().numel() // node.src(0).shape()[0]
    return inputs[0].view(node.shape(), start * cols)


def matmul_t(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var p = alloc[Float32](3)
    p[0] = Float32(inputs[0].shape()[0])
    p[1] = Float32(inputs[0].shape()[1])
    p[2] = Float32(inputs[1].shape()[0])
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[TernaryOp]("mograd_matmul_t")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    p.free()
    return out^


# ===-------------------------------------------------------------------===#
# CastOp
# ===-------------------------------------------------------------------===#

comptime CastOp = def(
    UnsafePointer[NoneType, MutAnyOrigin],
    UnsafePointer[NoneType, MutAnyOrigin],
    Int,
    DType,
    DType,
    DeviceContext,
) thin abi("C") raises -> None


def cast(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[CastOp]("mograd_cast")(
        inputs[0].data_ptr(), out.data_ptr(), node.shape().numel(), inputs[0].dtype(), node.dtype(), ctx.ctx
    )
    return out^


# ===-------------------------------------------------------------------===#
# Shape / metadata ops
# ===-------------------------------------------------------------------===#


def broadcast(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var p = alloc[Float32](1)
    p[0] = Float32(inputs[0].size())
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_broadcast")(
        inputs[0].data_ptr(), p.bitcast[NoneType](), out.data_ptr(), node.shape().numel(), node.dtype(), ctx.ctx
    )
    p.free()
    return out^


def uniform(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var params = alloc[Float32](3)
    params[0] = node.attrs()["low"][Float32]
    params[1] = node.attrs()["high"][Float32]
    params[2] = node.attrs()["seed"][Float32]
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[UnaryOp]("mograd_uniform")(
        params.bitcast[NoneType](), out.data_ptr(), node.shape().numel(), node.dtype(), ctx.ctx
    )
    params.free()
    return out^


def reshape(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    return inputs[0].reshape(node.shape())


# ===-------------------------------------------------------------------===#
# Grad ops
# ===-------------------------------------------------------------------===#


def relu_grad(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[BinaryOp]("mograd_relu_grad")(
        inputs[0].data_ptr(), inputs[1].data_ptr(), out.data_ptr(), node.shape().numel(), node.dtype(), ctx.ctx
    )
    return out^


def softmax_grad(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var shape = node.shape()
    var rank = len(shape)
    var N = 1 if rank == 1 else shape[0]
    var size = shape[rank - 1]
    var p = alloc[Float32](2)
    p[0] = Float32(N)
    p[1] = Float32(size)
    var out = AnyBuffer.empty(node.dtype(), shape, ctx)
    ctx.handle[].get_function[TernaryOp]("mograd_softmax_grad")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        shape.numel(),
        node.dtype(),
        ctx.ctx,
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
) thin abi("C") raises -> None


def cross_entropy_grad(node: OpRef, inputs: List[AnyBuffer], ctx: Device) raises -> AnyBuffer:
    var p = alloc[Float32](2)
    p[0] = Float32(inputs[0].shape()[0])
    p[1] = Float32(inputs[0].shape()[1])
    var out = AnyBuffer.empty(node.dtype(), node.shape(), ctx)
    ctx.handle[].get_function[QuaternaryOp]("mograd_cross_entropy_grad")(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        p.bitcast[NoneType](),
        out.data_ptr(),
        node.shape().numel(),
        node.dtype(),
        ctx.ctx,
    )
    p.free()
    return out^
