from max.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle
from std.pathlib.path import Path
from std.os.env import getenv
from std.random import randint as host_randint, seed as host_seed

from layout.int_tuple import IntTuple

from mograd import Device
from mograd.layout import Layout
from mograd.op import Op, OpRef, OpType, sink
from mograd.buffer import AnyBuffer, Buffer, BufferArm
from mograd.pattern_matcher import Rule, Pat
from mograd.scheduler import Scheduler, BoundExecFn, SchedulerRules
from mograd.simplify import Simplifier
from mograd.runtime.gpu.rewrites import (
    MATMUL_BT,
    MATMUL_BIAS_BT,
    MEAN,
    LAYER_NORM,
    LAYER_NORM_GRAD,
    FLASH_ATTN,
    FLASH_ATTN_GRAD,
    GPU_REWRITES,
)
from mograd.runtime.gpu.kernels.dispatch import KernelRegistry as K
from std.memory.alloc import unsafe_alloc

from mograd.runtime.gpu.kernels.dispatch import (
    unary_strided,
    binary_strided,
    axis_reduce_strided,
    matmul_strided,
    matmul_bias_strided,
    layer_norm_fwd_dispatch,
    layer_norm_bwd_dispatch,
    flash_attn_fwd_dispatch,
    flash_attn_bwd_dispatch,
    strided_copy,
)

# ===-------------------------------------------------------------------===#
# Runtime
# ===-------------------------------------------------------------------===#


trait Runtime:
    @staticmethod
    def run(
        root: OpRef,
        device: Optional[Device],
        simplifier: Bool = True,
        var extern_rules: Optional[SchedulerRules] = None,
    ) raises -> AnyBuffer:
        ...


# TODO: Move to GPURuntime, have seperate CPURuntime. Hence why we have a trait.
struct NativeRuntime(Runtime):
    @staticmethod
    def run(
        root: OpRef,
        device: Optional[Device],
        simplifier: Bool = True,
        var extern_rules: Optional[SchedulerRules] = None,
    ) raises -> AnyBuffer:
        if not device:
            raise Error("NativeRuntime requires a Device")

        var prepared = Self._prepare(extern_rules^)
        ref compound = prepared[0]
        ref extra_sched = prepared[1]
        var simplified = Simplifier(GPU_REWRITES()).run(root, compound.copy(), extra_sched) if simplifier else root
        return Scheduler(extra_sched.copy()).run(simplified, device.value())

    @staticmethod
    def run_many(
        var targets: List[OpRef],
        device: Optional[Device],
        simplifier: Bool = True,
        var extern_rules: Optional[SchedulerRules] = None,
    ) raises -> List[AnyBuffer]:
        """Evaluates all `targets` in one simplify/schedule pass, so shared
        bookkeeping (toposort, substitution, scheduling) and the final sync
        happen once instead of once per target.
        """
        if not device:
            raise Error("NativeRuntime requires a Device")

        var prepared = Self._prepare(extern_rules^)
        ref compound = prepared[0]
        ref extra_sched = prepared[1]
        var simplified_bundle = Simplifier(GPU_REWRITES()).run(
            sink(targets^), compound.copy(), extra_sched
        ) if simplifier else sink(targets^)
        var simplified_targets = List[OpRef]()
        for i in range(len(simplified_bundle.srcs())):
            simplified_targets.append(simplified_bundle.src(i))
        return Scheduler(extra_sched.copy()).run_many(simplified_targets^, device.value())

    @staticmethod
    def _prepare(var extern_rules: Optional[SchedulerRules]) raises -> Tuple[SchedulerRules, SchedulerRules]:
        var extra_sched = SchedulerRules()
        var compound = SchedulerRules()

        if extern_rules:
            for j in range(len(extern_rules.value())):
                if extern_rules.value()[j].pat.is_compound():
                    compound.append(extern_rules.value()[j].copy())
                else:
                    extra_sched.append(extern_rules.value()[j].copy())

        var scheduler_rules: List[Rule[BoundExecFn]] = [
            # Factory
            Rule(Pat(OpType.RANDN), randn),
            Rule(Pat(OpType.UNIFORM), uniform),
            Rule(Pat(OpType.RANDINT), randint),
            Rule(Pat(OpType.FULL), full),
            Rule(Pat(OpType.DISK), disk),
            # Unary Elementwise
            Rule(Pat(OpType.NEG), neg),
            Rule(Pat(OpType.LOG), log),
            Rule(Pat(OpType.EXP), exp),
            Rule(Pat(OpType.SQRT), sqrt),
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
            Rule(Pat(MEAN), mean),
            Rule(Pat(OpType.ARGMAX), argmax),
            # Linalg
            Rule(Pat(OpType.MATMUL), matmul),
            Rule(Pat(MATMUL_BT), matmul_t),
            Rule(Pat(MATMUL_BIAS_BT), matmul_bias_bt),
            Rule(Pat(LAYER_NORM), layer_norm_fwd),
            Rule(Pat(LAYER_NORM_GRAD), layer_norm_bwd),
            Rule(Pat(FLASH_ATTN), flash_attn_fwd_sched),
            Rule(Pat(FLASH_ATTN_GRAD), flash_attn_bwd_sched),
            # Indexing & Encoding
            Rule(Pat(OpType.ONE_HOT), one_hot),
            Rule(Pat(OpType.GATHER), gather),
            Rule(Pat(OpType.SCATTER_ADD), scatter_add),
            # Layout
            Rule(Pat(OpType.CONTIGUOUS), contiguous),
            Rule(Pat(OpType.EXPAND), view),
            Rule(Pat(OpType.VIEW), view),
            Rule(Pat(OpType.SLICE), view),
            Rule(Pat(OpType.RESHAPE), view),
            Rule(Pat(OpType.TRANSPOSE), view),
            Rule(Pat(OpType.SQUEEZE), view),
            Rule(Pat(OpType.UNSQUEEZE), view),
            Rule(Pat(OpType.TRIU), triu),
            Rule(Pat(OpType.CONCAT), concat),
            # TODO: Make layout aware.
            Rule(Pat(OpType.SOFTMAX), softmax),
            Rule(Pat(OpType.CROSS_ENTROPY), cross_entropy),
            Rule(Pat(OpType.SOFTMAX_GRAD), softmax_grad),
            Rule(Pat(OpType.CROSS_ENTROPY_GRAD), cross_entropy_grad),
        ]

        extra_sched += scheduler_rules^
        return (compound^, extra_sched^)


# ===-------------------------------------------------------------------===#
# Factory
# ===-------------------------------------------------------------------===#


def randn(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var params = unsafe_alloc[Float32](2)
    params[unsafe_offset=0] = node.attr("mean")
    params[unsafe_offset=1] = node.attr("std")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    K.randn.load(device)(
        params.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        UInt64(node.attr_int("seed")),
        device.ctx,
    )
    params.unsafe_free()
    return [out^]


def uniform(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var params = unsafe_alloc[Float32](2)
    params[unsafe_offset=0] = node.attr("low")
    params[unsafe_offset=1] = node.attr("high")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    K.uniform.load(device)(
        params.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        UInt64(node.attr_int("seed")),
        device.ctx,
    )
    params.unsafe_free()
    return [out^]


def randint(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    if node.dtype() != DType.int64:
        raise Error("randint: only int64 output is supported")
    var low = node.attr_int("low")
    var high = node.attr_int("high")
    host_seed(node.attr_int("seed"))
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    with out.unsafe_get[DType.int64]().buf().map_to_host() as host:
        host_randint(host.as_span(), low, high - 1)
    return [out^]


def full(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var v = unsafe_alloc[Float32](1)
    v[unsafe_offset=0] = node.attr("value")
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    K.full.load(device)(
        v.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        device.ctx,
    )
    v.unsafe_free()
    return [out^]


def gather(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [binary_strided(K.gather, node, inputs, device)]


def scatter_add(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel(), fill=0.0)
    K.scatter_add.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.dtype(),
        device.ctx,
    )
    return [out^]


def one_hot(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var la = node.src(0).layout()
    var ld = node.layout()
    var out = AnyBuffer.create(node.dtype(), device, ld.numel())
    K.one_hot.load(device)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        node.src(0).dtype(),
        node.dtype(),
        ld,
        la,
        device.ctx,
    )
    return [out^]


# ===-------------------------------------------------------------------===#
# Unary Elementwise
# ===-------------------------------------------------------------------===#


def neg(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [unary_strided(K.neg, node, inputs, device)]


def log(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [unary_strided(K.log, node, inputs, device)]


def exp(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [unary_strided(K.exp, node, inputs, device)]


def sqrt(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [unary_strided(K.sqrt, node, inputs, device)]


def relu(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [unary_strided(K.relu, node, inputs, device)]


def cast(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var layout = node.src(0).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    K.cast.load(device)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        layout,
        node.src(0).dtype(),
        node.dtype(),
        device.ctx,
    )
    return [out^]


# ===-------------------------------------------------------------------===#
# Binary Elementwise
# ===-------------------------------------------------------------------===#


def add(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    if la.is_contiguous() and lb.is_contiguous():
        var null = unsafe_alloc[Int64](1)
        K.add.load(device)(
            inputs[0].data_ptr(),
            inputs[1].data_ptr(),
            out.data_ptr(),
            node.numel(),
            node.dtype(),
            device.ctx,
        )
        null.unsafe_free()
        return [out^]

    return [binary_strided(K.add_strided, node, inputs, device)]


def mul(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [binary_strided(K.mul, node, inputs, device)]


def div(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [binary_strided(K.div, node, inputs, device)]


def eq(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [binary_strided(K.eq, node, inputs, device)]


def relu_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [binary_strided(K.relu_grad, node, inputs, device)]


def scale(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var la = node.src(0).layout()
    var s = unsafe_alloc[Float32](1)
    s[unsafe_offset=0] = node.attrs()["scalar"][Float32]
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    K.scale.load(device)(
        inputs[0].data_ptr(),
        s.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        out.data_ptr(),
        la,
        node.dtype(),
        device.ctx,
    )
    return [out^]


def slice_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var slice_layout = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel(), fill=0.0)
    var out_view = out.view(slice_layout)
    K.slice_grad.load(device)(
        inputs[0].data_ptr(),
        out_view.data_ptr(),
        slice_layout,
        node.dtype(),
        device.ctx,
    )
    return [out^]


# ===-------------------------------------------------------------------===#
# Binary Reduce
# ===-------------------------------------------------------------------===#


def sum(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    if "axis" in node.attrs():
        return [axis_reduce_strided(K.sum_axis, node, inputs, device)]
    return [unary_strided(K.sum, node, inputs, device)]


def mean(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    if "axis" in node.attrs():
        return [axis_reduce_strided(K.mean_axis, node, inputs, device)]
    return [unary_strided(K.mean, node, inputs, device)]


def argmax(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    if "axis" in node.attrs():
        return [axis_reduce_strided(K.argmax_axis, node, inputs, device)]
    return [unary_strided(K.argmax, node, inputs, device)]


# ===-------------------------------------------------------------------===#
# Matmul
# ===-------------------------------------------------------------------===#


def matmul(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [matmul_strided(K.matmul, node, inputs, device)]


def matmul_t(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [matmul_strided(K.matmul_bt, node, inputs, device)]


def matmul_bias_bt(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [matmul_bias_strided(K.matmul_bias_bt, node, inputs, device)]


def layer_norm_fwd(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [layer_norm_fwd_dispatch(K.layer_norm_fwd, node, inputs, device)]


def layer_norm_bwd(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return layer_norm_bwd_dispatch(K.layer_norm_bwd, node, inputs, device)


def flash_attn_fwd_sched(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return flash_attn_fwd_dispatch(K.flash_attn_fwd, node, inputs, device)


def flash_attn_bwd_sched(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return flash_attn_bwd_dispatch(K.flash_attn_bwd, node, inputs, device)


# ===-------------------------------------------------------------------===#
# Shape / metadata ops
# ===-------------------------------------------------------------------===#


def _is_last_two_swap(order: IntTuple, rank: Int) -> Bool:
    """True if `order` is the identity permutation except the last two axes."""
    if rank < 2:
        return False
    for i in range(rank - 2):
        if order.value(i) != i:
            return False
    return order.value(rank - 2) == rank - 1 and order.value(rank - 1) == rank - 2


def contiguous(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var layout = node.src(0).layout()
    var order = layout.permutation_of_contiguous()
    if order and _is_last_two_swap(order.value(), layout.rank()):
        return [unary_strided(K.transpose_last2, node, inputs, device)]
    return [unary_strided(K.contiguous, node, inputs, device)]


def view(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [inputs[0].view(node.layout())]


def concat(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var ax = node.attr_int("axis")
    var out_layout = node.layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    var offset = 0
    for i in range(len(node.srcs())):
        var src_layout = node.src(i).layout()
        var size = src_layout.shape(ax)
        var dst_layout = out_layout.slice_axis(ax, offset, offset + size)
        var dst_view = out.view(dst_layout)
        strided_copy(K.strided_copy, src_layout, dst_layout, inputs[i], dst_view, node.dtype(), device)
        offset += size
    return [out^]


def triu(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    var diagonal = node.attr_int("diagonal")
    K.triu.load(device)(
        inputs[0].data_ptr(),
        out.data_ptr(),
        node.src(0).layout(),
        diagonal,
        node.dtype(),
        device.ctx,
    )
    return [out^]


# _-_-_-_-_-
# TODO: All the below
# _-_-_-_-_-


def softmax(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    if node.src(0).layout().is_contiguous():
        return [unary_strided(K.softmax, node, inputs, device)]
    return [unary_strided(K.softmax_strided, node, inputs, device)]


# ===-------------------------------------------------------------------===#
# BinaryOps
# ===-------------------------------------------------------------------===#


def transpose(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var p = alloc[Float32](2)
    p[0] = Float32(node.src(0).layout().shape(0))
    p[1] = Float32(node.src(0).layout().shape(1))
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    K.transpose.load(device)(
        inputs[0].data_ptr(),
        p.bitcast[NoneType]().as_unsafe_any_origin(),
        out.data_ptr(),
        node.numel(),
        node.dtype(),
        device.ctx,
    )
    p.free()
    return [out^]


def cross_entropy(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    if node.src(0).layout().is_contiguous() and node.src(1).layout().is_contiguous():
        return [binary_strided(K.cross_entropy, node, inputs, device)]
    return [binary_strided(K.cross_entropy_strided, node, inputs, device)]


# TODO: Revisit this, this is slop and can probably be cleaned up.
def disk(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var size = node.numel()
    var bytes = Path(node.attrs()["path"][String]).read_bytes()
    comptime for k in range(AnyBuffer.BufVariant.Ts.length):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if node.dtype() == d:
            var ptr = bytes.unsafe_ptr().unsafe_bitcast[Scalar[d]]()
            var data = List[Scalar[d]]()
            data.reserve(size)
            for i in range(size):
                data.append(ptr[unsafe_offset=i])
            return [AnyBuffer(Buffer[d].from_data(device, data))]
    raise Error("unsupported dtype")


# ===-------------------------------------------------------------------===#
# Grad ops
# ===-------------------------------------------------------------------===#


def softmax_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    return [binary_strided(K.softmax_grad, node, inputs, device)]


def cross_entropy_grad(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    var la = node.src(0).layout()
    var lb = node.src(1).layout()
    var out = AnyBuffer.create(node.dtype(), device, node.numel())
    var sym = K.cross_entropy_grad
    if not (la.is_contiguous() and lb.is_contiguous()):
        sym = K.cross_entropy_grad_strided
    sym.load(device)(
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
        inputs[2].data_ptr(),
        out.data_ptr(),
        la,
        lb,
        node.dtype(),
        device.ctx,
    )
    return [out^]
