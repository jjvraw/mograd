from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer, AnyBuffer, BufferArm
from mograd.pattern_matcher import Rule, PatternMatcher, GraphUtils, Pat

# ===-------------------------------------------------------------------===#
# Exec function types
# ===-------------------------------------------------------------------===#

comptime ExecFn = def[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) thin raises -> AnyBuffer

comptime BoundExecFn = def(node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) thin raises -> AnyBuffer


def _make1[F: ExecFn, fp_only: Bool](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime dtype = T.node_dtype
        comptime if not fp_only or dtype.is_floating_point():
            if node.dtype() == dtype:
                return F[dtype](node, inputs, ctx)
    raise Error("unsupported dtype in make_bound")


def _make2[F: ExecFn](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime for j in range(AnyBuffer.BufVariant.Ts.size):
        comptime InT = AnyBuffer.BufVariant.Ts[j]
        comptime assert conforms_to(InT, BufferArm)
        comptime in_dtype = InT.node_dtype
        if inputs[0].isa[in_dtype]():
            comptime for k in range(AnyBuffer.BufVariant.Ts.size):
                comptime OutT = AnyBuffer.BufVariant.Ts[k]
                comptime assert conforms_to(OutT, BufferArm)
                comptime out_dtype = OutT.node_dtype
                if node.dtype() == out_dtype:
                    return F[in_dtype, out_dtype](node, inputs, ctx)
    raise Error("unsupported dtype combination in make_bound")


def make_bound[
    F: ExecFn, n_dtypes: Int = 1, fp_only: Bool = False
](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime if n_dtypes == 1:
        return _make1[F, fp_only](node, inputs, ctx)
    else:
        return _make2[F](node, inputs, ctx)


# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#


struct Scheduler[rules: List[Rule[BoundExecFn]]]:
    @staticmethod
    def run(root: OpRef, ctx: DeviceContext) raises -> AnyBuffer:
        var pm = PatternMatcher[BoundExecFn, Self.rules]()
        var bufs = Dict[OpRef, AnyBuffer]()
        var topo = GraphUtils.toposort(root)

        for i in range(len(topo)):
            var node = topo[i]
            if node.op().buf:
                bufs[node] = node.op().buf.value().copy()
            elif node.op_type() == OpType.BUFFER:
                raise Error("uninitialized BUFFER node")
            else:
                var inputs = List[AnyBuffer]()
                for j in range(len(node.srcs())):
                    inputs.append(bufs[node.src(j)].copy())
                var rule = pm.match(node)
                if not rule:
                    raise Error("no exec rule for op: " + node.op_type()._name)
                var result = rule.value()(node, inputs, ctx)
                node.op().buf = Optional[AnyBuffer](result.copy())
                bufs[node] = result^

        ctx.synchronize()
        return bufs[root].copy()
