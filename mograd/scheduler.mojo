from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType, AnyOpRef, NodeOps, HasDtype
from mograd.buffer import Buffer, AnyBuffer
from mograd.pattern_matcher import Rule, PatternMatcher, GraphUtils, Pat, build_rule_table

# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#

comptime ExecFn = def[dtype: DType](
    node: OpRef[dtype], inputs: List[AnyBuffer], ctx: DeviceContext
) thin raises -> Buffer[dtype]


struct Scheduler[rules: List[Rule[BoundExecFn]]]:
    @staticmethod
    def run(root: AnyOpRef, ctx: DeviceContext) raises -> AnyBuffer:
        var pm = PatternMatcher[BoundExecFn, Self.rules]()
        # var pm = PatternMatcher[ExecFn, Self.rules]()
        var bufs = Dict[AnyOpRef, AnyBuffer]()
        var topo = GraphUtils.toposort(root)

        for i in range(len(topo)):
            var node = topo[i]
            comptime for k in range(AnyOpRef.Ts.size):
                comptime T = AnyOpRef.Ts[k]
                if node.isa[T]():
                    comptime assert conforms_to(T, HasDtype)
                    comptime dtype = T.node_dtype
                    var typed = node.unsafe_get[OpRef[dtype]]()
                    if typed.op().buf:
                        bufs[node] = AnyBuffer(typed.op().buf.value().copy())
                    elif typed.op_type() == OpType.BUFFER:
                        raise Error("uninitialized BUFFER node")
                    else:
                        var inputs = List[AnyBuffer]()
                        for j in range(typed.srcs_count()):
                            inputs.append(bufs[typed.get_src(j)].copy())
                        var rule = pm.match(node)
                        if not rule:
                            raise Error("no exec rule for op: " + typed.op_type()._name)
                        bufs[node] = rule.value()(node, inputs, ctx)

        ctx.synchronize()
        return bufs[root].copy()


comptime BoundExecFn = def(node: AnyOpRef, inputs: List[AnyBuffer], ctx: DeviceContext) thin raises -> AnyBuffer


def bind_exec[F: ExecFn](node: AnyOpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime for k in range(AnyOpRef.Ts.size):
        comptime T = AnyOpRef.Ts[k]
        if node.isa[T]():
            comptime assert conforms_to(T, HasDtype)
            comptime dtype = T.node_dtype
            var typed = node.unsafe_get[OpRef[dtype]]()
            return AnyBuffer(F[dtype](typed, inputs, ctx))
    raise Error("unsupported dtype")


def bind_float_exec[F: ExecFn](node: AnyOpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime for k in range(AnyOpRef.Ts.size):
        comptime T = AnyOpRef.Ts[k]
        if node.isa[T]():
            comptime assert conforms_to(T, HasDtype)
            comptime dtype = T.node_dtype
            comptime if dtype.is_floating_point():
                var typed = node.unsafe_get[OpRef[dtype]]()
                return AnyBuffer(F[dtype](typed, inputs, ctx))
    raise Error("unsupported dtype")
