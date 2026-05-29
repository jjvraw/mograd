from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer
from mograd.pattern_matcher import Rule, PatternMatcher, GraphUtils

# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#

comptime ExecFn = def(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) thin raises -> Buffer


struct Scheduler[rules: List[Rule[ExecFn]]]:
    @staticmethod
    def run(root: OpRef, ctx: DeviceContext) raises -> Buffer:
        var pm = PatternMatcher[ExecFn, Self.rules]()
        var bufs = Dict[OpRef, Buffer]()
        var topo = GraphUtils.toposort(root)

        for i in range(len(topo)):
            var node = topo[i]
            if node.op().buf:
                bufs[node] = node.op().buf.value().copy()
            elif node.op_type() == OpType.BUFFER:
                raise Error("uninitialized BUFFER node")
            else:
                var inputs = List[Buffer]()
                for j in range(len(node.srcs())):
                    inputs.append(bufs[node.srcs()[j]].copy())
                var rule = pm.match(node)
                if not rule:
                    raise Error("no exec rule for op: " + node.op_type()._name)
                var result = rule.value()(node, inputs, ctx)
                node.op().buf = Optional[Buffer](result.copy())
                bufs[node] = result^

        ctx.synchronize()
        return bufs[root].copy()
