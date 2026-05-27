from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer
from mograd.pattern_matcher import Rule, PatternMatcher

# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#

comptime ExecFn = def(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) thin raises -> Buffer


struct Scheduler[rules: List[Rule[ExecFn]]]:
    @staticmethod
    def run(root: OpRef, ctx: DeviceContext) raises -> Buffer:
        var pm = PatternMatcher[ExecFn, Self.rules]()
        var bufs = Dict[OpRef, Buffer]()
        var topo = Self._toposort(root)

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

    @staticmethod
    def _toposort(root: OpRef) -> List[OpRef]:
        var visited = Dict[OpRef, Bool]()
        var result = List[OpRef]()
        Self._dfs(root, visited, result)
        return result^

    @staticmethod
    def _dfs(
        node: OpRef,
        mut visited: Dict[OpRef, Bool],
        mut result: List[OpRef],
    ):
        if node in visited:
            return
        visited[node] = True
        for i in range(len(node.srcs())):
            Self._dfs(node.srcs()[i], visited, result)
        result.append(node)
