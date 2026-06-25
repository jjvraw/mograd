from std.gpu.host import DeviceContext

from mograd import Device
from mograd.op import OpRef, OpType, sink
from mograd.buffer import Buffer, AnyBuffer, BufferArm
from mograd.pattern_matcher import Rule, PatternMatcher, GraphUtils, Pat

# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#

comptime BoundExecFn = def(node: OpRef, inputs: List[AnyBuffer], device: Device) thin raises -> AnyBuffer

comptime SchedulerRules = List[Rule[BoundExecFn]]


struct Scheduler:
    var rules: SchedulerRules

    def __init__(out self, var rules: SchedulerRules):
        self.rules = rules^

    def run(self, root: OpRef, device: Device) raises -> AnyBuffer:
        var bufs = self._compute(root, device)
        return bufs[root].copy()

    def run_many(self, var targets: List[OpRef], device: Device) raises -> List[AnyBuffer]:
        """Computes all `targets` in one pass: one toposort, one bufs dict,
        one synchronize - instead of one of each per separate `run` call.
        """
        var bundle = sink(targets^)
        var bufs = self._compute(bundle, device)
        var results = List[AnyBuffer]()
        for i in range(len(bundle.srcs())):
            results.append(bufs[bundle.src(i)].copy())
        return results^

    def _compute(self, root: OpRef, device: Device) raises -> Dict[OpRef, AnyBuffer]:
        var pm = PatternMatcher[BoundExecFn](self.rules)
        var bufs = Dict[OpRef, AnyBuffer]()
        var topo = GraphUtils.toposort(root)

        for i in range(len(topo)):
            var node = topo[i]
            if node.op_type() == OpType.SINK:
                continue
            elif node.op().buf:
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
                var result = rule.value().func(node, inputs, device)
                node.op().buf = Optional[AnyBuffer](result.copy())
                bufs[node] = result^

        device.ctx.synchronize()
        return bufs^
