from max.gpu.host import DeviceContext

from mograd import Device
from mograd.debug import Tracer
from mograd.op import OpRef, OpType, sink
from mograd.buffer import Buffer, AnyBuffer, BufferArm
from mograd.pattern_matcher import Rule, PatternMatcher, GraphUtils, Pat

# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#

comptime BoundExecFn = def(node: OpRef, inputs: List[AnyBuffer], device: Device) thin raises -> List[AnyBuffer]

comptime SchedulerRules = List[Rule[BoundExecFn]]


struct Scheduler:
    var rules: SchedulerRules

    def __init__(out self, var rules: SchedulerRules):
        self.rules = rules^

    def run(self, root: OpRef, device: Device) raises -> AnyBuffer:
        var bufs = self._compute(root, device)
        return bufs[root][0].copy()

    def run_many(self, var targets: List[OpRef], device: Device) raises -> List[AnyBuffer]:
        """Computes all `targets` in one pass: one toposort, one bufs dict,
        one synchronize - instead of one of each per separate `run` call.
        """
        var bundle = sink(targets^)
        var bufs = self._compute(bundle, device)
        var results = List[AnyBuffer]()
        for i in range(len(bundle.srcs())):
            results.append(bufs[bundle.src(i)][0].copy())
        return results^

    def _compute(self, root: OpRef, device: Device) raises -> Dict[OpRef, List[AnyBuffer]]:
        var pm = PatternMatcher[BoundExecFn](self.rules)
        var bufs = Dict[OpRef, List[AnyBuffer]]()
        var topo = GraphUtils.toposort(root)
        var tracer = Tracer()
        tracer.run_begin(root, device)

        for i in range(len(topo)):
            var node = topo[i]
            if node.op_type() == OpType.SINK:
                continue
            elif node.op_type() == OpType.GETTUPLE:
                var idx = node.attr_int("index")
                var out = bufs[node.src(0)][idx].copy()
                node.op().buf = [out.copy()]
                bufs[node] = [out^]
                continue
            elif len(node.op().buf) > 0:
                var cached = List[AnyBuffer]()
                for i in range(len(node.op().buf)):
                    cached.append(node.op().buf[i].copy())
                bufs[node] = cached^
                continue
            elif node.op_type() == OpType.BUFFER:
                raise Error("uninitialized BUFFER node")

            var inputs = List[AnyBuffer]()
            for j in range(len(node.srcs())):
                inputs.append(bufs[node.src(j)][0].copy())
            var rule = pm.match(node)
            if not rule:
                raise Error("no exec rule for op: " + node.op_type()._name)
            tracer.node_begin(node, device)
            var results = rule.value().func(node, inputs, device)
            tracer.node_end(node, inputs, results, device)
            var to_cache = List[AnyBuffer]()
            for res in results:
                to_cache.append(res.copy())
            node.op().buf = to_cache^
            bufs[node] = results^

        device.ctx.synchronize()
        tracer.run_end(device)
        return bufs^
