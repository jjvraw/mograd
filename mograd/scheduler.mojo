from std.gpu.host import DeviceContext

from mograd import Device
from mograd.op import OpRef, OpType
from mograd.buffer import Buffer, AnyBuffer, BufferArm
from mograd.pattern_matcher import Rule, PatternMatcher, GraphUtils, Pat

# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#

comptime BoundExecFn = def(node: OpRef, inputs: List[AnyBuffer], device: Device) thin raises -> AnyBuffer


struct Scheduler:
    var rules: List[Rule[BoundExecFn]]

    def __init__(out self, var rules: List[Rule[BoundExecFn]]):
        self.rules = rules^

    def run(self, root: OpRef, device: Device) raises -> AnyBuffer:
        var pm = PatternMatcher[BoundExecFn](self.rules)
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
                var result = rule.value()(node, inputs, device)
                node.op().buf = Optional[AnyBuffer](result.copy())
                bufs[node] = result^

        device.ctx.synchronize()
        return bufs[root].copy()
