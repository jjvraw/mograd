from std.math import ceildiv
from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer
from mograd.kernels import add_kernel, mul_kernel
from mograd.pattern_matcher import Rule, Pat, PatternMatcher

# ===-------------------------------------------------------------------===#
# Scheduler
# ===-------------------------------------------------------------------===#

comptime ExecFn = def(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) thin raises -> Buffer

comptime BLOCK_SIZE = 256

def add_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var c_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[add_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        c_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(c_buf^, node.shape().copy(), size)


def mul_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var c_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[mul_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        c_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(c_buf^, node.shape().copy(), size)


struct Scheduler:
    @staticmethod
    def run(root: OpRef, ctx: DeviceContext) raises -> Buffer:
        var pm = PatternMatcher[
            ExecFn,
            [
                Rule(Pat(OpType.ADD), add_exec),
                Rule(Pat(OpType.MUL), mul_exec),
            ],
        ]()

        var bufs = Dict[OpRef, Buffer]()
        var topo = Self._toposort(root)

        for i in range(len(topo)):
            var node = topo[i]
            if node.op_type() == OpType.BUFFER:
                if not node.op().buf:
                    raise Error("uninitialized BUFFER node")
                bufs[node] = node.op().buf.value().copy()
            else:
                var inputs = List[Buffer]()
                for j in range(len(node.srcs())):
                    inputs.append(bufs[node.srcs()[j]].copy())
                var rule = pm.match(node)
                if not rule:
                    raise Error("no exec rule for op: " + node.op_type()._name)
                bufs[node] = rule.value()(node, inputs, ctx)

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
