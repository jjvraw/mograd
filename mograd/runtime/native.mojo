from std.math import ceildiv
from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer
from mograd.kernels import add_kernel, mul_kernel, BLOCK_SIZE
from mograd.pattern_matcher import Rule, Pat
from mograd.scheduler import Scheduler, ExecFn

# ===-------------------------------------------------------------------===#
# Native Runtime
# ===-------------------------------------------------------------------===#


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


struct NativeRuntime:
    @staticmethod
    def run(root: OpRef, ctx: DeviceContext) raises -> Buffer:
        return Scheduler[
            [
                Rule(Pat(OpType.ADD), add_exec),
                Rule(Pat(OpType.MUL), mul_exec),
            ]
        ].run(root, ctx)
