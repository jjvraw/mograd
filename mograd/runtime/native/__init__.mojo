from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.pathlib import Path

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer
from mograd.runtime.native.kernels import (
    add_kernel,
    mul_kernel,
    relu_kernel,
    relu_grad_kernel,
    softmax_kernel,
    softmax_grad_kernel,
    exp_kernel,
    log_kernel,
    neg_kernel,
    div_kernel,
    sum_kernel,
    sum_grad_kernel,
    matmul_kernel,
    transpose_kernel,
    uniform_kernel,
    BLOCK_SIZE,
    TILE_DIM,
)
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime import Runtime
from mograd.scheduler import Scheduler, ExecFn

# ===-------------------------------------------------------------------===#
# Native Runtime
# ===-------------------------------------------------------------------===#


def add_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
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


def mul_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
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


def exp_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[exp_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def log_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[log_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def neg_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[neg_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def div_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[div_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def sum_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](1)
    out_buf.enqueue_fill(0.0)
    ctx.enqueue_function[sum_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, [1], 1)


def sum_grad_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = 1
    for i in range(len(node.shape())):
        size *= node.shape()[i]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[sum_grad_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def reshape_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var b = inputs[0].copy()
    b.shape = node.shape().copy()
    return b^


def matmul_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var M = inputs[0].shape[0]
    var K = inputs[0].shape[1]
    var N = inputs[1].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_function[matmul_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        M,
        N,
        K,
        grid_dim=(ceildiv(N, TILE_DIM), ceildiv(M, TILE_DIM)),
        block_dim=(TILE_DIM, TILE_DIM),
    )
    return Buffer(out_buf^, [M, N], M * N)


def transpose_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var M = inputs[0].shape[0]
    var N = inputs[0].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](M * N)
    ctx.enqueue_function[transpose_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        M,
        N,
        grid_dim=(ceildiv(N, TILE_DIM), ceildiv(M, TILE_DIM)),
        block_dim=(TILE_DIM, TILE_DIM),
    )
    return Buffer(out_buf^, [N, M], M * N)


def disk_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = 1
    for d in node.shape():
        size *= d
    var bytes = Path(node.str_attrs()[0]).read_bytes()
    var float_ptr = bytes.unsafe_ptr().bitcast[Float32]()
    var data = List[Float32]()
    data.reserve(size)
    for i in range(size):
        data.append(float_ptr[i])
    return Buffer.from_data(ctx, data, node.shape().copy())


def uniform_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = 1
    for d in node.shape():
        size *= d
    var low = node.attrs()[0]
    var high = node.attrs()[1]
    var seed = UInt32(Int(node.attrs()[2]))
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[uniform_kernel](
        out_buf.unsafe_ptr(),
        size,
        low,
        high,
        seed,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def relu_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[relu_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def relu_grad_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[relu_grad_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def softmax_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    if size > BLOCK_SIZE:
        raise Error(
            "softmax: size "
            + String(size)
            + " exceeds single-block limit ("
            + String(BLOCK_SIZE)
            + ")"
        )
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[softmax_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=1,
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


def softmax_grad_exec(
    node: OpRef, inputs: List[Buffer], ctx: DeviceContext
) raises -> Buffer:
    var size = inputs[0].size
    if size > BLOCK_SIZE:
        raise Error(
            "softmax_grad: size "
            + String(size)
            + " exceeds single-block limit ("
            + String(BLOCK_SIZE)
            + ")"
        )
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[softmax_grad_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=1,
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape().copy(), size)


struct NativeRuntime(Runtime):
    @staticmethod
    def run(root: OpRef, ctx: Optional[DeviceContext]) raises -> Buffer:
        if not ctx:
            raise Error("NativeRuntime requires a DeviceContext")
        return Scheduler[
            [
                Rule(Pat(OpType.ADD), add_exec),
                Rule(Pat(OpType.MUL), mul_exec),
                Rule(Pat(OpType.RELU), relu_exec),
                Rule(Pat(OpType.RELU_GRAD), relu_grad_exec),
                Rule(Pat(OpType.SOFTMAX), softmax_exec),
                Rule(Pat(OpType.SOFTMAX_GRAD), softmax_grad_exec),
                Rule(Pat(OpType.EXP), exp_exec),
                Rule(Pat(OpType.LOG), log_exec),
                Rule(Pat(OpType.NEG), neg_exec),
                Rule(Pat(OpType.DIV), div_exec),
                Rule(Pat(OpType.SUM), sum_exec),
                Rule(Pat(OpType.SUM_GRAD), sum_grad_exec),
                Rule(Pat(OpType.RESHAPE), reshape_exec),
                Rule(Pat(OpType.MATMUL), matmul_exec),
                Rule(Pat(OpType.TRANSPOSE), transpose_exec),
                Rule(Pat(OpType.UNIFORM), uniform_exec),
                Rule(Pat(OpType.DISK), disk_exec),
            ]
        ].run(root, ctx.value())
