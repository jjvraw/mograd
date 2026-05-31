from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.pathlib import Path

from layout import Coord, TileTensor, row_major
from linalg.matmul import matmul
from nn.argmaxmin_gpu import argmax_gpu
from nn.rand_normal import random_normal
from std.utils import IndexList

from mograd.op import Op, OpRef, OpType, AttrVal
from mograd.runtime.native.gpu.rewrites import MATMUL_T
from mograd.buffer import Buffer
from mograd.runtime.native.gpu.kernels import (
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
    transpose_kernel,
    uniform_kernel,
    slice_rows_kernel,
    cross_entropy_kernel,
    cross_entropy_grad_kernel,
    scale_kernel,
    eq_kernel,
    BLOCK_SIZE,
    TILE_DIM,
)
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime import Runtime
from mograd.scheduler import Scheduler, ExecFn

# ===-------------------------------------------------------------------===#
# GPURuntime
# ===-------------------------------------------------------------------===#


struct GPURuntime(Runtime):
    @staticmethod
    def run(root: OpRef, ctx: Optional[DeviceContext]) raises -> Buffer:
        if not ctx:
            raise Error("GPURuntime requires a DeviceContext")

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
                Rule(Pat(MATMUL_T), matmul_t_exec),
                Rule(Pat(OpType.TRANSPOSE), transpose_exec),
                Rule(Pat(OpType.UNIFORM), uniform_exec),
                Rule(Pat(OpType.DISK), disk_exec),
                Rule(Pat(OpType.SLICE), slice_exec),
                Rule(Pat(OpType.CROSS_ENTROPY), cross_entropy_exec),
                Rule(Pat(OpType.CROSS_ENTROPY_GRAD), cross_entropy_grad_exec),
                Rule(Pat(OpType.SCALE), scale_exec),
                Rule(Pat(OpType.ARGMAX), argmax_exec),
                Rule(Pat(OpType.EQ), eq_exec),
                Rule(Pat(OpType.FULL), full_exec),
                Rule(Pat(OpType.RANDN), randn_exec),
            ]
        ].run(root, ctx.value())


# ===-------------------------------------------------------------------===#
# Exec functions
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
    return Buffer(c_buf^, node.shape(), size)


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
    return Buffer(c_buf^, node.shape(), size)


def exp_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[exp_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def log_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[log_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def neg_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[neg_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def div_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
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
    return Buffer(out_buf^, node.shape(), size)


def sum_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
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
    return Buffer(out_buf^, (1,), 1)


def sum_grad_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[sum_grad_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def reshape_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var b = inputs[0].copy()
    b.shape = node.shape()
    return b^


def matmul_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var M = inputs[0].shape[0]
    var K = inputs[0].shape[1]
    var N = inputs[1].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a = TileTensor(inputs[0].buf().unsafe_ptr().as_any_origin(), row_major(Coord(M, K)))
    var b = TileTensor(inputs[1].buf().unsafe_ptr().as_any_origin(), row_major(Coord(K, N)))
    var c = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(M, N)))
    matmul[target="gpu"](c, a, b, ctx)
    return Buffer(out_buf^, (M, N), M * N)


def matmul_t_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var M = inputs[0].shape[0]
    var K = inputs[0].shape[1]
    var N = inputs[1].shape[0]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a = TileTensor(inputs[0].buf().unsafe_ptr().as_any_origin(), row_major(Coord(M, K)))
    var b = TileTensor(inputs[1].buf().unsafe_ptr().as_any_origin(), row_major(Coord(N, K)))
    var c = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(M, N)))
    matmul[target="gpu", transpose_b=True](c, a, b, ctx)
    return Buffer(out_buf^, (M, N), M * N)


def transpose_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
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
    return Buffer(out_buf^, (N, M), M * N)


def disk_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var path_attr = node.attrs()["path"]
    var bytes = Path(path_attr[String]).read_bytes()
    var float_ptr = bytes.unsafe_ptr().bitcast[Float32]()
    var data = List[Float32]()
    data.reserve(size)
    for i in range(size):
        data.append(float_ptr[i])
    return Buffer.from_data(ctx, data, node.shape())


def full_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var fill_value = node.attrs()["value"][Float32]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    out_buf.enqueue_fill(fill_value)
    return Buffer(out_buf^, node.shape(), size)


def randn_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var mean = node.attrs()["mean"][Float32]
    var std = node.attrs()["std"][Float32]
    var seed = UInt64(Int(node.attrs()["seed"][Float32]))
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(seed)
    var out_ptr = out_buf.unsafe_ptr()

    @parameter
    @always_inline
    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[DType.float32, width]):
        out_ptr.store(idx[0], val)

    random_normal[output_fn=store, target="gpu"](
        IndexList[1](size),
        mean,
        std,
        seed_buf.unsafe_ptr(),
        ctx,
    )
    # seed_buf must outlive the GPU kernel
    ctx.synchronize()
    return Buffer(out_buf^, node.shape(), size)


def uniform_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var low_attr = node.attrs()["low"]
    var high_attr = node.attrs()["high"]
    var seed_attr = node.attrs()["seed"]
    var low = low_attr[Float32]
    var high = high_attr[Float32]
    var seed = UInt32(Int(seed_attr[Float32]))
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
    return Buffer(out_buf^, node.shape(), size)


def relu_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[relu_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def relu_grad_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
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
    return Buffer(out_buf^, node.shape(), size)


def softmax_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    if size > BLOCK_SIZE:
        raise Error("softmax: size " + String(size) + " exceeds single-block limit (" + String(BLOCK_SIZE) + ")")
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[softmax_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=1,
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def softmax_grad_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    if size > BLOCK_SIZE:
        raise Error("softmax_grad: size " + String(size) + " exceeds single-block limit (" + String(BLOCK_SIZE) + ")")
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[softmax_grad_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=1,
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def eq_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[eq_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def argmax_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var N = inputs[0].shape[0]
    var C = inputs[0].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var inp = TileTensor(inputs[0].buf().unsafe_ptr().as_any_origin(), row_major(Coord(N, C)))
    var out = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(N, 1)))
    argmax_gpu[DType.float32, DType.float32](ctx, inp, out)
    return Buffer(out_buf^, (N,), N)


def scale_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var scalar = node.attrs()["scalar"][Float32]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[scale_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        scalar,
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def cross_entropy_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var N = inputs[0].shape[0]
    var C = inputs[0].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](1)
    out_buf.enqueue_fill(0.0)
    ctx.enqueue_function[cross_entropy_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        N,
        C,
        grid_dim=ceildiv(N, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, (1,), 1)


def cross_entropy_grad_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var N = inputs[0].shape[0]
    var C = inputs[0].shape[1]
    var size = N * C
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[cross_entropy_grad_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        inputs[2].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        N,
        C,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, (N, C), size)


def slice_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var start = Int(node.attrs()["start"][Float32])
    var cols = inputs[0].size // inputs[0].shape[0]
    var rows = node.shape()[0]
    var size = rows * cols
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[slice_rows_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        start,
        cols,
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)
