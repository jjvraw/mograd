from std.math import ceildiv
from std.gpu.host import DeviceContext
from std.pathlib import Path

from layout import Coord, TileTensor, row_major
from linalg.matmul import matmul
from nn.argmaxmin_gpu import argmax_gpu
from nn.rand_normal import random_normal
from nn.softmax import softmax as nn_softmax
from std.gpu.host import get_gpu_target
from std.sys.info import simd_width_of
from std.utils import IndexList

from mograd.op import Op, OpRef, OpType, AttrVal
from mograd.runtime.native.gpu.rewrites import MATMUL_T
from mograd.buffer import Buffer
from mograd.runtime.native.gpu.kernels import (
    add_kernel,
    mul_kernel,
    neg_kernel,
    div_kernel,
    scale_kernel,
    exp_kernel,
    log_kernel,
    eq_kernel,
    relu_kernel,
    relu_grad_kernel,
    softmax_grad_kernel,
    sum_kernel,
    sum_grad_kernel,
    transpose_kernel,
    slice_rows_kernel,
    uniform_kernel,
    cross_entropy_kernel,
    cross_entropy_grad_kernel,
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
                # Constructions
                Rule(Pat(OpType.FULL), full_exec),
                Rule(Pat(OpType.UNIFORM), uniform_exec),
                Rule(Pat(OpType.RANDN), randn_exec),
                Rule(Pat(OpType.DISK), disk_exec),
                # Pointwise
                Rule(Pat(OpType.ADD), add_exec),
                Rule(Pat(OpType.MUL), mul_exec),
                Rule(Pat(OpType.NEG), neg_exec),
                Rule(Pat(OpType.DIV), div_exec),
                Rule(Pat(OpType.SCALE), scale_exec),
                Rule(Pat(OpType.EXP), exp_exec),
                Rule(Pat(OpType.LOG), log_exec),
                Rule(Pat(OpType.EQ), eq_exec),
                # Activations
                Rule(Pat(OpType.RELU), relu_exec),
                Rule(Pat(OpType.RELU_GRAD), relu_grad_exec),
                Rule(Pat(OpType.SOFTMAX), softmax_exec),
                Rule(Pat(OpType.SOFTMAX_GRAD), softmax_grad_exec),
                # Reductions
                Rule(Pat(OpType.SUM), sum_exec),
                Rule(Pat(OpType.SUM_GRAD), sum_grad_exec),
                Rule(Pat(OpType.ARGMAX), argmax_exec),
                # Shape
                Rule(Pat(OpType.RESHAPE), reshape_exec),
                Rule(Pat(OpType.TRANSPOSE), transpose_exec),
                Rule(Pat(OpType.SLICE), slice_exec),
                # Contractions
                Rule(Pat(OpType.MATMUL), matmul_exec),
                Rule(Pat(MATMUL_T), matmul_t_exec),
                # Loss
                Rule(Pat(OpType.CROSS_ENTROPY), cross_entropy_exec),
                Rule(Pat(OpType.CROSS_ENTROPY_GRAD), cross_entropy_grad_exec),
            ]
        ].run(root, ctx.value())


# ===-------------------------------------------------------------------===#
# Constructions
# ===-------------------------------------------------------------------===#


def full_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    out_buf.enqueue_fill(node.attrs()["value"][Float32])
    return Buffer(out_buf^, node.shape(), size)


def uniform_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[uniform_kernel](
        out_buf.unsafe_ptr(),
        size,
        node.attrs()["low"][Float32],
        node.attrs()["high"][Float32],
        UInt32(Int(node.attrs()["seed"][Float32])),
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def randn_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(node.attrs()["seed"][Float32])))
    var out_ptr = out_buf.unsafe_ptr()

    @parameter
    @always_inline
    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[DType.float32, width]):
        out_ptr.store(idx[0], val)

    random_normal[output_fn=store, target="gpu"](
        IndexList[1](size),
        node.attrs()["mean"][Float32],
        node.attrs()["std"][Float32],
        seed_buf.unsafe_ptr(),
        ctx,
    )
    # seed_buf must outlive the GPU kernel
    ctx.synchronize()
    return Buffer(out_buf^, node.shape(), size)


def disk_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var bytes = Path(node.attrs()["path"][String]).read_bytes()
    var float_ptr = bytes.unsafe_ptr().bitcast[Float32]()
    var data = List[Float32]()
    data.reserve(size)
    for i in range(size):
        data.append(float_ptr[i])
    return Buffer.from_data(ctx, data, node.shape())


# ===-------------------------------------------------------------------===#
# Pointwise
# ===-------------------------------------------------------------------===#


def add_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[add_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


def mul_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[mul_kernel](
        inputs[0].buf().unsafe_ptr(),
        inputs[1].buf().unsafe_ptr(),
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


def scale_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    ctx.enqueue_function[scale_kernel](
        inputs[0].buf().unsafe_ptr(),
        out_buf.unsafe_ptr(),
        node.attrs()["scalar"][Float32],
        size,
        grid_dim=ceildiv(size, BLOCK_SIZE),
        block_dim=BLOCK_SIZE,
    )
    return Buffer(out_buf^, node.shape(), size)


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


# ===-------------------------------------------------------------------===#
# Activations
# ===-------------------------------------------------------------------===#


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
    var shape = inputs[0].shape
    var rank = len(shape)
    var rows = shape[0] if rank > 1 else 1
    var cols = shape[rank - 1]
    var size = rows * cols
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var out = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(rows, cols)))
    var inp_ptr = inputs[0].buf().unsafe_ptr()

    @parameter
    @always_inline
    def input_fn[width: Int, r: Int](coords: IndexList[r]) -> SIMD[DType.float32, width]:
        return inp_ptr.load[width=width](coords[0] * cols + coords[1])

    comptime simd_width = simd_width_of[DType.float32, target=get_gpu_target()]()
    nn_softmax[DType.float32, simd_width, 2, input_fn, "gpu"](
        IndexList[2](rows, cols),
        out,
        axis=1,
        context=ctx,
    )
    ctx.synchronize()
    return Buffer(out_buf^, node.shape(), size)


def softmax_grad_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
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


# ===-------------------------------------------------------------------===#
# Reductions
# ===-------------------------------------------------------------------===#


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


def argmax_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var N = inputs[0].shape[0]
    var C = inputs[0].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var inp = TileTensor(inputs[0].buf().unsafe_ptr().as_any_origin(), row_major(Coord(N, C)))
    var out = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(N, 1)))
    argmax_gpu[DType.float32, DType.float32](ctx, inp, out)
    return Buffer(out_buf^, (N,), N)


# ===-------------------------------------------------------------------===#
# Shape
# ===-------------------------------------------------------------------===#


def reshape_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var b = inputs[0].copy()
    b.shape = node.shape()
    return b^


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


# ===-------------------------------------------------------------------===#
# Contractions
# ===-------------------------------------------------------------------===#


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


# ===-------------------------------------------------------------------===#
# Loss
# ===-------------------------------------------------------------------===#


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
