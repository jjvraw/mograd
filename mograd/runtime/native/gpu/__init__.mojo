from std.math import ceildiv, exp, log
from std.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from std.sys.info import simd_width_of, size_of
from std.pathlib import Path
from std.algorithm.functional import elementwise
from std.algorithm.reduction import sum as reduce_sum
from std.utils import IndexList

from layout import Coord, TileTensor, row_major
from linalg.matmul import matmul
from nn.argmaxmin_gpu import argmax_gpu
from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform
from nn.softmax import softmax as nn_softmax

from mograd.op import Op, OpRef, OpType, AttrVal
from mograd.runtime.native.gpu.rewrites import MATMUL_T
from mograd.buffer import Buffer
from mograd.runtime.native.gpu.kernels import (
    softmax_grad_kernel,
    transpose_kernel,
    cross_entropy_kernel,
    cross_entropy_kernel_no_smem,
    cross_entropy_grad_kernel,
    cross_entropy_grad_kernel_no_smem,
    BLOCK_SIZE,
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
                Rule(Pat(OpType.ARGMAX), argmax_exec),
                # Shape
                Rule(Pat(OpType.RESHAPE), reshape_exec),
                Rule(Pat(OpType.TRANSPOSE), transpose_exec),
                Rule(Pat(OpType.SLICE), slice_exec),
                Rule(Pat(OpType.BROADCAST), broadcast_exec),
                Rule(Pat(OpType.ONE_HOT), one_hot_exec),
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
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(node.attrs()["seed"][Float32])))
    var out_ptr = out_buf.unsafe_ptr()

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[DType.float32, width]) capturing:
        out_ptr.store(idx[0], val)

    random_uniform[output_fn=store, target="gpu"](
        IndexList[1](size), node.attrs()["low"][Float32], node.attrs()["high"][Float32], seed_buf.unsafe_ptr(), ctx
    )
    ctx.synchronize()
    return Buffer(out_buf^, node.shape(), size)


def randn_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(node.attrs()["seed"][Float32])))
    var out_ptr = out_buf.unsafe_ptr()

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[DType.float32, width]) capturing:
        out_ptr.store(idx[0], val)

    random_normal[output_fn=store, target="gpu"](
        IndexList[1](size),
        node.attrs()["mean"][Float32],
        node.attrs()["std"][Float32],
        seed_buf.unsafe_ptr(),
        ctx,
    )
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
    var inp0_ptr = inputs[0].data_ptr()
    var inp1_ptr = inputs[1].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[
        width: Int, rank: Int, alignment: Int = 1
    ](coords: IndexList[rank]) {var inp0_ptr, var inp1_ptr, var out_ptr}:
        out_ptr.store(coords[0], inp0_ptr.load(coords[0]) + inp1_ptr.load(coords[0]))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)

    return Buffer(out_buf^, node.shape(), size)


def mul_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp0_ptr = inputs[0].data_ptr()
    var inp1_ptr = inputs[1].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[
        width: Int, rank: Int, alignment: Int = 1
    ](coords: IndexList[rank]) {var inp0_ptr, var inp1_ptr, var out_ptr}:
        out_ptr.store(coords[0], inp0_ptr.load(coords[0]) * inp1_ptr.load(coords[0]))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def neg_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp_ptr = inputs[0].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[width: Int, rank: Int, alignment: Int = 1](coords: IndexList[rank]) {var inp_ptr, var out_ptr}:
        out_ptr.store(coords[0], -inp_ptr.load(coords[0]))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def div_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp0_ptr = inputs[0].data_ptr()
    var inp1_ptr = inputs[1].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[
        width: Int, rank: Int, alignment: Int = 1
    ](coords: IndexList[rank]) {var inp0_ptr, var inp1_ptr, var out_ptr}:
        out_ptr.store(coords[0], inp0_ptr.load(coords[0]) / inp1_ptr.load(coords[0]))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def scale_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp_ptr = inputs[0].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var scalar = node.attrs()["scalar"][Float32]

    def apply[
        width: Int, rank: Int, alignment: Int = 1
    ](coords: IndexList[rank]) {var inp_ptr, var out_ptr, var scalar}:
        out_ptr.store(coords[0], inp_ptr.load(coords[0]) * scalar)

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def exp_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp_ptr = inputs[0].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[width: Int, rank: Int, alignment: Int = 1](coords: IndexList[rank]) {var inp_ptr, var out_ptr}:
        out_ptr.store(coords[0], exp(inp_ptr.load(coords[0])))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def log_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp_ptr = inputs[0].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[width: Int, rank: Int, alignment: Int = 1](coords: IndexList[rank]) {var inp_ptr, var out_ptr}:
        out_ptr.store(coords[0], log(inp_ptr.load(coords[0])))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def eq_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp0_ptr = inputs[0].data_ptr()
    var inp1_ptr = inputs[1].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[
        width: Int, rank: Int, alignment: Int = 1
    ](coords: IndexList[rank]) {var inp0_ptr, var inp1_ptr, var out_ptr}:
        var idx = coords[0]
        out_ptr.store(idx, Float32(1) if inp0_ptr.load(idx) == inp1_ptr.load(idx) else Float32(0))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


# ===-------------------------------------------------------------------===#
# Activations
# ===-------------------------------------------------------------------===#


def relu_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp_ptr = inputs[0].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[width: Int, rank: Int, alignment: Int = 1](coords: IndexList[rank]) {var inp_ptr, var out_ptr}:
        var x = inp_ptr.load(coords[0])
        out_ptr.store(coords[0], x if x > Float32(0) else Float32(0))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def relu_grad_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = inputs[0].size
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var x_ptr = inputs[0].data_ptr()
    var up_ptr = inputs[1].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[width: Int, rank: Int, alignment: Int = 1](coords: IndexList[rank]) {var x_ptr, var up_ptr, var out_ptr}:
        var idx = coords[0]
        out_ptr.store(idx, up_ptr.load(idx) if x_ptr.load(idx) > Float32(0) else Float32(0))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def softmax_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var shape = inputs[0].shape
    var rank = len(shape)
    var rows = shape[0] if rank > 1 else 1
    var cols = shape[rank - 1]
    var size = rows * cols
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var out = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(rows, cols)))
    var inp_ptr = inputs[0].data_ptr()

    def input_fn[width: Int, r: Int](coords: IndexList[r]) capturing -> SIMD[DType.float32, width]:
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
        inputs[0].data_ptr(),
        inputs[1].data_ptr(),
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
    var inp_ptr = inputs[0].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) capturing -> SIMD[DType.float32, width]:
        return inp_ptr.load[width=width](coords[0])

    def output_fn[width: Int, rank: Int](coords: IndexList[rank], val: SIMD[DType.float32, width]) capturing:
        out_ptr.store[width=width](coords[0], val)

    reduce_sum[DType.float32, input_fn, output_fn, target="gpu"](
        IndexList[1](size),
        reduce_dim=0,
        context=ctx,
    )
    return Buffer(out_buf^, (1,), 1)


def argmax_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var N = inputs[0].shape[0]
    var C = inputs[0].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var inp = TileTensor(inputs[0].data_ptr().as_any_origin(), row_major(Coord(N, C)))
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
    comptime TILE = 32
    comptime kernel = transpose_kernel[TILE]
    ctx.enqueue_function[kernel](
        inputs[0].data_ptr(),
        out_buf.unsafe_ptr(),
        M,
        N,
        grid_dim=(ceildiv(N, TILE), ceildiv(M, TILE)),
        block_dim=(TILE, TILE),
    )
    return Buffer(out_buf^, (N, M), M * N)


def slice_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var start = Int(node.attrs()["start"][Float32])
    var cols = inputs[0].size // inputs[0].shape[0]
    return Buffer(
        inputs[0]._ptr,
        node.shape(),
        inputs[0].strides,
        inputs[0].base_offset + start * cols,
    )


def broadcast_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[DType.float32](size)
    var inp_ptr = inputs[0].data_ptr()
    var inp_size = inputs[0].size
    var out_ptr = out_buf.unsafe_ptr()

    def apply[
        width: Int, rank: Int, alignment: Int = 1
    ](coords: IndexList[rank]) {var inp_ptr, var inp_size, var out_ptr}:
        out_ptr.store(coords[0], inp_ptr.load(coords[0] % inp_size))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](size), ctx)
    return Buffer(out_buf^, node.shape(), size)


def one_hot_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var N = inputs[0].size
    var C = Int(node.attrs()["num_classes"][Float32])
    var out_buf = ctx.enqueue_create_buffer[DType.float32](N * C)
    var inp_ptr = inputs[0].data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[width: Int, rank: Int, alignment: Int = 1](coords: IndexList[rank]) {var inp_ptr, var out_ptr, var C}:
        var idx = coords[0]
        var row = idx // C
        var col = idx % C
        out_ptr.store(idx, Float32(1) if Int(inp_ptr.load(row)) == col else Float32(0))

    elementwise[simd_width=1, target="gpu"](apply, IndexList[1](N * C), ctx)
    return Buffer(out_buf^, node.shape(), N * C)


# ===-------------------------------------------------------------------===#
# Contractions
# ===-------------------------------------------------------------------===#


def matmul_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var M = inputs[0].shape[0]
    var K = inputs[0].shape[1]
    var N = inputs[1].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a = TileTensor(inputs[0].data_ptr().as_any_origin(), row_major(Coord(M, K)))
    var b = TileTensor(inputs[1].data_ptr().as_any_origin(), row_major(Coord(K, N)))
    var c = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(M, N)))
    matmul[target="gpu"](c, a, b, ctx)
    return Buffer(out_buf^, (M, N), M * N)


def matmul_t_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var M = inputs[0].shape[0]
    var K = inputs[0].shape[1]
    var N = inputs[1].shape[0]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](M * N)
    var a = TileTensor(inputs[0].data_ptr().as_any_origin(), row_major(Coord(M, K)))
    var b = TileTensor(inputs[1].data_ptr().as_any_origin(), row_major(Coord(N, K)))
    var c = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(M, N)))
    matmul[target="gpu", transpose_b=True](c, a, b, ctx)
    return Buffer(out_buf^, (M, N), M * N)


# ===-------------------------------------------------------------------===#
# Loss
# ===-------------------------------------------------------------------===#


comptime CE_BLOCK = 32
comptime CE_SMEM_LIMIT = DeviceContext.default_device_info.shared_memory_per_multiprocessor - 1024


def cross_entropy_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    var N = inputs[0].shape[0]
    var C = inputs[0].shape[1]
    var row_buf = ctx.enqueue_create_buffer[DType.float32](N)
    var shared_mem_bytes = C * size_of[Float32]()

    if shared_mem_bytes <= CE_SMEM_LIMIT:
        ctx.enqueue_function[cross_entropy_kernel[CE_BLOCK]](
            inputs[0].data_ptr(),
            inputs[1].data_ptr(),
            row_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
            shared_mem_bytes=shared_mem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(shared_mem_bytes)),
        )
    else:
        ctx.enqueue_function[cross_entropy_kernel_no_smem[CE_BLOCK]](
            inputs[0].data_ptr(),
            inputs[1].data_ptr(),
            row_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )

    var scalar_buf = ctx.enqueue_create_buffer[DType.float32](1)
    var row_ptr = row_buf.unsafe_ptr()
    var scalar_ptr = scalar_buf.unsafe_ptr()

    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) capturing -> SIMD[DType.float32, width]:
        return row_ptr.load[width=width](coords[0])

    def output_fn[width: Int, rank: Int](coords: IndexList[rank], val: SIMD[DType.float32, width]) capturing:
        scalar_ptr.store[width=width](coords[0], val)

    reduce_sum[DType.float32, input_fn, output_fn, target="gpu"](
        IndexList[1](N),
        reduce_dim=0,
        context=ctx,
    )
    ctx.synchronize()
    return Buffer(scalar_buf^, (1,), 1)


def cross_entropy_grad_exec(node: OpRef, inputs: List[Buffer], ctx: DeviceContext) raises -> Buffer:
    # inputs[0] = logits, inputs[1] = labels, inputs[2] = upstream
    var N = inputs[0].shape[0]
    var C = inputs[0].shape[1]
    var out_buf = ctx.enqueue_create_buffer[DType.float32](N * C)
    var shared_mem_bytes = C * size_of[Float32]()

    if shared_mem_bytes <= CE_SMEM_LIMIT:
        ctx.enqueue_function[cross_entropy_grad_kernel[CE_BLOCK]](
            inputs[2].data_ptr(),
            inputs[0].data_ptr(),
            inputs[1].data_ptr(),
            out_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
            shared_mem_bytes=shared_mem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(shared_mem_bytes)),
        )
    else:
        ctx.enqueue_function[cross_entropy_grad_kernel_no_smem[CE_BLOCK]](
            inputs[2].data_ptr(),
            inputs[0].data_ptr(),
            inputs[1].data_ptr(),
            out_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )

    return Buffer(out_buf^, (N, C), N * C)
