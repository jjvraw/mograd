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
from mograd.buffer import Buffer, AnyBuffer, BufferArm
from mograd.runtime.native.gpu.kernels import (
    softmax_grad_kernel,
    transpose_kernel,
    cross_entropy_kernel,
    cross_entropy_kernel_no_smem,
    cross_entropy_grad_kernel,
    cross_entropy_grad_kernel_no_smem,
)
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime import Runtime
from mograd.scheduler import Scheduler, make_bound, BoundExecFn

# ===-------------------------------------------------------------------===#
# GPURuntime
# ===-------------------------------------------------------------------===#


struct GPURuntime(Runtime):
    @staticmethod
    def run(root: OpRef, ctx: Optional[DeviceContext]) raises -> AnyBuffer:
        if not ctx:
            raise Error("GPURuntime requires a DeviceContext")
        return Scheduler[
            [
                # Constructions
                Rule(Pat(OpType.FULL), make_bound[full_exec]),
                Rule(Pat(OpType.UNIFORM), make_bound[uniform_exec]),
                Rule(Pat(OpType.RANDN), make_bound[randn_exec]),
                Rule(Pat(OpType.DISK), make_bound[disk_exec]),
                # Pointwise
                Rule(Pat(OpType.ADD), make_bound[add_exec]),
                Rule(Pat(OpType.MUL), make_bound[mul_exec]),
                Rule(Pat(OpType.NEG), make_bound[neg_exec]),
                Rule(Pat(OpType.DIV), make_bound[div_exec]),
                Rule(Pat(OpType.SCALE), make_bound[scale_exec]),
                Rule(Pat(OpType.EXP, fp_only=True), make_bound[exp_exec, fp_only=True]),
                Rule(Pat(OpType.LOG, fp_only=True), make_bound[log_exec, fp_only=True]),
                Rule(Pat(OpType.EQ), make_bound[eq_exec]),
                # Activations
                Rule(Pat(OpType.RELU), make_bound[relu_exec]),
                Rule(Pat(OpType.RELU_GRAD), make_bound[relu_grad_exec]),
                Rule(Pat(OpType.SOFTMAX, fp_only=True), make_bound[softmax_exec, fp_only=True]),
                Rule(Pat(OpType.SOFTMAX_GRAD, fp_only=True), make_bound[softmax_grad_exec, fp_only=True]),
                # Reductions
                Rule(Pat(OpType.SUM), make_bound[sum_exec]),
                Rule(Pat(OpType.ARGMAX), make_bound[argmax_exec]),
                # Shape
                Rule(Pat(OpType.RESHAPE), make_bound[reshape_exec]),
                Rule(Pat(OpType.TRANSPOSE), make_bound[transpose_exec]),
                Rule(Pat(OpType.SLICE), make_bound[slice_exec]),
                Rule(Pat(OpType.BROADCAST), make_bound[broadcast_exec]),
                Rule(Pat(OpType.ONE_HOT), make_bound[one_hot_exec, n_dtypes=2]),
                Rule(Pat(OpType.CAST), make_bound[cast_exec, n_dtypes=2]),
                # Contractions
                Rule(Pat(OpType.MATMUL), make_bound[matmul_exec]),
                Rule(Pat(MATMUL_T), make_bound[matmul_t_exec]),
                # Loss
                Rule(Pat(OpType.CROSS_ENTROPY, fp_only=True), make_bound[cross_entropy_exec, fp_only=True]),
                Rule(Pat(OpType.CROSS_ENTROPY_GRAD, fp_only=True), make_bound[cross_entropy_grad_exec, fp_only=True]),
            ]
        ].run(root, ctx.value())


# ===-------------------------------------------------------------------===#
# Constructions
# ===-------------------------------------------------------------------===#


def full_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    out_buf.enqueue_fill(Scalar[dtype](node.attrs()["value"][Float32]))
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def uniform_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(node.attrs()["seed"][Float32])))
    var out_ptr = out_buf.unsafe_ptr()

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        out_ptr.store(idx[0], val)

    var low = Scalar[dtype](node.attrs()["low"][Float32])
    var high = Scalar[dtype](node.attrs()["high"][Float32])
    random_uniform[output_fn=store, target="gpu"](IndexList[1](size), low, high, seed_buf.unsafe_ptr(), ctx)
    ctx.synchronize()
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def randn_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(node.attrs()["seed"][Float32])))
    var out_ptr = out_buf.unsafe_ptr()

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        out_ptr.store(idx[0], val)

    random_normal[output_fn=store, target="gpu"](
        IndexList[1](size),
        node.attrs()["mean"][Float32],
        node.attrs()["std"][Float32],
        seed_buf.unsafe_ptr(),
        ctx,
    )
    ctx.synchronize()
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def disk_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    var size = node.shape().numel()
    var bytes = Path(node.attrs()["path"][String]).read_bytes()
    var ptr = bytes.unsafe_ptr().bitcast[Scalar[dtype]]()
    var data = List[Scalar[dtype]]()
    data.reserve(size)
    for i in range(size):
        data.append(ptr[i])
    return AnyBuffer(Buffer[dtype].from_data(ctx, data, node.shape()))


# ===-------------------------------------------------------------------===#
# Pointwise
# ===-------------------------------------------------------------------===#


def add_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp0 = inputs[0].unsafe_get[dtype]()
    ref inp1 = inputs[1].unsafe_get[dtype]()
    var size = inp0.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp0_ptr = inp0.data_ptr()
    var inp1_ptr = inp1.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, inp0_ptr.load(idx) + inp1_ptr.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def mul_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp0 = inputs[0].unsafe_get[dtype]()
    ref inp1 = inputs[1].unsafe_get[dtype]()
    var size = inp0.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp0_ptr = inp0.data_ptr()
    var inp1_ptr = inp1.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, inp0_ptr.load(idx) * inp1_ptr.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def neg_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var size = inp.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, -inp_ptr.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def div_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp0 = inputs[0].unsafe_get[dtype]()
    ref inp1 = inputs[1].unsafe_get[dtype]()
    var size = inp0.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp0_ptr = inp0.data_ptr()
    var inp1_ptr = inp1.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, inp0_ptr.load(idx) / inp1_ptr.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def scale_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var size = inp.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()
    var scalar = Scalar[dtype](node.attrs()["scalar"][Float32])

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, inp_ptr.load(idx) * scalar)

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def exp_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    comptime assert dtype.is_floating_point(), "exp_exec requires a floating point dtype"
    ref inp = inputs[0].unsafe_get[dtype]()
    var size = inp.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, exp(inp_ptr.load(idx)))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def log_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    comptime assert dtype.is_floating_point(), "log_exec requires a floating point dtype"
    ref inp = inputs[0].unsafe_get[dtype]()
    var size = inp.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, log(inp_ptr.load(idx)))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def eq_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp0 = inputs[0].unsafe_get[dtype]()
    ref inp1 = inputs[1].unsafe_get[dtype]()
    var size = inp0.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp0_ptr = inp0.data_ptr()
    var inp1_ptr = inp1.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, Scalar[dtype](1) if inp0_ptr.load(idx) == inp1_ptr.load(idx) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


# ===-------------------------------------------------------------------===#
# Activations
# ===-------------------------------------------------------------------===#


def relu_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var size = inp.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        var x = inp_ptr.load(idx)
        out_ptr.store(idx, x if x > Scalar[dtype](0) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def relu_grad_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref x_in = inputs[0].unsafe_get[dtype]()
    ref up_in = inputs[1].unsafe_get[dtype]()
    var size = x_in.size
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var x_ptr = x_in.data_ptr()
    var up_ptr = up_in.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, up_ptr.load(idx) if x_ptr.load(idx) > Scalar[dtype](0) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def softmax_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var shape = inp.shape
    var rank = len(shape)
    var rows = shape[0] if rank > 1 else 1
    var cols = shape[rank - 1]
    var size = rows * cols
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var out = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(rows, cols)))
    var inp_ptr = inp.data_ptr()

    def input_fn[width: Int, r: Int](coords: IndexList[r]) capturing -> SIMD[dtype, width]:
        return inp_ptr.load[width=width](coords[0] * cols + coords[1])

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    nn_softmax[dtype, simd_width, 2, input_fn, "gpu"](IndexList[2](rows, cols), out, axis=1, context=ctx)
    ctx.synchronize()
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def softmax_grad_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp0 = inputs[0].unsafe_get[dtype]()
    ref inp1 = inputs[1].unsafe_get[dtype]()
    var shape = inp0.shape
    var N = 1 if len(shape) == 1 else shape[0]
    var size = shape[-1]
    var out_buf = ctx.enqueue_create_buffer[dtype](N * size)
    comptime BLOCK_SIZE = 32
    ctx.enqueue_function[softmax_grad_kernel[dtype, BLOCK_SIZE]](
        inp0.data_ptr(),
        inp1.data_ptr(),
        out_buf.unsafe_ptr(),
        N,
        size,
        grid_dim=(N,),
        block_dim=(BLOCK_SIZE,),
    )
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), N * size))


# ===-------------------------------------------------------------------===#
# Reductions
# ===-------------------------------------------------------------------===#


def sum_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var size = inp.size
    var out_buf = ctx.enqueue_create_buffer[dtype](1)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) capturing -> SIMD[dtype, width]:
        return inp_ptr.load[width=width](coords[0])

    def output_fn[width: Int, rank: Int](coords: IndexList[rank], val: SIMD[dtype, width]) capturing:
        out_ptr.store[width=width](coords[0], val)

    reduce_sum[dtype, input_fn, output_fn, target="gpu", reduce_dim=0](IndexList[1](size), ctx)
    ctx.synchronize()
    return AnyBuffer(Buffer[dtype](out_buf^, (1,), 1))


def argmax_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var N = inp.shape[0]
    var C = inp.shape[1]
    var out_buf = ctx.enqueue_create_buffer[dtype](N)
    var a = TileTensor(inp.data_ptr().as_any_origin(), row_major(Coord(N, C)))
    var out = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(N, 1)))
    argmax_gpu[dtype, dtype](ctx, a, out)
    return AnyBuffer(Buffer[dtype](out_buf^, (N,), N))


# ===-------------------------------------------------------------------===#
# Shape
# ===-------------------------------------------------------------------===#


def reshape_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref src = inputs[0].unsafe_get[dtype]()
    return AnyBuffer(Buffer[dtype](src._ptr, node.shape(), node.shape().strides(), src.base_offset))


def transpose_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var M = inp.shape[0]
    var N = inp.shape[1]
    var out_buf = ctx.enqueue_create_buffer[dtype](M * N)
    comptime TILE = 32
    comptime kernel = transpose_kernel[dtype, TILE]
    ctx.enqueue_function[kernel](
        inp.data_ptr(),
        out_buf.unsafe_ptr(),
        M,
        N,
        grid_dim=(ceildiv(N, TILE), ceildiv(M, TILE)),
        block_dim=(TILE, TILE),
    )
    return AnyBuffer(Buffer[dtype](out_buf^, (N, M), M * N))


def slice_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var start = Int(node.attrs()["start"][Float32])
    var cols = inp.size // inp.shape[0]
    return AnyBuffer(Buffer[dtype](inp._ptr, node.shape(), inp.strides, inp.base_offset + start * cols))


def broadcast_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp = inputs[0].unsafe_get[dtype]()
    var size = node.shape().numel()
    var out_buf = ctx.enqueue_create_buffer[dtype](size)
    var inp_ptr = inp.data_ptr()
    var inp_size = inp.size
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, inp_ptr.load(idx % inp_size))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, node.shape(), size))


def one_hot_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime in_dtype = dtypes[0]
    comptime out_dtype = dtypes[1]
    ref inp = inputs[0].unsafe_get[in_dtype]()
    var C = Int(node.attrs()["num_classes"][Float32])
    var N = inp.size
    var out_buf = ctx.enqueue_create_buffer[out_dtype](N * C)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, Scalar[out_dtype](1) if Int(inp_ptr.load(idx // C)) == (idx % C) else Scalar[out_dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(N * C), ctx)
    return AnyBuffer(Buffer[out_dtype](out_buf^, node.shape(), N * C))


def cast_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime in_dtype = dtypes[0]
    comptime out_dtype = dtypes[1]
    ref inp = inputs[0].unsafe_get[in_dtype]()
    var size = inp.size
    var out_buf = ctx.enqueue_create_buffer[out_dtype](size)
    var inp_ptr = inp.data_ptr()
    var out_ptr = out_buf.unsafe_ptr()

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        out_ptr.store(idx, Scalar[out_dtype](inp_ptr.load(idx)))

    elementwise[simd_width=1, target="gpu"](apply, Coord(size), ctx)
    return AnyBuffer(Buffer[out_dtype](out_buf^, node.shape(), size))


# ===-------------------------------------------------------------------===#
# Contractions
# ===-------------------------------------------------------------------===#


def matmul_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp0 = inputs[0].unsafe_get[dtype]()
    ref inp1 = inputs[1].unsafe_get[dtype]()
    var M = inp0.shape[0]
    var K = inp0.shape[1]
    var N = inp1.shape[1]
    var out_buf = ctx.enqueue_create_buffer[dtype](M * N)
    var a = TileTensor(inp0.data_ptr().as_any_origin(), row_major(Coord(M, K)))
    var b = TileTensor(inp1.data_ptr().as_any_origin(), row_major(Coord(K, N)))
    var c = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(M, N)))
    matmul[target="gpu"](c, a, b, ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, (M, N), M * N))


def matmul_t_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    ref inp0 = inputs[0].unsafe_get[dtype]()
    ref inp1 = inputs[1].unsafe_get[dtype]()
    var M = inp0.shape[0]
    var K = inp0.shape[1]
    var N = inp1.shape[0]
    var out_buf = ctx.enqueue_create_buffer[dtype](M * N)
    var a = TileTensor(inp0.data_ptr().as_any_origin(), row_major(Coord(M, K)))
    var b = TileTensor(inp1.data_ptr().as_any_origin(), row_major(Coord(N, K)))
    var c = TileTensor(out_buf.unsafe_ptr().as_any_origin(), row_major(Coord(M, N)))
    matmul[target="gpu", transpose_b=True](c, a, b, ctx)
    return AnyBuffer(Buffer[dtype](out_buf^, (M, N), M * N))


# ===-------------------------------------------------------------------===#
# Loss
# ===-------------------------------------------------------------------===#


comptime CE_BLOCK = 32
comptime CE_SMEM_LIMIT = DeviceContext.default_device_info.shared_memory_per_multiprocessor - 1024


def cross_entropy_exec[*dtypes: DType](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    comptime assert dtype.is_floating_point(), "cross_entropy requires a floating point dtype"
    ref logits = inputs[0].unsafe_get[dtype]()
    ref labels = inputs[1].unsafe_get[dtype]()
    var N = logits.shape[0]
    var C = logits.shape[1]
    var row_buf = ctx.enqueue_create_buffer[dtype](N)
    var shared_mem_bytes = C * size_of[Scalar[dtype]]()
    if shared_mem_bytes <= CE_SMEM_LIMIT:
        ctx.enqueue_function[cross_entropy_kernel[dtype, CE_BLOCK]](
            logits.data_ptr(),
            labels.data_ptr(),
            row_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
            shared_mem_bytes=shared_mem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(shared_mem_bytes)),
        )
    else:
        ctx.enqueue_function[cross_entropy_kernel_no_smem[dtype, CE_BLOCK]](
            logits.data_ptr(),
            labels.data_ptr(),
            row_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )
    var scalar_buf = ctx.enqueue_create_buffer[dtype](1)
    var row_ptr = row_buf.unsafe_ptr()
    var scalar_ptr = scalar_buf.unsafe_ptr()

    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) capturing -> SIMD[dtype, width]:
        return row_ptr.load[width=width](coords[0])

    def output_fn[width: Int, rank: Int](coords: IndexList[rank], val: SIMD[dtype, width]) capturing:
        scalar_ptr.store[width=width](coords[0], val)

    reduce_sum[dtype, input_fn, output_fn, target="gpu", reduce_dim=0](IndexList[1](N), ctx)
    ctx.synchronize()
    return AnyBuffer(Buffer[dtype](scalar_buf^, (1,), 1))


def cross_entropy_grad_exec[
    *dtypes: DType
](node: OpRef, inputs: List[AnyBuffer], ctx: DeviceContext) raises -> AnyBuffer:
    comptime dtype = dtypes[0]
    comptime assert dtype.is_floating_point(), "cross_entropy requires a floating point dtype"
    ref logits = inputs[0].unsafe_get[dtype]()
    ref labels = inputs[1].unsafe_get[dtype]()
    ref upstream = inputs[2].unsafe_get[dtype]()
    var N = logits.shape[0]
    var C = logits.shape[1]
    var out_buf = ctx.enqueue_create_buffer[dtype](N * C)
    var shared_mem_bytes = C * size_of[Scalar[dtype]]()
    if shared_mem_bytes <= CE_SMEM_LIMIT:
        ctx.enqueue_function[cross_entropy_grad_kernel[dtype, CE_BLOCK]](
            upstream.data_ptr(),
            logits.data_ptr(),
            labels.data_ptr(),
            out_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
            shared_mem_bytes=shared_mem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(shared_mem_bytes)),
        )
    else:
        ctx.enqueue_function[cross_entropy_grad_kernel_no_smem[dtype, CE_BLOCK]](
            upstream.data_ptr(),
            logits.data_ptr(),
            labels.data_ptr(),
            out_buf.unsafe_ptr(),
            N,
            C,
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )
    return AnyBuffer(Buffer[dtype](out_buf^, (N, C), N * C))
