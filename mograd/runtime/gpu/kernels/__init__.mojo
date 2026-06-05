from mograd.buffer import AnyBuffer, BufferArm
from layout import Coord, TileTensor, row_major
from std.algorithm.functional import elementwise
from std.algorithm.reduction import sum as reduce_sum
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext, FuncAttribute, get_gpu_target
from std.sys.info import size_of
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import simd_width_of
from std.utils import IndexList
from linalg.matmul import matmul as linalg_matmul
from nn.argmaxmin_gpu import argmax_gpu
from nn.rand_normal import random_normal
from nn.rand_uniform import random_uniform
from nn.softmax import softmax as nn_softmax
from std.gpu.memory import external_memory
from std.gpu.primitives.warp import max as warp_max, sum as warp_sum


@export
def mograd_add(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = AnyBuffer.BufVariant.Ts[k].node_dtype
        if dtype == d:
            add[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def add[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, a.load(idx) + b.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_one_hot(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    in_dtype: DType,
    out_dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for ki in range(AnyBuffer.BufVariant.Ts.size):
        comptime Ti = AnyBuffer.BufVariant.Ts[ki]
        comptime assert conforms_to(Ti, BufferArm)
        comptime src_d = Ti.node_dtype
        if in_dtype == src_d:
            comptime for ko in range(AnyBuffer.BufVariant.Ts.size):
                comptime To = AnyBuffer.BufVariant.Ts[ko]
                comptime assert conforms_to(To, BufferArm)
                comptime dst_d = To.node_dtype
                comptime if dst_d.is_integral():
                    if out_dtype == dst_d:
                        var C = Int(b.bitcast[Float32]()[0])
                        one_hot[src_d, dst_d](a.bitcast[Scalar[src_d]](), dst.bitcast[Scalar[dst_d]](), n, C, ctx)
                        return
    raise Error("unsupported dtype combination")


def one_hot[
    in_dtype: DType, out_dtype: DType
](
    a: UnsafePointer[Scalar[in_dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[out_dtype], MutAnyOrigin],
    n: Int,
    C: Int,
    ctx: DeviceContext,
) raises where out_dtype.is_integral():
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        var class_idx = Int(a.load(idx // C))
        dst.store(idx, Scalar[out_dtype](1) if class_idx == (idx % C) else Scalar[out_dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_full(
    fill_val: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var val = Scalar[d](fill_val.bitcast[Float32]()[0])
            full[d](val, dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def full[
    dtype: DType
](val: Scalar[dtype], dst: UnsafePointer[Scalar[dtype], MutAnyOrigin], n: Int, ctx: DeviceContext,) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        dst.store(Int(coord[0].value()), val)

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_log(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if d.is_floating_point():
            if dtype == d:
                log[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
                return
    raise Error("unsupported dtype")


def log[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    from std.math import log as math_log

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, math_log(a.load(idx)))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_relu(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            relu[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def relu[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        var x = a.load(idx)
        dst.store(idx, x if x > Scalar[dtype](0) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_exp(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if d.is_floating_point():
            if dtype == d:
                exp[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
                return
    raise Error("unsupported dtype")


def exp[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    from std.math import exp as math_exp

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, math_exp(a.load(idx)))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_neg(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            neg[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def neg[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, -a.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_scale(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    scalar: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = AnyBuffer.BufVariant.Ts[k].node_dtype
        if dtype == d:
            scale[d](a.bitcast[Scalar[d]](), scalar.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def scale[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    var scalar_val = b.load(0)

    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, a.load(idx) * scalar_val)

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_eq(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            eq[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def eq[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, Scalar[dtype](1) if a.load(idx) == b.load(idx) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_mul(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            mul[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def mul[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, a.load(idx) * b.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_div(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            div[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def div[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, a.load(idx) / b.load(idx))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


@export
def mograd_randn(
    params: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var p = params.bitcast[Float32]()
            randn[d](dst.bitcast[Scalar[d]](), n, p[0], p[1], p[2], ctx)
            return
    raise Error("unsupported dtype")


def randn[
    dtype: DType
](
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    mean: Float32,
    std: Float32,
    seed: Float32,
    ctx: DeviceContext,
) raises:
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(seed)))

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        dst.store(idx[0], val)

    random_normal[output_fn=store, target="gpu"](
        IndexList[1](n),
        mean,
        std,
        seed_buf.unsafe_ptr(),
        ctx,
    )
    ctx.synchronize()


@export
def mograd_softmax(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if d.is_floating_point():
            if dtype == d:
                var p = shape_ptr.bitcast[Float32]()
                var rows = Int(p[0])
                var cols = Int(p[1])
                softmax[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), rows, cols, ctx)
                return
    raise Error("unsupported dtype")


def softmax[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rows: Int,
    cols: Int,
    ctx: DeviceContext,
) raises:
    var out = TileTensor(dst.as_any_origin(), row_major(Coord(rows, cols)))

    def input_fn[width: Int, r: Int](coords: IndexList[r]) capturing -> SIMD[dtype, width]:
        return a.load[width=width](coords[0] * cols + coords[1])

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    nn_softmax[dtype, simd_width, 2, input_fn, "gpu"](IndexList[2](rows, cols), out, axis=1, context=ctx)
    ctx.synchronize()


@export
def mograd_sum(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            sum[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def sum[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def input_fn[width: Int, rank: Int](coords: IndexList[rank]) capturing -> SIMD[dtype, width]:
        return a.load[width=width](coords[0])

    def output_fn[width: Int, rank: Int](coords: IndexList[rank], val: SIMD[dtype, width]) capturing:
        dst.store[width=width](coords[0], val)

    reduce_sum[dtype, input_fn, output_fn, target="gpu", reduce_dim=0](IndexList[1](n), ctx)
    ctx.synchronize()


@export
def mograd_matmul(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var p = shape_ptr.bitcast[Float32]()
            var M = Int(p[0])
            var K = Int(p[1])
            var N = Int(p[2])
            matmul[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), M, K, N, ctx)
            return
    raise Error("unsupported dtype")


def matmul[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    ctx: DeviceContext,
) raises:
    var ta = TileTensor(a.as_any_origin(), row_major(Coord(M, K)))
    var tb = TileTensor(b.as_any_origin(), row_major(Coord(K, N)))
    var tc = TileTensor(dst.as_any_origin(), row_major(Coord(M, N)))
    linalg_matmul[target="gpu"](tc, ta, tb, ctx)


@export
def mograd_matmul_t(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var p = shape_ptr.bitcast[Float32]()
            var M = Int(p[0])
            var K = Int(p[1])
            var N = Int(p[2])
            matmul_t[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), M, K, N, ctx)
            return
    raise Error("unsupported dtype")


def matmul_t[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    ctx: DeviceContext,
) raises:
    var ta = TileTensor(a.as_any_origin(), row_major(Coord(M, K)))
    var tb = TileTensor(b.as_any_origin(), row_major(Coord(N, K)))
    var tc = TileTensor(dst.as_any_origin(), row_major(Coord(M, N)))
    linalg_matmul[target="gpu", transpose_b=True](tc, ta, tb, ctx)


def transpose_kernel[
    dtype: DType, BLOCK_SIZE: Int
](src: UnsafePointer[Scalar[dtype], MutAnyOrigin], dst: UnsafePointer[Scalar[dtype], MutAnyOrigin], M: Int, N: Int,):
    var shmem = stack_allocation[BLOCK_SIZE * (BLOCK_SIZE + 1), dtype, address_space=AddressSpace.SHARED]()
    x = block_idx.x * BLOCK_SIZE + thread_idx.x
    y = block_idx.y * BLOCK_SIZE + thread_idx.y
    if x < N and y < M:
        shmem[thread_idx.y * (BLOCK_SIZE + 1) + thread_idx.x] = src[y * N + x]
    barrier()
    x_out = block_idx.y * BLOCK_SIZE + thread_idx.x
    y_out = block_idx.x * BLOCK_SIZE + thread_idx.y
    if x_out < M and y_out < N:
        dst[y_out * M + x_out] = shmem[thread_idx.x * (BLOCK_SIZE + 1) + thread_idx.y]


@export
def mograd_transpose(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var p = shape_ptr.bitcast[Float32]()
            var M = Int(p[0])
            var N = Int(p[1])
            comptime TILE = 32
            ctx.enqueue_function[transpose_kernel[d, TILE]](
                a.bitcast[Scalar[d]](),
                dst.bitcast[Scalar[d]](),
                M,
                N,
                grid_dim=(ceildiv(N, TILE), ceildiv(M, TILE)),
                block_dim=(TILE, TILE),
            )
            return
    raise Error("unsupported dtype")


@export
def mograd_argmax(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var p = shape_ptr.bitcast[Float32]()
            var N = Int(p[0])
            var C = Int(p[1])
            var inp = TileTensor(a.bitcast[Scalar[d]]().as_any_origin(), row_major(Coord(N, C)))
            var out = TileTensor(dst.bitcast[Scalar[d]]().as_any_origin(), row_major(Coord(N, 1)))
            argmax_gpu[d, d](ctx, inp, out)
            return
    raise Error("unsupported dtype")


def cross_entropy_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    from std.gpu.primitives.warp import max as warp_max, sum as warp_sum
    from std.gpu.memory import AddressSpace, external_memory
    from std.math import exp, log

    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tmp = external_memory[Scalar[dtype], address_space=AddressSpace.SHARED, alignment=4]()
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        tmp[i] = val
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)
    barrier()
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        sum_exp += exp(tmp[i] - max_val)
    sum_exp = warp_sum(sum_exp)
    var log_sum = log(sum_exp)
    var loss = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        loss += (tmp[i] - max_val - log_sum) * labels[row_offset + i]
    loss = -warp_sum(loss) / Scalar[dtype](N)
    if thread_idx.x == 0:
        dst[row] = loss


def cross_entropy_kernel_no_smem[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    from std.gpu.primitives.warp import max as warp_max, sum as warp_sum
    from std.math import exp, log

    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        sum_exp += exp(logits[row_offset + i] - max_val)
    sum_exp = warp_sum(sum_exp)
    var log_sum = log(sum_exp)
    var loss = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        loss += (logits[row_offset + i] - max_val - log_sum) * labels[row_offset + i]
    loss = -warp_sum(loss) / Scalar[dtype](N)
    if thread_idx.x == 0:
        dst[row] = loss


comptime CE_BLOCK = 32
comptime CE_SMEM_LIMIT = DeviceContext.default_device_info.shared_memory_per_multiprocessor - 1024


@export
def mograd_cross_entropy(
    logits: UnsafePointer[NoneType, MutAnyOrigin],
    labels: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if d.is_floating_point():
            if dtype == d:
                var p = shape_ptr.bitcast[Float32]()
                var N = Int(p[0])
                var C = Int(p[1])
                var row_buf = ctx.enqueue_create_buffer[d](N)
                var shared_mem_bytes = C * size_of[Scalar[d]]()
                if shared_mem_bytes <= CE_SMEM_LIMIT:
                    ctx.enqueue_function[cross_entropy_kernel[d, CE_BLOCK]](
                        logits.bitcast[Scalar[d]](),
                        labels.bitcast[Scalar[d]](),
                        row_buf.unsafe_ptr(),
                        N,
                        C,
                        grid_dim=(N,),
                        block_dim=(CE_BLOCK,),
                        shared_mem_bytes=shared_mem_bytes,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(shared_mem_bytes)),
                    )
                else:
                    ctx.enqueue_function[cross_entropy_kernel_no_smem[d, CE_BLOCK]](
                        logits.bitcast[Scalar[d]](),
                        labels.bitcast[Scalar[d]](),
                        row_buf.unsafe_ptr(),
                        N,
                        C,
                        grid_dim=(N,),
                        block_dim=(CE_BLOCK,),
                    )
                var row_ptr = row_buf.unsafe_ptr()
                var out_ptr = dst.bitcast[Scalar[d]]()

                def input_fn[width: Int, rank: Int](coords: IndexList[rank]) capturing -> SIMD[d, width]:
                    return row_ptr.load[width=width](coords[0])

                def output_fn[width: Int, rank: Int](coords: IndexList[rank], val: SIMD[d, width]) capturing:
                    out_ptr.store[width=width](coords[0], val)

                reduce_sum[d, input_fn, output_fn, target="gpu", reduce_dim=0](IndexList[1](N), ctx)
                ctx.synchronize()
                return
    raise Error("unsupported dtype")


@export
def mograd_cast(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    src_dtype: DType,
    dst_dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for ki in range(AnyBuffer.BufVariant.Ts.size):
        comptime Ti = AnyBuffer.BufVariant.Ts[ki]
        comptime assert conforms_to(Ti, BufferArm)
        comptime src_d = Ti.node_dtype
        if src_dtype == src_d:
            comptime for ko in range(AnyBuffer.BufVariant.Ts.size):
                comptime To = AnyBuffer.BufVariant.Ts[ko]
                comptime assert conforms_to(To, BufferArm)
                comptime dst_d = To.node_dtype
                if dst_dtype == dst_d:
                    cast[src_d, dst_d](a.bitcast[Scalar[src_d]](), dst.bitcast[Scalar[dst_d]](), n, ctx)
                    return
    raise Error("unsupported dtype combination")


def cast[
    src_dtype: DType, dst_dtype: DType
](
    a: UnsafePointer[Scalar[src_dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dst_dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, a.load(idx).cast[dst_dtype]())

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Broadcast
# ===-------------------------------------------------------------------===#


@export
def mograd_broadcast(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var inp_size = Int(b.bitcast[Float32]()[0])
            broadcast[d](a.bitcast[Scalar[d]](), inp_size, dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def broadcast[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    inp_size: Int,
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, a.load(idx % inp_size))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Uniform
# ===-------------------------------------------------------------------===#


@export
def mograd_uniform(
    params: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if d.is_floating_point():
            if dtype == d:
                uniform[d](params.bitcast[Float32](), dst.bitcast[Scalar[d]](), n, ctx)
                return
    raise Error("unsupported dtype")


def uniform[
    dtype: DType
](
    params: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    var low = Scalar[dtype](params[0])
    var high = Scalar[dtype](params[1])
    var seed_buf = ctx.enqueue_create_buffer[DType.uint64](1)
    seed_buf.enqueue_fill(UInt64(Int(params[2])))

    def store[width: Int, rank: Int](idx: IndexList[rank], val: SIMD[dtype, width]) capturing:
        dst.store(idx[0], val)

    random_uniform[output_fn=store, target="gpu"](IndexList[1](n), low, high, seed_buf.unsafe_ptr(), ctx)
    ctx.synchronize()


# ===-------------------------------------------------------------------===#
# ReLU Grad
# ===-------------------------------------------------------------------===#


@export
def mograd_relu_grad(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            relu_grad[d](a.bitcast[Scalar[d]](), b.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), n, ctx)
            return
    raise Error("unsupported dtype")


def relu_grad[
    dtype: DType
](
    x: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    upstream: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var idx = Int(coord[0].value())
        dst.store(idx, upstream.load(idx) if x.load(idx) > Scalar[dtype](0) else Scalar[dtype](0))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Softmax Grad
# ===-------------------------------------------------------------------===#


def softmax_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    upstream: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    size: Int,
):
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * size
    var dot = Scalar[dtype](0.0)
    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dot += y[row_offset + i] * upstream[row_offset + i]
    dot = warp_sum(dot)
    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dst[row_offset + i] = y[row_offset + i] * (upstream[row_offset + i] - dot)


@export
def mograd_softmax_grad(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    c: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        comptime if d.is_floating_point():
            if dtype == d:
                var p = c.bitcast[Float32]()
                var N = Int(p[0])
                var size = Int(p[1])
                comptime BLOCK_SIZE = 32
                ctx.enqueue_function[softmax_grad_kernel[d, BLOCK_SIZE]](
                    a.bitcast[Scalar[d]](),
                    b.bitcast[Scalar[d]](),
                    dst.bitcast[Scalar[d]](),
                    N,
                    size,
                    grid_dim=(N,),
                    block_dim=(BLOCK_SIZE,),
                )
                return
    raise Error("unsupported dtype")


# ===-------------------------------------------------------------------===#
# Cross Entropy Grad
# ===-------------------------------------------------------------------===#


def cross_entropy_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    from std.math import exp as math_exp

    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tmp = external_memory[Scalar[dtype], address_space=AddressSpace.SHARED, alignment=4]()
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        tmp[i] = val
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = math_exp(tmp[i] - max_val)
        sum_exp += val
        tmp[i] = val
    sum_exp = warp_sum(sum_exp)
    var sm_scale = Scalar[dtype](1.0) / sum_exp
    var d_by_nrows = grad[0] / Scalar[dtype](N)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        dst[row_offset + i] = (tmp[i] * sm_scale - labels[row_offset + i]) * d_by_nrows


def cross_entropy_grad_kernel_no_smem[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    from std.math import exp as math_exp

    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = math_exp(logits[row_offset + i] - max_val)
        sum_exp += val
        dst[row_offset + i] = val
    sum_exp = warp_sum(sum_exp)
    var sm_scale = Scalar[dtype](1.0) / sum_exp
    var d_by_nrows = grad[0] / Scalar[dtype](N)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        dst[row_offset + i] = (dst[row_offset + i] * sm_scale - labels[row_offset + i]) * d_by_nrows


@export
def mograd_cross_entropy_grad(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    c: UnsafePointer[NoneType, MutAnyOrigin],
    d: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime kd = T.node_dtype
        comptime if kd.is_floating_point():
            if dtype == kd:
                var p = d.bitcast[Float32]()
                var N = Int(p[0])
                var C = Int(p[1])
                var shared_mem_bytes = C * size_of[Scalar[kd]]()
                comptime BLOCK_SIZE = CE_BLOCK
                if shared_mem_bytes <= CE_SMEM_LIMIT:
                    ctx.enqueue_function[cross_entropy_grad_kernel[kd, BLOCK_SIZE]](
                        c.bitcast[Scalar[kd]](),
                        a.bitcast[Scalar[kd]](),
                        b.bitcast[Scalar[kd]](),
                        dst.bitcast[Scalar[kd]](),
                        N,
                        C,
                        grid_dim=(N,),
                        block_dim=(BLOCK_SIZE,),
                        shared_mem_bytes=shared_mem_bytes,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(shared_mem_bytes)),
                    )
                else:
                    ctx.enqueue_function[cross_entropy_grad_kernel_no_smem[kd, BLOCK_SIZE]](
                        c.bitcast[Scalar[kd]](),
                        a.bitcast[Scalar[kd]](),
                        b.bitcast[Scalar[kd]](),
                        dst.bitcast[Scalar[kd]](),
                        N,
                        C,
                        grid_dim=(N,),
                        block_dim=(BLOCK_SIZE,),
                    )
                return
    raise Error("unsupported dtype")
