from std.gpu.primitives.warp import max as warp_max, sum as warp_sum
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext, get_gpu_target
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import simd_width_of

from layout import Coord, MixedLayout, TileTensor, row_major

from linalg.matmul import matmul as linalg_matmul
from linalg.matmul.gpu import matmul_kernel_naive

from nn.argmaxmin_gpu import argmax_gpu
from nn.softmax import softmax as nn_softmax

from mograd.buffer import AnyBuffer, BufferArm
from mograd.runtime.gpu.kernels.factory import *
from mograd.runtime.gpu.kernels.elementwise import *
from mograd.runtime.gpu.kernels.reduce import *
from mograd.runtime.gpu.kernels.utils import (
    dispatch_binary_contiguous,
    dispatch_binary_map,
    dispatch_binary_scalar_map,
    dispatch_dtype,
    dispatch_unary,
    dispatch_unary_map,
)

# ===-------------------------------------------------------------------===#
# Factory Kernels
# ===-------------------------------------------------------------------===#


@export
def mograd_randn(
    params: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var p = params.bitcast[Float32]()
        randn[d](dst.bitcast[Scalar[d]](), n, p[0], p[1], p[2], ctx)

    dispatch_dtype[body](dtype)


@export
def mograd_uniform(
    params: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        comptime assert d.is_floating_point()
        uniform[d](params.bitcast[Float32](), dst.bitcast[Scalar[d]](), n, ctx)

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_full(
    fill_val: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var val = Scalar[d](fill_val.bitcast[Float32]()[0])
        full[d](val, dst.bitcast[Scalar[d]](), n, ctx)

    dispatch_dtype[body](dtype)


@export
def mograd_one_hot(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    in_dtype: DType,
    out_dtype: DType,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sd: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    ctx: DeviceContext,
) abi("Mojo") raises:
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
                        one_hot[src_d, dst_d](
                            a.bitcast[Scalar[src_d]](),
                            dst.bitcast[Scalar[dst_d]](),
                            n,
                            rank,
                            inner,
                            sd,
                            sa,
                            ctx,
                        )
                        return
    raise Error("unsupported dtype combination")


# ===-------------------------------------------------------------------===#
# Unary Maps
# ===-------------------------------------------------------------------===#


@export
def mograd_neg(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_unary_map[neg_op](a, dst, numel, rank, inner, sa, dtype, ctx)


@export
def mograd_log(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_unary_map[log_op, float_only=True](a, dst, numel, rank, inner, sa, dtype, ctx)


@export
def mograd_exp(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_unary_map[exp_op, float_only=True](a, dst, numel, rank, inner, sa, dtype, ctx)


@export
def mograd_relu(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_unary_map[relu_op](a, dst, numel, rank, inner, sa, dtype, ctx)


@export
def mograd_slice_grad(
    upstream: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_unary[slice_grad](upstream, dst, numel, rank, inner, sa, dtype, ctx)


@export
def mograd_cast(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    in_dtype: DType,
    out_dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    comptime for ki in range(AnyBuffer.BufVariant.Ts.size):
        comptime Ti = AnyBuffer.BufVariant.Ts[ki]
        comptime assert conforms_to(Ti, BufferArm)
        comptime src_d = Ti.node_dtype
        if in_dtype == src_d:
            comptime for ko in range(AnyBuffer.BufVariant.Ts.size):
                comptime To = AnyBuffer.BufVariant.Ts[ko]
                comptime assert conforms_to(To, BufferArm)
                comptime dst_d = To.node_dtype
                if out_dtype == dst_d:
                    cast[src_d, dst_d](
                        a.bitcast[Scalar[src_d]](), dst.bitcast[Scalar[dst_d]](), rank, inner, sa, numel, ctx
                    )
                    return
    raise Error("unsupported dtype combination")


@export
def mograd_contiguous(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    numel: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_unary_map[identity_op](a, dst, numel, rank, inner, sa, dtype, ctx)


# ===-------------------------------------------------------------------===#
# Binary Maps
# ===-------------------------------------------------------------------===#


@export
def mograd_add(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_contiguous[add](a, b, dst, n, dtype, ctx)


@export
def mograd_add_strided(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_map[add_op](a, b, dst, n, rank, inner, sa, sb, dtype, ctx)


@export
def mograd_mul(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_map[mul_op](a, b, dst, n, rank, inner, sa, sb, dtype, ctx)


@export
def mograd_div(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_map[div_op](a, b, dst, n, rank, inner, sa, sb, dtype, ctx)


@export
def mograd_eq(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_map[eq_op](a, b, dst, n, rank, inner, sa, sb, dtype, ctx)


@export
def mograd_relu_grad(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_map[relu_grad_op](a, b, dst, n, rank, inner, sa, sb, dtype, ctx)


@export
def mograd_scale(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_scalar_map[mul_op](a, b, dst, n, rank, inner, sa, dtype, ctx)


# ===-------------------------------------------------------------------===#
# Matmul
# ===-------------------------------------------------------------------===#
# TODO: Support vendors https://github.com/pytorch/pytorch/pull/184248


@export
def mograd_matmul(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    lda: Int,
    lda1: Int,
    ldb: Int,
    ldb1: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        if lda == K and lda1 == 1 and ldb == N and ldb1 == 1:
            var ta = TileTensor(a.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(M, K)))
            var tb = TileTensor(b.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(K, N)))
            var tc = TileTensor(dst.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(M, N)))
            linalg_matmul[target="gpu"](tc, ta, tb, ctx)
        else:
            var a_imm: UnsafePointer[Scalar[d], ImmutAnyOrigin] = a.bitcast[Scalar[d]]()
            var b_imm: UnsafePointer[Scalar[d], ImmutAnyOrigin] = b.bitcast[Scalar[d]]()
            var ta = TileTensor(a_imm, MixedLayout(Coord(M, K), Coord(lda, lda1)))
            var tb = TileTensor(b_imm, MixedLayout(Coord(K, N), Coord(ldb, ldb1)))
            var tc = TileTensor(dst.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(M, N)))
            comptime BLOCK = 16
            comptime naive = matmul_kernel_naive[
                d, d, d, type_of(tc).LayoutType, type_of(ta).LayoutType, type_of(tb).LayoutType, BLOCK
            ]
            ctx.enqueue_function[naive](
                tc, ta, tb, M, N, K, grid_dim=(ceildiv(M, BLOCK), ceildiv(N, BLOCK)), block_dim=(BLOCK, BLOCK)
            )

    dispatch_dtype[body](dtype)


@export
def mograd_matmul_t(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    b: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    M: Int,
    K: Int,
    N: Int,
    lda: Int,
    lda1: Int,
    ldb: Int,
    ldb1: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        if lda == K and lda1 == 1 and ldb == K and ldb1 == 1:
            var ta = TileTensor(a.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(M, K)))
            var tb = TileTensor(b.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(N, K)))
            var tc = TileTensor(dst.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(M, N)))
            linalg_matmul[target="gpu", transpose_b=True](tc, ta, tb, ctx)
        else:
            var a_imm: UnsafePointer[Scalar[d], ImmutAnyOrigin] = a.bitcast[Scalar[d]]()
            var b_imm: UnsafePointer[Scalar[d], ImmutAnyOrigin] = b.bitcast[Scalar[d]]()
            var ta = TileTensor(a_imm, MixedLayout(Coord(M, K), Coord(lda, lda1)))
            var tb = TileTensor(b_imm, MixedLayout(Coord(N, K), Coord(ldb, ldb1)))
            var tc = TileTensor(dst.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(M, N)))
            comptime BLOCK = 16
            comptime naive = matmul_kernel_naive[
                d, d, d, type_of(tc).LayoutType, type_of(ta).LayoutType, type_of(tb).LayoutType, BLOCK, True
            ]
            ctx.enqueue_function[naive](
                tc, ta, tb, M, N, K, grid_dim=(ceildiv(M, BLOCK), ceildiv(N, BLOCK)), block_dim=(BLOCK, BLOCK)
            )

    dispatch_dtype[body](dtype)


# ===-------------------------------------------------------------------===#
# Softmax
# ===-------------------------------------------------------------------===#


@export
def mograd_softmax(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
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
) abi("Mojo") raises:
    var out = TileTensor(dst.as_unsafe_any_origin(), row_major(Coord(rows, cols)))

    def input_fn[width: Int](coords: Coord) capturing -> SIMD[dtype, width]:
        return a.load[width=width](Int(coords[0].value()) * cols + Int(coords[1].value()))

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    nn_softmax[dtype, simd_width, 2, input_fn, "gpu"](Coord(rows, cols), out, axis=1, context=ctx)
    ctx.synchronize()


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
) abi("Mojo") raises:
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
# Transpose
# ===-------------------------------------------------------------------===#


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
) abi("Mojo") raises:
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


# ===-------------------------------------------------------------------===#
# Sum
# ===-------------------------------------------------------------------===#


@export
def mograd_sum(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_unary[sum](a, dst, n, rank, inner, sa, dtype, ctx)


@export
def mograd_sum_axis(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    outer: Int,
    reduce_size: Int,
    inner: Int,
    rank: Int,
    inner_sizes: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    contiguous: Bool,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        sum_axis[d](
            a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), outer, reduce_size, inner, rank, inner_sizes, sa, ctx
        )

    dispatch_dtype[body](dtype)


# ===-------------------------------------------------------------------===#
# Argmax
# ===-------------------------------------------------------------------===#


@export
def mograd_argmax(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        argmax[d](a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), rank, inner, sa, n, ctx)

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_argmax_axis(
    a: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    outer: Int,
    reduce_size: Int,
    inner: Int,
    rank: Int,
    inner_sizes: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    contiguous: Bool,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        if contiguous and inner == 1 and rank == 2:
            var a_imm: UnsafePointer[Scalar[d], ImmutAnyOrigin] = a.bitcast[Scalar[d]]()
            var inp = TileTensor(a_imm, row_major(Coord(outer, reduce_size)))
            var out = TileTensor(dst.bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(outer, 1)))
            argmax_gpu[d, d](ctx, inp, out)
        else:
            # TODO: use nn.argmaxmin when generalised to mid-axis (modular/.../nn/argmaxmin.mojo:59)
            argmax_axis[d](
                a.bitcast[Scalar[d]](), dst.bitcast[Scalar[d]](), outer, reduce_size, inner, rank, inner_sizes, sa, ctx
            )

    dispatch_dtype[body, float_only=True](dtype)


comptime CE_BLOCK = 256

# ===-------------------------------------------------------------------===#
# Cross Entropy
# ===-------------------------------------------------------------------===#


def cross_entropy_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    from std.math import exp, log
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tid = thread_idx.x

    var local_max = Scalar[dtype].MIN
    for i in range(tid, C, BLOCK_SIZE):
        var v = logits[row_offset + i]
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[w] > smem[0]:
                smem[0] = smem[w]
    barrier()
    var global_max = smem[0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += exp(logits[row_offset + i] - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        smem[0] = log(smem[0])
    barrier()
    var log_sum_exp = smem[0]

    var local_loss = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_loss += (logits[row_offset + i] - global_max - log_sum_exp) * labels[row_offset + i]
    local_loss = warp_sum(local_loss)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_loss
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        dst[row] = -smem[0] / Scalar[dtype](N)


def sum_rows_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    src: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
) where dtype.is_floating_point():
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var local = Scalar[dtype](0)
    for i in range(tid, N, BLOCK_SIZE):
        local += src[i]
    local = warp_sum(local)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        dst[0] = smem[0]


@export
def mograd_cross_entropy(
    logits: UnsafePointer[NoneType, MutAnyOrigin],
    labels: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
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
                ctx.enqueue_function[cross_entropy_kernel[d, CE_BLOCK]](
                    logits.bitcast[Scalar[d]](),
                    labels.bitcast[Scalar[d]](),
                    row_buf.unsafe_ptr(),
                    N,
                    C,
                    grid_dim=(N,),
                    block_dim=(CE_BLOCK,),
                )
                ctx.enqueue_function[sum_rows_kernel[d, CE_BLOCK]](
                    row_buf.unsafe_ptr(),
                    dst.bitcast[Scalar[d]](),
                    N,
                    grid_dim=(1,),
                    block_dim=(CE_BLOCK,),
                )
                ctx.synchronize()
                return
    raise Error("unsupported dtype")


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
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tid = thread_idx.x

    var local_max = Scalar[dtype].MIN
    for i in range(tid, C, BLOCK_SIZE):
        var v = logits[row_offset + i]
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[w] > smem[0]:
                smem[0] = smem[w]
    barrier()
    var global_max = smem[0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += math_exp(logits[row_offset + i] - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
    barrier()
    var sm_scale = Scalar[dtype](1) / smem[0]
    var d_by_nrows = grad[0] / Scalar[dtype](N)

    for i in range(tid, C, BLOCK_SIZE):
        dst[row_offset + i] = (
            math_exp(logits[row_offset + i] - global_max) * sm_scale - labels[row_offset + i]
        ) * d_by_nrows


@export
def mograd_cross_entropy_grad(
    logits: UnsafePointer[NoneType, MutAnyOrigin],
    labels: UnsafePointer[NoneType, MutAnyOrigin],
    grad: UnsafePointer[NoneType, MutAnyOrigin],
    shape_ptr: UnsafePointer[NoneType, MutAnyOrigin],
    dst: UnsafePointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.size):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime kd = T.node_dtype
        comptime if kd.is_floating_point():
            if dtype == kd:
                var p = shape_ptr.bitcast[Float32]()
                var N = Int(p[0])
                var C = Int(p[1])
                ctx.enqueue_function[cross_entropy_grad_kernel[kd, CE_BLOCK]](
                    grad.bitcast[Scalar[kd]](),
                    logits.bitcast[Scalar[kd]](),
                    labels.bitcast[Scalar[kd]](),
                    dst.bitcast[Scalar[kd]](),
                    N,
                    C,
                    grid_dim=(N,),
                    block_dim=(CE_BLOCK,),
                )
                return
    raise Error("unsupported dtype")
