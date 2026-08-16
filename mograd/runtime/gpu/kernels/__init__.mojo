from std.gpu import global_idx, thread_idx, block_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from std.memory import AddressSpace
from std.math import ceildiv
from std.memory import stack_allocation
from std.sys.info import has_apple_gpu_accelerator
from std.utils.index import IndexList

from layout import Coord, Idx, MixedLayout, TensorLayout, TileTensor, row_major
from layout import stack_allocation as tile_stack_allocation

from linalg.bmm import batched_matmul, elementwise_epilogue_type, naive_batched_matmul_kernel

from nn.argmaxmin_gpu import argmax_gpu

from mograd.buffer import AnyBuffer, BufferArm
from mograd.layout import Layout
from mograd.memory import scratch_take
from mograd.runtime.gpu.kernels.factory import *
from mograd.runtime.gpu.kernels.elementwise import *
from mograd.runtime.gpu.kernels.reduce import *
from mograd.runtime.gpu.kernels.softmax import *
from mograd.runtime.gpu.kernels.gather_scatter import *
from mograd.runtime.gpu.kernels.cross_entropy import *
from mograd.runtime.gpu.kernels.normalization import layer_norm_fwd, layer_norm_bwd
from mograd.runtime.gpu.kernels.attention import flash_attn_fwd, flash_attn_bwd
from mograd.runtime.gpu.kernels.dispatch import (
    dispatch_binary_contiguous,
    dispatch_binary_map,
    dispatch_binary_scalar_map,
    dispatch_dtype,
    dispatch_unary,
    dispatch_unary_map,
)
from mograd.runtime.gpu.kernels.strided import strided_copy_map

# ===-------------------------------------------------------------------===#
# Factory Kernels
# ===-------------------------------------------------------------------===#


@export
def mograd_randn(
    params: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    seed: UInt64,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var p = params.unsafe_bitcast[Float32]()
        randn[d](dst.unsafe_bitcast[Scalar[d]](), n, p[unsafe_offset=0], p[unsafe_offset=1], seed, ctx)

    dispatch_dtype[body](dtype)


@export
def mograd_uniform(
    params: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    seed: UInt64,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        comptime assert d.is_floating_point()
        uniform[d](params.unsafe_bitcast[Float32](), dst.unsafe_bitcast[Scalar[d]](), n, seed, ctx)

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_full(
    fill_val: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var val = Scalar[d](fill_val.unsafe_bitcast[Float32]()[unsafe_offset=0])
        full[d](val, dst.unsafe_bitcast[Scalar[d]](), n, ctx)

    dispatch_dtype[body](dtype)


@export
def mograd_one_hot(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    in_dtype: DType,
    out_dtype: DType,
    imm ld: Layout,
    imm la: Layout,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = ld.inner_sizes_buffer(ctx)
    var sd_buf = ld.strides_buffer(ctx)
    var sa_buf = la.strides_buffer(ctx)
    comptime for ki in range(AnyBuffer.BufVariant.Ts.length):
        comptime Ti = AnyBuffer.BufVariant.Ts[ki]
        comptime assert conforms_to(Ti, BufferArm)
        comptime src_d = Ti.node_dtype
        if in_dtype == src_d:
            comptime for ko in range(AnyBuffer.BufVariant.Ts.length):
                comptime To = AnyBuffer.BufVariant.Ts[ko]
                comptime assert conforms_to(To, BufferArm)
                comptime dst_d = To.node_dtype
                comptime if dst_d.is_integral():
                    if out_dtype == dst_d:
                        one_hot[src_d, dst_d](
                            a.unsafe_bitcast[Scalar[src_d]](),
                            dst.unsafe_bitcast[Scalar[dst_d]](),
                            ld.numel(),
                            ld.rank(),
                            inner_buf.unsafe_ptr(),
                            sd_buf.unsafe_ptr(),
                            sa_buf.unsafe_ptr(),
                            ctx,
                        )
                        return
    raise Error("unsupported dtype combination")


@export
def mograd_gather(
    src: Pointer[NoneType, ImmutAnyOrigin],
    indices: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var row_size = la.shape(la.rank() - 1)
        var n = lb.numel() * row_size
        var idx_inner_buf = lb.inner_sizes_buffer(ctx)
        var idx_strides_buf = lb.strides_buffer(ctx)

        gather[d](
            src.unsafe_bitcast[Scalar[d]](),
            indices.unsafe_bitcast[Scalar[DType.int64]](),
            dst.unsafe_bitcast[Scalar[d]](),
            n,
            row_size,
            la.stride(0),
            la.stride(1),
            lb.rank(),
            idx_inner_buf.unsafe_ptr(),
            idx_strides_buf.unsafe_ptr(),
            ctx,
        )

    dispatch_dtype[body](dtype)


@export
def mograd_scatter_add(
    indices: Pointer[NoneType, ImmutAnyOrigin],
    values: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var row_size = lb.shape(lb.rank() - 1)
        var n = lb.numel()
        var idx_inner_buf = la.inner_sizes_buffer(ctx)
        var idx_strides_buf = la.strides_buffer(ctx)
        var values_inner_buf = lb.inner_sizes_buffer(ctx)
        var values_strides_buf = lb.strides_buffer(ctx)

        comptime assert d.is_floating_point()
        scatter_add[d](
            indices.unsafe_bitcast[Scalar[DType.int64]](),
            values.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            n,
            row_size,
            la.rank(),
            idx_inner_buf.unsafe_ptr(),
            idx_strides_buf.unsafe_ptr(),
            lb.rank(),
            values_inner_buf.unsafe_ptr(),
            values_strides_buf.unsafe_ptr(),
            ctx,
        )

    comptime if has_apple_gpu_accelerator():
        if dtype != DType.float32:
            raise Error("scatter_add: only float32 is supported on Apple GPU")
        body[DType.float32]()
    else:
        dispatch_dtype[body, float_only=True](dtype)


# ===-------------------------------------------------------------------===#
# Unary Maps
# ===-------------------------------------------------------------------===#


@export
def mograd_neg(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary_map[neg_op](
        a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_log(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary_map[log_op, float_only=True](
        a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_exp(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary_map[exp_op, float_only=True](
        a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_sqrt(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary_map[sqrt_op, float_only=True](
        a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_relu(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary_map[relu_op](
        a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_slice_grad(
    upstream: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary[slice_grad](
        upstream, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_cast(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    in_dtype: DType,
    out_dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var strides_buf = layout.strides_buffer(ctx)
    comptime for ki in range(AnyBuffer.BufVariant.Ts.length):
        comptime Ti = AnyBuffer.BufVariant.Ts[ki]
        comptime assert conforms_to(Ti, BufferArm)
        comptime src_d = Ti.node_dtype
        if in_dtype == src_d:
            comptime for ko in range(AnyBuffer.BufVariant.Ts.length):
                comptime To = AnyBuffer.BufVariant.Ts[ko]
                comptime assert conforms_to(To, BufferArm)
                comptime dst_d = To.node_dtype
                if out_dtype == dst_d:
                    cast[src_d, dst_d](
                        a.unsafe_bitcast[Scalar[src_d]](),
                        dst.unsafe_bitcast[Scalar[dst_d]](),
                        layout.rank(),
                        inner_buf.unsafe_ptr(),
                        strides_buf.unsafe_ptr(),
                        layout.numel(),
                        ctx,
                    )
                    return
    raise Error("unsupported dtype combination")


@export
def mograd_contiguous(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary_map[identity_op](
        a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_strided_copy(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm ld: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var inner_buf = la.inner_sizes_buffer(ctx)
        var sa_buf = la.strides_buffer(ctx)
        var sd_buf = ld.strides_buffer(ctx)
        strided_copy_map[d](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            la.rank(),
            inner_buf.unsafe_ptr(),
            sa_buf.unsafe_ptr(),
            sd_buf.unsafe_ptr(),
            la.numel(),
            ctx,
        )

    dispatch_dtype[body](dtype)


# ===-------------------------------------------------------------------===#
# Transpose (last two dims)
# ===-------------------------------------------------------------------===#

comptime TRANSPOSE_TILE = 32


def transpose_last2_kernel[
    dtype: DType,
    InLayout: TensorLayout,
    OutLayout: TensorLayout,
](
    input: TileTensor[dtype, InLayout, ImmutAnyOrigin],
    output: TileTensor[dtype, OutLayout, MutAnyOrigin],
    M_arg: Int64,
    N_arg: Int64,
):
    var M = Int(M_arg)
    var N = Int(N_arg)
    comptime assert input.flat_rank == 3 and output.flat_rank == 3

    var b = block_idx.z
    var tx = thread_idx.x
    var ty = thread_idx.y
    var tile_m = block_idx.x * TRANSPOSE_TILE
    var tile_n = block_idx.y * TRANSPOSE_TILE

    var tile = tile_stack_allocation[dtype, address_space=AddressSpace.SHARED](
        row_major[TRANSPOSE_TILE, TRANSPOSE_TILE + 1]()
    )

    var m = tile_m + tx
    var n = tile_n + ty
    if m < M and n < N:
        tile[ty, tx] = input[b, m, n]
    barrier()

    var m2 = tile_m + ty
    var n2 = tile_n + tx
    if m2 < M and n2 < N:
        output[b, m2, n2] = tile[tx, ty]


@export
def mograd_transpose_last2(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var rank = layout.rank()
        var M = layout.shape(rank - 2)
        var N = layout.shape(rank - 1)
        var sm = layout.stride(rank - 2)
        var sn = layout.stride(rank - 1)
        var batch = layout.numel() // (M * N)

        var input = TileTensor(a.unsafe_bitcast[Scalar[d]](), MixedLayout(Coord(batch, M, N), Coord(M * N, sm, sn)))
        var output = TileTensor(dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, N)))

        comptime kernel = transpose_last2_kernel[d, type_of(input).LayoutType, type_of(output).LayoutType]
        ctx.enqueue_function[kernel](
            input,
            output,
            Int64(M),
            Int64(N),
            grid_dim=(ceildiv(M, TRANSPOSE_TILE), ceildiv(N, TRANSPOSE_TILE), batch),
            block_dim=(TRANSPOSE_TILE, TRANSPOSE_TILE),
        )

    dispatch_dtype[body](dtype)


# ===-------------------------------------------------------------------===#
# Binary Maps
# ===-------------------------------------------------------------------===#


@export
def mograd_add(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    dispatch_binary_contiguous[add](a, b, dst, n, dtype, ctx)


@export
def mograd_add_strided(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = la.inner_sizes_buffer(ctx)
    var sa_buf = la.strides_buffer(ctx)
    var sb_buf = lb.strides_buffer(ctx)
    dispatch_binary_map[add_op](
        a, b, dst, la.numel(), la.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), sb_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_mul(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = la.inner_sizes_buffer(ctx)
    var sa_buf = la.strides_buffer(ctx)
    var sb_buf = lb.strides_buffer(ctx)
    dispatch_binary_map[mul_op](
        a, b, dst, la.numel(), la.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), sb_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_div(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = la.inner_sizes_buffer(ctx)
    var sa_buf = la.strides_buffer(ctx)
    var sb_buf = lb.strides_buffer(ctx)
    dispatch_binary_map[div_op](
        a, b, dst, la.numel(), la.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), sb_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_eq(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = la.inner_sizes_buffer(ctx)
    var sa_buf = la.strides_buffer(ctx)
    var sb_buf = lb.strides_buffer(ctx)
    dispatch_binary_map[eq_op](
        a, b, dst, la.numel(), la.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), sb_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_relu_grad(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = la.inner_sizes_buffer(ctx)
    var sa_buf = la.strides_buffer(ctx)
    var sb_buf = lb.strides_buffer(ctx)
    dispatch_binary_map[relu_grad_op](
        a, b, dst, la.numel(), la.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), sb_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_scale(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_binary_scalar_map[mul_op](
        a, b, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


# ===-------------------------------------------------------------------===#
# Matmul
# ===-------------------------------------------------------------------===#
# TODO: Support vendors https://github.com/modular/modular/issues/6672

comptime MATMUL_KN_BUCKETS: List[Int] = [128, 256, 512, 1024, 2048, 4096]


@always_inline
def _try_static_square_batched_matmul[
    d: DType, transpose_b: Bool
](
    a: Pointer[Scalar[d], ImmutAnyOrigin],
    b: Pointer[Scalar[d], ImmutAnyOrigin],
    dst: Pointer[Scalar[d], MutAnyOrigin],
    batch: Int,
    M: Int,
    K: Int,
    N: Int,
    ctx: DeviceContext,
) raises -> Bool:
    """Tries the comptime-static-(K, N) fast path for K == N at a handful
    of typical sizes.
    """
    if K != N:
        return False
    comptime for i in range(len(MATMUL_KN_BUCKETS)):
        comptime S = MATMUL_KN_BUCKETS[i]
        if K == S:
            var ta = TileTensor(a, row_major(Coord(batch, M, Idx[S])))
            var tb = TileTensor(b, row_major(Coord(batch, Idx[S], Idx[S])))
            var tc = TileTensor(dst, row_major(Coord(batch, M, Idx[S])))
            batched_matmul[target="gpu", transpose_b=transpose_b](tc, ta, tb, context=ctx)
            return True
    return False


@export
def mograd_matmul(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    N: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var rank = la.rank()
        # Defensive: op.mojo's matmul() should have already materialised
        # any non-collapsible operand via .contiguous() before it ever
        # reaches this kernel.
        if not (la.batch_dims_collapsible() and lb.batch_dims_collapsible()):
            raise Error("matmul: received non-collapsible batch dims, expected op.mojo to materialise these first")

        var M = la.shape(rank - 2)
        var K = la.shape(rank - 1)
        var lda0 = la.stride(rank - 2)
        var lda1 = la.stride(rank - 1)
        var ldb0 = lb.stride(rank - 2)
        var ldb1 = lb.stride(rank - 1)
        var batch = la.numel() // (M * K)
        var batch_stride_a = la.stride(rank - 3) if rank >= 3 else M * K
        var batch_stride_b = lb.stride(rank - 3) if rank >= 3 else K * N

        if lda0 == K and lda1 == 1 and ldb0 == N and ldb1 == 1:
            comptime if d.is_floating_point():
                if _try_static_square_batched_matmul[d, False](
                    a.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(),
                    b.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(),
                    dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(),
                    batch,
                    M,
                    K,
                    N,
                    ctx,
                ):
                    return
            var ta = TileTensor(a.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, K)))
            var tb = TileTensor(b.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, K, N)))
            var tc = TileTensor(dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, N)))
            batched_matmul[target="gpu"](tc, ta, tb, context=ctx)
        else:
            var ta = TileTensor(
                a.unsafe_bitcast[Scalar[d]](), MixedLayout(Coord(batch, M, K), Coord(batch_stride_a, lda0, lda1))
            )
            var tb = TileTensor(
                b.unsafe_bitcast[Scalar[d]](), MixedLayout(Coord(batch, K, N), Coord(batch_stride_b, ldb0, ldb1))
            )
            var tc = TileTensor(dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, N)))
            comptime BLOCK = 16
            comptime naive = naive_batched_matmul_kernel[
                3, d, d, d, type_of(tc).LayoutType, type_of(ta).LayoutType, type_of(tb).LayoutType
            ]
            ctx.enqueue_function[naive](
                tc,
                ta,
                tb,
                IndexList[3](batch, M, N),
                grid_dim=(ceildiv(N, BLOCK), ceildiv(M, BLOCK), batch),
                block_dim=(BLOCK, BLOCK),
            )

    dispatch_dtype[body](dtype)


@export
def mograd_matmul_bt(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    N: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var rank = la.rank()
        # Defensive: op.mojo's matmul() should have already materialised
        # any non-collapsible operand via .contiguous() before it ever
        # reaches this kernel.
        if not (la.batch_dims_collapsible() and lb.batch_dims_collapsible()):
            raise Error("matmul: received non-collapsible batch dims, expected op.mojo to materialise these first")

        var M = la.shape(rank - 2)
        var K = la.shape(rank - 1)
        var lda0 = la.stride(rank - 2)
        var lda1 = la.stride(rank - 1)
        var ldb0 = lb.stride(rank - 2)
        var ldb1 = lb.stride(rank - 1)
        var batch = la.numel() // (M * K)
        var batch_stride_a = la.stride(rank - 3) if rank >= 3 else M * K
        var batch_stride_b = lb.stride(rank - 3) if rank >= 3 else N * K

        if lda0 == K and lda1 == 1 and ldb0 == K and ldb1 == 1:
            comptime if d.is_floating_point():
                if _try_static_square_batched_matmul[d, True](
                    a.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(),
                    b.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(),
                    dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(),
                    batch,
                    M,
                    K,
                    N,
                    ctx,
                ):
                    return
            var ta = TileTensor(a.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, K)))
            var tb = TileTensor(b.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, N, K)))
            var tc = TileTensor(dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, N)))
            batched_matmul[target="gpu", transpose_b=True](tc, ta, tb, context=ctx)
        else:
            var ta = TileTensor(
                a.unsafe_bitcast[Scalar[d]](), MixedLayout(Coord(batch, M, K), Coord(batch_stride_a, lda0, lda1))
            )
            var tb = TileTensor(
                b.unsafe_bitcast[Scalar[d]](), MixedLayout(Coord(batch, N, K), Coord(batch_stride_b, ldb0, ldb1))
            )
            var tc = TileTensor(dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, N)))
            comptime BLOCK = 16
            comptime naive = naive_batched_matmul_kernel[
                3, d, d, d, type_of(tc).LayoutType, type_of(ta).LayoutType, type_of(tb).LayoutType, transpose_b=True
            ]
            ctx.enqueue_function[naive](
                tc,
                ta,
                tb,
                IndexList[3](batch, M, N),
                grid_dim=(ceildiv(N, BLOCK), ceildiv(M, BLOCK), batch),
                block_dim=(BLOCK, BLOCK),
            )

    dispatch_dtype[body](dtype)


@export
def mograd_matmul_bias_bt(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    bias: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    N: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    # Fused `(a @ b.T) + bias` (bias broadcast along the last axis).

    @always_inline
    def body[d: DType]() capturing raises:
        var rank = la.rank()
        if not (la.batch_dims_collapsible() and lb.batch_dims_collapsible()):
            raise Error("matmul_bias: received non-collapsible batch dims, expected op.mojo to materialise these first")

        var M = la.shape(rank - 2)
        var K = la.shape(rank - 1)
        var lda0 = la.stride(rank - 2)
        var lda1 = la.stride(rank - 1)
        var ldb0 = lb.stride(rank - 2)
        var ldb1 = lb.stride(rank - 1)
        var batch = la.numel() // (M * K)
        var batch_stride_a = la.stride(rank - 3) if rank >= 3 else M * K
        var batch_stride_b = lb.stride(rank - 3) if rank >= 3 else N * K

        var bias_d = bias.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin()
        var dst_d = dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin()

        @always_inline
        @__copy_capture(bias_d, dst_d, M, N)
        def epilogue[
            c_type: DType, width: SIMDLength, erank: Int, *, alignment: Int = 1
        ](coord: IndexList[erank], val: SIMD[c_type, width]) capturing -> None:
            var n = coord[erank - 1]
            var flat = coord[0] * M * N + coord[1] * N + n
            var result = val._refine[d]() + bias_d.unsafe_load[width=width, alignment=1](n)
            dst_d.unsafe_store[width=width, alignment=1](flat, result)

        if lda0 == K and lda1 == 1 and ldb0 == K and ldb1 == 1:
            var ta = TileTensor(a.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, M, K)))
            var tb = TileTensor(b.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(batch, N, K)))
            var tc = TileTensor(dst_d, row_major(Coord(batch, M, N)))
            batched_matmul[
                target="gpu",
                transpose_b=True,
                elementwise_epilogue_fn=Optional[elementwise_epilogue_type](epilogue),
            ](tc, ta, tb, context=ctx)
        else:
            # Strided fallback
            var ta = TileTensor(
                a.unsafe_bitcast[Scalar[d]](), MixedLayout(Coord(batch, M, K), Coord(batch_stride_a, lda0, lda1))
            )
            var tb = TileTensor(
                b.unsafe_bitcast[Scalar[d]](), MixedLayout(Coord(batch, N, K), Coord(batch_stride_b, ldb0, ldb1))
            )
            var tc = TileTensor(dst_d, row_major(Coord(batch, M, N)))
            comptime BLOCK = 16
            comptime naive = naive_batched_matmul_kernel[
                3,
                d,
                d,
                d,
                type_of(tc).LayoutType,
                type_of(ta).LayoutType,
                type_of(tb).LayoutType,
                transpose_b=True,
                elementwise_lambda_fn=Optional[elementwise_epilogue_type](epilogue),
            ]
            ctx.enqueue_function[naive](
                tc,
                ta,
                tb,
                IndexList[3](batch, M, N),
                grid_dim=(ceildiv(N, BLOCK), ceildiv(M, BLOCK), batch),
                block_dim=(BLOCK, BLOCK),
            )

    dispatch_dtype[body](dtype)


# ===-------------------------------------------------------------------===#
# Flash Attention
# ===-------------------------------------------------------------------===#


@export
def mograd_flash_attn_fwd(
    q: Pointer[NoneType, ImmutAnyOrigin],
    k: Pointer[NoneType, ImmutAnyOrigin],
    v: Pointer[NoneType, ImmutAnyOrigin],
    mask: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    lse: Pointer[NoneType, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    causal: Int,
    has_bias: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        if causal != 0:
            # The causal kernels (fwd and bwd) never read the mask, so a causal
            # + bias combination would silently drop the bias. Fusion upholds
            # has_bias == not is_causal (rewrites.mojo). Reject any violation.
            if has_bias != 0:
                raise Error("flash_attn_fwd: causal attention with additive bias is not supported")
            else:
                flash_attn_fwd[d, True, False](
                    q.unsafe_bitcast[Scalar[d]](),
                    k.unsafe_bitcast[Scalar[d]](),
                    v.unsafe_bitcast[Scalar[d]](),
                    mask.unsafe_bitcast[Scalar[d]](),
                    dst.unsafe_bitcast[Scalar[d]](),
                    lse.unsafe_bitcast[Float32](),
                    B,
                    S,
                    H,
                    D,
                    scale,
                    ctx,
                )
        else:
            if has_bias != 0:
                flash_attn_fwd[d, False, True](
                    q.unsafe_bitcast[Scalar[d]](),
                    k.unsafe_bitcast[Scalar[d]](),
                    v.unsafe_bitcast[Scalar[d]](),
                    mask.unsafe_bitcast[Scalar[d]](),
                    dst.unsafe_bitcast[Scalar[d]](),
                    lse.unsafe_bitcast[Float32](),
                    B,
                    S,
                    H,
                    D,
                    scale,
                    ctx,
                )
            else:
                flash_attn_fwd[d, False, False](
                    q.unsafe_bitcast[Scalar[d]](),
                    k.unsafe_bitcast[Scalar[d]](),
                    v.unsafe_bitcast[Scalar[d]](),
                    mask.unsafe_bitcast[Scalar[d]](),
                    dst.unsafe_bitcast[Scalar[d]](),
                    lse.unsafe_bitcast[Float32](),
                    B,
                    S,
                    H,
                    D,
                    scale,
                    ctx,
                )

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_flash_attn_bwd(
    dy: Pointer[NoneType, ImmutAnyOrigin],
    o: Pointer[NoneType, ImmutAnyOrigin],
    q: Pointer[NoneType, ImmutAnyOrigin],
    k: Pointer[NoneType, ImmutAnyOrigin],
    v: Pointer[NoneType, ImmutAnyOrigin],
    mask: Pointer[NoneType, ImmutAnyOrigin],
    lse: Pointer[NoneType, ImmutAnyOrigin],
    dq: Pointer[NoneType, MutAnyOrigin],
    dk: Pointer[NoneType, MutAnyOrigin],
    dv: Pointer[NoneType, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    causal: Int,
    has_bias: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        if causal != 0:
            flash_attn_bwd[d, True, False](
                dy.unsafe_bitcast[Scalar[d]](),
                o.unsafe_bitcast[Scalar[d]](),
                q.unsafe_bitcast[Scalar[d]](),
                k.unsafe_bitcast[Scalar[d]](),
                v.unsafe_bitcast[Scalar[d]](),
                mask.unsafe_bitcast[Scalar[d]](),
                lse.unsafe_bitcast[Float32](),
                dq.unsafe_bitcast[Scalar[d]](),
                dk.unsafe_bitcast[Scalar[d]](),
                dv.unsafe_bitcast[Scalar[d]](),
                B,
                S,
                H,
                D,
                scale,
                ctx,
            )
        elif has_bias != 0:
            flash_attn_bwd[d, False, True](
                dy.unsafe_bitcast[Scalar[d]](),
                o.unsafe_bitcast[Scalar[d]](),
                q.unsafe_bitcast[Scalar[d]](),
                k.unsafe_bitcast[Scalar[d]](),
                v.unsafe_bitcast[Scalar[d]](),
                mask.unsafe_bitcast[Scalar[d]](),
                lse.unsafe_bitcast[Float32](),
                dq.unsafe_bitcast[Scalar[d]](),
                dk.unsafe_bitcast[Scalar[d]](),
                dv.unsafe_bitcast[Scalar[d]](),
                B,
                S,
                H,
                D,
                scale,
                ctx,
            )
        else:
            flash_attn_bwd[d, False, False](
                dy.unsafe_bitcast[Scalar[d]](),
                o.unsafe_bitcast[Scalar[d]](),
                q.unsafe_bitcast[Scalar[d]](),
                k.unsafe_bitcast[Scalar[d]](),
                v.unsafe_bitcast[Scalar[d]](),
                mask.unsafe_bitcast[Scalar[d]](),
                lse.unsafe_bitcast[Float32](),
                dq.unsafe_bitcast[Scalar[d]](),
                dk.unsafe_bitcast[Scalar[d]](),
                dv.unsafe_bitcast[Scalar[d]](),
                B,
                S,
                H,
                D,
                scale,
                ctx,
            )

    dispatch_dtype[body, float_only=True](dtype)


# ===-------------------------------------------------------------------===#
# Softmax
# ===-------------------------------------------------------------------===#


@export
def mograd_softmax(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var rows = layout.numel() // layout.shape(layout.rank() - 1)
        var cols = layout.shape(layout.rank() - 1)

        softmax[d](a.unsafe_bitcast[Scalar[d]](), dst.unsafe_bitcast[Scalar[d]](), rows, cols, ctx)

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_softmax_strided(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var rows = layout.numel() // layout.shape(layout.rank() - 1)
        var cols = layout.shape(layout.rank() - 1)
        var inner_buf = layout.inner_sizes_buffer(ctx)
        var sa_buf = layout.strides_buffer(ctx)

        softmax_strided[d](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            rows,
            cols,
            layout.rank(),
            inner_buf.unsafe_ptr(),
            sa_buf.unsafe_ptr(),
            ctx,
        )

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_softmax_grad(
    a: Pointer[NoneType, ImmutAnyOrigin],
    b: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    # TODO: both a and b are always contiguous and same-shaped by
    # construction (softmax_grad forces upstream.contiguous()).
    # This should handle arbitrary layout.
    @always_inline
    def body[d: DType]() capturing raises:
        var N = la.numel() // la.shape(la.rank() - 1)
        var size = la.shape(la.rank() - 1)

        comptime BLOCK_SIZE = 32
        ctx.enqueue_function[softmax_grad_kernel[d, BLOCK_SIZE]](
            a.unsafe_bitcast[Scalar[d]](),
            b.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            Int64(N),
            Int64(size),
            grid_dim=(N,),
            block_dim=(BLOCK_SIZE,),
        )

    dispatch_dtype[body, float_only=True](dtype)


# ===-------------------------------------------------------------------===#
# Transpose
# ===-------------------------------------------------------------------===#


def transpose_kernel[
    dtype: DType, BLOCK_SIZE: Int
](src: Pointer[Scalar[dtype], ImmutAnyOrigin], dst: Pointer[Scalar[dtype], MutAnyOrigin], M_arg: Int64, N_arg: Int64,):
    var M = Int(M_arg)
    var N = Int(N_arg)
    var shmem = stack_allocation[BLOCK_SIZE * (BLOCK_SIZE + 1), dtype, address_space=AddressSpace.SHARED]()
    var x = block_idx.x * BLOCK_SIZE + thread_idx.x
    var y = block_idx.y * BLOCK_SIZE + thread_idx.y
    if x < N and y < M:
        shmem[unsafe_offset=thread_idx.y * (BLOCK_SIZE + 1) + thread_idx.x] = src[unsafe_offset=y * N + x]
    barrier()
    var x_out = block_idx.y * BLOCK_SIZE + thread_idx.x
    var y_out = block_idx.x * BLOCK_SIZE + thread_idx.y
    if x_out < M and y_out < N:
        dst[unsafe_offset=y_out * M + x_out] = shmem[unsafe_offset=thread_idx.x * (BLOCK_SIZE + 1) + thread_idx.y]


@export
def mograd_transpose(
    a: Pointer[NoneType, ImmutAnyOrigin],
    shape_ptr: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    n: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    comptime for k in range(AnyBuffer.BufVariant.Ts.length):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if dtype == d:
            var p = shape_ptr.unsafe_bitcast[Float32]()
            var M = Int(p[unsafe_offset=0])
            var N = Int(p[unsafe_offset=1])
            comptime TILE = 32
            ctx.enqueue_function[transpose_kernel[d, TILE]](
                a.unsafe_bitcast[Scalar[d]](),
                dst.unsafe_bitcast[Scalar[d]](),
                Int64(M),
                Int64(N),
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
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary[sum](a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx)


@export
def mograd_sum_axis(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    axis: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var outer, reduce_size, inner = layout.reduce_dims(axis)
        var inner_buf = layout.inner_sizes_buffer(ctx)
        var sa_buf = layout.strides_buffer(ctx)
        sum_axis[d](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            outer,
            reduce_size,
            inner,
            layout.rank(),
            inner_buf.unsafe_ptr(),
            sa_buf.unsafe_ptr(),
            ctx,
        )

    dispatch_dtype[body](dtype)


@export
def mograd_mean(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var inner_buf = layout.inner_sizes_buffer(ctx)
        var sa_buf = layout.strides_buffer(ctx)
        mean[d](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            layout.rank(),
            inner_buf.unsafe_ptr(),
            sa_buf.unsafe_ptr(),
            layout.numel(),
            Scalar[d](1) / Scalar[d](layout.numel()),
            ctx,
        )

    dispatch_dtype[body](dtype)


@export
def mograd_mean_axis(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    axis: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var outer, reduce_size, inner = layout.reduce_dims(axis)
        var inner_buf = layout.inner_sizes_buffer(ctx)
        var sa_buf = layout.strides_buffer(ctx)
        mean_axis[d](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            outer,
            reduce_size,
            inner,
            layout.rank(),
            inner_buf.unsafe_ptr(),
            sa_buf.unsafe_ptr(),
            Scalar[d](1) / Scalar[d](reduce_size),
            ctx,
        )

    dispatch_dtype[body](dtype)


# ===-------------------------------------------------------------------===#
# Argmax
# ===-------------------------------------------------------------------===#


@export
def mograd_argmax(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    var inner_buf = layout.inner_sizes_buffer(ctx)
    var sa_buf = layout.strides_buffer(ctx)
    dispatch_unary[argmax, float_only=True](
        a, dst, layout.numel(), layout.rank(), inner_buf.unsafe_ptr(), sa_buf.unsafe_ptr(), dtype, ctx
    )


@export
def mograd_argmax_axis(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    axis: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var outer, reduce_size, inner = layout.reduce_dims(axis)
        var rank = layout.rank()
        if layout.is_contiguous() and inner == 1 and rank == 2:
            var inp = TileTensor(a.unsafe_bitcast[Scalar[d]](), row_major(Coord(outer, reduce_size)))
            var out = TileTensor(dst.unsafe_bitcast[Scalar[d]]().as_unsafe_any_origin(), row_major(Coord(outer, 1)))
            argmax_gpu[d, d](ctx, inp, out)
        else:
            # TODO: use nn.argmaxmin when generalised to mid-axis (modular/.../nn/argmaxmin.mojo:59)
            var inner_buf = layout.inner_sizes_buffer(ctx)
            var sa_buf = layout.strides_buffer(ctx)
            argmax_axis[d](
                a.unsafe_bitcast[Scalar[d]](),
                dst.unsafe_bitcast[Scalar[d]](),
                outer,
                reduce_size,
                inner,
                rank,
                inner_buf.unsafe_ptr(),
                sa_buf.unsafe_ptr(),
                ctx,
            )

    dispatch_dtype[body, float_only=True](dtype)


# ===-------------------------------------------------------------------===#
# LayerNorm
# ===-------------------------------------------------------------------===#


@export
def mograd_layer_norm_fwd(
    x: Pointer[NoneType, ImmutAnyOrigin],
    gamma: Pointer[NoneType, ImmutAnyOrigin],
    beta: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Float32,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        layer_norm_fwd[d](
            x.unsafe_bitcast[Scalar[d]](),
            gamma.unsafe_bitcast[Scalar[d]](),
            beta.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            rows,
            cols,
            Scalar[d](eps),
            ctx,
        )

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_layer_norm_bwd(
    dy: Pointer[NoneType, ImmutAnyOrigin],
    x: Pointer[NoneType, ImmutAnyOrigin],
    gamma: Pointer[NoneType, ImmutAnyOrigin],
    dx: Pointer[NoneType, MutAnyOrigin],
    dgamma: Pointer[NoneType, MutAnyOrigin],
    dbeta: Pointer[NoneType, MutAnyOrigin],
    rows: Int,
    cols: Int,
    eps: Float32,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        layer_norm_bwd[d](
            dy.unsafe_bitcast[Scalar[d]](),
            x.unsafe_bitcast[Scalar[d]](),
            gamma.unsafe_bitcast[Scalar[d]](),
            dx.unsafe_bitcast[Scalar[d]](),
            dgamma.unsafe_bitcast[Scalar[d]](),
            dbeta.unsafe_bitcast[Scalar[d]](),
            rows,
            cols,
            Scalar[d](eps),
            ctx,
        )

    dispatch_dtype[body, float_only=True](dtype)


# ===-------------------------------------------------------------------===#
# Cross Entropy
# ===-------------------------------------------------------------------===#


@export
def mograd_cross_entropy(
    logits: Pointer[NoneType, ImmutAnyOrigin],
    labels: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var N = la.numel() // la.shape(la.rank() - 1)
        var C = la.shape(la.rank() - 1)
        var row_buf = scratch_take[d](ctx, N)
        comptime assert d.is_floating_point()
        ctx.enqueue_function[cross_entropy_kernel[d, CE_BLOCK]](
            logits.unsafe_bitcast[Scalar[d]](),
            labels.unsafe_bitcast[Scalar[d]](),
            row_buf.unsafe_ptr().as_unsafe_any_origin(),
            Int64(N),
            Int64(C),
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )
        ctx.enqueue_function[sum_rows_kernel[d, CE_BLOCK]](
            row_buf.unsafe_ptr().as_unsafe_any_origin(),
            dst.unsafe_bitcast[Scalar[d]](),
            Int64(N),
            grid_dim=(1,),
            block_dim=(CE_BLOCK,),
        )
        ctx.synchronize()

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_cross_entropy_strided(
    logits: Pointer[NoneType, ImmutAnyOrigin],
    labels: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var N = la.numel() // la.shape(la.rank() - 1)
        var C = la.shape(la.rank() - 1)
        var la_inner_buf = la.inner_sizes_buffer(ctx)
        var la_sa_buf = la.strides_buffer(ctx)
        var lb_inner_buf = lb.inner_sizes_buffer(ctx)
        var lb_sa_buf = lb.strides_buffer(ctx)
        var row_buf = scratch_take[d](ctx, N)
        comptime assert d.is_floating_point()
        ctx.enqueue_function[cross_entropy_kernel_strided[d, CE_BLOCK]](
            logits.unsafe_bitcast[Scalar[d]](),
            labels.unsafe_bitcast[Scalar[d]](),
            row_buf.unsafe_ptr().as_unsafe_any_origin(),
            Int64(N),
            Int64(C),
            Int64(la.rank()),
            la_inner_buf.unsafe_ptr(),
            la_sa_buf.unsafe_ptr(),
            lb_inner_buf.unsafe_ptr(),
            lb_sa_buf.unsafe_ptr(),
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )
        ctx.enqueue_function[sum_rows_kernel[d, CE_BLOCK]](
            row_buf.unsafe_ptr().as_unsafe_any_origin(),
            dst.unsafe_bitcast[Scalar[d]](),
            Int64(N),
            grid_dim=(1,),
            block_dim=(CE_BLOCK,),
        )
        ctx.synchronize()

    dispatch_dtype[body, float_only=True](dtype)


# ===-------------------------------------------------------------------===#
# Cross Entropy grad
# ===-------------------------------------------------------------------===#


@export
def mograd_cross_entropy_grad(
    logits: Pointer[NoneType, ImmutAnyOrigin],
    labels: Pointer[NoneType, ImmutAnyOrigin],
    grad: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var N = la.numel() // la.shape(la.rank() - 1)
        var C = la.shape(la.rank() - 1)
        comptime assert d.is_floating_point()
        ctx.enqueue_function[cross_entropy_grad_kernel[d, CE_BLOCK]](
            grad.unsafe_bitcast[Scalar[d]](),
            logits.unsafe_bitcast[Scalar[d]](),
            labels.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            Int64(N),
            Int64(C),
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_cross_entropy_grad_strided(
    logits: Pointer[NoneType, ImmutAnyOrigin],
    labels: Pointer[NoneType, ImmutAnyOrigin],
    grad: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm la: Layout,
    imm lb: Layout,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var N = la.numel() // la.shape(la.rank() - 1)
        var C = la.shape(la.rank() - 1)
        var la_inner_buf = la.inner_sizes_buffer(ctx)
        var la_sa_buf = la.strides_buffer(ctx)
        var lb_inner_buf = lb.inner_sizes_buffer(ctx)
        var lb_sa_buf = lb.strides_buffer(ctx)
        comptime assert d.is_floating_point()
        ctx.enqueue_function[cross_entropy_grad_kernel_strided[d, CE_BLOCK]](
            grad.unsafe_bitcast[Scalar[d]](),
            logits.unsafe_bitcast[Scalar[d]](),
            labels.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            Int64(N),
            Int64(C),
            Int64(la.rank()),
            la_inner_buf.unsafe_ptr(),
            la_sa_buf.unsafe_ptr(),
            lb_inner_buf.unsafe_ptr(),
            lb_sa_buf.unsafe_ptr(),
            grid_dim=(N,),
            block_dim=(CE_BLOCK,),
        )

    dispatch_dtype[body, float_only=True](dtype)


@export
def mograd_triu(
    a: Pointer[NoneType, ImmutAnyOrigin],
    dst: Pointer[NoneType, MutAnyOrigin],
    imm layout: Layout,
    diagonal: Int,
    dtype: DType,
    ctx: DeviceContext,
) abi("Mojo") raises:
    @always_inline
    def body[d: DType]() capturing raises:
        var rank = layout.rank()

        if rank < 2:
            raise Error("triu requires rank >= 2")

        var rows = layout.shape(rank - 2)
        var cols = layout.shape(rank - 1)
        var inner_buf = layout.inner_sizes_buffer(ctx)
        var sa_buf = layout.strides_buffer(ctx)

        triu_impl[d](
            a.unsafe_bitcast[Scalar[d]](),
            dst.unsafe_bitcast[Scalar[d]](),
            layout.numel(),
            rows,
            cols,
            rank,
            inner_buf.unsafe_ptr().as_unsafe_any_origin(),
            sa_buf.unsafe_ptr().as_unsafe_any_origin(),
            Int64(diagonal),
            ctx,
        )

    dispatch_dtype[body](dtype)
