from std.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle
from std.os.env import getenv
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of

from internal_utils import CacheBustingBuffer

from mograd.runtime import BinaryElementWise, BinaryScalarElementWiseStrided
from mograd.runtime.gpu.kernels.utils import UnaryStrided, BinaryStrided


# ===-------------------------------------------------------------------===#
# Unary strided (neg, log, exp, relu, contiguous)
# ===-------------------------------------------------------------------===#


@no_inline
def bench_unary[dtype: DType, N: Int](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 2 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = lib.get_function[UnaryStrided](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    var inner = ctx.enqueue_create_buffer[DType.int64](1)
    var sa = ctx.enqueue_create_buffer[DType.int64](1)
    var one: List[Int64] = [1]
    ctx.enqueue_copy(inner, one)
    ctx.enqueue_copy(sa, one)

    func(
        a.offset_ptr(0).bitcast[NoneType](),
        dst.offset_ptr(0).bitcast[NoneType](),
        N,
        1,
        inner.unsafe_ptr().bitcast[Int64](),
        sa.unsafe_ptr().bitcast[Int64](),
        dtype,
        ctx,
    )
    ctx.synchronize()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            func(
                a.offset_ptr(iteration).bitcast[NoneType](),
                dst.offset_ptr(iteration).bitcast[NoneType](),
                N,
                1,
                inner.unsafe_ptr().bitcast[Int64](),
                sa.unsafe_ptr().bitcast[Int64](),
                dtype,
                dc,
            )

        bench.iter_custom[kernel_launch](ctx)
        keep(dst.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId(symbol, input_id=String(dtype_str, "/", N // (1024 * 1024), "M")),
        [
            ThroughputMeasure(BenchMetric.elements, N),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )


# ===-------------------------------------------------------------------===#
# Binary contiguous (add fast path)
# ===-------------------------------------------------------------------===#


@no_inline
def bench_binary[dtype: DType, N: Int](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 3 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = lib.get_function[BinaryElementWise](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var b = CacheBustingBuffer[dtype](N, 1, ctx)
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    func(
        a.offset_ptr(0).bitcast[NoneType](),
        b.offset_ptr(0).bitcast[NoneType](),
        dst.offset_ptr(0).bitcast[NoneType](),
        N,
        dtype,
        ctx,
    )
    ctx.synchronize()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            func(
                a.offset_ptr(iteration).bitcast[NoneType](),
                b.offset_ptr(iteration).bitcast[NoneType](),
                dst.offset_ptr(iteration).bitcast[NoneType](),
                N,
                dtype,
                dc,
            )

        bench.iter_custom[kernel_launch](ctx)
        keep(dst.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId(symbol, input_id=String(dtype_str, "/", N // (1024 * 1024), "M")),
        [
            ThroughputMeasure(BenchMetric.elements, N),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )


# ===-------------------------------------------------------------------===#
# Binary strided (add_strided, mul, div, eq, relu_grad)
# NOTE: run with contiguous strides (rank=1, stride=[1]) to measure overhead
# vs the fast path above.
# ===-------------------------------------------------------------------===#


@no_inline
def bench_binary_strided[
    dtype: DType, N: Int
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 3 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = lib.get_function[BinaryStrided](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var b = CacheBustingBuffer[dtype](N, 1, ctx)
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    var inner = ctx.enqueue_create_buffer[DType.int64](1)
    var sa = ctx.enqueue_create_buffer[DType.int64](1)
    var sb = ctx.enqueue_create_buffer[DType.int64](1)
    var one: List[Int64] = [1]
    ctx.enqueue_copy(inner, one)
    ctx.enqueue_copy(sa, one)
    ctx.enqueue_copy(sb, one)

    func(
        a.offset_ptr(0).bitcast[NoneType](),
        b.offset_ptr(0).bitcast[NoneType](),
        dst.offset_ptr(0).bitcast[NoneType](),
        N,
        1,
        inner.unsafe_ptr().bitcast[Int64](),
        sa.unsafe_ptr().bitcast[Int64](),
        sb.unsafe_ptr().bitcast[Int64](),
        dtype,
        ctx,
    )
    ctx.synchronize()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            func(
                a.offset_ptr(iteration).bitcast[NoneType](),
                b.offset_ptr(iteration).bitcast[NoneType](),
                dst.offset_ptr(iteration).bitcast[NoneType](),
                N,
                1,
                inner.unsafe_ptr().bitcast[Int64](),
                sa.unsafe_ptr().bitcast[Int64](),
                sb.unsafe_ptr().bitcast[Int64](),
                dtype,
                dc,
            )

        bench.iter_custom[kernel_launch](ctx)
        keep(dst.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId(symbol, input_id=String(dtype_str, "/", N // (1024 * 1024), "M")),
        [
            ThroughputMeasure(BenchMetric.elements, N),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )


# ===-------------------------------------------------------------------===#
# Binary scalar strided (scale)
# ===-------------------------------------------------------------------===#


@no_inline
def bench_binary_scalar_strided[
    dtype: DType, N: Int
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 2 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = lib.get_function[BinaryScalarElementWiseStrided](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var scalar = alloc[Float32](1)
    scalar[0] = 2.0
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    var inner = ctx.enqueue_create_buffer[DType.int64](1)
    var sa = ctx.enqueue_create_buffer[DType.int64](1)
    var one: List[Int64] = [1]
    ctx.enqueue_copy(inner, one)
    ctx.enqueue_copy(sa, one)

    func(
        a.offset_ptr(0).bitcast[NoneType](),
        scalar.bitcast[NoneType](),
        dst.offset_ptr(0).bitcast[NoneType](),
        N,
        1,
        inner.unsafe_ptr().bitcast[Int64](),
        sa.unsafe_ptr().bitcast[Int64](),
        dtype,
        ctx,
    )
    ctx.synchronize()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            func(
                a.offset_ptr(iteration).bitcast[NoneType](),
                scalar.bitcast[NoneType](),
                dst.offset_ptr(iteration).bitcast[NoneType](),
                N,
                1,
                inner.unsafe_ptr().bitcast[Int64](),
                sa.unsafe_ptr().bitcast[Int64](),
                dtype,
                dc,
            )

        bench.iter_custom[kernel_launch](ctx)
        keep(dst.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId(symbol, input_id=String(dtype_str, "/", N // (1024 * 1024), "M")),
        [
            ThroughputMeasure(BenchMetric.elements, N),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )
    scalar.free()


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=5, max_runtime_secs=10))
    var lib = OwnedDLHandle(getenv("MOGRAD_SO"))

    comptime dtypes = (DType.float32,)
    comptime sizes = (1024 * 1024, 1024 * 1024 * 16, 1024 * 1024 * 64)

    comptime unary_syms = ("mograd_neg", "mograd_exp", "mograd_log", "mograd_relu", "mograd_contiguous")
    comptime binary_strided_syms = ("mograd_add_strided", "mograd_mul", "mograd_div", "mograd_eq", "mograd_relu_grad")

    with DeviceContext() as ctx:
        comptime for di in range(len(dtypes)):
            comptime for si in range(len(sizes)):
                comptime dtype = dtypes[di]
                comptime N = sizes[si]

                # Unary strided
                comptime for ki in range(len(unary_syms)):
                    bench_unary[dtype, N](m, lib=lib, ctx=ctx, symbol=unary_syms[ki])

                # Binary contiguous fast path vs strided overhead
                bench_binary[dtype, N](m, lib=lib, ctx=ctx, symbol="mograd_add")
                comptime for ki in range(len(binary_strided_syms)):
                    bench_binary_strided[dtype, N](m, lib=lib, ctx=ctx, symbol=binary_strided_syms[ki])

                # Binary scalar strided
                bench_binary_scalar_strided[dtype, N](m, lib=lib, ctx=ctx, symbol="mograd_scale")

    m.dump_report()
