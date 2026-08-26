from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of

from internal_utils import CacheBustingBuffer

from std.memory.alloc import unsafe_alloc

from mograd import Device
from mograd.layout import Layout
from mograd.runtime.gpu.kernels.dispatch import BinaryElementWise, BinaryScalarElementWiseStrided
from mograd.runtime.gpu.kernels.dispatch import UnaryStrided, BinaryStrided


# ===-------------------------------------------------------------------===#
# Unary strided (neg, log, exp, relu, contiguous)
# ===-------------------------------------------------------------------===#


@no_inline
def bench_unary[dtype: DType, N: Int](mut m: Bench, *, device: Device, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 2 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = device.get_function[UnaryStrided](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    var layout = Layout(N)

    _ = func(
        a.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        layout,
        dtype,
        ctx,
    )
    ctx.synchronize()

    @always_inline
    def bench_func(mut bench: Bencher) raises {imm}:
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises {imm}:
            _ = func(
                a.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                dst.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                layout,
                dtype,
                dc,
            )

        bencher_iter_custom(bench, kernel_launch, ctx)
        keep(dst.unsafe_ptr())

    m.bench_function(
        bench_func,
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
def bench_binary[dtype: DType, N: Int](mut m: Bench, *, device: Device, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 3 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = device.get_function[BinaryElementWise](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var b = CacheBustingBuffer[dtype](N, 1, ctx)
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    _ = func(
        a.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        b.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        N,
        dtype,
        ctx,
    )
    ctx.synchronize()

    @always_inline
    def bench_func(mut bench: Bencher) raises {imm}:
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises {imm}:
            _ = func(
                a.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                b.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                dst.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                N,
                dtype,
                dc,
            )

        bencher_iter_custom(bench, kernel_launch, ctx)
        keep(dst.unsafe_ptr())

    m.bench_function(
        bench_func,
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
](mut m: Bench, *, device: Device, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 3 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = device.get_function[BinaryStrided](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var b = CacheBustingBuffer[dtype](N, 1, ctx)
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    var layout = Layout(N)

    _ = func(
        a.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        b.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        layout,
        layout,
        dtype,
        ctx,
    )
    ctx.synchronize()

    @always_inline
    def bench_func(mut bench: Bencher) raises {imm}:
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises {imm}:
            _ = func(
                a.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                b.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                dst.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                layout,
                layout,
                dtype,
                dc,
            )

        bencher_iter_custom(bench, kernel_launch, ctx)
        keep(dst.unsafe_ptr())

    m.bench_function(
        bench_func,
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
](mut m: Bench, *, device: Device, ctx: DeviceContext, symbol: String) raises:
    comptime bytes_moved = N * 2 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = device.get_function[BinaryScalarElementWiseStrided](symbol)
    var a = CacheBustingBuffer[dtype](N, 1, ctx)
    var scalar = unsafe_alloc[Float32](1)
    scalar[unsafe_offset=0] = 2.0
    var dst = CacheBustingBuffer[dtype](N, 1, ctx)

    var layout = Layout(N)

    _ = func(
        a.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        scalar.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        layout,
        dtype,
        ctx,
    )
    ctx.synchronize()

    @always_inline
    def bench_func(mut bench: Bencher) raises {imm}:
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises {imm}:
            _ = func(
                a.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                scalar.unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                dst.offset_ptr(iteration).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
                layout,
                dtype,
                dc,
            )

        bencher_iter_custom(bench, kernel_launch, ctx)
        keep(dst.unsafe_ptr())

    m.bench_function(
        bench_func,
        BenchId(symbol, input_id=String(dtype_str, "/", N // (1024 * 1024), "M")),
        [
            ThroughputMeasure(BenchMetric.elements, N),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )
    scalar.unsafe_free()


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=5, max_runtime_secs=10))
    var device = Device()

    comptime dtypes = (DType.float32,)
    comptime sizes = (1024 * 1024, 1024 * 1024 * 16, 1024 * 1024 * 64, 1024 * 1024 * 128, 1024 * 1024 * 256)

    comptime unary_syms = ("mograd_neg", "mograd_exp", "mograd_log", "mograd_relu", "mograd_contiguous")
    comptime binary_strided_syms = ("mograd_add_strided", "mograd_mul", "mograd_div", "mograd_eq", "mograd_relu_grad")

    var ctx = device.ctx
    comptime for di in range(len(dtypes)):
        comptime dtype = rebind[DType](dtypes[di])
        comptime for si in range(len(sizes)):
            comptime N = rebind[Int](sizes[si])

            # Unary strided
            comptime for ki in range(len(unary_syms)):
                bench_unary[dtype, N](m, device=device, ctx=ctx, symbol=rebind[String](unary_syms[ki]))

            # Binary contiguous fast path vs strided overhead
            bench_binary[dtype, N](m, device=device, ctx=ctx, symbol="mograd_add")
            comptime for ki in range(len(binary_strided_syms)):
                bench_binary_strided[dtype, N](
                    m, device=device, ctx=ctx, symbol=rebind[String](binary_strided_syms[ki])
                )

            # Binary scalar strided
            bench_binary_scalar_strided[dtype, N](m, device=device, ctx=ctx, symbol="mograd_scale")

    m.dump_report()
