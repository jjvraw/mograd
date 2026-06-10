from std.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle
from std.os.env import getenv
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of

from internal_utils import CacheBustingBuffer

from mograd.runtime import BinaryElementWise, BinaryElementWiseStrided


@no_inline
def bench_binary[
    dtype: DType, N: Int
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext, symbol: String,) raises:
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


# NOTE: Bench impact of strided kernel on contiguous data. Is it worth having two paths?
@no_inline
def bench_binary_strided[
    dtype: DType, N: Int
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext, symbol: String,) raises:
    comptime bytes_moved = N * 3 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = lib.get_function[BinaryElementWiseStrided](symbol)
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


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=5, max_runtime_secs=10))
    var lib = OwnedDLHandle(getenv("MOGRAD_SO"))

    comptime dtypes = (DType.float32,)
    comptime sizes = (1024 * 1024, 1024 * 1024 * 16, 1024 * 1024 * 64)
    # (contiguous_symbol, strided_symbol) pairs
    comptime symbols = (("mograd_add", "mograd_add_strided"),)

    with DeviceContext() as ctx:
        comptime for di in range(len(dtypes)):
            comptime for si in range(len(sizes)):
                comptime for ki in range(len(symbols)):
                    comptime dtype = dtypes[di]
                    comptime N = sizes[si]
                    comptime sym = symbols[ki][0]
                    comptime sym_strided = symbols[ki][1]
                    bench_binary[dtype, N](m, lib=lib, ctx=ctx, symbol=sym)
                    bench_binary_strided[dtype, N](m, lib=lib, ctx=ctx, symbol=sym_strided)

    m.dump_report()
