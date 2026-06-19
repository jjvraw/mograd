from std.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle
from std.os.env import getenv
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of

from layout.int_tuple import IntTuple

from internal_utils import CacheBustingBuffer

from mograd.layout import Layout
from mograd.runtime.gpu.kernels.utils import UnaryStrided


def _transposed_view_layout(batch: Int, M: Int, N: Int) -> Layout:
    if batch == 1:
        return Layout(2, IntTuple(N, M), IntTuple(1, N), 0)
    return Layout(3, IntTuple(batch, N, M), IntTuple(M * N, 1, N), 0)


@no_inline
def bench_transpose[
    dtype: DType, batch: Int, M: Int, N: Int
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext, symbol: String) raises:
    comptime numel = batch * M * N
    comptime bytes_moved = numel * 2 * size_of[dtype]()
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"

    var func = lib.get_function[UnaryStrided](symbol)
    var a = CacheBustingBuffer[dtype](numel, 1, ctx)
    var dst = CacheBustingBuffer[dtype](numel, 1, ctx)

    var layout = _transposed_view_layout(batch, M, N)

    func(
        a.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        layout,
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
                a.offset_ptr(iteration).bitcast[NoneType]().as_unsafe_any_origin(),
                dst.offset_ptr(iteration).bitcast[NoneType]().as_unsafe_any_origin(),
                layout,
                dtype,
                dc,
            )

        bench.iter_custom[kernel_launch](ctx)
        keep(dst.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId(symbol, input_id=String(dtype_str, "/", batch, "x", M, "x", N)),
        [
            ThroughputMeasure(BenchMetric.elements, numel),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=5, max_runtime_secs=10))
    var lib = OwnedDLHandle(getenv("MOGRAD_SO"))

    comptime dtypes = (DType.float32,)
    comptime syms = ("mograd_transpose_last2", "mograd_contiguous")

    comptime sizes_2d_m = (1024, 4096, 4096, 8192, 16384, 16384)
    comptime sizes_2d_n = (1024, 1024, 4096, 8192, 8192, 16384)

    comptime sizes_3d_b = (8, 32, 64)
    comptime sizes_3d_m = (1024, 512, 1024)
    comptime sizes_3d_n = (1024, 2048, 2048)

    with DeviceContext() as ctx:
        comptime for di in range(len(dtypes)):
            comptime dtype = dtypes[di]

            comptime for si in range(len(sizes_2d_m)):
                comptime M = sizes_2d_m[si]
                comptime N = sizes_2d_n[si]
                comptime for ki in range(len(syms)):
                    bench_transpose[dtype, 1, M, N](m, lib=lib, ctx=ctx, symbol=syms[ki])

            comptime for si in range(len(sizes_3d_b)):
                comptime B = sizes_3d_b[si]
                comptime M = sizes_3d_m[si]
                comptime N = sizes_3d_n[si]
                comptime for ki in range(len(syms)):
                    bench_transpose[dtype, B, M, N](m, lib=lib, ctx=ctx, symbol=syms[ki])

    m.dump_report()
