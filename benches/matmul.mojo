from max.benchmark import bencher_iter_custom
from max.gpu.host import DeviceContext
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of


from internal_utils import CacheBustingBuffer

from mograd import Device
from mograd.layout import Layout
from mograd.runtime.gpu.kernels.dispatch import MatmulStrided


def _row_major_layout(batch: Int, M: Int, K: Int) -> Layout:
    if batch == 1:
        return Layout(M, K)
    return Layout(batch, M, K)


def _padded_layout(batch: Int, M: Int, K: Int, pad: Int) raises -> Layout:
    var ld = K + pad
    if batch == 1:
        return Layout(M, ld).slice_axis(1, 0, K)
    return Layout(batch, M, ld).slice_axis(2, 0, K)


@no_inline
def bench_matmul[
    dtype: DType, batch: Int, M: Int, K: Int, N: Int, pad: Int
](mut m: Bench, *, device: Device, ctx: DeviceContext, symbol: String) raises:
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    comptime flops = 2 * batch * M * N * K
    comptime bytes_moved = batch * (M * K + K * N + M * N) * size_of[dtype]()

    var func = device.get_function[MatmulStrided](symbol)

    var a_numel = batch * M * (K + pad)
    var b_numel = batch * K * (N + pad)
    var c_numel = batch * M * N

    var a = CacheBustingBuffer[dtype](a_numel, 1, ctx)
    var b = CacheBustingBuffer[dtype](b_numel, 1, ctx)
    var dst = CacheBustingBuffer[dtype](c_numel, 1, ctx)

    var la = _row_major_layout(batch, M, K) if pad == 0 else _padded_layout(batch, M, K, pad)
    var lb = _row_major_layout(batch, K, N) if pad == 0 else _padded_layout(batch, K, N, pad)

    _ = func(
        a.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        b.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        la,
        lb,
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
                la,
                lb,
                N,
                dtype,
                dc,
            )

        bencher_iter_custom(bench, kernel_launch, ctx)
        keep(dst.unsafe_ptr())

    comptime path = "fast" if pad == 0 else "naive"
    m.bench_function(
        bench_func,
        BenchId(symbol, input_id=String(dtype_str, "/", path, "/", batch, "x", M, "x", K, "x", N)),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )


@no_inline
def bench_matmul_t[
    dtype: DType, batch: Int, M: Int, K: Int, N: Int, pad: Int
](mut m: Bench, *, device: Device, ctx: DeviceContext, symbol: String) raises:
    # mograd_matmul_bt computes A @ B^T, so B is stored (N, K) rather than (K, N).
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    comptime flops = 2 * batch * M * N * K
    comptime bytes_moved = batch * (M * K + N * K + M * N) * size_of[dtype]()

    var func = device.get_function[MatmulStrided](symbol)

    var a_numel = batch * M * (K + pad)
    var b_numel = batch * N * (K + pad)
    var c_numel = batch * M * N

    var a = CacheBustingBuffer[dtype](a_numel, 1, ctx)
    var b = CacheBustingBuffer[dtype](b_numel, 1, ctx)
    var dst = CacheBustingBuffer[dtype](c_numel, 1, ctx)

    var la = _row_major_layout(batch, M, K) if pad == 0 else _padded_layout(batch, M, K, pad)
    var lb = _row_major_layout(batch, N, K) if pad == 0 else _padded_layout(batch, N, K, pad)

    _ = func(
        a.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        b.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).unsafe_bitcast[NoneType]().as_unsafe_any_origin(),
        la,
        lb,
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
                la,
                lb,
                N,
                dtype,
                dc,
            )

        bencher_iter_custom(bench, kernel_launch, ctx)
        keep(dst.unsafe_ptr())

    comptime path = "fast" if pad == 0 else "naive"
    m.bench_function(
        bench_func,
        BenchId(symbol, input_id=String(dtype_str, "/", path, "/", batch, "x", M, "x", K, "x", N)),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, bytes_moved),
        ],
    )


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=5, max_runtime_secs=10))
    var device = Device()

    comptime dtypes = (DType.float32,)
    comptime PAD = 8

    comptime sizes_b = (1, 1, 1, 32, 8)
    comptime sizes_m = (1024, 4096, 8192, 128, 512)
    comptime sizes_k = (1024, 4096, 1024, 64, 512)
    comptime sizes_n = (1024, 4096, 8192, 128, 2048)

    comptime sweep_square = (32, 64, 128, 256, 512, 1024)

    var ctx = device.ctx
    comptime for di in range(len(dtypes)):
        comptime dtype = rebind[DType](dtypes[di])
        comptime for si in range(len(sizes_b)):
            comptime B = rebind[Int](sizes_b[si])
            comptime M = rebind[Int](sizes_m[si])
            comptime K = rebind[Int](sizes_k[si])
            comptime N = rebind[Int](sizes_n[si])
            bench_matmul[dtype, B, M, K, N, 0](m, device=device, ctx=ctx, symbol="mograd_matmul")
            bench_matmul[dtype, B, M, K, N, PAD](m, device=device, ctx=ctx, symbol="mograd_matmul")
            bench_matmul_t[dtype, B, M, K, N, 0](m, device=device, ctx=ctx, symbol="mograd_matmul_bt")
            bench_matmul_t[dtype, B, M, K, N, PAD](m, device=device, ctx=ctx, symbol="mograd_matmul_bt")

        comptime for si in range(len(sweep_square)):
            comptime S = rebind[Int](sweep_square[si])
            bench_matmul[dtype, 32, S, S, S, 0](m, device=device, ctx=ctx, symbol="mograd_matmul")
            bench_matmul[dtype, 32, S, S, S, PAD](m, device=device, ctx=ctx, symbol="mograd_matmul")

    m.dump_report()
