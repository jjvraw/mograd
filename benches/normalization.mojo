from std.gpu.host import DeviceContext, DeviceAttribute
from std.ffi import OwnedDLHandle
from std.os.env import getenv
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of

from internal_utils import CacheBustingBuffer

from mograd.runtime.gpu.kernels.utils import LayerNormFwdKernel, LayerNormBwdKernel


@no_inline
def bench_layer_norm_fwd[
    dtype: DType, rows: Int, cols: Int
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext) raises:
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    comptime bytes_moved = (2 * rows * cols + 3 * cols) * size_of[dtype]()

    var func = lib.get_function[LayerNormFwdKernel]("mograd_layer_norm_fwd")
    var x = CacheBustingBuffer[dtype](rows * cols, 1, ctx)
    var gamma = CacheBustingBuffer[dtype](cols, 1, ctx)
    var beta = CacheBustingBuffer[dtype](cols, 1, ctx)
    var dst = CacheBustingBuffer[dtype](rows * cols, 1, ctx)

    func(
        x.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        gamma.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        beta.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        dst.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        rows,
        cols,
        Float32(1e-5),
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
                x.offset_ptr(iteration).bitcast[NoneType]().as_unsafe_any_origin(),
                gamma.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
                beta.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
                dst.offset_ptr(iteration).bitcast[NoneType]().as_unsafe_any_origin(),
                rows,
                cols,
                Float32(1e-5),
                dtype,
                dc,
            )

        bench.iter_custom[kernel_launch](ctx)
        keep(dst.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId("layer_norm_fwd", input_id=String(dtype_str, "/", rows, "x", cols)),
        [ThroughputMeasure(BenchMetric.bytes, bytes_moved)],
    )


@no_inline
def bench_layer_norm_bwd[
    dtype: DType, rows: Int, cols: Int
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext) raises:
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    var num_sm = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var actual_groups = min(rows, num_sm)
    var bytes_moved = (3 * rows * cols + 2 * actual_groups * cols + 3 * cols) * size_of[dtype]()

    var func = lib.get_function[LayerNormBwdKernel]("mograd_layer_norm_bwd")
    var dy = CacheBustingBuffer[dtype](rows * cols, 1, ctx)
    var x = CacheBustingBuffer[dtype](rows * cols, 1, ctx)
    var gamma = CacheBustingBuffer[dtype](cols, 1, ctx)
    var dx = CacheBustingBuffer[dtype](rows * cols, 1, ctx)
    var dgamma = CacheBustingBuffer[dtype](cols, 1, ctx)
    var dbeta = CacheBustingBuffer[dtype](cols, 1, ctx)

    func(
        dy.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        x.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        gamma.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        dx.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        dgamma.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        dbeta.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
        rows,
        cols,
        Float32(1e-5),
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
                dy.offset_ptr(iteration).bitcast[NoneType]().as_unsafe_any_origin(),
                x.offset_ptr(iteration).bitcast[NoneType]().as_unsafe_any_origin(),
                gamma.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
                dx.offset_ptr(iteration).bitcast[NoneType]().as_unsafe_any_origin(),
                dgamma.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
                dbeta.offset_ptr(0).bitcast[NoneType]().as_unsafe_any_origin(),
                rows,
                cols,
                Float32(1e-5),
                dtype,
                dc,
            )

        bench.iter_custom[kernel_launch](ctx)
        keep(dx.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId("layer_norm_bwd", input_id=String(dtype_str, "/", rows, "x", cols)),
        [ThroughputMeasure(BenchMetric.bytes, bytes_moved)],
    )


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=2, max_runtime_secs=5))
    var lib = OwnedDLHandle(getenv("MOGRAD_SO"))

    comptime shapes = (
        (512, 512),
        (512, 768),
        (512, 1024),
        (512, 2048),
        (512, 4096),
        (2048, 512),
        (2048, 768),
        (2048, 1024),
        (2048, 2048),
        (2048, 4096),
        (4096, 4096),
        (8192, 512),
        (8192, 1024),
        (8192, 4096),
        (8192, 8192),
    )

    with DeviceContext() as ctx:
        comptime for si in range(len(shapes)):
            comptime rows = shapes[si][0]
            comptime cols = shapes[si][1]
            bench_layer_norm_fwd[DType.float32, rows, cols](m, lib=lib, ctx=ctx)
            bench_layer_norm_bwd[DType.float32, rows, cols](m, lib=lib, ctx=ctx)

    m.dump_report()
