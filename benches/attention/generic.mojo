# Benchmark for the generic SIMT attention kernels, launched directly.

# The main benches/attention/dispatch.mojo measures the shipping FFI entry point, which
# dispatch routes to the MMA kernels on NVIDIA, so the generic fallback never
# runs there. This bench imports the generic launchers directly.
#
# These are portability kernels rather than tuned ones, and absolute numbers
# on NVIDIA are a proxy tripwire rather than a target. The grid is
# deliberately small, with just enough shape and variant coverage to catch
# codegen regressions without bloating bench time.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of

from internal_utils import CacheBustingBuffer

from mograd.runtime.gpu.kernels.attention.generic import _flash_attn_fwd_launch, _flash_attn_bwd_launch


@always_inline
def _imm[dtype: DType](buf: CacheBustingBuffer[dtype], i: Int) -> UnsafePointer[Scalar[dtype], ImmutAnyOrigin]:
    return buf.offset_ptr(i).as_immutable().as_unsafe_any_origin()


@always_inline
def _mut[dtype: DType](buf: CacheBustingBuffer[dtype], i: Int) -> UnsafePointer[Scalar[dtype], MutAnyOrigin]:
    return buf.offset_ptr(i).as_unsafe_any_origin()


@no_inline
def bench_generic_fwd[
    dtype: DType, B: Int, S: Int, H: Int, D: Int, CAUSAL: Bool, BIAS: Bool = False
](mut m: Bench, *, ctx: DeviceContext) raises:
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    comptime causal_str = "causal" if CAUSAL else ("bias" if BIAS else "full")
    comptime fwd_flop_scale = 2 if CAUSAL else 4
    comptime flops = fwd_flop_scale * B * H * S * S * D
    comptime hbm_bytes = 4 * B * H * S * D * size_of[dtype]() + B * H * S * size_of[DType.float32]() + (
        B * H * S * S * size_of[dtype]() if BIAS else 0
    )

    var scale = Float32(1.0) / sqrt(Float32(D))
    var q = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var k = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var v = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var mask = CacheBustingBuffer[dtype](B * H * S * S, 1, ctx)
    var dst = CacheBustingBuffer[dtype](B * H * S * D, 1, ctx)
    var lse = CacheBustingBuffer[DType.float32](B * H * S, 1, ctx)

    @parameter
    @always_inline
    def kernel_launch(dc: DeviceContext, iteration: Int) raises:
        _flash_attn_fwd_launch[dtype, D, CAUSAL, BIAS](
            _imm(q, iteration),
            _imm(k, iteration),
            _imm(v, iteration),
            _imm(mask, 0),
            _mut(dst, iteration),
            _mut(lse, iteration),
            B,
            S,
            H,
            D,
            scale,
            dc,
        )

    kernel_launch(ctx, 0)  # warmup
    ctx.synchronize()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        bench.iter_custom[kernel_launch](ctx)
        keep(dst.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId(
            "generic_attn_fwd",
            input_id=String(dtype_str, "/", causal_str, "/B", B, "S", S, "H", H, "D", D),
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, hbm_bytes),
        ],
    )


@no_inline
def bench_generic_bwd[
    dtype: DType, B: Int, S: Int, H: Int, D: Int, CAUSAL: Bool, BIAS: Bool = not CAUSAL
](mut m: Bench, *, ctx: DeviceContext) raises:
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    comptime causal_str = "causal" if CAUSAL else ("bias" if BIAS else "full")
    # FA2 convention (see benches/attention/dispatch.mojo). The split dq/dkdv scheme
    # recomputes scores, which correctly shows up as lost efficiency.
    comptime bwd_flop_scale = 5 if CAUSAL else 10
    comptime flops = bwd_flop_scale * B * H * S * S * D
    comptime hbm_bytes = 8 * B * H * S * D * size_of[dtype]() + B * H * S * size_of[DType.float32]() + (
        B * H * S * S * size_of[dtype]() if BIAS else 0
    )

    var scale = Float32(1.0) / sqrt(Float32(D))
    var dy = CacheBustingBuffer[dtype](B * H * S * D, 1, ctx)
    var o = CacheBustingBuffer[dtype](B * H * S * D, 1, ctx)
    var q = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var k = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var v = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var mask = CacheBustingBuffer[dtype](B * H * S * S, 1, ctx)
    var lse = CacheBustingBuffer[DType.float32](B * H * S, 1, ctx)
    var dq = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var dk = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var dv = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)

    @parameter
    @always_inline
    def kernel_launch(dc: DeviceContext, iteration: Int) raises:
        _flash_attn_bwd_launch[dtype, D, CAUSAL, BIAS](
            _imm(dy, iteration),
            _imm(o, iteration),
            _imm(q, iteration),
            _imm(k, iteration),
            _imm(v, iteration),
            _imm(mask, 0),
            _imm(lse, 0),
            _mut(dq, iteration),
            _mut(dk, iteration),
            _mut(dv, iteration),
            B,
            S,
            H,
            D,
            scale,
            dc,
        )

    kernel_launch(ctx, 0)  # warmup
    ctx.synchronize()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        bench.iter_custom[kernel_launch](ctx)
        keep(dq.unsafe_ptr())

    m.bench_function[bench_func](
        BenchId(
            "generic_attn_bwd",
            input_id=String(dtype_str, "/", causal_str, "/B", B, "S", S, "H", H, "D", D),
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, hbm_bytes),
        ],
    )


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=2, max_runtime_secs=5))

    comptime shapes = (
        (1, 256, 8, 64),
        (4, 512, 12, 64),
        (8, 512, 16, 128),
        (2, 512, 8, 256),
    )
    comptime dtypes = (DType.float32, DType.float16)

    with DeviceContext() as ctx:
        comptime for si in range(len(shapes)):
            comptime B = shapes[si][0]
            comptime S = shapes[si][1]
            comptime H = shapes[si][2]
            comptime D = shapes[si][3]
            comptime for di in range(len(dtypes)):
                comptime dtype = dtypes[di]
                bench_generic_fwd[dtype, B, S, H, D, False, False](m, ctx=ctx)
                bench_generic_fwd[dtype, B, S, H, D, False, True](m, ctx=ctx)
                bench_generic_fwd[dtype, B, S, H, D, True](m, ctx=ctx)
                bench_generic_bwd[dtype, B, S, H, D, False, False](m, ctx=ctx)
                bench_generic_bwd[dtype, B, S, H, D, False, True](m, ctx=ctx)
                bench_generic_bwd[dtype, B, S, H, D, True](m, ctx=ctx)

    m.dump_report()
