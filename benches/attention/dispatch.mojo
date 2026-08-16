from std.gpu.host import DeviceContext
from std.ffi import OwnedDLHandle
from std.math import sqrt
from std.os.env import getenv
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure
from std.sys import size_of

from internal_utils import CacheBustingBuffer

from mograd.runtime.gpu.kernels.utils import FlashAttnFwdKernel, FlashAttnBwdKernel


@always_inline
def _arg[dtype: DType](buf: CacheBustingBuffer[dtype], i: Int) -> Pointer[NoneType, MutAnyOrigin]:
    return buf.offset_ptr(i).bitcast[NoneType]().as_unsafe_any_origin()


@no_inline
def bench_flash_attn_fwd[
    dtype: DType, B: Int, S: Int, H: Int, D: Int, CAUSAL: Bool, BIAS: Bool = False
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext) raises:
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    comptime causal_str = "causal" if CAUSAL else ("bias" if BIAS else "full")
    comptime fwd_flop_scale = 2 if CAUSAL else 4
    comptime flops = fwd_flop_scale * B * H * S * S * D
    comptime hbm_bytes = 4 * B * H * S * D * size_of[dtype]() + B * H * S * size_of[DType.float32]() + (
        B * H * S * S * size_of[dtype]() if BIAS else 0  # additive mask, read only on the bias path
    )

    var scale = Float32(1.0) / sqrt(Float32(D))
    var func = lib.get_function[FlashAttnFwdKernel]("mograd_flash_attn_fwd")

    var q = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var k = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var v = CacheBustingBuffer[dtype](B * S * H * D, 1, ctx)
    var mask = CacheBustingBuffer[dtype](B * H * S * S, 1, ctx)
    var dst = CacheBustingBuffer[dtype](B * H * S * D, 1, ctx)
    var lse = CacheBustingBuffer[DType.float32](B * H * S, 1, ctx)

    @parameter
    @always_inline
    def kernel_launch(dc: DeviceContext, iteration: Int) raises:
        func(
            _arg(q, iteration),
            _arg(k, iteration),
            _arg(v, iteration),
            _arg(mask, 0),
            _arg(dst, iteration),
            _arg(lse, iteration),
            B,
            S,
            H,
            D,
            scale,
            Int(CAUSAL),
            Int(BIAS),
            dtype,
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
            "flash_attn_fwd",
            input_id=String(dtype_str, "/", causal_str, "/B", B, "S", S, "H", H, "D", D),
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, hbm_bytes),
        ],
    )


@no_inline
def bench_flash_attn_bwd[
    dtype: DType, B: Int, S: Int, H: Int, D: Int, CAUSAL: Bool, BIAS: Bool = not CAUSAL
](mut m: Bench, *, lib: OwnedDLHandle, ctx: DeviceContext) raises:
    comptime dtype_str = "f32" if dtype == DType.float32 else "f16" if dtype == DType.float16 else "bf16"
    comptime causal_str = "causal" if CAUSAL else ("bias" if BIAS else "full")
    # We use FA2 convention here, the backward is 5 matmuls (score, dP, dV, dK, dQ) = 2.5x the forward's 2,
    # so TFLOPS are comparable with published numbers.
    # Implementations that recompute work (e.g. the f32 split path, which runs 7 matmuls) correctly show the
    # recomputation as lost efficiency rather than free flops.
    comptime bwd_flop_scale = 5 if CAUSAL else 10
    comptime flops = bwd_flop_scale * B * H * S * S * D
    # Ideal algorithm I/O only: 5 reads (dO, O, Q, K, V) + 3 writes
    # (dQ, dK, dV) + LSE + the additive mask. Implementation temporaries
    # (delta buffer, the fused half path's fp32 dq_accum round trip) are
    # deliberately excluded, since counting overhead as throughput would inflate
    # the metric.
    comptime hbm_bytes = 8 * B * H * S * D * size_of[dtype]() + B * H * S * size_of[DType.float32]() + (
        B * H * S * S * size_of[dtype]() if BIAS else 0  # additive mask, read only on the bias path
    )

    var scale = Float32(1.0) / sqrt(Float32(D))
    var func = lib.get_function[FlashAttnBwdKernel]("mograd_flash_attn_bwd")

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
        func(
            _arg(dy, iteration),
            _arg(o, iteration),
            _arg(q, iteration),
            _arg(k, iteration),
            _arg(v, iteration),
            _arg(mask, 0),
            _arg(lse, 0),
            _arg(dq, iteration),
            _arg(dk, iteration),
            _arg(dv, iteration),
            B,
            S,
            H,
            D,
            scale,
            Int(CAUSAL),
            Int(BIAS),
            dtype,
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
            "flash_attn_bwd",
            input_id=String(dtype_str, "/", causal_str, "/B", B, "S", S, "H", H, "D", D),
        ),
        [
            ThroughputMeasure(BenchMetric.flops, flops),
            ThroughputMeasure(BenchMetric.bytes, hbm_bytes),
        ],
    )


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=2, max_runtime_secs=5))
    var lib = OwnedDLHandle(getenv("MOGRAD_SO"))

    # (B, S, H, D) spans both D buckets and small to large sequence lengths.
    comptime shapes = (
        (1, 256, 8, 64),
        (1, 1024, 8, 64),
        (4, 512, 12, 64),
        (8, 1024, 16, 64),
        (8, 256, 16, 128),
        (8, 512, 16, 128),
        (8, 1024, 16, 128),
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
                bench_flash_attn_fwd[dtype, B, S, H, D, False, False](m, lib=lib, ctx=ctx)
                bench_flash_attn_fwd[dtype, B, S, H, D, False, True](m, lib=lib, ctx=ctx)
                bench_flash_attn_fwd[dtype, B, S, H, D, True](m, lib=lib, ctx=ctx)
                bench_flash_attn_bwd[dtype, B, S, H, D, False, False](m, lib=lib, ctx=ctx)
                bench_flash_attn_bwd[dtype, B, S, H, D, False, True](m, lib=lib, ctx=ctx)
                bench_flash_attn_bwd[dtype, B, S, H, D, True](m, lib=lib, ctx=ctx)

    m.dump_report()
