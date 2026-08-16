from max.gpu.host import DeviceContext
from std.math import sqrt
from std.sys.info import has_apple_gpu_accelerator
from std.testing import TestSuite, assert_true
from harness import bwd_case, fwd_case
from mograd.runtime.gpu.kernels.attention.apple import _flash_attn_fwd_launch_apple, _flash_attn_bwd_launch_apple

comptime IS_APPLE = has_apple_gpu_accelerator()


# ===-------------------------------------------------------------------===#
# Forward
# ===-------------------------------------------------------------------===#


def test_fwd_f32_full() raises:
    @__parameter
    @always_inline
    def launch(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch](2, 2, 96, 64, 10)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 full fwd error")


def test_fwd_f32_causal() raises:
    @__parameter
    @always_inline
    def launch(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, True, False, launch](2, 2, 96, 64, 20)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 causal fwd error")


def test_fwd_f32_bias() raises:
    @__parameter
    @always_inline
    def launch(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, True, launch](2, 2, 96, 64, 30)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 bias fwd error")


def test_fwd_f32_ragged_d() raises:
    # D=48 in the 64 bucket exercises the d < D lane guards
    @__parameter
    @always_inline
    def launch(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, True, False, launch](1, 2, 80, 48, 40)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 ragged-D fwd error")


def test_fwd_f32_d256() raises:
    @__parameter
    @always_inline
    def launch(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 256, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, True, False, launch](1, 1, 64, 256, 50)
    assert_true(e.o < 5e-4 and e.lse < 5e-4, "f32 d256 fwd error")


def test_fwd_f16_causal() raises:
    @__parameter
    @always_inline
    def launch(
        q: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, True, False, launch](2, 2, 96, 64, 60)
    assert_true(e.o < 5e-3 and e.lse < 1e-3, "f16 causal fwd error")


# ===-------------------------------------------------------------------===#
# Backward
# ===-------------------------------------------------------------------===#


def test_bwd_f32_full() raises:
    @__parameter
    @always_inline
    def fwd(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    @__parameter
    @always_inline
    def bwd(
        dy: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        o: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        lse: Pointer[Float32, ImmutAnyOrigin],
        dq: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dk: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dv: Pointer[Scalar[DType.float32], MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_bwd_launch_apple[DType.float32, 64, False, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float32, False, False, fwd, bwd](2, 2, 96, 64, 110)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 full bwd error")


def test_bwd_f32_causal() raises:
    @__parameter
    @always_inline
    def fwd(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    @__parameter
    @always_inline
    def bwd(
        dy: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        o: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        lse: Pointer[Float32, ImmutAnyOrigin],
        dq: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dk: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dv: Pointer[Scalar[DType.float32], MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_bwd_launch_apple[DType.float32, 64, True, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float32, True, False, fwd, bwd](2, 2, 96, 64, 120)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 causal bwd error")


def test_bwd_f32_bias() raises:
    @__parameter
    @always_inline
    def fwd(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    @__parameter
    @always_inline
    def bwd(
        dy: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        o: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        lse: Pointer[Float32, ImmutAnyOrigin],
        dq: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dk: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dv: Pointer[Scalar[DType.float32], MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_bwd_launch_apple[DType.float32, 64, False, True](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float32, False, True, fwd, bwd](2, 2, 96, 64, 130)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 bias bwd error")


def test_bwd_f32_multi_tile() raises:
    # S=128 spans multiple APPLE_QROWS tiles
    @__parameter
    @always_inline
    def fwd(
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float32], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float32, 32, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    @__parameter
    @always_inline
    def bwd(
        dy: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        o: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        q: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float32], ImmutAnyOrigin],
        lse: Pointer[Float32, ImmutAnyOrigin],
        dq: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dk: Pointer[Scalar[DType.float32], MutAnyOrigin],
        dv: Pointer[Scalar[DType.float32], MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_bwd_launch_apple[DType.float32, 32, True, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float32, True, False, fwd, bwd](1, 1, 128, 8, 140)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 multi-tile bwd error")


def test_bwd_f16_causal() raises:
    @__parameter
    @always_inline
    def fwd(
        q: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        dst: Pointer[Scalar[DType.float16], MutAnyOrigin],
        lse: Pointer[Float32, MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_fwd_launch_apple[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    @__parameter
    @always_inline
    def bwd(
        dy: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        o: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        q: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        k: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        v: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        mask: Pointer[Scalar[DType.float16], ImmutAnyOrigin],
        lse: Pointer[Float32, ImmutAnyOrigin],
        dq: Pointer[Scalar[DType.float16], MutAnyOrigin],
        dk: Pointer[Scalar[DType.float16], MutAnyOrigin],
        dv: Pointer[Scalar[DType.float16], MutAnyOrigin],
        B: Int,
        S: Int,
        H: Int,
        D: Int,
        scale: Float32,
        ctx: DeviceContext,
    ) raises:
        _flash_attn_bwd_launch_apple[DType.float16, 64, True, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float16, True, False, fwd, bwd](2, 2, 96, 64, 150)
    assert_true(e.dq < 2e-2 and e.dk < 2e-2 and e.dv < 2e-2, "f16 causal bwd error")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run(skip_all=not IS_APPLE)
