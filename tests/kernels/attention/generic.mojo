from max.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import has_accelerator
from std.sys.info import has_apple_gpu_accelerator
from std.testing import TestSuite, assert_true
from mograd import Device, Tensor
from harness import assert_gate, bwd_case, fwd_case, run_fwd
from mograd.runtime.gpu.kernels.attention.generic import _flash_attn_fwd_launch, _flash_attn_bwd_launch

comptime IS_GENERIC_TARGET = has_accelerator() and not has_apple_gpu_accelerator()


def _fwd[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    q: Pointer[Scalar[dtype], ImmutAnyOrigin],
    k: Pointer[Scalar[dtype], ImmutAnyOrigin],
    v: Pointer[Scalar[dtype], ImmutAnyOrigin],
    mask: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    lse: Pointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    comptime if IS_GENERIC_TARGET:
        _flash_attn_fwd_launch[dtype, D_BUCKET, CAUSAL, HAS_BIAS](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)
    else:
        raise Error("generic kernels require a non-Apple accelerator")


def _bwd[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: Pointer[Scalar[dtype], ImmutAnyOrigin],
    o: Pointer[Scalar[dtype], ImmutAnyOrigin],
    q: Pointer[Scalar[dtype], ImmutAnyOrigin],
    k: Pointer[Scalar[dtype], ImmutAnyOrigin],
    v: Pointer[Scalar[dtype], ImmutAnyOrigin],
    mask: Pointer[Scalar[dtype], ImmutAnyOrigin],
    lse: Pointer[Float32, ImmutAnyOrigin],
    dq: Pointer[Scalar[dtype], MutAnyOrigin],
    dk: Pointer[Scalar[dtype], MutAnyOrigin],
    dv: Pointer[Scalar[dtype], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    comptime if IS_GENERIC_TARGET:
        _flash_attn_bwd_launch[dtype, D_BUCKET, CAUSAL, HAS_BIAS](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )
    else:
        raise Error("generic kernels require a non-Apple accelerator")


# ===-------------------------------------------------------------------===#
# Forward
# ===-------------------------------------------------------------------===#


def test_fwd_f16_full() raises:
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
        _fwd[DType.float16, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, False, False, launch, WITH_NAIVE=True](2, 2, 96, 64, 100)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 full generic")


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
        _fwd[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, True, False, launch, WITH_NAIVE=True](2, 2, 96, 64, 110)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 causal generic")


def test_fwd_f16_bias() raises:
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
        _fwd[DType.float16, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, False, True, launch, WITH_NAIVE=True](2, 2, 96, 64, 120)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 bias generic")


def test_fwd_f16_causal_d128() raises:
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
        _fwd[DType.float16, 128, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, True, False, launch, WITH_NAIVE=True](2, 2, 96, 128, 130)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 causal d128 generic")


def test_fwd_f16_large_s() raises:
    # S=768 forces the large-grid launch geometry
    @__parameter
    @always_inline
    def full(
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
        _fwd[DType.float16, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, False, False, full, WITH_NAIVE=True](1, 4, 768, 64, 300)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 large-S full generic")

    @__parameter
    @always_inline
    def causal(
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
        _fwd[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var ec = fwd_case[DType.float16, True, False, causal, WITH_NAIVE=True](1, 4, 768, 64, 310)
    assert_gate(ec.o, ec.naive, 2.0, 1e-4, "f16 large-S causal generic")


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
        _fwd[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch, WITH_NAIVE=True](2, 2, 96, 64, 140)
    assert_gate(e.o, e.naive, 4.0, 1e-7, "f32 full generic")


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
        _fwd[DType.float32, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, True, False, launch, WITH_NAIVE=True](2, 2, 96, 64, 150)
    assert_gate(e.o, e.naive, 4.0, 1e-7, "f32 causal generic")


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
        _fwd[DType.float32, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, True, launch, WITH_NAIVE=True](2, 2, 96, 64, 160)
    assert_gate(e.o, e.naive, 4.0, 1e-7, "f32 bias generic")


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
        _fwd[DType.float32, 256, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch, WITH_NAIVE=True](2, 2, 96, 256, 400)
    assert_gate(e.o, e.naive, 4.0, 1e-6, "f32 d256 fwd generic")


def test_fwd_large_logits_stable() raises:
    # std=20 pushes logits to ~|400|. True f32 FMA, so error tracks
    # 400 * 2^-24 amplified (measured 6.9e-5) rather than TF32 physics.
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
        _fwd[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch](2, 2, 96, 64, 170, std=20.0)
    assert_true(e.o < 2e-4, "large-logit f32 generic inaccurate")


def test_fwd_bias_shift_invariance() raises:
    # softmax(s + c) == softmax(s)
    var device = Device()
    var Qt = Tensor.randn(device, (2, 96, 2, 64), seed=180)
    var Kt = Tensor.randn(device, (2, 96, 2, 64), seed=181)
    var Vt = Tensor.randn(device, (2, 96, 2, 64), seed=182)
    var Mt = Tensor.randn(device, (2, 2, 96, 96), seed=183)
    var ql = Qt.to_list[DType.float32]()
    var kl = Kt.to_list[DType.float32]()
    var vl = Vt.to_list[DType.float32]()
    var ml = Mt.to_list[DType.float32]()
    var ml_shift = List[Scalar[DType.float32]](capacity=len(ml))
    for m in ml:
        ml_shift.append(m + 8.0)
    var scale = Float32(1.0) / sqrt(Float32(64))

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
        _fwd[DType.float32, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var g1 = run_fwd[DType.float32, True, launch](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    var g2 = run_fwd[DType.float32, True, launch](ql, kl, vl, ml_shift, 2, 96, 2, 64, scale)
    var d = Float64(0)
    for i in range(len(g1[0])):
        var e = Float64(g1[0][i]) - Float64(g2[0][i])
        e = e if e >= 0 else -e
        if e > d:
            d = e
    print(t"    shift-invariance max deviation: generic = {d}")
    if d > 1e-5:
        raise Error(t"generic bias shift invariance violated: {d} > 1e-5")


def test_fwd_d512() raises:
    # D > 256 has no MMA kernel, dispatch routes it here
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
        _fwd[DType.float32, 512, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch](2, 2, 96, 512, 500)
    assert_true(e.o < 1e-5 and e.lse < 1e-5, "f32 d512 generic fwd error")


# ===-------------------------------------------------------------------===#
# Backward
# ===-------------------------------------------------------------------===#


def test_bwd_f16_causal_d64() raises:
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
        _fwd[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _bwd[DType.float16, 64, True, False](dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx)

    var e = bwd_case[DType.float16, True, False, fwd, bwd](2, 2, 96, 64, 200)
    assert_true(e.dq < 5e-3 and e.dk < 5e-3 and e.dv < 5e-3, "f16 causal bwd generic error")


def test_bwd_f16_causal_d128() raises:
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
        _fwd[DType.float16, 128, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _bwd[DType.float16, 128, True, False](dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx)

    var e = bwd_case[DType.float16, True, False, fwd, bwd](2, 2, 96, 128, 210)
    assert_true(e.dq < 5e-3 and e.dk < 5e-3 and e.dv < 5e-3, "f16 causal d128 bwd generic error")


def test_bwd_f16_bias_d64() raises:
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
        _fwd[DType.float16, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _bwd[DType.float16, 64, False, True](dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx)

    var e = bwd_case[DType.float16, False, True, fwd, bwd](2, 2, 96, 64, 220)
    assert_true(e.dq < 5e-3 and e.dk < 5e-3 and e.dv < 5e-3, "f16 bias bwd generic error")


def test_bwd_f32_full_d64() raises:
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
        _fwd[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _bwd[DType.float32, 64, False, False](dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx)

    var e = bwd_case[DType.float32, False, False, fwd, bwd](2, 2, 96, 64, 230)
    assert_true(e.dq < 5e-6 and e.dk < 5e-6 and e.dv < 5e-6, "f32 full bwd generic error")


def test_bwd_f32_full_d256() raises:
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
        _fwd[DType.float32, 256, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _bwd[DType.float32, 256, False, False](dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx)

    var e = bwd_case[DType.float32, False, False, fwd, bwd](2, 2, 96, 256, 410)
    assert_true(e.dq < 2e-5 and e.dk < 2e-5 and e.dv < 2e-5, "f32 d256 bwd generic error")


def test_bwd_d512() raises:
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
        _fwd[DType.float32, 512, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _bwd[DType.float32, 512, False, False](dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx)

    var e = bwd_case[DType.float32, False, False, fwd, bwd](2, 2, 96, 512, 503)
    assert_true(e.dq < 5e-5 and e.dk < 5e-5 and e.dv < 5e-5, "f32 d512 generic bwd error")


def main() raises:
    comptime assert has_accelerator(), "GPU required to run kernel accuracy tests"
    TestSuite.discover_tests[__functions_in_module()]().run(skip_all=not IS_GENERIC_TARGET)
