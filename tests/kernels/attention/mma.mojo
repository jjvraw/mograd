"""Numerical accuracy tests for the NVIDIA MMA flash-attention kernels,
called directly (no graph, no dispatch routing) via harness.mojo.

A kernel's max error vs the f64 oracle may not exceed a small factor of the
naive baseline's error, adapting the tolerance to dtype and shape instead of
hand-picking epsilons. f32 is the exception: MMA runs TF32 matmuls (10-bit
mantissa) while the naive f32 baseline does not, so it gets an absolute
TF32-class cap instead.
"""
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import has_accelerator
from std.sys.info import has_nvidia_gpu_accelerator
from std.testing import TestSuite, assert_true
from mograd import Device, Tensor
from harness import assert_gate, bwd_case, fwd_case, run_fwd, run_bwd
from mograd.runtime.gpu.kernels.attention.nvidia_fwd import _flash_attn_fwd_launch_mma
from mograd.runtime.gpu.kernels.attention.nvidia_bwd import (
    _flash_attn_bwd_launch_mma,
    _flash_attn_bwd_launch_mma_half,
)

# The MMA kernels require NVIDIA sm_80+ (TF32 m16n8k8, m16n8k16 HMMA,
# cp.async). The suite skips on other targets.
comptime IS_NVIDIA = has_nvidia_gpu_accelerator()


# ===-------------------------------------------------------------------===#
# Forward. S=96 is ragged (not a multiple of Br_MMA=64) and multi-tile.
# D=64 and D=128 hit both head-dim buckets.
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
        _flash_attn_fwd_launch_mma[DType.float16, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, False, False, launch, WITH_NAIVE=True](2, 2, 96, 64, 100)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 full mma")
    assert_true(e.lse < 1e-4, "f16 full LSE error")


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
        _flash_attn_fwd_launch_mma[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, True, False, launch, WITH_NAIVE=True](2, 2, 96, 64, 110)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 causal mma")
    assert_true(e.lse < 1e-4, "f16 causal LSE error")


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
        _flash_attn_fwd_launch_mma[DType.float16, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, False, True, launch, WITH_NAIVE=True](2, 2, 96, 64, 120)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 bias mma")


def test_fwd_f16_causal_d128() raises:
    # exercises the register-capped fwd-half MMA variant
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
        _flash_attn_fwd_launch_mma[DType.float16, 128, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, True, False, launch, WITH_NAIVE=True](2, 2, 96, 128, 130)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 causal d128 mma")


def test_fwd_f16_large_s() raises:
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
        _flash_attn_fwd_launch_mma[DType.float16, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float16, False, False, full, WITH_NAIVE=True](1, 4, 768, 64, 300)
    assert_gate(e.o, e.naive, 2.0, 1e-4, "f16 large-S full mma")

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
        _flash_attn_fwd_launch_mma[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var ec = fwd_case[DType.float16, True, False, causal, WITH_NAIVE=True](1, 4, 768, 64, 310)
    assert_gate(ec.o, ec.naive, 2.0, 1e-4, "f16 large-S causal mma")


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
        _flash_attn_fwd_launch_mma[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch](2, 2, 96, 64, 140)
    assert_true(e.o < 5e-3, "f32(TF32) full mma error above TF32-class cap")


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
        _flash_attn_fwd_launch_mma[DType.float32, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, True, False, launch](2, 2, 96, 64, 150)
    assert_true(e.o < 5e-3, "f32(TF32) causal mma error above TF32-class cap")


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
        _flash_attn_fwd_launch_mma[DType.float32, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, True, launch](2, 2, 96, 64, 160)
    assert_true(e.o < 5e-3, "f32(TF32) bias mma error above TF32-class cap")


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
        _flash_attn_fwd_launch_mma[DType.float32, 256, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch](2, 2, 96, 256, 400)
    assert_true(e.o < 2e-2, "f32(TF32) d256 fwd mma error")


def test_fwd_large_logits_stable() raises:
    # std=20 pushes logits to ~|400|, where TF32's 10-bit mantissa gives
    # logit error ~400 * 2^-11 ~= 0.2 and exp amplifies it into probability
    # weight (measured 0.19). Asserts stability, not precision.
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
        _flash_attn_fwd_launch_mma[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var e = fwd_case[DType.float32, False, False, launch](2, 2, 96, 64, 170, std=20.0)
    assert_true(e.o < 0.5, "large-logit f32 mma unstable (NaN/Inf or gross error)")


def test_fwd_deterministic_bitwise() raises:
    # forward has no atomics, so identical inputs must agree bit for bit
    var device = Device()
    var Qt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=190)
    var Kt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=191)
    var Vt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=192)
    var ql = Qt.to_list[DType.float16]()
    var kl = Kt.to_list[DType.float16]()
    var vl = Vt.to_list[DType.float16]()
    var ml = List[Scalar[DType.float16]]()
    var scale = Float32(1.0) / sqrt(Float32(64))

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
        _flash_attn_fwd_launch_mma[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var a = run_fwd[DType.float16, False, launch](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    var b = run_fwd[DType.float16, False, launch](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    for i in range(len(a[0])):
        assert_true(a[0][i] == b[0][i], "forward kernel is not bitwise deterministic")


def test_fwd_bias_shift_invariance() raises:
    # softmax(s + c) == softmax(s). P@V re-quantizes to TF32, so the two
    # runs decorrelate at ~sqrt(S) * P * 2^-11 (measured 1.6e-4).
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
        _flash_attn_fwd_launch_mma[DType.float32, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

    var o1 = run_fwd[DType.float32, True, launch](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    var o2 = run_fwd[DType.float32, True, launch](ql, kl, vl, ml_shift, 2, 96, 2, 64, scale)
    var d = Float64(0)
    for i in range(len(o1[0])):
        var e = Float64(o1[0][i]) - Float64(o2[0][i])
        e = e if e >= 0 else -e
        if e > d:
            d = e
    print(t"    shift-invariance max deviation: mma(TF32) = {d}")
    if d > 1e-3:
        raise Error(t"mma bias shift invariance violated: {d} > 1e-3")


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
        _flash_attn_fwd_launch_mma[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _flash_attn_bwd_launch_mma_half[DType.float16, 64, True, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float16, True, False, fwd, bwd](2, 2, 96, 64, 200)
    assert_true(e.dq < 8e-3 and e.dk < 8e-3 and e.dv < 8e-3, "f16 causal bwd mma error")


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
        _flash_attn_fwd_launch_mma[DType.float16, 128, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _flash_attn_bwd_launch_mma_half[DType.float16, 128, True, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float16, True, False, fwd, bwd](2, 2, 96, 128, 210)
    assert_true(e.dq < 8e-3 and e.dk < 8e-3 and e.dv < 8e-3, "f16 causal d128 bwd mma error")


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
        _flash_attn_fwd_launch_mma[DType.float16, 64, False, True](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _flash_attn_bwd_launch_mma_half[DType.float16, 64, False, True](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float16, False, True, fwd, bwd](2, 2, 96, 64, 220)
    assert_true(e.dq < 8e-3 and e.dk < 8e-3 and e.dv < 8e-3, "f16 bias bwd mma error")


def test_bwd_f32_full_d64() raises:
    # FA2 has no f32 backward to mirror, hence the absolute tolerance
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
        _flash_attn_fwd_launch_mma[DType.float32, 64, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _flash_attn_bwd_launch_mma[DType.float32, 64, False, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float32, False, False, fwd, bwd](2, 2, 96, 64, 230)
    assert_true(e.dq < 1.5e-2 and e.dk < 1.5e-2 and e.dv < 1.5e-2, "f32(TF32) full bwd mma error")


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
        _flash_attn_fwd_launch_mma[DType.float32, 256, False, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _flash_attn_bwd_launch_mma[DType.float32, 256, False, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var e = bwd_case[DType.float32, False, False, fwd, bwd](2, 2, 96, 256, 410)
    assert_true(e.dq < 5e-2 and e.dk < 5e-2 and e.dv < 5e-2, "f32(TF32) d256 bwd mma error")


def test_bwd_dq_atomic_spread() raises:
    # dQ accumulates via red.global.add (unordered), so it's bounded, not
    # bitwise. dK/dV have no atomics and ARE asserted bitwise.
    var device = Device()
    var Qt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=240)
    var Kt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=241)
    var Vt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=242)
    var Dyt = Tensor.randn(device, (2, 2, 96, 64), dtype=DType.float16, seed=243)
    var ql = Qt.to_list[DType.float16]()
    var kl = Kt.to_list[DType.float16]()
    var vl = Vt.to_list[DType.float16]()
    var ml = List[Scalar[DType.float16]]()
    var dyl = Dyt.to_list[DType.float16]()
    var scale = Float32(1.0) / sqrt(Float32(64))

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
        _flash_attn_fwd_launch_mma[DType.float16, 64, True, False](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

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
        _flash_attn_bwd_launch_mma_half[DType.float16, 64, True, False](
            dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
        )

    var r1 = run_bwd[DType.float16, False, fwd, bwd](ql, kl, vl, ml, dyl, 2, 96, 2, 64, scale)
    var r2 = run_bwd[DType.float16, False, fwd, bwd](ql, kl, vl, ml, dyl, 2, 96, 2, 64, scale)
    var spread = Float64(0)
    for i in range(len(r1[0])):
        var e = Float64(r1[0][i]) - Float64(r2[0][i])
        e = e if e >= 0 else -e
        if e > spread:
            spread = e
    print(t"    dq run-to-run atomic spread = {spread}")
    assert_true(spread < 1e-2, "dq atomic-order spread unexpectedly large")
    for i in range(len(r1[1])):
        assert_true(r1[1][i] == r2[1][i], "dk must be bitwise deterministic (no atomics)")
        assert_true(r1[2][i] == r2[2][i], "dv must be bitwise deterministic (no atomics)")


def main() raises:
    comptime assert has_accelerator(), "GPU required to run kernel accuracy tests"
    TestSuite.discover_tests[__functions_in_module()]().run(skip_all=not IS_NVIDIA)
