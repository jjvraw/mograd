"""Numerical accuracy tests for the flash attention kernels, called DIRECTLY.

The kernel launchers are invoked with raw device buffers, with no graph,
no GPU_REWRITES routing, and no dispatch heuristics. This tests exactly the
kernel named in each case and makes the generic SIMT kernels testable on
NVIDIA, where dispatch would otherwise always route to the MMA path.

Three consumers, one source of truth: seeded Tensor.randn leaves provide the
input bits (extracted via to_list), which feed
  oracle   a float64 CPU reference (ground truth, includes LSE)
  naive    the unfused mograd.Tensor graph in the working dtype, evaluated
           with simplifier=False as the same-precision accuracy baseline
  kernels  _flash_attn_fwd_launch_mma (NVIDIA MMA) and _flash_attn_fwd_launch
           (generic SIMT), launched directly

Gate (FlashAttention's methodology): a kernel's max error vs the f64 oracle
may not exceed a small factor of the naive baseline's error. This adapts the
tolerance to dtype and shape instead of hand-picking epsilons. The exception
is the f32 MMA path, which runs TF32 matmuls (10-bit mantissa) while the
naive f32 baseline does not, so it gets an absolute TF32-class cap instead.

Layouts (kernel contract): Q/K/V are BSHD, O is BHSD, mask BHSS, LSE (B,H,S)
float32 in natural log.
"""
from std.gpu.host import DeviceContext
from std.math import exp, log, sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true
from std.utils.numerics import neg_inf
from mograd import Device, Tensor
from std.sys.info import has_nvidia_gpu_accelerator
from mograd.runtime.gpu.kernels.attention.generic import _flash_attn_fwd_launch, _flash_attn_bwd_launch
from mograd.runtime.gpu.kernels.attention.nvidia_fwd import _flash_attn_fwd_launch_mma
from mograd.runtime.gpu.kernels.attention.nvidia_bwd import _flash_attn_bwd_launch_mma, _flash_attn_bwd_launch_mma_half

# The MMA kernels require NVIDIA sm_80+ (TF32 m16n8k8, m16n8k16 HMMA,
# cp.async). Tests comptime-skip elsewhere so the file compiles anywhere.
comptime IS_NVIDIA = has_nvidia_gpu_accelerator()


# ===-------------------------------------------------------------------===#
# f64 CPU oracle + error helpers
# ===-------------------------------------------------------------------===#


def _to_f64[dtype: DType](vals: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(vals))
    for v in vals:
        out.append(Float64(v))
    return out^


def _ref_sdpa_f64(
    q: List[Float64],
    k: List[Float64],
    v: List[Float64],
    mask: List[Float64],
    has_mask: Bool,
    causal: Bool,
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float64,
) -> Tuple[List[Float64], List[Float64]]:
    """Naive O(S^2) attention in float64. Q/K/V are BSHD, output is BHSD,
    mask BHSS, LSE (B,H,S) in natural log, matching the kernel contract.
    Softmax is max-subtracted, so the oracle is stable at any logit scale."""
    var out = List[Float64](capacity=B * H * S * D)
    for _ in range(B * H * S * D):
        out.append(0.0)
    var lse = List[Float64](capacity=B * H * S)
    for _ in range(B * H * S):
        lse.append(0.0)
    var scores = List[Float64](capacity=S)
    for _ in range(S):
        scores.append(0.0)
    for b in range(B):
        for h in range(H):
            var obase = (b * H + h) * S * D
            var mbase = (b * H + h) * S * S
            for i in range(S):
                var j_end = (i + 1) if causal else S
                var row_max = -1e300
                for j in range(j_end):
                    var s = 0.0
                    for d in range(D):
                        # BSHD: element (b, s, h, d)
                        s += q[((b * S + i) * H + h) * D + d] * k[((b * S + j) * H + h) * D + d]
                    s *= scale
                    if has_mask:
                        s += mask[mbase + i * S + j]
                    scores[j] = s
                    if s > row_max:
                        row_max = s
                var denom = 0.0
                for j in range(j_end):
                    scores[j] = exp(scores[j] - row_max)
                    denom += scores[j]
                for d in range(D):
                    var acc = 0.0
                    for j in range(j_end):
                        acc += scores[j] * v[((b * S + j) * H + h) * D + d]
                    out[obase + i * D + d] = acc / denom
                lse[(b * H + h) * S + i] = row_max + log(denom)
    return (out^, lse^)


def _max_abs_err[dtype: DType](got: List[Scalar[dtype]], want: List[Float64]) raises -> Float64:
    assert_true(len(got) == len(want), "length mismatch vs oracle")
    var worst = 0.0
    for i in range(len(got)):
        var g = Float64(got[i])
        if g != g:  # NaN must fail loudly, not slip through comparisons
            return 1e300
        var e = g - want[i]
        e = e if e >= 0 else -e
        if e > worst:
            worst = e
    return worst


# ===-------------------------------------------------------------------===#
# Direct kernel launch: raw buffers in, (O, LSE) out. No graph, no routing.
# ===-------------------------------------------------------------------===#


def _run_fwd_kernel[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool, MMA: Bool
](
    q: List[Scalar[dtype]],
    k: List[Scalar[dtype]],
    v: List[Scalar[dtype]],
    mask: List[Scalar[dtype]],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
) raises -> Tuple[List[Scalar[dtype]], List[Float32]]:
    with DeviceContext() as ctx:
        var q_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var k_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var v_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var m_buf = ctx.enqueue_create_buffer[dtype](len(mask) if HAS_BIAS else 8)
        var o_buf = ctx.enqueue_create_buffer[dtype](B * H * S * D)
        var lse_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S)

        with q_buf.map_to_host() as host:
            for i in range(len(q)):
                host[i] = q[i]
        with k_buf.map_to_host() as host:
            for i in range(len(k)):
                host[i] = k[i]
        with v_buf.map_to_host() as host:
            for i in range(len(v)):
                host[i] = v[i]
        comptime if HAS_BIAS:
            with m_buf.map_to_host() as host:
                for i in range(len(mask)):
                    host[i] = mask[i]
        o_buf.enqueue_fill(Scalar[dtype](0))
        lse_buf.enqueue_fill(Float32(0))
        ctx.synchronize()

        comptime launch = _flash_attn_fwd_launch_mma[
            dtype, D_BUCKET, CAUSAL, HAS_BIAS
        ] if MMA else _flash_attn_fwd_launch[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
        launch(
            q_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            k_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            v_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            m_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            o_buf.unsafe_ptr().as_unsafe_any_origin(),
            lse_buf.unsafe_ptr().as_unsafe_any_origin(),
            B,
            S,
            H,
            D,
            scale,
            ctx,
        )
        ctx.synchronize()

        var o_list = List[Scalar[dtype]](capacity=B * H * S * D)
        with o_buf.map_to_host() as host:
            for i in range(B * H * S * D):
                o_list.append(host[i])
        var lse_list = List[Float32](capacity=B * H * S)
        with lse_buf.map_to_host() as host:
            for i in range(B * H * S):
                lse_list.append(host[i])
        return (o_list^, lse_list^)


# ===-------------------------------------------------------------------===#
# Shared driver: all errors vs the f64 oracle for one configuration
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct FwdErrs(Copyable, Movable):
    var mma: Float64
    var generic: Float64
    var naive: Float64
    var lse_mma: Float64
    var lse_generic: Float64


def _fwd_case[
    dtype: DType, CAUSAL: Bool, HAS_BIAS: Bool, D_BUCKET: Int
](B: Int, H: Int, S: Int, D: Int, seed: UInt32, std: Float32 = 1.0) raises -> FwdErrs:
    # Seeded BSHD randn leaves are the single source of truth. to_list()
    # extracts the exact bits every consumer sees.
    var device = Device()
    var Qt = Tensor.randn(device, (B, S, H, D), dtype=dtype, std=std, seed=seed)
    var Kt = Tensor.randn(device, (B, S, H, D), dtype=dtype, std=std, seed=seed + 1)
    var Vt = Tensor.randn(device, (B, S, H, D), dtype=dtype, seed=seed + 2)
    var Mt = Tensor.randn(device, (B, H, S, S), dtype=dtype, seed=seed + 3)
    var ql = Qt.to_list[dtype]()
    var kl = Kt.to_list[dtype]()
    var vl = Vt.to_list[dtype]()
    var ml = Mt.to_list[dtype]()
    var scale = Float32(1.0) / sqrt(Float32(D))

    var oracle = _ref_sdpa_f64(
        _to_f64(ql), _to_f64(kl), _to_f64(vl), _to_f64(ml), HAS_BIAS, CAUSAL, B, S, H, D, Float64(scale)
    )

    # Naive same-dtype baseline: unfused tensor graph, simplifier=False so
    # GPU rewrites cannot silently turn this into the kernel under test.
    var Qb = Qt.transpose(1, 2)
    var Kb = Kt.transpose(1, 2)
    var Vb = Vt.transpose(1, 2)
    var scores = (Qb @ Kb.transpose(-2, -1)) * scale
    comptime if CAUSAL:
        scores = scores + Tensor.full_like(scores, neg_inf[DType.float32]()).triu(1)
    comptime if HAS_BIAS:
        scores = scores + Mt
    var naive = scores.softmax() @ Vb
    var err_naive = _max_abs_err(naive.to_list[dtype](simplifier=False), oracle[0])

    # The kernels, called directly.
    var mma = _run_fwd_kernel[dtype, D_BUCKET, CAUSAL, HAS_BIAS, True](ql, kl, vl, ml, B, S, H, D, scale)
    var gen = _run_fwd_kernel[dtype, D_BUCKET, CAUSAL, HAS_BIAS, False](ql, kl, vl, ml, B, S, H, D, scale)
    var errs = FwdErrs(
        _max_abs_err(mma[0], oracle[0]),
        _max_abs_err(gen[0], oracle[0]),
        err_naive,
        _max_abs_err(mma[1], oracle[1]),
        _max_abs_err(gen[1], oracle[1]),
    )
    print(
        t"    err_mma={errs.mma}  err_generic={errs.generic}  err_naive={errs.naive}"
        t"  lse_mma={errs.lse_mma}  lse_generic={errs.lse_generic}"
    )
    return errs^


def _assert_gate(err: Float64, baseline: Float64, factor: Float64, floor: Float64, name: String) raises:
    if err > factor * baseline + floor:
        raise Error(t"accuracy gate failed [{name}]: err={err} > {factor} * baseline={baseline} + {floor}")


# ===-------------------------------------------------------------------===#
# Accuracy gates. S=96 is deliberately ragged (not a multiple of Br_MMA=64)
# and multi-tile. D=64 and D=128 hit both head-dim buckets.
# ===-------------------------------------------------------------------===#


def test_fwd_f16_full() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _fwd_case[DType.float16, False, False, 64](2, 2, 96, 64, 100)
    _assert_gate(e.mma, e.naive, 2.0, 1e-4, "f16 full mma")
    _assert_gate(e.generic, e.naive, 2.0, 1e-4, "f16 full generic")
    assert_true(e.lse_mma < 1e-4 and e.lse_generic < 1e-4, "f16 full LSE error")


def test_fwd_f16_causal() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _fwd_case[DType.float16, True, False, 64](2, 2, 96, 64, 110)
    _assert_gate(e.mma, e.naive, 2.0, 1e-4, "f16 causal mma")
    _assert_gate(e.generic, e.naive, 2.0, 1e-4, "f16 causal generic")
    assert_true(e.lse_mma < 1e-4 and e.lse_generic < 1e-4, "f16 causal LSE error")


def test_fwd_f16_bias() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _fwd_case[DType.float16, False, True, 64](2, 2, 96, 64, 120)
    _assert_gate(e.mma, e.naive, 2.0, 1e-4, "f16 bias mma")
    _assert_gate(e.generic, e.naive, 2.0, 1e-4, "f16 bias generic")


def test_fwd_f16_causal_d128() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # D=128 bucket: exercises the register-capped fwd-half MMA variant.
    var e = _fwd_case[DType.float16, True, False, 128](2, 2, 96, 128, 130)
    _assert_gate(e.mma, e.naive, 2.0, 1e-4, "f16 causal d128 mma")
    _assert_gate(e.generic, e.naive, 2.0, 1e-4, "f16 causal d128 generic")


def test_fwd_f16_large_s() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # S=768 pushes the generic launcher onto its large-grid geometry (GR=8,
    # 32-row blocks). The small shapes above all take the GR=4 branch, so
    # without this case GR=8 would be covered by history only. Causal is the
    # sharpest check because the diagonal straddles two 16-wide tiles per
    # block.
    var e = _fwd_case[DType.float16, False, False, 64](1, 4, 768, 64, 300)
    _assert_gate(e.mma, e.naive, 2.0, 1e-4, "f16 large-S full mma")
    _assert_gate(e.generic, e.naive, 2.0, 1e-4, "f16 large-S full generic")
    var ec = _fwd_case[DType.float16, True, False, 64](1, 4, 768, 64, 310)
    _assert_gate(ec.mma, ec.naive, 2.0, 1e-4, "f16 large-S causal mma")
    _assert_gate(ec.generic, ec.naive, 2.0, 1e-4, "f16 large-S causal generic")


def test_fwd_f32_full() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # MMA f32 means TF32 matmuls (~1e-3-class), so it gets an absolute cap
    # rather than a naive-relative one. The generic SIMT kernel is true f32
    # FMA and must stay near the naive baseline (~1e-6-class), a tight
    # routing-independent check.
    var e = _fwd_case[DType.float32, False, False, 64](2, 2, 96, 64, 140)
    assert_true(e.mma < 5e-3, "f32(TF32) full mma error above TF32-class cap")
    _assert_gate(e.generic, e.naive, 4.0, 1e-7, "f32 full generic")


def test_fwd_f32_causal() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _fwd_case[DType.float32, True, False, 64](2, 2, 96, 64, 150)
    assert_true(e.mma < 5e-3, "f32(TF32) causal mma error above TF32-class cap")
    _assert_gate(e.generic, e.naive, 4.0, 1e-7, "f32 causal generic")


def test_fwd_f32_bias() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _fwd_case[DType.float32, False, True, 64](2, 2, 96, 64, 160)
    assert_true(e.mma < 5e-3, "f32(TF32) bias mma error above TF32-class cap")
    _assert_gate(e.generic, e.naive, 4.0, 1e-7, "f32 bias generic")


# ===-------------------------------------------------------------------===#
# Stress + property tests
# ===-------------------------------------------------------------------===#


def test_fwd_f32_d256() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # At D=256 MMA routes to the TF32 kernel. Generic gets its only coverage
    # at this bucket here (no shipping workload uses it yet).
    var e = _fwd_case[DType.float32, False, False, 256](2, 2, 96, 256, 400)
    assert_true(e.mma < 2e-2, "f32(TF32) d256 fwd mma error")
    _assert_gate(e.generic, e.naive, 4.0, 1e-6, "f32 d256 fwd generic")


def test_bwd_f32_full_d256() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _bwd_case[DType.float32, False, False, 256](2, 2, 96, 256, 410)
    assert_true(e.dq_mma < 5e-2 and e.dk_mma < 5e-2 and e.dv_mma < 5e-2, "f32(TF32) d256 bwd mma error")
    assert_true(e.dq_gen < 2e-5 and e.dk_gen < 2e-5 and e.dv_gen < 2e-5, "f32 d256 bwd generic error")


def test_d512_generic_only() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # D > 256 has no MMA kernels (dispatch routes it to generic), so this
    # gates the generic launchers directly at the 512 bucket, the only
    # coverage that bucket has.
    var device = Device()
    var B = 2
    var H = 2
    var S = 96
    var D = 512
    var Qt = Tensor.randn(device, (B, S, H, D), seed=500)
    var Kt = Tensor.randn(device, (B, S, H, D), seed=501)
    var Vt = Tensor.randn(device, (B, S, H, D), seed=502)
    var Dyt = Tensor.randn(device, (B, H, S, D), seed=503)
    var ql = Qt.to_list[DType.float32]()
    var kl = Kt.to_list[DType.float32]()
    var vl = Vt.to_list[DType.float32]()
    var dyl = Dyt.to_list[DType.float32]()
    var ml = List[Scalar[DType.float32]]()
    var scale = Float32(1.0) / sqrt(Float32(D))
    var oracle = _ref_sdpa_f64(
        _to_f64(ql), _to_f64(kl), _to_f64(vl), List[Float64](), False, False, B, S, H, D, Float64(scale)
    )
    var fwd = _run_fwd_kernel[DType.float32, 512, False, False, False](ql, kl, vl, ml, B, S, H, D, scale)
    var err_o = _max_abs_err(fwd[0], oracle[0])
    var err_lse = _max_abs_err(fwd[1], oracle[1])
    var grads = _ref_sdpa_bwd_f64(
        _to_f64(ql), _to_f64(kl), _to_f64(vl), _to_f64(dyl), List[Float64](), False, False, B, S, H, D, Float64(scale)
    )
    var bwd = _run_bwd_kernel[DType.float32, 512, False, False, False](ql, kl, vl, ml, dyl, B, S, H, D, scale)
    var err_dq = _max_abs_err(bwd[0], grads[0])
    var err_dk = _max_abs_err(bwd[1], grads[1])
    var err_dv = _max_abs_err(bwd[2], grads[2])
    print(t"    d512 generic: o={err_o} lse={err_lse} dq={err_dq} dk={err_dk} dv={err_dv}")
    assert_true(err_o < 1e-5 and err_lse < 1e-5, "f32 d512 generic fwd error")
    assert_true(err_dq < 5e-5 and err_dk < 5e-5 and err_dv < 5e-5, "f32 d512 generic bwd error")


def test_fwd_large_logits_stable() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # std=20 pushes logits to ~O(100). An unstable softmax overflows exp
    # and returns NaN/Inf (caught as err=1e300), so the online-softmax max
    # tracking must keep error at the same magnitude as the unit-scale case.
    var e = _fwd_case[DType.float32, False, False, 64](2, 2, 96, 64, 170, std=20.0)
    # TF32 physics, measured 0.19. Logits reach ~|400|, TF32's 10-bit
    # mantissa gives logit error ~400 * 2^-11 ~= 0.2, and exp turns logit
    # error directly into probability-weight error. This asserts stability
    # (no NaN/Inf, bounded error) for the MMA path. True precision at large
    # logits requires the generic full-f32 kernel, gated tightly below
    # (measured 6.9e-5, matching 400 * 2^-24 amplified).
    assert_true(e.mma < 0.5, "large-logit f32 mma unstable (NaN/Inf or gross error)")
    assert_true(e.generic < 2e-4, "large-logit f32 generic inaccurate")


def test_fwd_deterministic_bitwise() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # The forward has no atomics, so two launches on identical inputs must
    # agree bit for bit. The backward's dQ is atomic-accumulated and
    # explicitly NOT bitwise reproducible, so do not add it here.
    var device = Device()
    var Qt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=190)
    var Kt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=191)
    var Vt = Tensor.randn(device, (2, 96, 2, 64), dtype=DType.float16, seed=192)
    var ql = Qt.to_list[DType.float16]()
    var kl = Kt.to_list[DType.float16]()
    var vl = Vt.to_list[DType.float16]()
    var ml = List[Scalar[DType.float16]]()
    var scale = Float32(1.0) / sqrt(Float32(64))
    var a = _run_fwd_kernel[DType.float16, 64, True, False, True](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    var b = _run_fwd_kernel[DType.float16, 64, True, False, True](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    for i in range(len(a[0])):
        assert_true(a[0][i] == b[0][i], "forward kernel is not bitwise deterministic")


def test_fwd_bias_shift_invariance() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # softmax(s + c) == softmax(s): adding a constant to every bias entry
    # must leave O unchanged. Catches max-tracking bugs, no oracle needed.
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
    # On the generic true-f32 kernel the property must hold tightly. This
    # is the sharp algorithmic check on the online-softmax max tracking.
    var g1 = _run_fwd_kernel[DType.float32, 64, False, True, False](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    var g2 = _run_fwd_kernel[DType.float32, 64, False, True, False](ql, kl, vl, ml_shift, 2, 96, 2, 64, scale)
    var gmax = 0.0
    for i in range(len(g1[0])):
        var d = Float64(g1[0][i]) - Float64(g2[0][i])
        d = d if d >= 0 else -d
        if d > gmax:
            gmax = d
    print(t"    shift-invariance max deviation: generic = {gmax}")
    if gmax > 1e-5:
        raise Error(t"generic bias shift invariance violated: {gmax} > 1e-5")

    # MMA kernel: P@V re-quantizes the attention weights to TF32, so the two
    # runs' roundings decorrelate at ~sqrt(S) * P * 2^-11 ~= 1e-4 (measured
    # 1.6e-4). The property holds only to tensor-core precision here.
    var o1 = _run_fwd_kernel[DType.float32, 64, False, True, True](ql, kl, vl, ml, 2, 96, 2, 64, scale)
    var o2 = _run_fwd_kernel[DType.float32, 64, False, True, True](ql, kl, vl, ml_shift, 2, 96, 2, 64, scale)
    var dmax = 0.0
    for i in range(len(o1[0])):
        var d = Float64(o1[0][i]) - Float64(o2[0][i])
        d = d if d >= 0 else -d
        if d > dmax:
            dmax = d
    print(t"    shift-invariance max deviation: mma(TF32) = {dmax}")
    if dmax > 1e-3:
        raise Error(t"mma bias shift invariance violated: {dmax} > 1e-3")


# ===-------------------------------------------------------------------===#
# Backward: f64 oracle gradients + direct kernel calls
# ===-------------------------------------------------------------------===#


def _ref_sdpa_bwd_f64(
    q: List[Float64],
    k: List[Float64],
    v: List[Float64],
    dy: List[Float64],
    mask: List[Float64],
    has_mask: Bool,
    causal: Bool,
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float64,
) -> Tuple[List[Float64], List[Float64], List[Float64]]:
    """f64 CPU gradients (dQ, dK, dV), all BSHD; dy is BHSD (dO). Standard
    flash-attention backward math: dV = P^T dO, dS = P o (dP - delta) * scale
    with dP = dO V^T and delta_i = dO_i . O_i, then dQ = dS K, dK = dS^T Q."""
    var dq = List[Float64](capacity=B * S * H * D)
    var dk = List[Float64](capacity=B * S * H * D)
    var dv = List[Float64](capacity=B * S * H * D)
    for _ in range(B * S * H * D):
        dq.append(0.0)
        dk.append(0.0)
        dv.append(0.0)
    var p = List[Float64](capacity=S)
    for _ in range(S):
        p.append(0.0)
    for b in range(B):
        for h in range(H):
            var mbase = (b * H + h) * S * S
            var dybase = (b * H + h) * S * D
            for i in range(S):
                var j_end = (i + 1) if causal else S
                var row_max = -1e300
                for j in range(j_end):
                    var s = 0.0
                    for d in range(D):
                        s += q[((b * S + i) * H + h) * D + d] * k[((b * S + j) * H + h) * D + d]
                    s *= scale
                    if has_mask:
                        s += mask[mbase + i * S + j]
                    p[j] = s
                    if s > row_max:
                        row_max = s
                var denom = 0.0
                for j in range(j_end):
                    p[j] = exp(p[j] - row_max)
                    denom += p[j]
                for j in range(j_end):
                    p[j] /= denom
                # delta_i = dO_i . O_i with O_i = sum_j P_ij V_j
                var delta = 0.0
                for d in range(D):
                    var o_d = 0.0
                    for j in range(j_end):
                        o_d += p[j] * v[((b * S + j) * H + h) * D + d]
                    delta += dy[dybase + i * D + d] * o_d
                for j in range(j_end):
                    var dp = 0.0
                    for d in range(D):
                        dp += dy[dybase + i * D + d] * v[((b * S + j) * H + h) * D + d]
                    var ds = p[j] * (dp - delta) * scale
                    for d in range(D):
                        dq[((b * S + i) * H + h) * D + d] += ds * k[((b * S + j) * H + h) * D + d]
                        dk[((b * S + j) * H + h) * D + d] += ds * q[((b * S + i) * H + h) * D + d]
                        dv[((b * S + j) * H + h) * D + d] += p[j] * dy[dybase + i * D + d]
    return (dq^, dk^, dv^)


def _run_bwd_kernel[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool, MMA: Bool
](
    q: List[Scalar[dtype]],
    k: List[Scalar[dtype]],
    v: List[Scalar[dtype]],
    mask: List[Scalar[dtype]],
    dy: List[Scalar[dtype]],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
) raises -> Tuple[List[Scalar[dtype]], List[Scalar[dtype]], List[Scalar[dtype]]]:
    """Same-path forward (for O/LSE) chained into the backward kernel, all
    called directly with raw buffers. Returns (dQ, dK, dV) as BSHD lists."""
    with DeviceContext() as ctx:
        var q_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var k_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var v_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var m_buf = ctx.enqueue_create_buffer[dtype](len(mask) if HAS_BIAS else 8)
        var o_buf = ctx.enqueue_create_buffer[dtype](B * H * S * D)
        var lse_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S)
        var dy_buf = ctx.enqueue_create_buffer[dtype](B * H * S * D)
        var dq_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var dk_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var dv_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)

        with q_buf.map_to_host() as host:
            for i in range(len(q)):
                host[i] = q[i]
        with k_buf.map_to_host() as host:
            for i in range(len(k)):
                host[i] = k[i]
        with v_buf.map_to_host() as host:
            for i in range(len(v)):
                host[i] = v[i]
        with dy_buf.map_to_host() as host:
            for i in range(len(dy)):
                host[i] = dy[i]
        comptime if HAS_BIAS:
            with m_buf.map_to_host() as host:
                for i in range(len(mask)):
                    host[i] = mask[i]
        o_buf.enqueue_fill(Scalar[dtype](0))
        lse_buf.enqueue_fill(Float32(0))
        dq_buf.enqueue_fill(Scalar[dtype](0))
        dk_buf.enqueue_fill(Scalar[dtype](0))
        dv_buf.enqueue_fill(Scalar[dtype](0))
        ctx.synchronize()

        comptime fwd = _flash_attn_fwd_launch_mma[dtype, D_BUCKET, CAUSAL, HAS_BIAS] if MMA else _flash_attn_fwd_launch[
            dtype, D_BUCKET, CAUSAL, HAS_BIAS
        ]
        fwd(
            q_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            k_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            v_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            m_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            o_buf.unsafe_ptr().as_unsafe_any_origin(),
            lse_buf.unsafe_ptr().as_unsafe_any_origin(),
            B,
            S,
            H,
            D,
            scale,
            ctx,
        )
        ctx.synchronize()

        comptime bwd = _flash_attn_bwd_launch_mma_half[
            dtype, D_BUCKET, CAUSAL, HAS_BIAS
        ] if MMA and dtype.is_half_float() and D_BUCKET <= 128 else (
            _flash_attn_bwd_launch_mma[dtype, D_BUCKET, CAUSAL, HAS_BIAS] if MMA else _flash_attn_bwd_launch[
                dtype, D_BUCKET, CAUSAL, HAS_BIAS
            ]
        )
        bwd(
            dy_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            o_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            q_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            k_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            v_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            m_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            lse_buf.unsafe_ptr().as_immutable().as_unsafe_any_origin(),
            dq_buf.unsafe_ptr().as_unsafe_any_origin(),
            dk_buf.unsafe_ptr().as_unsafe_any_origin(),
            dv_buf.unsafe_ptr().as_unsafe_any_origin(),
            B,
            S,
            H,
            D,
            scale,
            ctx,
        )
        ctx.synchronize()

        var dq_l = List[Scalar[dtype]](capacity=B * S * H * D)
        with dq_buf.map_to_host() as host:
            for i in range(B * S * H * D):
                dq_l.append(host[i])
        var dk_l = List[Scalar[dtype]](capacity=B * S * H * D)
        with dk_buf.map_to_host() as host:
            for i in range(B * S * H * D):
                dk_l.append(host[i])
        var dv_l = List[Scalar[dtype]](capacity=B * S * H * D)
        with dv_buf.map_to_host() as host:
            for i in range(B * S * H * D):
                dv_l.append(host[i])
        return (dq_l^, dk_l^, dv_l^)


@fieldwise_init
struct BwdErrs(Copyable, Movable):
    var dq_mma: Float64
    var dk_mma: Float64
    var dv_mma: Float64
    var dq_gen: Float64
    var dk_gen: Float64
    var dv_gen: Float64


def _bwd_case[
    dtype: DType, CAUSAL: Bool, HAS_BIAS: Bool, D_BUCKET: Int
](B: Int, H: Int, S: Int, D: Int, seed: UInt32) raises -> BwdErrs:
    var device = Device()
    var Qt = Tensor.randn(device, (B, S, H, D), dtype=dtype, seed=seed)
    var Kt = Tensor.randn(device, (B, S, H, D), dtype=dtype, seed=seed + 1)
    var Vt = Tensor.randn(device, (B, S, H, D), dtype=dtype, seed=seed + 2)
    var Mt = Tensor.randn(device, (B, H, S, S), dtype=dtype, seed=seed + 3)
    var Dyt = Tensor.randn(device, (B, H, S, D), dtype=dtype, seed=seed + 4)
    var ql = Qt.to_list[dtype]()
    var kl = Kt.to_list[dtype]()
    var vl = Vt.to_list[dtype]()
    var ml = Mt.to_list[dtype]()
    var dyl = Dyt.to_list[dtype]()
    var scale = Float32(1.0) / sqrt(Float32(D))

    var oracle = _ref_sdpa_bwd_f64(
        _to_f64(ql), _to_f64(kl), _to_f64(vl), _to_f64(dyl), _to_f64(ml), HAS_BIAS, CAUSAL, B, S, H, D, Float64(scale)
    )
    var mma = _run_bwd_kernel[dtype, D_BUCKET, CAUSAL, HAS_BIAS, True](ql, kl, vl, ml, dyl, B, S, H, D, scale)
    var gen = _run_bwd_kernel[dtype, D_BUCKET, CAUSAL, HAS_BIAS, False](ql, kl, vl, ml, dyl, B, S, H, D, scale)
    var errs = BwdErrs(
        _max_abs_err(mma[0], oracle[0]),
        _max_abs_err(mma[1], oracle[1]),
        _max_abs_err(mma[2], oracle[2]),
        _max_abs_err(gen[0], oracle[0]),
        _max_abs_err(gen[1], oracle[1]),
        _max_abs_err(gen[2], oracle[2]),
    )
    print(
        t"    dq_mma={errs.dq_mma}  dk_mma={errs.dk_mma}  dv_mma={errs.dv_mma}"
        t"  dq_gen={errs.dq_gen}  dk_gen={errs.dk_gen}  dv_gen={errs.dv_gen}"
    )
    return errs^


def test_bwd_f16_causal_d64() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # Calibrated 2026-07-07: mma <= 1.65e-3 (f16 P/dS casts), gen <= 1.0e-3.
    var e = _bwd_case[DType.float16, True, False, 64](2, 2, 96, 64, 200)
    assert_true(e.dq_mma < 8e-3 and e.dk_mma < 8e-3 and e.dv_mma < 8e-3, "f16 causal bwd mma error")
    assert_true(e.dq_gen < 5e-3 and e.dk_gen < 5e-3 and e.dv_gen < 5e-3, "f16 causal bwd generic error")


def test_bwd_f16_causal_d128() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _bwd_case[DType.float16, True, False, 128](2, 2, 96, 128, 210)
    assert_true(e.dq_mma < 8e-3 and e.dk_mma < 8e-3 and e.dv_mma < 8e-3, "f16 causal d128 bwd mma error")
    assert_true(e.dq_gen < 5e-3 and e.dk_gen < 5e-3 and e.dv_gen < 5e-3, "f16 causal d128 bwd generic error")


def test_bwd_f16_bias_d64() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    var e = _bwd_case[DType.float16, False, True, 64](2, 2, 96, 64, 220)
    assert_true(e.dq_mma < 8e-3 and e.dk_mma < 8e-3 and e.dv_mma < 8e-3, "f16 bias bwd mma error")
    assert_true(e.dq_gen < 5e-3 and e.dk_gen < 5e-3 and e.dv_gen < 5e-3, "f16 bias bwd generic error")


def test_bwd_f32_full_d64() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # f32 routes to the TF32 split path while generic is true-f32 FMA, so
    # the generic gate is the tight one at 4 orders below the TF32 path
    # (calibrated at 3.1e-3 for TF32 and 6.7e-7 for generic).
    var e = _bwd_case[DType.float32, False, False, 64](2, 2, 96, 64, 230)
    assert_true(e.dq_mma < 1.5e-2 and e.dk_mma < 1.5e-2 and e.dv_mma < 1.5e-2, "f32(TF32) full bwd mma error")
    assert_true(e.dq_gen < 5e-6 and e.dk_gen < 5e-6 and e.dv_gen < 5e-6, "f32 full bwd generic error")


def test_bwd_dq_atomic_spread() raises:
    comptime if not IS_NVIDIA:
        print("    skipped: requires NVIDIA sm_80+ (MMA kernels)")
        return
    # dQ is accumulated with red.global.add, whose ordering is not
    # guaranteed, so dQ reproducibility is bounded rather than asserted
    # bitwise. The spread measures 0.0 back to back because the f32
    # accumulator and f16 convert absorb small order variation, but the
    # semantics permit nonzero. dK/dV have no atomics and ARE asserted
    # bitwise.
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
    var r1 = _run_bwd_kernel[DType.float16, 64, True, False, True](ql, kl, vl, ml, dyl, 2, 96, 2, 64, scale)
    var r2 = _run_bwd_kernel[DType.float16, 64, True, False, True](ql, kl, vl, ml, dyl, 2, 96, 2, 64, scale)
    var dq_spread = 0.0
    for i in range(len(r1[0])):
        var d = Float64(r1[0][i]) - Float64(r2[0][i])
        d = d if d >= 0 else -d
        if d > dq_spread:
            dq_spread = d
    print(t"    dq run-to-run atomic spread = {dq_spread}")
    assert_true(dq_spread < 1e-2, "dq atomic-order spread unexpectedly large")
    for i in range(len(r1[1])):
        assert_true(r1[1][i] == r2[1][i], "dk must be bitwise deterministic (no atomics)")
        assert_true(r1[2][i] == r2[2][i], "dv must be bitwise deterministic (no atomics)")


def main() raises:
    comptime assert has_accelerator(), "GPU required to run kernel accuracy tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
