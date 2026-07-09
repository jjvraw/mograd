from std.gpu.host import DeviceContext
from std.math import exp, log, sqrt
from std.testing import TestSuite, assert_true
from mograd import Device, Tensor
from std.sys.info import has_apple_gpu_accelerator
from mograd.runtime.gpu.kernels.attention.apple import (
    _flash_attn_fwd_launch_apple,
    _flash_attn_bwd_launch_apple,
)

comptime IS_APPLE = has_apple_gpu_accelerator()


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
    """Naive O(S^2) attention in float64. Output is BHSD, LSE natural log."""
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
    """f64 CPU gradients (dQ, dK, dV), all BSHD, with dy (dO) BHSD."""
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
# Direct kernel launch: raw buffers in, results out. No graph, no routing.
# ===-------------------------------------------------------------------===#


def _run_fwd[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
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

        _flash_attn_fwd_launch_apple[dtype, D_BUCKET, CAUSAL, HAS_BIAS](
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


def _run_bwd[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
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
    with DeviceContext() as ctx:
        var q_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var k_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var v_buf = ctx.enqueue_create_buffer[dtype](B * S * H * D)
        var m_buf = ctx.enqueue_create_buffer[dtype](len(mask) if HAS_BIAS else 8)
        var dy_buf = ctx.enqueue_create_buffer[dtype](B * H * S * D)
        var o_buf = ctx.enqueue_create_buffer[dtype](B * H * S * D)
        var lse_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S)
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

        # The backward consumes the forward's O and LSE, produced by the
        # same Apple kernel under test.
        _flash_attn_fwd_launch_apple[dtype, D_BUCKET, CAUSAL, HAS_BIAS](
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
        _flash_attn_bwd_launch_apple[dtype, D_BUCKET, CAUSAL, HAS_BIAS](
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


# ===-------------------------------------------------------------------===#
# Cases
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct FwdErrs(Copyable, Movable):
    var o: Float64
    var lse: Float64


def _fwd_case[
    dtype: DType, CAUSAL: Bool, HAS_BIAS: Bool, D_BUCKET: Int
](B: Int, H: Int, S: Int, D: Int, seed: UInt32) raises -> FwdErrs:
    var device = Device()
    var Qt = Tensor.randn(device, (B, S, H, D), dtype=dtype, seed=seed)
    var Kt = Tensor.randn(device, (B, S, H, D), dtype=dtype, seed=seed + 1)
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
    var got = _run_fwd[dtype, D_BUCKET, CAUSAL, HAS_BIAS](ql, kl, vl, ml, B, S, H, D, scale)
    var errs = FwdErrs(_max_abs_err(got[0], oracle[0]), _max_abs_err(got[1], oracle[1]))
    print(t"    err_o={errs.o}  err_lse={errs.lse}")
    return errs^


@fieldwise_init
struct BwdErrs(Copyable, Movable):
    var dq: Float64
    var dk: Float64
    var dv: Float64


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
    var got = _run_bwd[dtype, D_BUCKET, CAUSAL, HAS_BIAS](ql, kl, vl, ml, dyl, B, S, H, D, scale)
    var errs = BwdErrs(
        _max_abs_err(got[0], oracle[0]),
        _max_abs_err(got[1], oracle[1]),
        _max_abs_err(got[2], oracle[2]),
    )
    print(t"    err_dq={errs.dq}  err_dk={errs.dk}  err_dv={errs.dv}")
    return errs^


# ===-------------------------------------------------------------------===#
# Forward
# ===-------------------------------------------------------------------===#


def test_fwd_f32_full() raises:
    var e = _fwd_case[DType.float32, False, False, 64](2, 2, 96, 64, 10)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 full fwd error")


def test_fwd_f32_causal() raises:
    var e = _fwd_case[DType.float32, True, False, 64](2, 2, 96, 64, 20)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 causal fwd error")


def test_fwd_f32_bias() raises:
    var e = _fwd_case[DType.float32, False, True, 64](2, 2, 96, 64, 30)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 bias fwd error")


def test_fwd_f32_ragged_d() raises:
    # D=48 in the 64 bucket exercises the d < D lane guards.
    var e = _fwd_case[DType.float32, True, False, 64](1, 2, 80, 48, 40)
    assert_true(e.o < 1e-4 and e.lse < 1e-4, "f32 ragged-D fwd error")


def test_fwd_f32_d256() raises:
    var e = _fwd_case[DType.float32, True, False, 256](1, 1, 64, 256, 50)
    assert_true(e.o < 5e-4 and e.lse < 5e-4, "f32 d256 fwd error")


def test_fwd_f16_causal() raises:
    var e = _fwd_case[DType.float16, True, False, 64](2, 2, 96, 64, 60)
    assert_true(e.o < 5e-3 and e.lse < 1e-3, "f16 causal fwd error")


# ===-------------------------------------------------------------------===#
# Backward
# ===-------------------------------------------------------------------===#


def test_bwd_f32_full() raises:
    var e = _bwd_case[DType.float32, False, False, 64](2, 2, 96, 64, 110)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 full bwd error")


def test_bwd_f32_causal() raises:
    var e = _bwd_case[DType.float32, True, False, 64](2, 2, 96, 64, 120)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 causal bwd error")


def test_bwd_f32_bias() raises:
    var e = _bwd_case[DType.float32, False, True, 64](2, 2, 96, 64, 130)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 bias bwd error")


def test_bwd_f32_multi_tile() raises:
    # S=128 spans multiple APPLE_QROWS tiles.
    var e = _bwd_case[DType.float32, True, False, 32](1, 1, 128, 8, 140)
    assert_true(e.dq < 1e-3 and e.dk < 1e-3 and e.dv < 1e-3, "f32 multi-tile bwd error")


def test_bwd_f16_causal() raises:
    var e = _bwd_case[DType.float16, True, False, 64](2, 2, 96, 64, 150)
    assert_true(e.dq < 2e-2 and e.dk < 2e-2 and e.dv < 2e-2, "f16 causal bwd error")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run(skip_all=not IS_APPLE)
