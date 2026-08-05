# Shared accuracy harness for the flash-attention kernels.
#
# Layouts: Q/K/V and dQ/dK/dV are BSHD, O and dO are BHSD, mask BHSS,
# LSE (B, H, S) float32 in natural log.
from std.gpu.host import DeviceBuffer, DeviceContext
from std.math import exp, log, sqrt
from std.testing import assert_true
from std.utils.numerics import neg_inf
from mograd import Device, Tensor

comptime FwdLaunch[dtype: DType] = def(
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], MutAnyOrigin],
    UnsafePointer[Float32, MutAnyOrigin],
    Int,
    Int,
    Int,
    Int,
    Float32,
    DeviceContext,
) raises capturing

comptime BwdLaunch[dtype: DType] = def(
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    UnsafePointer[Float32, ImmutAnyOrigin],
    UnsafePointer[Scalar[dtype], MutAnyOrigin],
    UnsafePointer[Scalar[dtype], MutAnyOrigin],
    UnsafePointer[Scalar[dtype], MutAnyOrigin],
    Int,
    Int,
    Int,
    Int,
    Float32,
    DeviceContext,
) raises capturing


# ===-------------------------------------------------------------------===#
# f64 CPU oracle + error helpers
# ===-------------------------------------------------------------------===#


def to_f64[dtype: DType](vals: List[Scalar[dtype]]) -> List[Float64]:
    var out = List[Float64](capacity=len(vals))
    for v in vals:
        out.append(Float64(v))
    return out^


def ref_sdpa_f64(
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
    """Naive O(S^2) attention in float64. Output is BHSD, LSE natural log.
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


def ref_sdpa_bwd_f64(
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


def max_abs_err[dtype: DType](got: List[Scalar[dtype]], want: List[Float64]) raises -> Float64:
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


def assert_gate(err: Float64, baseline: Float64, factor: Float64, floor: Float64, name: String) raises:
    if err > factor * baseline + floor:
        raise Error(t"accuracy gate failed [{name}]: err={err} > {factor} * baseline={baseline} + {floor}")


# ===-------------------------------------------------------------------===#
# Direct kernel launch: raw buffers in, results out
# ===-------------------------------------------------------------------===#


def _fill[dtype: DType](buf: DeviceBuffer[dtype], vals: List[Scalar[dtype]]) raises:
    with buf.map_to_host() as host:
        for i in range(len(vals)):
            host[i] = vals[i]


def _read_back[dtype: DType](buf: DeviceBuffer[dtype], n: Int) raises -> List[Scalar[dtype]]:
    var out = List[Scalar[dtype]](capacity=n)
    with buf.map_to_host() as host:
        for i in range(n):
            out.append(host[i])
    return out^


def run_fwd[
    dtype: DType, HAS_BIAS: Bool, launch: FwdLaunch[dtype]
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

        _fill(q_buf, q)
        _fill(k_buf, k)
        _fill(v_buf, v)
        comptime if HAS_BIAS:
            _fill(m_buf, mask)
        o_buf.enqueue_fill(Scalar[dtype](0))
        lse_buf.enqueue_fill(Float32(0))
        ctx.synchronize()

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
        return (_read_back(o_buf, B * H * S * D), _read_back(lse_buf, B * H * S))


def run_bwd[
    dtype: DType, HAS_BIAS: Bool, fwd_launch: FwdLaunch[dtype], bwd_launch: BwdLaunch[dtype]
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
    """Same-path forward (for O/LSE) chained into the backward kernel."""
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

        _fill(q_buf, q)
        _fill(k_buf, k)
        _fill(v_buf, v)
        _fill(dy_buf, dy)
        comptime if HAS_BIAS:
            _fill(m_buf, mask)
        o_buf.enqueue_fill(Scalar[dtype](0))
        lse_buf.enqueue_fill(Float32(0))
        dq_buf.enqueue_fill(Scalar[dtype](0))
        dk_buf.enqueue_fill(Scalar[dtype](0))
        dv_buf.enqueue_fill(Scalar[dtype](0))
        ctx.synchronize()

        fwd_launch(
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
        bwd_launch(
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
        return (
            _read_back(dq_buf, B * S * H * D),
            _read_back(dk_buf, B * S * H * D),
            _read_back(dv_buf, B * S * H * D),
        )


# ===-------------------------------------------------------------------===#
# Cases: seeded inputs, oracle, kernel, max errors
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct FwdErrs(Copyable, Movable):
    var o: Float64
    var lse: Float64
    var naive: Float64


@fieldwise_init
struct BwdErrs(Copyable, Movable):
    var dq: Float64
    var dk: Float64
    var dv: Float64


def fwd_case[
    dtype: DType, CAUSAL: Bool, HAS_BIAS: Bool, launch: FwdLaunch[dtype], WITH_NAIVE: Bool = False
](B: Int, H: Int, S: Int, D: Int, seed: Int, std: Float32 = 1.0) raises -> FwdErrs:
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

    var oracle = ref_sdpa_f64(
        to_f64(ql), to_f64(kl), to_f64(vl), to_f64(ml), HAS_BIAS, CAUSAL, B, S, H, D, Float64(scale)
    )

    # Same-dtype baseline: unfused tensor graph, simplifier=False so GPU
    # rewrites cannot silently turn this into the kernel under test.
    var err_naive = Float64(0)
    comptime if WITH_NAIVE:
        var Qb = Qt.transpose(1, 2)
        var Kb = Kt.transpose(1, 2)
        var Vb = Vt.transpose(1, 2)
        var scores = (Qb @ Kb.transpose(-2, -1)) * scale
        comptime if CAUSAL:
            scores = scores + Tensor.full_like(scores, neg_inf[DType.float32]()).triu(1)
        comptime if HAS_BIAS:
            scores = scores + Mt
        var naive = scores.softmax() @ Vb
        err_naive = max_abs_err(naive.to_list[dtype](simplifier=False), oracle[0])

    var got = run_fwd[dtype, HAS_BIAS, launch](ql, kl, vl, ml, B, S, H, D, scale)
    var errs = FwdErrs(max_abs_err(got[0], oracle[0]), max_abs_err(got[1], oracle[1]), err_naive)
    print(t"    err_o={errs.o}  err_lse={errs.lse}  err_naive={errs.naive}")
    return errs^


def bwd_case[
    dtype: DType, CAUSAL: Bool, HAS_BIAS: Bool, fwd_launch: FwdLaunch[dtype], bwd_launch: BwdLaunch[dtype]
](B: Int, H: Int, S: Int, D: Int, seed: Int) raises -> BwdErrs:
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

    var oracle = ref_sdpa_bwd_f64(
        to_f64(ql), to_f64(kl), to_f64(vl), to_f64(dyl), to_f64(ml), HAS_BIAS, CAUSAL, B, S, H, D, Float64(scale)
    )
    var got = run_bwd[dtype, HAS_BIAS, fwd_launch, bwd_launch](ql, kl, vl, ml, dyl, B, S, H, D, scale)
    var errs = BwdErrs(
        max_abs_err(got[0], oracle[0]),
        max_abs_err(got[1], oracle[1]),
        max_abs_err(got[2], oracle[2]),
    )
    print(t"    err_dq={errs.dq}  err_dk={errs.dk}  err_dv={errs.dv}")
    return errs^
