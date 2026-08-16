"""Host-side dispatch: dtype / head-dim bucket / routing."""
from max.gpu.host import DeviceContext
from std.sys.info import has_apple_gpu_accelerator, has_nvidia_gpu_accelerator
from mograd.runtime.gpu.kernels.attention.generic import (
    _flash_attn_fwd_launch,
    _flash_attn_bwd_launch,
)

# TODO: AMD CDNA support.


def flash_attn_fwd[
    d: DType, CAUSAL: Bool = False, HAS_BIAS: Bool = True
](
    q: Pointer[Scalar[d], ImmutAnyOrigin],
    k: Pointer[Scalar[d], ImmutAnyOrigin],
    v: Pointer[Scalar[d], ImmutAnyOrigin],
    mask: Pointer[Scalar[d], ImmutAnyOrigin],
    dst: Pointer[Scalar[d], MutAnyOrigin],
    lse: Pointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    if D <= 0 or D > 512:
        raise Error("flash_attn_fwd: D=", D, " out of range (1..512)")
    # The MMA path needs sm_80 or newer (TF32 mma, m16n8k16 HMMA, cp.async).
    # Pre-Ampere NVIDIA takes the generic SIMT path. Half dtypes with
    # D <= 128 run the HMMA kernel and everything else runs TF32. Half
    # buckets are limited to 64 and 128 because the ldmatrix swizzle needs a
    # row of D_BUCKET halves to divide into or be a multiple of 128 bytes.
    comptime if has_nvidia_gpu_accelerator() and ctx.default_device_info.compute >= 8:
        from mograd.runtime.gpu.kernels.attention.nvidia_fwd import (
            _flash_attn_fwd_launch_mma,
        )

        @__parameter
        @always_inline
        def mma[DB: Int]() raises:
            _flash_attn_fwd_launch_mma[d, DB, CAUSAL, HAS_BIAS](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

        if D <= 64:
            mma[64]()
        elif D <= 128:
            mma[128]()
        elif D <= 256:
            mma[256]()
        else:
            # D > 256 exceeds the MMA kernels' register/smem envelope (the
            # same hdim <= 256 limit as FA2) and runs on the generic path.
            _flash_attn_fwd_launch[d, 512, CAUSAL, HAS_BIAS](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)
    elif has_apple_gpu_accelerator():
        # Metal: the smem-tiled SIMT kernels miscompile, and SMEM staging
        # measures as a net loss on Apple anyway (see apple.mojo). Route
        # to the register-streaming kernel. No shared memory means no
        # head-dim tile limit either.
        from mograd.runtime.gpu.kernels.attention.apple import _flash_attn_fwd_launch_apple

        @__parameter
        @always_inline
        def apple[DB: Int]() raises:
            _flash_attn_fwd_launch_apple[d, DB, CAUSAL, HAS_BIAS](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

        if D <= 32:
            apple[32]()
        elif D <= 64:
            apple[64]()
        elif D <= 96:
            apple[96]()
        elif D <= 128:
            apple[128]()
        elif D <= 256:
            apple[256]()
        else:
            apple[512]()
    else:
        # Scalar fallback
        @__parameter
        @always_inline
        def simt[DB: Int]() raises:
            _flash_attn_fwd_launch[d, DB, CAUSAL, HAS_BIAS](q, k, v, mask, dst, lse, B, S, H, D, scale, ctx)

        if D <= 32:
            simt[32]()
        elif D <= 64:
            simt[64]()
        elif D <= 96:
            simt[96]()
        elif D <= 128:
            simt[128]()
        elif D <= 256:
            simt[256]()
        else:
            simt[512]()


def flash_attn_bwd[
    d: DType, CAUSAL: Bool = False, HAS_BIAS: Bool = not CAUSAL
](
    dy: Pointer[Scalar[d], ImmutAnyOrigin],
    o: Pointer[Scalar[d], ImmutAnyOrigin],
    q: Pointer[Scalar[d], ImmutAnyOrigin],
    k: Pointer[Scalar[d], ImmutAnyOrigin],
    v: Pointer[Scalar[d], ImmutAnyOrigin],
    mask: Pointer[Scalar[d], ImmutAnyOrigin],
    lse: Pointer[Float32, ImmutAnyOrigin],
    dq: Pointer[Scalar[d], MutAnyOrigin],
    dk: Pointer[Scalar[d], MutAnyOrigin],
    dv: Pointer[Scalar[d], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    # HAS_BIAS selects whether the non-causal kernels read `mask` as an
    # additive bias of shape (B, H, S, S). Causal kernels never read it
    # (has_bias implies not is_causal, enforced at the op layer).
    comptime assert not (CAUSAL and HAS_BIAS), "causal attention takes no bias mask"
    if D <= 0 or D > 512:
        raise Error("flash_attn_bwd: D=", D, " out of range (1..512)")
    # The MMA kernels need sm_80 or newer (TF32 mma, m16n8k16 HMMA, cp.async).
    comptime if has_nvidia_gpu_accelerator() and ctx.default_device_info.compute >= 8:
        from mograd.runtime.gpu.kernels.attention.nvidia_bwd import (
            _flash_attn_bwd_launch_mma,
            _flash_attn_bwd_launch_mma_half,
        )

        @__parameter
        @always_inline
        def mma_half[DB: Int]() raises:
            _flash_attn_bwd_launch_mma_half[d, DB, CAUSAL, HAS_BIAS](
                dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
            )

        @__parameter
        @always_inline
        def mma[DB: Int]() raises:
            _flash_attn_bwd_launch_mma[d, DB, CAUSAL, HAS_BIAS](
                dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
            )

        comptime if d.is_half_float():
            # The FA2-shaped fused half kernel needs D <= 128 for resident
            # tiles. D > 128 falls back to the TF32 split path. Evict-first
            # mask staging makes fused the winner at every bias size.
            if D <= 64:
                mma_half[64]()
            elif D <= 128:
                mma_half[128]()
            elif D <= 256:
                mma[256]()
            else:
                _flash_attn_bwd_launch[d, 512, CAUSAL, HAS_BIAS](
                    dy,
                    o,
                    q,
                    k,
                    v,
                    mask,
                    lse,
                    dq,
                    dk,
                    dv,
                    B,
                    S,
                    H,
                    D,
                    scale,
                    ctx,
                )
        else:
            # TF32 split path (FA2 has no f32 backward to mirror).
            if D <= 64:
                mma[64]()
            elif D <= 128:
                mma[128]()
            elif D <= 256:
                mma[256]()
            else:
                _flash_attn_bwd_launch[d, 512, CAUSAL, HAS_BIAS](
                    dy,
                    o,
                    q,
                    k,
                    v,
                    mask,
                    lse,
                    dq,
                    dk,
                    dv,
                    B,
                    S,
                    H,
                    D,
                    scale,
                    ctx,
                )
    elif has_apple_gpu_accelerator():
        # Metal miscompiles the smem-staged SIMT kernels, so Apple runs the
        # register-streaming pair in apple.mojo. No shared memory means no
        # head-dim tile limit.
        from mograd.runtime.gpu.kernels.attention.apple import _flash_attn_bwd_launch_apple

        @__parameter
        @always_inline
        def apple[DB: Int]() raises:
            _flash_attn_bwd_launch_apple[d, DB, CAUSAL, HAS_BIAS](
                dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
            )

        if D <= 32:
            apple[32]()
        elif D <= 64:
            apple[64]()
        elif D <= 96:
            apple[96]()
        elif D <= 128:
            apple[128]()
        elif D <= 256:
            apple[256]()
        else:
            apple[512]()
    else:

        @__parameter
        @always_inline
        def simt[DB: Int]() raises:
            _flash_attn_bwd_launch[d, DB, CAUSAL, HAS_BIAS](
                dy, o, q, k, v, mask, lse, dq, dk, dv, B, S, H, D, scale, ctx
            )

        if D <= 32:
            simt[32]()
        elif D <= 64:
            simt[64]()
        elif D <= 96:
            simt[96]()
        elif D <= 128:
            simt[128]()
        elif D <= 256:
            simt[256]()
        else:
            simt[512]()
