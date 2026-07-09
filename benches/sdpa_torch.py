# Reference numbers: PyTorch SDPA on the same shape grid and row axis as
# benches/attention.mojo (causal / full / bias). Run with a torch+CUDA env:
#   ~/miniconda3/envs/torch-pr/bin/python benches/sdpa_torch.py
#
# Conventions match benches/attention.mojo: FA2 flop counting (fwd = 2
# matmuls, bwd = 5 matmuls = 2.5x fwd, and causal halves both).
#
# Backends: causal and full run the FLASH_ATTENTION backend (vendored FA2).
# The flash backend rejects attn_mask, so bias rows run torch's best
# available path for additive masks, EFFICIENT_ATTENTION (cutlass
# memory-efficient attention). The backend is part of the row so the
# comparison stays honest.
#
# Cache busting: mirrors internal_utils.CacheBustingBuffer used by mograd's
# benches/attention.mojo and by modular's own bench_mha.mojo. Each iteration
# reads a fresh q/k/v copy from a pool sized past 2x the GPU L2, so numbers
# reflect cold-cache bandwidth instead of L2-resident reuse. `mask` is left
# unbusted to match the mograd bench (which pins it at offset 0). Set
# CACHE_BUST_BYTES = 0 to fall back to the old warm-cache behavior.
import math

import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

torch.manual_seed(0)

# 512 MiB per tensor, matching modular's internal_utils default. Exceeds 2x
# the L2 on every current NVIDIA GPU (4090=72MB, H100=50MB, A100=40MB).
CACHE_BUST_BYTES = 512 * 1024 * 1024

SHAPES = [
    (1, 256, 8, 64),
    (1, 1024, 8, 64),
    (4, 512, 12, 64),
    (8, 1024, 16, 64),
    (8, 256, 16, 128),
    (8, 512, 16, 128),
    (8, 1024, 16, 128),
]

MODES = [
    ("causal", SDPBackend.FLASH_ATTENTION, "flash"),
    ("full", SDPBackend.FLASH_ATTENTION, "flash"),
    ("bias", SDPBackend.EFFICIENT_ATTENTION, "mem_eff"),
]


def _num_copies(tensor_bytes):
    """Copies needed so the rotation pool exceeds the cache-bust budget."""
    if CACHE_BUST_BYTES <= 0 or tensor_bytes <= 0:
        return 1
    return max(1, math.ceil(CACHE_BUST_BYTES / tensor_bytes))


def bench(B, S, H, D, mode, backend, backend_name, iters=1000):
    is_causal = mode == "causal"

    # Rotate q/k/v through `ncopies` distinct allocations. tensor_bytes is one
    # fp16 (B,H,S,D) copy. The pool is ncopies * that, sized past 2x L2.
    tensor_bytes = B * H * S * D * 2
    ncopies = _num_copies(tensor_bytes)

    def make_qkv():
        q = torch.randn(B, H, S, D, dtype=torch.float16, device="cuda", requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        return q, k, v

    qkvs = [make_qkv() for _ in range(ncopies)]

    # mask is unbusted (single copy at "offset 0"), matching the mograd bench.
    mask = None
    if mode == "bias":
        mask = torch.randn(B, H, S, S, dtype=torch.float16, device="cuda")

    def call(i):
        q, k, v = qkvs[i % ncopies]
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=is_causal)

    with sdpa_kernel([backend]):
        for i in range(30):
            o = call(i)
            torch.autograd.grad(o.sum(), qkvs[i % ncopies])
        torch.cuda.synchronize()

        s0, e0 = torch.cuda.Event(True), torch.cuda.Event(True)
        s0.record()
        for i in range(iters):
            o = call(i)
        e0.record()
        torch.cuda.synchronize()
        fwd_ms = s0.elapsed_time(e0) / iters

        s1, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        s1.record()
        for i in range(iters):
            o = call(i)
            torch.autograd.grad(o.sum(), qkvs[i % ncopies])
        e1.record()
        torch.cuda.synchronize()
        bwd_ms = s1.elapsed_time(e1) / iters - fwd_ms

    scale = 2 if is_causal else 4
    fwd_tf = scale * B * H * S * S * D / (fwd_ms * 1e9)
    bwd_tf = 2.5 * scale * B * H * S * S * D / (bwd_ms * 1e9)
    print(
        f"| sdpa_torch/{mode}/B{B}S{S}H{H}D{D} ({backend_name})"
        f" | fwd {fwd_ms:8.4f} ms ({fwd_tf:6.1f} TF)"
        f" | bwd {bwd_ms:8.4f} ms ({bwd_tf:6.1f} TF) | ratio {bwd_ms / fwd_ms:.2f}"
        f" | copies {ncopies} |"
    )


if __name__ == "__main__":
    assert torch.cuda.is_available()
    bust = "off" if CACHE_BUST_BYTES <= 0 else f"{CACHE_BUST_BYTES // (1024 * 1024)} MiB/tensor"
    print(f"# torch {torch.__version__} on {torch.cuda.get_device_name(0)} | cache-bust {bust}")
    for mode, backend, backend_name in MODES:
        for B, S, H, D in SHAPES:
            try:
                bench(B, S, H, D, mode, backend, backend_name)
            except RuntimeError as e:
                print(f"| sdpa_torch/{mode}/B{B}S{S}H{H}D{D} ({backend_name}) | UNSUPPORTED: {str(e)[:60]} |")
