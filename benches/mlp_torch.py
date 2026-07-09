import torch
import torch.utils.benchmark as benchmark
from dataclasses import dataclass


@dataclass
class Config:
    B: int
    D_in: int
    D_h: int
    D_out: int

    @property
    def fwd_flops(self) -> int:
        return 2 * self.B * self.D_in * self.D_h + 2 * self.B * self.D_h * self.D_out

    def label(self) -> str:
        return f"{self.B}x{self.D_in}->{self.D_h}->{self.D_out}"


def make_tensors(cfg: Config, device: torch.device):
    x  = torch.randn(cfg.B, cfg.D_in, device=device, dtype=torch.float32)
    w1 = torch.randn(cfg.D_in, cfg.D_h, device=device, dtype=torch.float32, requires_grad=True)
    w2 = torch.randn(cfg.D_h, cfg.D_out, device=device, dtype=torch.float32, requires_grad=True)
    return x, w1, w2


def mlp_forward(x, w1, w2):
    return (x @ w1).relu() @ w2


def mlp_forward_backward(x, w1, w2):
    out = (x @ w1).relu() @ w2
    out.sum().backward()
    return out


def bench(cfg: Config, device: torch.device, compiled: bool = False) -> None:
    x, w1, w2 = make_tensors(cfg, device)
    fn = torch.compile(mlp_forward) if compiled else mlp_forward
    mode = "compiled" if compiled else "eager"

    # Forward
    t_fwd = benchmark.Timer(
        stmt="fn(x, w1, w2)",
        globals={"fn": fn, "x": x, "w1": w1, "w2": w2},
        label=f"mlp_forward/{mode}",
        sub_label=cfg.label(),
        num_threads=1,
    ).blocked_autorange(min_run_time=5)

    ms = t_fwd.median * 1e3
    gflops = cfg.fwd_flops / (t_fwd.median * 1e9)
    print(f"mlp_forward/{mode:<8} {cfg.label():<26} {ms:.4f} ms   {gflops:.1f} GFLOPS/s")

    # Forward + backward (only float ops, not compiled backward for now)
    fn_bwd = torch.compile(mlp_forward_backward) if compiled else mlp_forward_backward
    t_bwd = benchmark.Timer(
        stmt="fn_bwd(x, w1, w2); w1.grad = None; w2.grad = None",
        globals={"fn_bwd": fn_bwd, "x": x, "w1": w1, "w2": w2},
        label=f"mlp_fwd_bwd/{mode}",
        sub_label=cfg.label(),
        num_threads=1,
    ).blocked_autorange(min_run_time=5)

    ms_bwd = t_bwd.median * 1e3
    gflops_bwd = (cfg.fwd_flops * 3) / (t_bwd.median * 1e9)
    print(f"mlp_fwd_bwd/{mode:<8} {cfg.label():<26} {ms_bwd:.4f} ms   {gflops_bwd:.1f} GFLOPS/s")


def main():
    device = torch.device("cuda")
    assert torch.cuda.is_available(), "CUDA required"

    configs = [
        Config(32,  768,  3072, 768),
        Config(64,  768,  3072, 768),
        Config(128, 768,  3072, 768),
        Config(256, 768,  3072, 768),
        Config(512, 768,  3072, 768),
        Config(32,  1024, 4096, 1024),
        Config(128, 1024, 4096, 1024),
        Config(512, 1024, 4096, 1024),
    ]

    print(f"\n{'--- eager ':->60}")
    for cfg in configs:
        bench(cfg, device, compiled=False)

    print(f"\n{'--- torch.compile ':->60}")
    for cfg in configs:
        bench(cfg, device, compiled=True)


if __name__ == "__main__":
    main()
