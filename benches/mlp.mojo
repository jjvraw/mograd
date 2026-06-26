from std.gpu.host import DeviceContext
from std.benchmark import Bench, Bencher, BenchId, keep, BenchConfig
from std.benchmark import BenchMetric, ThroughputMeasure

from mograd import Tensor, Device
import mograd.nn as nn

comptime N_BATCHES = 8


# ===-------------------------------------------------------------------===#
# Raw matmul (no bias, no optimizer)
# ===-------------------------------------------------------------------===#


@no_inline
def bench_mlp_forward[B: Int, D_in: Int, D_h: Int, D_out: Int](mut m: Bench, device: Device) raises:
    comptime flops = 2 * B * D_in * D_h + 2 * B * D_h * D_out

    var xs = List[Tensor]()
    for i in range(N_BATCHES):
        var xb = Tensor.randn(device, (B, D_in), seed=UInt32(i))
        _ = xb.value()
        xs.append(xb)
    var w1 = Tensor.randn(device, (D_in, D_h), seed=100)
    var w2 = Tensor.randn(device, (D_h, D_out), seed=101)
    _ = w1.value()
    _ = w2.value()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            var x = xs[iteration % N_BATCHES]
            var h = (x @ w1).relu()
            _ = (h @ w2).value()

        bench.iter_custom[kernel_launch](device.ctx)

    m.bench_function[bench_func](
        BenchId("mlp_forward", input_id=String(B, "x", D_in, "->", D_h, "->", D_out)),
        [ThroughputMeasure(BenchMetric.flops, flops)],
    )


@no_inline
def bench_mlp_forward_backward[B: Int, D_in: Int, D_h: Int, D_out: Int](mut m: Bench, device: Device) raises:
    comptime flops = 2 * B * D_in * D_h + 2 * B * D_h * D_out

    var xs = List[Tensor]()
    for i in range(N_BATCHES):
        var xb = Tensor.randn(device, (B, D_in), seed=UInt32(i))
        _ = xb.value()
        xs.append(xb)
    var w1 = Tensor.randn(device, (D_in, D_h), seed=100, requires_grad=True)
    var w2 = Tensor.randn(device, (D_h, D_out), seed=101, requires_grad=True)
    _ = w1.value()
    _ = w2.value()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            var x = xs[iteration % N_BATCHES]
            var h = (x @ w1).relu()
            var loss = (h @ w2).sum()
            var grads = loss.gradient([w1, w2])
            _ = Tensor.values(grads)

        bench.iter_custom[kernel_launch](device.ctx)

    m.bench_function[bench_func](
        BenchId("mlp_fwd_bwd", input_id=String(B, "x", D_in, "->", D_h, "->", D_out)),
        [ThroughputMeasure(BenchMetric.flops, flops * 3)],
    )


# ===-------------------------------------------------------------------===#
# nn.Linear with bias — full MNIST training stack
# ===-------------------------------------------------------------------===#


@no_inline
def bench_mnist_mlp_forward[B: Int, D_in: Int, D_h1: Int, D_h2: Int, D_out: Int](mut m: Bench, device: Device) raises:
    comptime flops = 2 * B * D_in * D_h1 + 2 * B * D_h1 * D_h2 + 2 * B * D_h2 * D_out

    var xs = List[Tensor]()
    for i in range(N_BATCHES):
        var xb = Tensor.randn(device, (B, D_in), seed=UInt32(i))
        _ = xb.value()
        xs.append(xb)

    var l1 = nn.Linear(D_in, D_h1)
    var l2 = nn.Linear(D_h1, D_h2)
    var l3 = nn.Linear(D_h2, D_out)
    _ = l3(l2(l1(xs[0]).relu()).relu()).value()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            var x = xs[iteration % N_BATCHES]
            var h = l1(x).relu()
            h = l2(h).relu()
            _ = l3(h).value()

        bench.iter_custom[kernel_launch](device.ctx)

    m.bench_function[bench_func](
        BenchId("mnist_mlp_forward", input_id=String(B, "x", D_in, "->", D_h1, "->", D_h2, "->", D_out)),
        [ThroughputMeasure(BenchMetric.flops, flops)],
    )


@no_inline
def bench_mnist_mlp_training[B: Int, D_in: Int, D_h1: Int, D_h2: Int, D_out: Int](mut m: Bench, device: Device) raises:
    comptime flops = 2 * B * D_in * D_h1 + 2 * B * D_h1 * D_h2 + 2 * B * D_h2 * D_out
    # SGD: w -= lr * g → 2 flops/params
    #      params = weights + biases across all layers
    comptime n_params = (D_in * D_h1 + D_h1) + (D_h1 * D_h2 + D_h2) + (D_h2 * D_out + D_out)
    comptime sgd_flops = n_params * 2

    var xs = List[Tensor]()
    var ys = List[Tensor]()
    for b in range(N_BATCHES):
        var xb = Tensor.randn(device, (B, D_in), seed=UInt32(b))
        var yb = Tensor(device, [Float32(b % D_out)] * B, (B,))
        _ = xb.value()
        _ = yb.value()
        xs.append(xb)
        ys.append(yb)

    var l1 = nn.Linear(D_in, D_h1)
    var l2 = nn.Linear(D_h1, D_h2)
    var l3 = nn.Linear(D_h2, D_out)
    var ps = l1.parameters()
    ps += l2.parameters()
    ps += l3.parameters()
    var opt = nn.SGD(ps^, lr=Float32(0.01))
    _ = l3(l2(l1(xs[0]).relu()).relu()).value()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            var b = iteration % N_BATCHES
            var h = l1(xs[b]).relu()
            h = l2(h).relu()
            var logits = l3(h)
            var loss = logits.cross_entropy(ys[b].one_hot(D_out).cast(DType.float32))
            var grads = loss.gradient(opt.params())
            opt.step(grads)

        bench.iter_custom[kernel_launch](device.ctx)

    m.bench_function[bench_func](
        BenchId("mnist_mlp_training", input_id=String(B, "x", D_in, "->", D_h1, "->", D_h2, "->", D_out)),
        [ThroughputMeasure(BenchMetric.flops, flops * 3 + sgd_flops)],
    )


@no_inline
def bench_mnist_mlp_training_adam[
    B: Int, D_in: Int, D_h1: Int, D_h2: Int, D_out: Int
](mut m: Bench, device: Device) raises:
    comptime flops = 2 * B * D_in * D_h1 + 2 * B * D_h1 * D_h2 + 2 * B * D_h2 * D_out
    # Adam: new_m, new_v, m_hat, v_hat, sqrt, div, mul, w -= → ~14 flops/param
    comptime n_params = (D_in * D_h1 + D_h1) + (D_h1 * D_h2 + D_h2) + (D_h2 * D_out + D_out)
    comptime adam_flops = n_params * 14

    var xs = List[Tensor]()
    var ys = List[Tensor]()
    for b in range(N_BATCHES):
        var xb = Tensor.randn(device, (B, D_in), seed=UInt32(b))
        var yb = Tensor(device, [Float32(b % D_out)] * B, (B,))
        _ = xb.value()
        _ = yb.value()
        xs.append(xb)
        ys.append(yb)

    var l1 = nn.Linear(D_in, D_h1)
    var l2 = nn.Linear(D_h1, D_h2)
    var l3 = nn.Linear(D_h2, D_out)
    var ps = l1.parameters()
    ps += l2.parameters()
    ps += l3.parameters()
    var opt = nn.Adam(ps^, lr=Float32(3e-4))
    _ = l3(l2(l1(xs[0]).relu()).relu()).value()

    @parameter
    @always_inline
    def bench_func(mut bench: Bencher) raises:
        @parameter
        @always_inline
        def kernel_launch(dc: DeviceContext, iteration: Int) raises:
            var b = iteration % N_BATCHES
            var h = l1(xs[b]).relu()
            h = l2(h).relu()
            var logits = l3(h)
            var loss = logits.cross_entropy(ys[b].one_hot(D_out).cast(DType.float32))
            var grads = loss.gradient(opt.params())
            opt.step(grads)

        bench.iter_custom[kernel_launch](device.ctx)

    m.bench_function[bench_func](
        BenchId("mnist_mlp_training_adam", input_id=String(B, "x", D_in, "->", D_h1, "->", D_h2, "->", D_out)),
        [ThroughputMeasure(BenchMetric.flops, flops * 3 + adam_flops)],
    )


def main() raises:
    var m = Bench(BenchConfig(max_iters=1000, min_runtime_secs=5, max_runtime_secs=10))
    var device = Device()

    bench_mlp_forward[32, 768, 3072, 768](m, device)
    bench_mlp_forward[128, 768, 3072, 768](m, device)
    bench_mlp_forward[512, 768, 3072, 768](m, device)
    bench_mlp_forward[32, 1024, 4096, 1024](m, device)
    bench_mlp_forward[512, 1024, 4096, 1024](m, device)

    bench_mlp_forward_backward[32, 768, 3072, 768](m, device)
    bench_mlp_forward_backward[128, 768, 3072, 768](m, device)
    bench_mlp_forward_backward[512, 768, 3072, 768](m, device)

    bench_mnist_mlp_forward[32, 784, 256, 128, 10](m, device)
    bench_mnist_mlp_training[32, 784, 256, 128, 10](m, device)
    bench_mnist_mlp_training_adam[32, 784, 256, 128, 10](m, device)

    m.dump_report()
