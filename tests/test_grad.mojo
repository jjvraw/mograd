from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true

import mograd.nn as nn
from mograd import Tensor, DeviceContext
from mograd.shape import Shape
from mograd.testing import assert_allclose, assert_close


comptime FwdFn = def(x: Tensor) thin raises -> Tensor


def numerical_grad[
    fwd: FwdFn
](ctx: DeviceContext, data: List[Float32], shape: Shape, eps: Float32 = 1e-3,) raises -> List[Float32]:
    var grads = List[Float32]()
    for i in range(len(data)):
        var d_plus = data.copy()
        var d_minus = data.copy()
        d_plus[i] += eps
        d_minus[i] -= eps
        var f_plus = fwd(Tensor(ctx, d_plus, shape)).item()
        var f_minus = fwd(Tensor(ctx, d_minus, shape)).item()
        grads.append((f_plus - f_minus) / (Float32(2.0) * eps))
    return grads^


def test_scale_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return (x * Float32(3.0)).sum()

    var ctx = DeviceContext()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](ctx, data, (4,))
    var x = Tensor(ctx, data, (4,), requires_grad=True)
    var loss = (x * Float32(3.0)).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(3), 3, 3, 3])


def test_sum_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.sum()

    var ctx = DeviceContext()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](ctx, data, (4,))
    var x = Tensor(ctx, data, (4,), requires_grad=True)
    var loss = x.sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1])


def test_relu_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.relu().sum()

    var ctx = DeviceContext()
    var data: List[Float32] = [-1.0, -0.5, 0.5, 1.0]
    var num = numerical_grad[fwd](ctx, data, (4,))
    var x = Tensor(ctx, data, (4,), requires_grad=True)
    var loss = x.relu().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(0), 0, 1, 1])


def test_relu_zero_boundary() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(0.0)], (1,), requires_grad=True)
    var loss = x.relu().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], [Float32(0.0)])


def test_log_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.log().sum()

    var ctx = DeviceContext()
    var data: List[Float32] = [1.0, 2.0, 4.0]
    var num = numerical_grad[fwd](ctx, data, (3,))
    var x = Tensor(ctx, data, (3,), requires_grad=True)
    var loss = x.log().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1.0), 0.5, 0.25], tol=1e-4)


def test_matmul_grad() raises:
    def fwd(a: Tensor) raises -> Tensor:
        var b = Tensor(a.ctx.value(), [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
        return (a @ b).sum()

    var ctx = DeviceContext()
    var a_data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](ctx, a_data, (2, 3))
    var a = Tensor(ctx, a_data, (2, 3), requires_grad=True)
    var b = Tensor(ctx, [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
    var loss = (a @ b).sum()
    var grads = loss.gradient([a])
    assert_allclose(grads[0], num, tol=0.05)


def test_matmul_grad_b() raises:
    def fwd(b: Tensor) raises -> Tensor:
        var a = Tensor(b.ctx.value(), [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
        return (a @ b).sum()

    var ctx = DeviceContext()
    var b_data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](ctx, b_data, (3, 2))
    var a = Tensor(ctx, [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
    var b = Tensor(ctx, b_data, (3, 2), requires_grad=True)
    var loss = (a @ b).sum()
    var grads = loss.gradient([b])
    assert_allclose(grads[0], num, tol=0.05)


def test_softmax_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var e0 = Tensor(x.ctx.value(), [Float32(1), 0, 0, 0], (4,))
        return (x.softmax() * e0).sum()

    var ctx = DeviceContext()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](ctx, data, (4,))
    var x = Tensor(ctx, data, (4,), requires_grad=True)
    var e0 = Tensor(ctx, [Float32(1), 0, 0, 0], (4,))
    var loss = (x.softmax() * e0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)


def test_cross_entropy_grad_row_sums() raises:
    var ctx = DeviceContext()
    var logits_data = List[Float32]()
    for _ in range(40):
        logits_data.append(Float32(1.0))
    var logits = Tensor(ctx, logits_data, (4, 10), requires_grad=True)
    var labels = Tensor(ctx, [Float32(0), 1, 2, 3], (4,))
    var loss = logits.cross_entropy(labels)
    var grads = loss.gradient([logits])
    var grad_vals = grads[0].to_list()
    for row in range(4):
        var row_sum = Float32(0.0)
        for col in range(10):
            row_sum += grad_vals[row * 10 + col]
        assert_almost_equal(row_sum, Float32(0.0), atol=1e-4)


def test_cross_entropy_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var labels = Tensor(x.ctx.value(), [Float32(0), 1, 2, 3], (4,))
        return x.cross_entropy(labels)

    var ctx = DeviceContext()
    var logits_data = List[Float32]()
    for _ in range(40):
        logits_data.append(Float32(1.0))

    var num = numerical_grad[fwd](ctx, logits_data, (4, 10))
    var logits = Tensor(ctx, logits_data, (4, 10), requires_grad=True)
    var labels = Tensor(ctx, [Float32(0), 1, 2, 3], (4,))
    var loss = logits.cross_entropy(labels)
    var grads = loss.gradient([logits])
    assert_allclose(grads[0], num, tol=0.05)


def test_simple_mlp_grad() raises:
    def fwd_w1(w1: Tensor) raises -> Tensor:
        var ctx = w1.ctx.value()
        var x = Tensor(ctx, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
        var labels = Tensor(ctx, [Float32(0)], (1,))
        var h1 = (x @ w1.transpose()).relu()
        var l2 = nn.Linear(8, 4)
        var l3 = nn.Linear(4, 3)
        return l3(l2(h1).relu()).cross_entropy(labels)

    def fwd_w2(w2: Tensor) raises -> Tensor:
        var ctx = w2.ctx.value()
        var x = Tensor(ctx, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
        var labels = Tensor(ctx, [Float32(0)], (1,))
        var l1 = nn.Linear(4, 8)
        var h2 = (l1(x).relu() @ w2.transpose()).relu()
        var l3 = nn.Linear(4, 3)
        return l3(h2).cross_entropy(labels)

    def fwd_w3(w3: Tensor) raises -> Tensor:
        var ctx = w3.ctx.value()
        var x = Tensor(ctx, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
        var labels = Tensor(ctx, [Float32(0)], (1,))
        var l1 = nn.Linear(4, 8)
        var l2 = nn.Linear(8, 4)
        return (l2(l1(x).relu()).relu() @ w3.transpose()).cross_entropy(labels)

    var ctx = DeviceContext()

    # Linear seeds are deterministic (UInt32(out * in)), so the same weights
    # are produced every call: numerical and analytical grads see the same network
    var l1 = nn.Linear(4, 8)
    var l2 = nn.Linear(8, 4)
    var l3 = nn.Linear(4, 3)
    var opt = nn.SGD([l1, l2, l3], lr=Float32(0.1))

    var x = Tensor(ctx, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
    var labels = Tensor(ctx, [Float32(0)], (1,))
    var h1 = l1(x).relu()
    var h2 = l2(h1).relu()
    var loss = l3(h2).cross_entropy(labels)
    var params = opt.params()
    var grads = loss.gradient(params)

    var w1_data = l1._weight[].value().to_list()
    var w2_data = l2._weight[].value().to_list()
    var w3_data = l3._weight[].value().to_list()
    assert_allclose(grads[0], numerical_grad[fwd_w1](ctx, w1_data, (8, 4)), tol=0.05)
    assert_allclose(grads[1], numerical_grad[fwd_w2](ctx, w2_data, (4, 8)), tol=0.05)
    assert_allclose(grads[2], numerical_grad[fwd_w3](ctx, w3_data, (3, 4)), tol=0.05)


def main() raises:
    comptime assert has_accelerator(), "GPU required to run gradient tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
