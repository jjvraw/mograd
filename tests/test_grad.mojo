from std.sys import has_accelerator
from std.testing import TestSuite
from std.math import abs

from mograd import Tensor, DeviceContext
from mograd.testing import assert_allclose, assert_close


comptime FwdFn = def(x: Tensor) thin raises -> Tensor


def numerical_grad[
    fwd: FwdFn
](ctx: DeviceContext, data: List[Float32], shape: List[Int], eps: Float32 = 1e-3,) raises -> List[Float32]:
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


def fwd_scale(x: Tensor) raises -> Tensor:
    return (x * Float32(3.0)).sum()


def fwd_sum(x: Tensor) raises -> Tensor:
    return x.sum()


def fwd_relu_sum(x: Tensor) raises -> Tensor:
    return x.relu().sum()


def fwd_log_sum(x: Tensor) raises -> Tensor:
    return x.log().sum()


def test_scale_grad() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd_scale](ctx, data, [4])
    var x = Tensor(ctx, data, [4], requires_grad=True)
    var loss = (x * Float32(3.0)).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(3), 3, 3, 3])


def test_sum_grad() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd_sum](ctx, data, [4])
    var x = Tensor(ctx, data, [4], requires_grad=True)
    var loss = x.sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1])


def test_relu_grad() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [-1.0, -0.5, 0.5, 1.0]
    var num = numerical_grad[fwd_relu_sum](ctx, data, [4])
    var x = Tensor(ctx, data, [4], requires_grad=True)
    var loss = x.relu().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(0), 0, 1, 1])


def test_log_grad() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [1.0, 2.0, 4.0]
    var num = numerical_grad[fwd_log_sum](ctx, data, [3])
    var x = Tensor(ctx, data, [3], requires_grad=True)
    var loss = x.log().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1.0), 0.5, 0.25], tol=1e-4)


def test_matmul_grad() raises:
    var ctx = DeviceContext()
    var a_data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var b_data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var eps = Float32(1e-2)
    var b = Tensor(ctx, b_data, [3, 2])
    var num = List[Float32]()
    for i in range(6):
        var d_plus = a_data.copy()
        var d_minus = a_data.copy()
        d_plus[i] += eps
        d_minus[i] -= eps
        var f_plus = (Tensor(ctx, d_plus, [2, 3]) @ b).sum().item()
        var f_minus = (Tensor(ctx, d_minus, [2, 3]) @ b).sum().item()
        num.append((f_plus - f_minus) / (Float32(2.0) * eps))
    var a = Tensor(ctx, a_data, [2, 3], requires_grad=True)
    var b2 = Tensor(ctx, b_data, [3, 2])
    var loss = (a @ b2).sum()
    var grads = loss.gradient([a])
    assert_allclose(grads[0], num, tol=0.05)


def test_cross_entropy_grad_row_sums() raises:
    var ctx = DeviceContext()
    var logits_data = List[Float32]()
    for _ in range(40):
        logits_data.append(Float32(1.0))
    var logits = Tensor(ctx, logits_data, [4, 10], requires_grad=True)
    var labels = Tensor(ctx, [Float32(0), 1, 2, 3], [4])
    var loss = logits.cross_entropy(labels)
    var grads = loss.gradient([logits])
    var grad_vals = grads[0].to_list()
    for row in range(4):
        var row_sum = Float32(0.0)
        for col in range(10):
            row_sum += grad_vals[row * 10 + col]
        if abs(row_sum) >= Float32(1e-4):
            raise Error("CE grad row " + String(row) + " sum != 0: " + String(row_sum))


def main() raises:
    comptime assert has_accelerator(), "GPU required to run gradient tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
