from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true

import mograd.nn as nn
from mograd import Tensor, Device
from mograd.grad import Grad
from mograd.layout import Layout
from mograd.op import OpRef
from mograd.runtime.gpu.rewrites import FLASH_ATTN_GRAD, GPU_REWRITES
from mograd.simplify import Simplifier
from mograd.testing import assert_allclose, assert_close


comptime FwdFn = def(x: Tensor) thin raises -> Tensor


def numerical_grad[
    fwd: FwdFn
](device: Device, data: List[Float32], shape: Layout, eps: Float32 = 1e-3,) raises -> List[Float32]:
    var grads = List[Float32]()
    for i in range(len(data)):
        var d_plus = data.copy()
        var d_minus = data.copy()
        d_plus[i] += eps
        d_minus[i] -= eps
        var f_plus = fwd(Tensor(device, d_plus, shape)).item()
        var f_minus = fwd(Tensor(device, d_minus, shape)).item()
        grads.append((f_plus - f_minus) / (Float32(2.0) * eps))
    return grads^


def test_scale_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return (x * Float32(3.0)).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var loss = (x * Float32(3.0)).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(3), 3, 3, 3])


def test_sum_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var loss = x.sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1])


def test_sum_axis0_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.sum(axis=0).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, data, (2, 3))
    var x = Tensor(device, data, (2, 3), requires_grad=True)
    var loss = x.sum(axis=0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], Tensor.ones(device, (2, 3)))


def test_sum_axis1_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.sum(axis=1).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, data, (2, 3))
    var x = Tensor(device, data, (2, 3), requires_grad=True)
    var loss = x.sum(axis=1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], Tensor.ones(device, (2, 3)))


def test_mean_axis1_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.mean(axis=1).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, data, (2, 3))
    var x = Tensor(device, data, (2, 3), requires_grad=True)
    var loss = x.mean(axis=1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], Tensor.full(device, (2, 3), 1.0 / 3.0), tol=1e-4)


def test_sum_axis_keepdim_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.sum(axis=1, keepdim=True).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, data, (2, 3))
    var x = Tensor(device, data, (2, 3), requires_grad=True)
    var loss = x.sum(axis=1, keepdim=True).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], Tensor.ones(device, (2, 3)))


def test_sum_axis_3d_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.sum(axis=1).sum()

    var device = Device()
    var data = List[Float32]()
    for i in range(24):
        data.append(Float32(i + 1))
    var num = numerical_grad[fwd](device, data, (2, 3, 4))
    var x = Tensor(device, data, (2, 3, 4), requires_grad=True)
    var loss = x.sum(axis=1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], Tensor.ones(device, (2, 3, 4)))


def test_relu_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.relu().sum()

    var device = Device()
    var data: List[Float32] = [-1.0, -0.5, 0.5, 1.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var loss = x.relu().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(0), 0, 1, 1])


def test_sum_large_grad() raises:
    var device = Device()
    var n = 4096
    var x = Tensor.ones(device, (n,), requires_grad=True)
    var loss = x.sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], Tensor.ones(device, (n,)))


def test_sum_of_softmax_grad_is_zero() raises:
    var device = Device()
    var x = Tensor.randn(device, (8,), seed=42, requires_grad=True)
    var loss = x.softmax().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], Tensor.full(device, (8,), 0.0), tol=1e-4)


def test_relu_zero_boundary() raises:
    var device = Device()
    var x = Tensor(device, [Float32(0.0)], (1,), requires_grad=True)
    var loss = x.relu().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], [Float32(0.0)])


def test_log_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.log().sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 4.0]
    var num = numerical_grad[fwd](device, data, (3,))
    var x = Tensor(device, data, (3,), requires_grad=True)
    var loss = x.log().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1.0), 0.5, 0.25], tol=1e-4)


def test_neg_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return (-x).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var loss = (-x).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(-1), -1, -1, -1])


def test_exp_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.exp().sum()

    var device = Device()
    var data: List[Float32] = [0.0, 1.0, 2.0]
    var num = numerical_grad[fwd](device, data, (3,))
    var x = Tensor(device, data, (3,), requires_grad=True)
    var loss = x.exp().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    # grad of sum(exp(x)) w.r.t. x is exp(x)
    assert_allclose(grads[0], [Float32(1.0), 2.718282, 7.389056], tol=1e-3)


def test_sqrt_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.sqrt().sum()

    var device = Device()
    var data: List[Float32] = [1.0, 4.0, 9.0]
    var num = numerical_grad[fwd](device, data, (3,))
    var x = Tensor(device, data, (3,), requires_grad=True)
    var loss = x.sqrt().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(0.5), 0.25, 1.0 / 6.0], tol=1e-3)


def test_div_grad_numerator() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var b = Tensor(x.device.value(), [Float32(2.0), 4.0, 8.0], (3,))
        return (x / b).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 4.0]
    var num = numerical_grad[fwd](device, data, (3,))
    var x = Tensor(device, data, (3,), requires_grad=True)
    var b = Tensor(device, [Float32(2.0), 4.0, 8.0], (3,))
    var loss = (x / b).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(0.5), 0.25, 0.125], tol=1e-4)


def test_div_grad_denominator() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var a = Tensor(x.device.value(), [Float32(4.0), 8.0, 16.0], (3,))
        return (a / x).sum()

    var device = Device()
    var data: List[Float32] = [2.0, 2.0, 2.0]
    var num = numerical_grad[fwd](device, data, (3,))
    var a = Tensor(device, [Float32(4.0), 8.0, 16.0], (3,))
    var x = Tensor(device, data, (3,), requires_grad=True)
    var loss = (a / x).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    # d(a/x)/dx = -a/x^2
    assert_allclose(grads[0], [Float32(-1.0), -2.0, -4.0], tol=1e-4)


def test_reshape_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.reshape((2, 2)).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var loss = x.reshape((2, 2)).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1])


def test_transpose_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.transpose().sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, data, (2, 3))
    var x = Tensor(device, data, (2, 3), requires_grad=True)
    var loss = x.transpose().sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1, 1, 1])


def test_matmul_grad() raises:
    def fwd(a: Tensor) raises -> Tensor:
        var b = Tensor(a.device.value(), [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
        return (a @ b).sum()

    var device = Device()
    var a_data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, a_data, (2, 3))
    var a = Tensor(device, a_data, (2, 3), requires_grad=True)
    var b = Tensor(device, [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (3, 2))
    var loss = (a @ b).sum()
    var grads = loss.gradient([a])
    assert_allclose(grads[0], num, tol=0.05)


def test_matmul_grad_b() raises:
    def fwd(b: Tensor) raises -> Tensor:
        var a = Tensor(b.device.value(), [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
        return (a @ b).sum()

    var device = Device()
    var b_data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, b_data, (3, 2))
    var a = Tensor(device, [Float32(1.0), 2.0, 3.0, 4.0, 5.0, 6.0], (2, 3))
    var b = Tensor(device, b_data, (3, 2), requires_grad=True)
    var loss = (a @ b).sum()
    var grads = loss.gradient([b])
    assert_allclose(grads[0], num, tol=0.05)


def test_batched_matmul_grad() raises:
    def fwd(a: Tensor) raises -> Tensor:
        var b = Tensor(a.device.value(), [Float32(1), 2, 3, 4, 5, 6, 7, 8], (2, 2, 2))
        return (a @ b).sum()

    var device = Device()
    var a_data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8]
    var num = numerical_grad[fwd](device, a_data, (2, 2, 2))
    var a = Tensor(device, a_data, (2, 2, 2), requires_grad=True)
    var b = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (2, 2, 2))
    var loss = (a @ b).sum()
    var grads = loss.gradient([a])
    assert_allclose(grads[0], num, tol=0.05)


def test_softmax_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var e0 = Tensor(x.device.value(), [Float32(1), 0, 0, 0], (4,))
        return (x.softmax() * e0).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var e0 = Tensor(device, [Float32(1), 0, 0, 0], (4,))
    var loss = (x.softmax() * e0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)


def test_softmax_axis0_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var e0 = Tensor(x.device.value(), [Float32(1), 0, 0, 0], (2, 2))
        return (x.softmax(axis=0) * e0).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (2, 2))
    var x = Tensor(device, data, (2, 2), requires_grad=True)
    var e0 = Tensor(device, [Float32(1), 0, 0, 0], (2, 2))
    var loss = (x.softmax(axis=0) * e0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)


def test_softmax_middle_axis_rank3_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.softmax(axis=1).sum()

    var device = Device()
    var data = List[Float32]()
    for i in range(24):
        data.append(Float32(i + 1))
    var num = numerical_grad[fwd](device, data, (2, 3, 4))
    var x = Tensor(device, data, (2, 3, 4), requires_grad=True)
    var loss = x.softmax(axis=1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)


def test_cross_entropy_grad_row_sums() raises:
    var device = Device()
    var logits_data = List[Float32]()
    for _ in range(40):
        logits_data.append(Float32(1.0))
    var logits = Tensor(device, logits_data, (4, 10), requires_grad=True)
    var labels = Tensor(device, [Float32(0), 1, 2, 3], (4,)).cast(DType.float32)
    var loss = logits.cross_entropy(labels.one_hot(10).cast(DType.float32))
    var grads = loss.gradient([logits])
    var grad_vals = grads[0].to_list()
    for row in range(4):
        var row_sum = Float32(0.0)
        for col in range(10):
            row_sum += grad_vals[row * 10 + col]
        assert_almost_equal(row_sum, Float32(0.0), atol=1e-4)


def test_cross_entropy_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var labels = Tensor(x.device.value(), [Float32(0), 1, 2, 3], (4,)).cast(DType.float32)
        return x.cross_entropy(labels.one_hot(10).cast(DType.float32))

    var device = Device()
    var logits_data = List[Float32]()
    for _ in range(40):
        logits_data.append(Float32(1.0))

    var num = numerical_grad[fwd](device, logits_data, (4, 10))
    var logits = Tensor(device, logits_data, (4, 10), requires_grad=True)
    var labels = Tensor(device, [Float32(0), 1, 2, 3], (4,)).cast(DType.float32)
    var loss = logits.cross_entropy(labels.one_hot(10).cast(DType.float32))
    var grads = loss.gradient([logits])
    assert_allclose(grads[0], num, tol=0.05)


def test_cross_entropy_grad_rank3_matches_reshape() raises:
    var device = Device()
    var logits_data = List[Float32]()
    for i in range(24):
        logits_data.append(Float32(i) * 0.1)
    var label_ids = Tensor(device, [Int64(0), 2, 1, 3, 0, 1], (6,))
    var labels = label_ids.one_hot(4).cast(DType.float32).reshape((2, 3, 4))

    var x_rank3 = Tensor(device, logits_data, (2, 3, 4), requires_grad=True)
    var loss_rank3 = x_rank3.cross_entropy(labels)
    var grads_rank3 = loss_rank3.gradient([x_rank3])

    var x_reshaped = Tensor(device, logits_data, (6, 4), requires_grad=True)
    var loss_reshaped = x_reshaped.cross_entropy(labels.reshape((6, 4)))
    var grads_reshaped = loss_reshaped.gradient([x_reshaped])

    assert_allclose(grads_rank3[0], grads_reshaped[0], tol=1e-4)


def test_gather_grad_scatter_adds_repeated_indices() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var indices = Tensor(x.device.value(), [Int64(1), 3, 1, 0], (4,))
        return x.gather(indices).sum()

    var device = Device()
    var data: List[Float32] = [Float32(i) for i in range(12)]
    var num = numerical_grad[fwd](device, data, (4, 3))
    var table = Tensor(device, data, (4, 3), requires_grad=True)
    var indices = Tensor(device, [Int64(1), 3, 1, 0], (4,))
    var loss = table.gather(indices).sum()
    var grads = loss.gradient([table])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1])


def test_scatter_add_grad_is_gather() raises:
    def fwd(x: Tensor) raises -> Tensor:
        var indices = Tensor(x.device.value(), [Int64(1), 3, 1, 0], (4,))
        var weights = Tensor(x.device.value(), [Float32(i) for i in range(15)], (5, 3))
        return (x.scatter_add(indices, 5) * weights).sum()

    var device = Device()
    var data: List[Float32] = [Float32(i) for i in range(12)]
    var num = numerical_grad[fwd](device, data, (4, 3))
    var values = Tensor(device, data, (4, 3), requires_grad=True)
    var indices = Tensor(device, [Int64(1), 3, 1, 0], (4,))
    var weights = Tensor(device, [Float32(i) for i in range(15)], (5, 3))
    var loss = (values.scatter_add(indices, 5) * weights).sum()
    var grads = loss.gradient([values])
    assert_allclose(grads[0], num, tol=0.05)
    # d(loss)/d(values[i]) == weights[indices[i]] (the SCATTER_ADD grad is GATHER)
    assert_allclose(grads[0], weights.gather(indices))


def test_simple_mlp_grad() raises:
    def linear_bias(device: Device, in_features: Int, out_features: Int) raises -> Tensor:
        var bound = Float32(sqrt(Float32(6) / Float32(in_features)))
        var seed = UInt32(out_features * in_features) + 1
        return Tensor.uniform(device, (out_features,), low=-bound, high=bound, seed=seed)

    def fwd_w1(w1: Tensor) raises -> Tensor:
        var device = w1.device.value()
        var x = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
        var labels = Tensor(device, [Float32(0)], (1,)).cast(DType.float32)
        var b1 = linear_bias(device, 4, 8)
        var h1 = (x @ w1.transpose() + b1.expand(1, 8)).relu()
        var l2 = nn.Linear(8, 4)
        var l3 = nn.Linear(4, 3)
        return l3(l2(h1).relu()).cross_entropy(labels.one_hot(3).cast(DType.float32))

    def fwd_w2(w2: Tensor) raises -> Tensor:
        var device = w2.device.value()
        var x = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
        var labels = Tensor(device, [Float32(0)], (1,)).cast(DType.float32)
        var l1 = nn.Linear(4, 8)
        var b2 = linear_bias(device, 8, 4)
        var h2 = (l1(x).relu() @ w2.transpose() + b2.expand(1, 4)).relu()
        var l3 = nn.Linear(4, 3)
        return l3(h2).cross_entropy(labels.one_hot(3).cast(DType.float32))

    def fwd_w3(w3: Tensor) raises -> Tensor:
        var device = w3.device.value()
        var x = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
        var labels = Tensor(device, [Float32(0)], (1,))
        var l1 = nn.Linear(4, 8)
        var l2 = nn.Linear(8, 4)
        var b3 = linear_bias(device, 4, 3)
        return (l2(l1(x).relu()).relu() @ w3.transpose() + b3.expand(1, 3)).cross_entropy(
            labels.one_hot(3).cast(DType.float32)
        )

    var device = Device()

    var l1 = nn.Linear(4, 8)
    var l2 = nn.Linear(8, 4)
    var l3 = nn.Linear(4, 3)
    var ps = l1.parameters()
    ps += l2.parameters()
    ps += l3.parameters()
    var opt = nn.SGD(ps^, lr=Float32(0.1))

    var x = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
    var labels = Tensor(device, [Float32(0)], (1,))
    var h1 = l1(x).relu()
    var h2 = l2(h1).relu()
    var loss = l3(h2).cross_entropy(labels.one_hot(3).cast(DType.float32))
    var params = opt.params()
    var grads = loss.gradient(params)

    var w1_data = l1._weight[].value().to_list()
    var w2_data = l2._weight[].value().to_list()
    var w3_data = l3._weight[].value().to_list()
    # opt.params() interleaves each layer's bias after its weight:
    # [w1, b1, w2, b2, w3, b3], so the weight grads sit at indices 0, 2, 4.
    assert_allclose(grads[0], numerical_grad[fwd_w1](device, w1_data, (8, 4)), tol=0.05)
    assert_allclose(grads[2], numerical_grad[fwd_w2](device, w2_data, (4, 8)), tol=0.05)
    assert_allclose(grads[4], numerical_grad[fwd_w3](device, w3_data, (3, 4)), tol=0.05)


def test_unsqueeze_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.unsqueeze(0).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var loss = x.unsqueeze(0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1])


def test_unsqueeze_grad_2d() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.unsqueeze(1).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, data, (2, 3))
    var x = Tensor(device, data, (2, 3), requires_grad=True)
    var loss = x.unsqueeze(1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], Tensor.ones(device, (2, 3)))


def test_squeeze_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.squeeze(0).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (1, 4))
    var x = Tensor(device, data, (1, 4), requires_grad=True)
    var loss = x.squeeze(0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1])


def test_squeeze_grad_2d() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.squeeze(1).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    var num = numerical_grad[fwd](device, data, (2, 1, 3))
    var x = Tensor(device, data, (2, 1, 3), requires_grad=True)
    var loss = x.squeeze(1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], Tensor.ones(device, (2, 1, 3)))


def test_squeeze_unsqueeze_roundtrip_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.unsqueeze(0).squeeze(0).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0, 4.0]
    var num = numerical_grad[fwd](device, data, (4,))
    var x = Tensor(device, data, (4,), requires_grad=True)
    var loss = x.unsqueeze(0).squeeze(0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1])


def test_expand_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.expand(3, 4).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0]
    var num = numerical_grad[fwd](device, data, (3, 1))
    var x = Tensor(device, data, (3, 1), requires_grad=True)
    var loss = x.expand(3, 4).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(4), 4, 4])


def test_expand_grad_multi_axis() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.expand(2, 4, 3).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0]
    var num = numerical_grad[fwd](device, data, (1, 1, 3))
    var x = Tensor(device, data, (1, 1, 3), requires_grad=True)
    var loss = x.expand(2, 4, 3).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(8), 8, 8])


def test_expand_grad_pads_rank() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.expand(4, -1).sum()

    var device = Device()
    var data: List[Float32] = [1.0, 2.0, 3.0]
    var num = numerical_grad[fwd](device, data, (3,))
    var x = Tensor(device, data, (3,), requires_grad=True)
    var loss = x.expand(4, -1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(4), 4, 4])


def test_triu_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.triu(0).sum()

    var device = Device()
    var data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    var num = numerical_grad[fwd](device, data, (3, 3))
    var x = Tensor(device, data, (3, 3), requires_grad=True)
    var loss = x.triu(0).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)


def test_triu_diagonal_grad() raises:
    def fwd(x: Tensor) raises -> Tensor:
        return x.triu(1).sum()

    var device = Device()
    var data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    var num = numerical_grad[fwd](device, data, (3, 3))
    var x = Tensor(device, data, (3, 3), requires_grad=True)
    var loss = x.triu(1).sum()
    var grads = loss.gradient([x])
    assert_allclose(grads[0], num, tol=0.05)


def test_concat_grad_axis0() raises:
    def fwd(a: Tensor) raises -> Tensor:
        var device = a.device.value()
        var b = Tensor(device, [Float32(7), 8, 9, 10, 11, 12], (2, 3))
        return Tensor.concat([a, b], 0).sum()

    var device = Device()
    var a_data: List[Float32] = [1, 2, 3, 4, 5, 6]
    var num = numerical_grad[fwd](device, a_data, (2, 3))
    var a = Tensor(device, a_data, (2, 3), requires_grad=True)
    var b = Tensor(device, [Float32(7), 8, 9, 10, 11, 12], (2, 3))
    var loss = Tensor.concat([a, b], 0).sum()
    var grads = loss.gradient([a])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(1), 1, 1, 1, 1, 1])


def test_concat_grad_axis1() raises:
    def fwd(a: Tensor) raises -> Tensor:
        var device = a.device.value()
        var b = Tensor(device, [Float32(5), 6, 7, 8, 9, 10], (2, 3))
        var c = Tensor.concat([a, b], 1)
        return (c * c).sum()

    var device = Device()
    var a_data: List[Float32] = [1, 2, 3, 4]
    var num = numerical_grad[fwd](device, a_data, (2, 2))
    var a = Tensor(device, a_data, (2, 2), requires_grad=True)
    var b = Tensor(device, [Float32(5), 6, 7, 8, 9, 10], (2, 3))
    var c = Tensor.concat([a, b], 1)
    var loss = (c * c).sum()
    var grads = loss.gradient([a])
    assert_allclose(grads[0], num, tol=0.05)
    assert_allclose(grads[0], [Float32(2), 4, 6, 8])


def test_layer_norm_grad_dx() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4), requires_grad=True)
    var dy = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = Tensor(device, [Float32(1), 2, 1, 2], (4,), requires_grad=True)
    ln._bias[] = Tensor(device, [Float32(0), 0, 0, 0], (4,), requires_grad=True)
    var loss = (ln(x) * dy).sum()
    var grads = loss.gradient([x])
    assert_allclose(
        grads[0],
        [Float32(0.0), Float32(0.089443), Float32(-0.178886), Float32(0.089443)],
        tol=Float32(2e-4),
    )


def test_layer_norm_grad_dgamma() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4), requires_grad=True)
    var gamma = Tensor(device, [Float32(1), 2, 1, 2], (4,), requires_grad=True)
    var dy = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = gamma
    ln._bias[] = Tensor(device, [Float32(0), 0, 0, 0], (4,), requires_grad=True)
    var loss = (ln(x) * dy).sum()
    var grads = loss.gradient([gamma])
    assert_allclose(
        grads[0],
        [Float32(-0.134164), Float32(-0.089443), Float32(0.134164), Float32(0.536656)],
        tol=Float32(2e-4),
    )


def test_layer_norm_grad_dbeta() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4), requires_grad=True)
    var beta = Tensor(device, [Float32(0), 0, 0, 0], (4,), requires_grad=True)
    var dy = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4], (1, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = Tensor(device, [Float32(1), 2, 1, 2], (4,), requires_grad=True)
    ln._bias[] = beta
    var loss = (ln(x) * dy).sum()
    var grads = loss.gradient([beta])
    assert_allclose(
        grads[0],
        [Float32(0.1), Float32(0.2), Float32(0.3), Float32(0.4)],
        tol=Float32(1e-5),
    )


def test_layer_norm_grad_dbeta_two_rows_accumulates() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 1, 2, 3, 4], (2, 4), requires_grad=True)
    var beta = Tensor(device, [Float32(0), 0, 0, 0], (4,), requires_grad=True)
    var dy = Tensor(device, [Float32(0.1), 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4], (2, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = Tensor(device, [Float32(1), 1, 1, 1], (4,), requires_grad=True)
    ln._bias[] = beta
    var loss = (ln(x) * dy).sum()
    var grads = loss.gradient([beta])
    assert_allclose(
        grads[0],
        [Float32(0.2), Float32(0.4), Float32(0.6), Float32(0.8)],
        tol=Float32(1e-5),
    )


def test_layer_norm_grad_uniform_dy_gives_zero_dx() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (1, 4), requires_grad=True)
    var dy = Tensor(device, [Float32(1), 1, 1, 1], (1, 4))
    var ln = nn.LayerNorm(4)
    ln._weight[] = Tensor(device, [Float32(1), 1, 1, 1], (4,), requires_grad=True)
    ln._bias[] = Tensor(device, [Float32(0), 0, 0, 0], (4,), requires_grad=True)
    var loss = (ln(x) * dy).sum()
    var grads = loss.gradient([x])
    assert_allclose(
        grads[0],
        [Float32(0), Float32(0), Float32(0), Float32(0)],
        tol=Float32(1e-5),
    )


# ===-------------------------------------------------------------------===#
# Flash attention backward (SDPA gradient)
#
# Q/K/V are created in BSHD layout and transposed to BHSD, matching the model
# pattern that triggers fuse_flash_attention.  This ensures gradient() goes
# through flash_attn_bwd (via FLASH_ATTN_GRAD), not the unfused scalar ops.
#
# Dimensions are B=1, T=4, H=1, Dh=8. Dh=8 is the smallest MMA tile size.
#
# ===-------------------------------------------------------------------===#


def test_flash_attn_bwd_uses_flash_attn_grad_op() raises:
    # Verify that the backward graph for fused SDPA contains FLASH_ATTN_GRAD,
    # not decomposed scalar op grads.
    from mograd.op import OpRef
    from mograd.runtime.gpu.rewrites import FLASH_ATTN_GRAD
    from mograd.simplify import Simplifier
    from mograd.runtime.gpu.rewrites import GPU_REWRITES
    from mograd.grad import Grad
    from mograd.pattern_matcher import GraphUtils

    var device = Device()
    var Q_bshd = Tensor.randn(device, Layout(1, 4, 1, 8), seed=10, requires_grad=True)
    var K_bshd = Tensor.randn(device, Layout(1, 4, 1, 8), seed=11)
    var V_bshd = Tensor.randn(device, Layout(1, 4, 1, 8), seed=12)
    var loss = (
        Q_bshd.transpose(1, 2)
        .scaled_dot_product_attention(K_bshd.transpose(1, 2), V_bshd.transpose(1, 2), is_causal=True)
        .sum()
    )

    var fwd_simplified = Simplifier(GPU_REWRITES()).run(loss.op)
    var target_ops = List[OpRef]()
    target_ops.append(Q_bshd.op)
    var initial_grad = Tensor.ones(device, (1,)).op
    var grad_ops = Grad.compute(fwd_simplified, initial_grad, target_ops)

    # Toposort the backward graph and scan for FLASH_ATTN_GRAD
    assert_true(grad_ops[0] is not None)
    var topo = GraphUtils.toposort(grad_ops[0].value())
    var found = False
    for i in range(len(topo)):
        if topo[i].op_type() == FLASH_ATTN_GRAD:
            found = True
            break
    assert_true(found)


def _fused_grad(loss_op: OpRef, target: Tensor) raises -> Tensor:
    # Compute gradient through the FUSED (FLASH_ATTN → FLASH_ATTN_GRAD) path.
    var fwd_simplified = Simplifier(GPU_REWRITES()).run(loss_op)
    var initial_grad = Tensor.ones(target.device.value(), (1,)).op
    var targets = List[OpRef]()
    targets.append(target.op)
    var grads = Grad.compute(fwd_simplified, initial_grad, targets)
    return Tensor(target.device.value(), grads[0].value())


def test_flash_attn_bwd_dq_causal() raises:
    var device = Device()

    def fwd(q: Tensor) raises -> Tensor:
        var K = Tensor.randn(q.device.value(), Layout(1, 4, 1, 8), seed=21)
        var V = Tensor.randn(q.device.value(), Layout(1, 4, 1, 8), seed=22)
        return (
            q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
        )

    var Q = Tensor.randn(device, Layout(1, 4, 1, 8), seed=20, requires_grad=True)
    var K = Tensor.randn(device, Layout(1, 4, 1, 8), seed=21)
    var V = Tensor.randn(device, Layout(1, 4, 1, 8), seed=22)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
    )
    var dQ = _fused_grad(loss.op, Q)
    var num_dQ = numerical_grad[fwd](device, Q.to_list[DType.float32](), Layout(1, 4, 1, 8), eps=Float32(0.05))
    assert_allclose(dQ, num_dQ, tol=Float32(5e-2))


def test_flash_attn_bwd_dk_causal() raises:
    var device = Device()

    def fwd(k: Tensor) raises -> Tensor:
        var Q = Tensor.randn(k.device.value(), Layout(1, 4, 1, 8), seed=30)
        var V = Tensor.randn(k.device.value(), Layout(1, 4, 1, 8), seed=32)
        return (
            Q.transpose(1, 2).scaled_dot_product_attention(k.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
        )

    var Q = Tensor.randn(device, Layout(1, 4, 1, 8), seed=30)
    var K = Tensor.randn(device, Layout(1, 4, 1, 8), seed=31, requires_grad=True)
    var V = Tensor.randn(device, Layout(1, 4, 1, 8), seed=32)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
    )
    var dK = _fused_grad(loss.op, K)
    var num_dK = numerical_grad[fwd](device, K.to_list[DType.float32](), Layout(1, 4, 1, 8), eps=Float32(0.05))
    assert_allclose(dK, num_dK, tol=Float32(5e-2))


def test_flash_attn_bwd_dv_causal() raises:
    var device = Device()

    def fwd(v: Tensor) raises -> Tensor:
        var Q = Tensor.randn(v.device.value(), Layout(1, 4, 1, 8), seed=40)
        var K = Tensor.randn(v.device.value(), Layout(1, 4, 1, 8), seed=41)
        return (
            Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), v.transpose(1, 2), is_causal=True).sum()
        )

    var Q = Tensor.randn(device, Layout(1, 4, 1, 8), seed=40)
    var K = Tensor.randn(device, Layout(1, 4, 1, 8), seed=41)
    var V = Tensor.randn(device, Layout(1, 4, 1, 8), seed=42, requires_grad=True)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
    )
    var dV = _fused_grad(loss.op, V)
    var num_dV = numerical_grad[fwd](device, V.to_list[DType.float32](), Layout(1, 4, 1, 8), eps=Float32(0.05))
    assert_allclose(dV, num_dV, tol=Float32(5e-2))


def test_flash_attn_bwd_dv_no_mask() raises:
    var device = Device()

    def fwd(v: Tensor) raises -> Tensor:
        var Q = Tensor.randn(v.device.value(), Layout(1, 4, 1, 8), seed=50)
        var K = Tensor.randn(v.device.value(), Layout(1, 4, 1, 8), seed=51)
        return Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), v.transpose(1, 2)).sum()

    var Q = Tensor.randn(device, Layout(1, 4, 1, 8), seed=50)
    var K = Tensor.randn(device, Layout(1, 4, 1, 8), seed=51)
    var V = Tensor.randn(device, Layout(1, 4, 1, 8), seed=52, requires_grad=True)
    var loss = Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2)).sum()
    var dV = _fused_grad(loss.op, V)
    # eps=0.05: the fast-path P@V uses TF32 (precision ~1e-3), so eps=1e-3 causes
    # catastrophic cancellation in the finite difference.  Use a larger eps so the
    # signal (2*eps*col_sum(P) ~ 0.1) dominates TF32 rounding noise.
    var num_dV = numerical_grad[fwd](device, V.to_list[DType.float32](), Layout(1, 4, 1, 8), eps=Float32(0.05))
    assert_allclose(dV, num_dV, tol=Float32(5e-2))


def test_flash_attn_bwd_dv_causal_multi_tile() raises:
    # S=128 spans two 64-row tiles in both the Q and KV grids of the MMA
    # backward kernels, so dV must accumulate across Q-tile iterations.
    # With Q = K = 0 the causal softmax is exactly uniform: P[i,j] = 1/(i+1)
    # for j <= i, so for loss = sum(O), dV[j, :] = sum_{i>=j} 1/(i+1).
    var device = Device()
    var S = 128
    var Dh = 8
    var Q = Tensor.zeros(device, Layout(1, S, 1, Dh))
    var K = Tensor.zeros(device, Layout(1, S, 1, Dh))
    var V = Tensor.randn(device, Layout(1, S, 1, Dh), seed=70, requires_grad=True)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
    )
    var dV = _fused_grad(loss.op, V)
    var expected = List[Float32]()
    for j in range(S):
        var tail = Float32(0)
        for i in range(j, S):
            tail += Float32(1) / Float32(i + 1)
        for _ in range(Dh):
            expected.append(tail)
    assert_allclose(dV, expected, tol=Float32(2e-2))


def test_flash_attn_bwd_causal_multi_tile_matches_unfused() raises:
    # Fused FLASH_ATTN_GRAD vs the decomposed backward graph at S larger than
    # one 64-row MMA tile, catching per-tile accumulation bugs in dQ/dK/dV.
    # H=2 also exercises the BSHD (row stride H*D) indexing of Q/K/V/dQ/dK/dV.
    var device = Device()
    var S = 96
    var H = 2
    var Dh = 16
    var Q = Tensor.randn(device, Layout(1, S, H, Dh), seed=80, requires_grad=True)
    var K = Tensor.randn(device, Layout(1, S, H, Dh), seed=81, requires_grad=True)
    var V = Tensor.randn(device, Layout(1, S, H, Dh), seed=82, requires_grad=True)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
    )

    # Unfused reference: gradients of the raw graph, evaluated without GPU
    # rewrites so no flash kernels are involved.
    var initial = Tensor.ones(device, (1,)).op
    var targets = List[OpRef]()
    targets.append(Q.op)
    targets.append(K.op)
    targets.append(V.op)
    var ref_grads = Grad.compute(loss.op, initial, targets)

    var fused_dQ = _fused_grad(loss.op, Q)
    var fused_dK = _fused_grad(loss.op, K)
    var fused_dV = _fused_grad(loss.op, V)
    var ref_dQ = Tensor(device, ref_grads[0].value())
    var ref_dK = Tensor(device, ref_grads[1].value())
    var ref_dV = Tensor(device, ref_grads[2].value())
    assert_allclose(ref_dQ.value(simplifier=False), ref_dQ.op.layout(), fused_dQ, tol=5e-2)
    assert_allclose(ref_dK.value(simplifier=False), ref_dK.op.layout(), fused_dK, tol=5e-2)
    assert_allclose(ref_dV.value(simplifier=False), ref_dV.op.layout(), fused_dV, tol=5e-2)


def test_flash_attn_bwd_dq_bias_mask() raises:
    # Non-causal SDPA with a non-zero additive mask (bias): the backward must
    # include the bias when reconstructing P = exp(score + mask - lse).
    var device = Device()

    def fwd(q: Tensor) raises -> Tensor:
        var dev = q.device.value()
        var K = Tensor.randn(dev, Layout(1, 4, 1, 8), seed=91)
        var V = Tensor.randn(dev, Layout(1, 4, 1, 8), seed=92)
        var mask = Tensor.randn(dev, Layout(1, 1, 4, 4), seed=93)
        return (
            q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), attn_mask=mask).sum()
        )

    var Q = Tensor.randn(device, Layout(1, 4, 1, 8), seed=90, requires_grad=True)
    var K = Tensor.randn(device, Layout(1, 4, 1, 8), seed=91)
    var V = Tensor.randn(device, Layout(1, 4, 1, 8), seed=92)
    var mask = Tensor.randn(device, Layout(1, 1, 4, 4), seed=93)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), attn_mask=mask).sum()
    )
    var dQ = _fused_grad(loss.op, Q)
    var num_dQ = numerical_grad[fwd](device, Q.to_list[DType.float32](), Layout(1, 4, 1, 8), eps=Float32(0.05))
    assert_allclose(dQ, num_dQ, tol=Float32(5e-2))


def test_flash_attn_bwd_dv_bias_mask() raises:
    var device = Device()

    def fwd(v: Tensor) raises -> Tensor:
        var dev = v.device.value()
        var Q = Tensor.randn(dev, Layout(1, 4, 1, 8), seed=95)
        var K = Tensor.randn(dev, Layout(1, 4, 1, 8), seed=96)
        var mask = Tensor.randn(dev, Layout(1, 1, 4, 4), seed=98)
        return (
            Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), v.transpose(1, 2), attn_mask=mask).sum()
        )

    var Q = Tensor.randn(device, Layout(1, 4, 1, 8), seed=95)
    var K = Tensor.randn(device, Layout(1, 4, 1, 8), seed=96)
    var V = Tensor.randn(device, Layout(1, 4, 1, 8), seed=97, requires_grad=True)
    var mask = Tensor.randn(device, Layout(1, 1, 4, 4), seed=98)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), attn_mask=mask).sum()
    )
    var dV = _fused_grad(loss.op, V)
    var num_dV = numerical_grad[fwd](device, V.to_list[DType.float32](), Layout(1, 4, 1, 8), eps=Float32(0.05))
    assert_allclose(dV, num_dV, tol=Float32(5e-2))


def test_flash_attn_bwd_f16_causal_multi_tile_matches_unfused() raises:
    # f16 exercises the fused half-precision backward kernel (FA2-shaped,
    # single kernel + atomic dQ accumulation). S=96 spans two 64-row tiles
    # (ragged second tile), H=2 exercises BSHD strides, Dh=16 → D_BUCKET=64.
    var device = Device()
    var S = 96
    var H = 2
    var Dh = 16
    var Q = Tensor.randn(device, Layout(1, S, H, Dh), seed=180).cast(DType.float16)
    var K = Tensor.randn(device, Layout(1, S, H, Dh), seed=181).cast(DType.float16)
    var V = Tensor.randn(device, Layout(1, S, H, Dh), seed=182).cast(DType.float16)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), is_causal=True).sum()
    )

    var initial = Tensor.ones(device, (1,)).op
    var targets = List[OpRef]()
    targets.append(Q.op)
    targets.append(K.op)
    targets.append(V.op)
    var ref_grads = Grad.compute(loss.op, initial, targets)

    var fused_dQ = _fused_grad(loss.op, Q)
    var fused_dK = _fused_grad(loss.op, K)
    var fused_dV = _fused_grad(loss.op, V)
    var ref_dQ = Tensor(device, ref_grads[0].value())
    var ref_dK = Tensor(device, ref_grads[1].value())
    var ref_dV = Tensor(device, ref_grads[2].value())
    assert_allclose(ref_dQ.value(simplifier=False), ref_dQ.op.layout(), fused_dQ, tol=7e-2)
    assert_allclose(ref_dK.value(simplifier=False), ref_dK.op.layout(), fused_dK, tol=7e-2)
    assert_allclose(ref_dV.value(simplifier=False), ref_dV.op.layout(), fused_dV, tol=7e-2)


def test_flash_attn_bwd_f16_bias_mask_d128_matches_unfused() raises:
    # f16 half backward with an additive bias mask at Dh=128 (D_BUCKET=128,
    # the largest resident-tile configuration) and ragged S.
    var device = Device()
    var S = 80
    var H = 2
    var Dh = 128
    var Q = Tensor.randn(device, Layout(1, S, H, Dh), seed=190).cast(DType.float16)
    var K = Tensor.randn(device, Layout(1, S, H, Dh), seed=191).cast(DType.float16)
    var V = Tensor.randn(device, Layout(1, S, H, Dh), seed=192).cast(DType.float16)
    var mask = Tensor.randn(device, Layout(1, H, S, S), seed=193).cast(DType.float16)
    var loss = (
        Q.transpose(1, 2).scaled_dot_product_attention(K.transpose(1, 2), V.transpose(1, 2), attn_mask=mask).sum()
    )

    var initial = Tensor.ones(device, (1,)).op
    var targets = List[OpRef]()
    targets.append(Q.op)
    targets.append(K.op)
    targets.append(V.op)
    var ref_grads = Grad.compute(loss.op, initial, targets)

    var fused_dQ = _fused_grad(loss.op, Q)
    var fused_dK = _fused_grad(loss.op, K)
    var fused_dV = _fused_grad(loss.op, V)
    var ref_dQ = Tensor(device, ref_grads[0].value())
    var ref_dK = Tensor(device, ref_grads[1].value())
    var ref_dV = Tensor(device, ref_grads[2].value())
    assert_allclose(ref_dQ.value(simplifier=False), ref_dQ.op.layout(), fused_dQ, tol=7e-2)
    assert_allclose(ref_dK.value(simplifier=False), ref_dK.op.layout(), fused_dK, tol=7e-2)
    assert_allclose(ref_dV.value(simplifier=False), ref_dV.op.layout(), fused_dV, tol=7e-2)


def test_flash_attn_bwd_shapes() raises:
    var device = Device()
    var B = 2
    var T = 8
    var H = 4
    var Dh = 16
    var bshd = Layout(B, T, H, Dh)
    var Q_bshd = Tensor.randn(device, bshd, seed=60, requires_grad=True)
    var K_bshd = Tensor.randn(device, bshd, seed=61, requires_grad=True)
    var V_bshd = Tensor.randn(device, bshd, seed=62, requires_grad=True)
    var loss = (
        Q_bshd.transpose(1, 2)
        .scaled_dot_product_attention(K_bshd.transpose(1, 2), V_bshd.transpose(1, 2), is_causal=True)
        .sum()
    )
    var grads = loss.gradient([Q_bshd, K_bshd, V_bshd])
    assert_true(grads[0].shape() == Q_bshd.shape())
    assert_true(grads[1].shape() == K_bshd.shape())
    assert_true(grads[2].shape() == V_bshd.shape())


def main() raises:
    comptime assert has_accelerator(), "GPU required to run gradient tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
