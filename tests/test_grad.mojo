from std.math import sqrt
from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_equal, assert_true

import mograd.nn as nn
from mograd import Tensor, Device
from mograd.layout import Layout
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


def main() raises:
    comptime assert has_accelerator(), "GPU required to run gradient tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
