from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_true

from mograd import Tensor, DeviceContext
from mograd.testing import assert_allclose, assert_close

# ===-------------------------------------------------------------------===#
# Pointwise operations
# ===-------------------------------------------------------------------===#


def test_basic_mixed_add_sub_neg() raises:
    var ctx = DeviceContext()
    var a = Tensor.ones(ctx, (8, 16))
    var b = Tensor.ones(ctx, (8, 16))
    var c = ((a + b) * 2) - (3 * a) + (2 + b) + a - a
    assert_allclose(c, Tensor.full(ctx, (8, 16), 4))


def test_basic_mixed_mul_div() raises:
    var ctx = DeviceContext()
    var a = Tensor.ones(ctx, (8, 16))
    var b = Tensor.full(ctx, (8, 16), 2)
    var c = (((((a * b) + 1) * -b) * a) / a) / b
    assert_allclose(c, Tensor.full(ctx, (8, 16), -3))


def test_matmul() raises:
    var ctx = DeviceContext()
    var a = Tensor(ctx, [1, 2, 3, 4], (2, 2))
    var identity = Tensor(ctx, [1, 0, 0, 1], (2, 2))
    assert_allclose(a @ identity, [1, 2, 3, 4])


def test_slice_preserves_shape_for_matmul() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [1, 2, 3, 4, 5, 6], (3, 2))
    var identity = Tensor(ctx, [1, 0, 0, 1], (2, 2))
    assert_allclose(x[1:3] @ identity, [Float32(3), 4, 5, 6])


def test_softmax_sums_to_one() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], (4,))
    assert_close(x.softmax().sum(), Float32(1.0))


def test_one_hot_identity() raises:
    var ctx = DeviceContext()
    var labels = Tensor(ctx, [Float32(0), 1, 2], (3,))
    assert_allclose(labels.one_hot(3), [Float32(1), 0, 0, 0, 1, 0, 0, 0, 1])


def test_one_hot_shape() raises:
    var ctx = DeviceContext()
    var labels = Tensor(ctx, [Float32(0), 1, 2, 3], (4,))
    var oh = labels.one_hot(10)
    assert_true(oh.shape(0) == 4 and oh.shape(1) == 10)


def test_one_hot_values() raises:
    var ctx = DeviceContext()
    var labels = Tensor(ctx, [Float32(2), 0, 1], (3,))
    assert_allclose(
        labels.one_hot(4),
        [
            Float32(0),
            0,
            1,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
        ],
    )


def test_cross_entropy_uniform() raises:
    var ctx = DeviceContext()
    var logits = Tensor.full(ctx, (4, 10), Float32(0))
    var labels = Tensor(ctx, [Float32(0), 1, 2, 3], (4,))
    assert_close(logits.cross_entropy(labels.one_hot(10)), Float32(2.302585), tol=1e-3)


def test_cross_entropy_soft_labels() raises:
    var ctx = DeviceContext()
    var logits = Tensor.full(ctx, (2, 4), Float32(0))
    var labels = Tensor.full(ctx, (2, 4), Float32(0.25))
    assert_close(logits.cross_entropy(labels), Float32(1.386294), tol=1e-3)


def test_cross_entropy_certain_prediction() raises:
    var ctx = DeviceContext()
    var logits = Tensor(ctx, [Float32(100), 0, 0, 0, 100, 0], (2, 3))
    var labels = Tensor(ctx, [Float32(1), 0, 0, 0, 1, 0], (2, 3))
    assert_close(logits.cross_entropy(labels), Float32(0.0), tol=1e-3)


def test_slice_rows() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    var x = Tensor(ctx, data, (4, 3))
    assert_allclose(x[1:3], [Float32(4), 5, 6, 7, 8, 9])


def test_argmax() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(0.1), 0.9, 0.2, 0.3], (1, 4))
    assert_allclose(x.argmax(), [Float32(1)])


def test_log_exp_inverse() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [0.5, 1.0, 2.0]
    var x = Tensor(ctx, data, (3,))
    assert_allclose(x.exp().log(), data)


def test_relu_zeros_negatives() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(-2), -1, 0, 1, 2], (5,))
    assert_allclose(x.relu(), [Float32(0), 0, 0, 1, 2])


def test_eq_produces_zero_one() raises:
    var ctx = DeviceContext()
    var a = Tensor(ctx, [Float32(1), 2, 3], (3,))
    var b = Tensor(ctx, [Float32(1), 0, 3], (3,))
    assert_allclose(a == b, [Float32(1), 0, 1])


def test_scale() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], (4,))
    assert_allclose(x * Float32(3.0), [Float32(3), 6, 9, 12])


def test_sum() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], (4,))
    assert_close(x.sum(), Float32(10.0))


def test_sum_large() raises:
    var ctx = DeviceContext()
    var n = 4096
    var x = Tensor.ones(ctx, (n,))
    assert_close(x.sum(), Float32(n))


def test_sum_of_softmax_is_one() raises:
    var ctx = DeviceContext()
    var x = Tensor.randn(ctx, (1024,))
    assert_close(x.softmax().sum(), Float32(1.0), tol=1e-4)


def test_randn_mean_and_std() raises:
    var ctx = DeviceContext()
    var x = Tensor.randn(ctx, (10000,), mean=0.0, std=1.0, seed=42)
    var vals = x.to_list()
    var mean = Float32(0.0)
    var sq = Float32(0.0)
    var n = Float32(len(vals))
    for i in range(len(vals)):
        mean += vals[i]
    mean /= n
    for i in range(len(vals)):
        sq += (vals[i] - mean) * (vals[i] - mean)
    var std = (sq / n) ** Float32(0.5)
    assert_almost_equal(mean, Float32(0.0), atol=0.05)
    assert_almost_equal(std, Float32(1.0), atol=0.05)


def test_randn_reproducible() raises:
    var ctx = DeviceContext()
    var a = Tensor.randn(ctx, (64,), seed=42)
    var b = Tensor.randn(ctx, (64,), seed=42)
    assert_allclose(a, b)


def test_randn_different_seeds_differ() raises:
    var ctx = DeviceContext()
    var a = Tensor.randn(ctx, (64,), seed=1).to_list()
    var b = Tensor.randn(ctx, (64,), seed=2).to_list()
    var any_diff = False
    for i in range(len(a)):
        if a[i] != b[i]:
            any_diff = True
            break
    assert_true(any_diff)


def test_transpose_2x3() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    assert_allclose(x.transpose(), [Float32(1), 4, 2, 5, 3, 6])


def test_transpose_shape() raises:
    var ctx = DeviceContext()
    var x = Tensor.ones(ctx, (4, 7))
    var t = x.transpose()
    assert_true(t.shape(0) == 7 and t.shape(1) == 4)


def test_transpose_square() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], (2, 2))
    assert_allclose(x.transpose(), [Float32(1), 3, 2, 4])


def test_transpose_tranpose() raises:
    var ctx = DeviceContext()
    var x = Tensor.randn(ctx, (113, 257), seed=7)
    assert_allclose(x.transpose().transpose(), x)


def main() raises:
    comptime assert has_accelerator(), "GPU required to run tensor op tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
