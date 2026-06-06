from std.sys import has_accelerator
from std.testing import TestSuite, assert_almost_equal, assert_true

from mograd import Tensor, Device
from mograd.testing import assert_allclose, assert_close

# TODO: Uniform, reshape, broadcast

# ===-------------------------------------------------------------------===#
# Pointwise operations
# ===-------------------------------------------------------------------===#


def test_basic_mixed_add_sub_neg() raises:
    var device = Device()
    var a = Tensor.ones(device, (8, 16))
    var b = Tensor.ones(device, (8, 16))
    var c = ((a + b) * 2) - (3 * a) + (2 + b) + a - a
    assert_allclose(c, Tensor.full(device, (8, 16), 4))


def test_basic_mixed_mul_div() raises:
    var device = Device()
    var a = Tensor.ones(device, (8, 16))
    var b = Tensor.full(device, (8, 16), 2)
    var c = (((((a * b) + 1) * -b) * a) / a) / b
    assert_allclose(c, Tensor.full(device, (8, 16), -3))


def test_matmul() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4], (2, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1], (2, 2))
    assert_allclose(a @ identity, [1, 2, 3, 4])


def test_slice_preserves_shape_for_matmul() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (3, 2))
    var identity = Tensor(device, [Float32(1), 0, 0, 1], (2, 2))
    assert_allclose(x[1:3] @ identity, [Float32(3), 4, 5, 6])


def test_softmax_sums_to_one() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (4,))
    assert_close(x.softmax().sum(), Float32(1.0))


def test_one_hot_identity() raises:
    var device = Device()
    var labels = Tensor[DType.int64](device, [Int64(0), 1, 2], (3,))
    assert_allclose(labels.one_hot(3), [Int64(1), 0, 0, 0, 1, 0, 0, 0, 1])


def test_one_hot_shape() raises:
    var device = Device()
    var labels = Tensor[DType.int64](device, [Int64(0), 1, 2, 3], (4,))
    var oh = labels.one_hot(10)
    assert_true(oh.shape(0) == 4 and oh.shape(1) == 10)


def test_one_hot_values() raises:
    var device = Device()
    var labels = Tensor[DType.int64](device, [Int64(2), 0, 1], (3,))
    assert_allclose(
        labels.one_hot(4),
        [Int64(0), 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    )


def test_cross_entropy_uniform() raises:
    var device = Device()
    var logits = Tensor.full(device, (4, 10), Float32(0))
    var labels = Tensor[DType.int64](device, [Int64(0), 1, 2, 3], (4,))
    assert_close(logits.cross_entropy(labels.one_hot(10).cast[DType.float32]()), Float32(2.302585), tol=1e-3)


def test_cross_entropy_soft_labels() raises:
    var device = Device()
    var logits = Tensor.full(device, (2, 4), Float32(0))
    var labels = Tensor.full(device, (2, 4), Float32(0.25))
    assert_close(logits.cross_entropy(labels), Float32(1.386294), tol=1e-3)


def test_cross_entropy_certain_prediction() raises:
    var device = Device()
    var logits = Tensor(device, [Float32(100), 0, 0, 0, 100, 0], (2, 3))
    var labels = Tensor(device, [Int64(1), 0, 0, 0, 1, 0], (2, 3))
    assert_close(logits.cross_entropy(labels.cast[DType.float32]()), Float32(0.0), tol=1e-3)


def test_slice_rows() raises:
    var device = Device()
    var data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    var x = Tensor(device, data, (4, 3))
    assert_allclose(x[1:3], [Float32(4), 5, 6, 7, 8, 9])


def test_argmax() raises:
    var device = Device()
    var x = Tensor(device, [Float32(0.1), 0.9, 0.2, 0.3], (1, 4))
    assert_allclose(x.argmax(), [Float32(1)])


def test_log_exp_inverse() raises:
    var device = Device()
    var data: List[Float32] = [0.5, 1.0, 2.0]
    var x = Tensor(device, data, (3,))
    assert_allclose(x.exp().log(), data)


def test_relu_zeros_negatives() raises:
    var device = Device()
    var x = Tensor(device, [Float32(-2), -1, 0, 1, 2], (5,))
    assert_allclose(x.relu(), [Float32(0), 0, 0, 1, 2])


def test_eq_produces_zero_one() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3], (3,))
    var b = Tensor(device, [Float32(1), 0, 3], (3,))
    assert_allclose(a == b, [Float32(1), 0, 1])


def test_scale() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (4,))
    assert_allclose(x * Float32(3.0), [Float32(3), 6, 9, 12])


def test_sum() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (4,))
    assert_close(x.sum(), Float32(10.0))


def test_sum_large() raises:
    var device = Device()
    var n = 4096
    var x = Tensor.ones(device, (n,))
    assert_close(x.sum(), Float32(n))


def test_sum_of_softmax_is_one() raises:
    var device = Device()
    var x = Tensor.randn(device, (1024,))
    assert_close(x.softmax().sum(), Float32(1.0), tol=1e-4)


def test_randn_mean_and_std() raises:
    var device = Device()
    var x = Tensor.randn(device, (10000,), mean=0.0, std=1.0, seed=42)
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
    var device = Device()
    var a = Tensor.randn(device, (64,), seed=42)
    var b = Tensor.randn(device, (64,), seed=42)
    assert_allclose(a, b)


def test_randn_different_seeds_differ() raises:
    var device = Device()
    var a = Tensor.randn(device, (64,), seed=1).to_list()
    var b = Tensor.randn(device, (64,), seed=2).to_list()
    var any_diff = False
    for i in range(len(a)):
        if a[i] != b[i]:
            any_diff = True
            break
    assert_true(any_diff)


def test_transpose_2x3() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4, 5, 6], (2, 3))
    assert_allclose(x.transpose(), [Float32(1), 4, 2, 5, 3, 6])


def test_transpose_shape() raises:
    var device = Device()
    var x = Tensor.ones(device, (4, 7))
    var t = x.transpose()
    assert_true(t.shape(0) == 7 and t.shape(1) == 4)


def test_transpose_square() raises:
    var device = Device()
    var x = Tensor(device, [Float32(1), 2, 3, 4], (2, 2))
    assert_allclose(x.transpose(), [Float32(1), 3, 2, 4])


def test_mean_multi_batch_slice() raises:
    var device = Device()
    var x = Tensor.ones(device, (512, 8))
    var n = x.shape(0)
    var batch_size = 32
    for step in range(n // batch_size):
        var xb = x[step * batch_size : (step + 1) * batch_size]
        var m = xb.mean().item()
        assert_true(m > 0.99 and m < 1.01)


def test_transpose_tranpose() raises:
    var device = Device()
    var x = Tensor.randn(device, (113, 257), seed=7)
    assert_allclose(x.transpose().transpose(), x)


def test_cast_int64_to_float32() raises:
    var device = Device()
    var x = Tensor[DType.int64](device, [Int64(0), 1, 2, 3], (4,))
    var y = x.cast[DType.float32]()
    assert_allclose(y, [Float32(0), 1, 2, 3])


def test_cast_one_hot_to_float32() raises:
    var device = Device()
    var labels = Tensor[DType.int64](device, [Int64(0), 1, 2], (3,))
    var oh = labels.one_hot(3).cast[DType.float32]()
    assert_allclose(oh, [Float32(1), 0, 0, 0, 1, 0, 0, 0, 1])


def main() raises:
    comptime assert has_accelerator(), "GPU required to run tensor op tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
