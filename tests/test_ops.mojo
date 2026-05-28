from std.sys import has_accelerator
from std.testing import TestSuite

from mograd import Tensor, DeviceContext
from mograd.testing import assert_allclose, assert_close

# ===-------------------------------------------------------------------===#
# Pointwise operations
# ===-------------------------------------------------------------------===#


def test_basic_mixed_add_sub_neg() raises:
    var ctx = DeviceContext()
    var a = Tensor.ones(ctx, [8, 16])
    var b = Tensor.ones(ctx, [8, 16])
    var c = ((a + b) * 2) - (3 * a) + (2 + b) + a - a
    assert_allclose(c, Tensor.full(ctx, [8, 16], 4))


def test_basic_mixed_mul_div() raises:
    var ctx = DeviceContext()
    var a = Tensor.ones(ctx, [8, 16])
    var b = Tensor.full(ctx, [8, 16], 2)
    var c = (((((a * b) + 1) * -b) * a) / a) / b
    assert_allclose(c, Tensor.full(ctx, [8, 16], -3))


def test_matmul() raises:
    var ctx = DeviceContext()
    var a = Tensor(ctx, [Float32(1), 2, 3, 4], [2, 2])
    var identity = Tensor(ctx, [Float32(1), 0, 0, 1], [2, 2])
    assert_allclose(a @ identity, [Float32(1), 2, 3, 4])


def test_softmax_sums_to_one() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], [4])
    assert_close(x.softmax().sum(), Float32(1.0))


def test_cross_entropy_uniform() raises:
    var ctx = DeviceContext()
    var logits_data = List[Float32]()
    for _ in range(40):
        logits_data.append(Float32(0.0))
    var logits = Tensor(ctx, logits_data, [4, 10])
    var labels = Tensor(ctx, [Float32(0), 1, 2, 3], [4])
    assert_close(logits.cross_entropy(labels), Float32(2.302585), tol=1e-3)


def test_slice_rows() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    var x = Tensor(ctx, data, [4, 3])
    assert_allclose(x[1:3], [Float32(4), 5, 6, 7, 8, 9])


def test_argmax() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(0.1), 0.9, 0.2, 0.3], [1, 4])
    assert_allclose(x.argmax(), [Float32(1)])


def test_log_exp_inverse() raises:
    var ctx = DeviceContext()
    var data: List[Float32] = [0.5, 1.0, 2.0]
    var x = Tensor(ctx, data, [3])
    assert_allclose(x.exp().log(), data)


def test_relu_zeros_negatives() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(-2), -1, 0, 1, 2], [5])
    assert_allclose(x.relu(), [Float32(0), 0, 0, 1, 2])


def test_eq_produces_zero_one() raises:
    var ctx = DeviceContext()
    var a = Tensor(ctx, [Float32(1), 2, 3], [3])
    var b = Tensor(ctx, [Float32(1), 0, 3], [3])
    assert_allclose(a == b, [Float32(1), 0, 1])


def test_scale() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], [4])
    assert_allclose(x * Float32(3.0), [Float32(3), 6, 9, 12])


def test_sum() raises:
    var ctx = DeviceContext()
    var x = Tensor(ctx, [Float32(1), 2, 3, 4], [4])
    assert_close(x.sum(), Float32(10.0))


def main() raises:
    comptime assert has_accelerator(), "GPU required to run tensor op tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
