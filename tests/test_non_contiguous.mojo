from std.sys import has_accelerator
from std.testing import TestSuite, assert_true, assert_false

from mograd import Tensor, Device
from mograd.layout import Layout
from mograd.testing import assert_allclose, assert_close


def test_add_strided_step2() raises:
    device = Device()
    a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    b = Tensor(device, [Float32(10), 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], (6, 2))
    sa = a[0:6:2]
    sb = b[0:6:2]
    c = sa + sb
    assert_true(not sa.is_contiguous() or not sb.is_contiguous() or c.is_contiguous())
    assert_allclose(c, [Float32(11), 22, 55, 66, 99, 110])


def test_add_strided_lhs_only() raises:
    device = Device()
    a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (4, 2))
    b = Tensor.full(device, (2, 2), Float32(100))
    sa = a[0:4:2]
    c = sa + b
    assert_true(not sa.is_contiguous() or b.is_contiguous() or c.is_contiguous())
    assert_allclose(c, [Float32(101), 102, 105, 106])


def test_add_strided_step3() raises:
    device = Device()
    a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    zeros = Tensor.full(device, (2, 2), Float32(0))
    sa = a[0:6:3]
    c = sa + zeros
    assert_true(not sa.is_contiguous() or zeros.is_contiguous() or c.is_contiguous())
    assert_allclose(c, [Float32(1), 2, 7, 8])


def test_add_both_strided_different_offsets() raises:
    var device = Device()
    a = Tensor(device, [Float32(1), 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6], (6, 2))
    b = Tensor(device, [Float32(10), 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60], (6, 2))
    c = a[0:6:2] + b[1:6:2]
    assert_allclose(c, [Float32(21), 21, 43, 43, 65, 65])

def test_sub_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(10), 20, 30, 40, 50, 60], (3, 2))
    var b = Tensor.full(device, (2, 2), Float32(1))
    var c = a[0:3:2] - b
    assert_allclose(c, [Float32(9), 19, 49, 59])

def test_log_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2.718282, 0, 0, 7.389056, 1, 0, 0], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = sa.log()
    assert_allclose(c, [Float32(0), 1, 2, 0], tol=1e-5)


def test_exp_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(0), 1, 0, 0, 2, 0, 0, 0], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = sa.exp()
    assert_allclose(c, [Float32(1), 2.718282, 7.389056, 1], tol=1e-5)


def test_relu_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(-1), 2, 0, 0, -3, 4, 0, 0], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = sa.relu()
    assert_allclose(c, [Float32(0), 2, 0, 4])


def test_neg_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = -sa
    assert_allclose(c, [Float32(-1), -2, -5, -6])


def test_eq_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 0, 2, 0, 1, 0, 3, 0], (4, 2))
    var b = Tensor.full(device, (2, 2), Float32(1))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = sa == b
    assert_allclose(c, [Float32(1), 0, 1, 0])


def test_cast_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = sa.cast[DType.float16]()
    assert_allclose(c, [Float16(1), 2, 5, 6])


def test_scale_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = sa * Float32(3)
    assert_allclose(c, [Float32(3), 6, 15, 18])


def test_grad_relu_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(-1), 2, 0, 0, -3, 4, 0, 0], (4, 2), requires_grad=True)
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var c = sa.relu().sum()
    var grads = c.gradient([a])
    assert_allclose(grads[0], [Float32(0), 1, 0, 0, 0, 1, 0, 0])


def test_grad_add_strided() raises:
    var device = Device()
    var a_data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8]
    var a = Tensor(device, a_data, (4, 2), requires_grad=True)
    var b = Tensor.ones(device, (2, 2))
    var c = (a[0:4:2] + b).sum()
    var grads = c.gradient([a])
    assert_allclose(grads[0], [Float32(1), 1, 0, 0, 1, 1, 0, 0])


def test_mul_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    var b = Tensor.full(device, (3, 2), Float32(2))
    var sa = a[0:6:2]
    assert_true(not sa.is_contiguous())
    var c = sa * b
    assert_allclose(c, [Float32(2), 4, 10, 12, 18, 20])


def test_div_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(2), 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24], (6, 2))
    var b = Tensor.full(device, (3, 2), Float32(2))
    var sa = a[0:6:2]
    assert_true(not sa.is_contiguous())
    var c = sa / b
    assert_allclose(c, [Float32(1), 2, 5, 6, 9, 10])


def test_one_hot_strided_labels() raises:
    var device = Device()
    var labels = Tensor(device, [Int64(0), 99, 2, 99, 1, 99], (6,))
    var sliced = labels[0:6:2]
    assert_true(not sliced.is_contiguous())
    var oh = sliced.one_hot(3)
    assert_allclose(oh, [Int64(1), 0, 0, 0, 0, 1, 0, 1, 0])


def test_transpose_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    var sa = a[0:6:2]
    assert_true(not sa.is_contiguous())
    assert_allclose(sa.transpose(), [Float32(1), 5, 9, 2, 6, 10])


def test_matmul_strided_lhs() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    var sa = a[0:6:2]
    assert_true(not sa.is_contiguous())
    var identity = Tensor(device, [Float32(1), 0, 0, 1], (2, 2))
    assert_allclose(sa @ identity, [Float32(1), 2, 5, 6, 9, 10])


def test_matmul_strided_rhs() raises:
    var device = Device()
    var b = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    var sb = b[0:6:2]
    assert_true(not sb.is_contiguous())
    var ones = Tensor.ones(device, (2, 3))
    assert_allclose(ones @ sb, [Float32(15), 18, 15, 18])


def test_sum_axis_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    var sa = a[0:6:2]
    assert_true(not sa.is_contiguous())
    assert_allclose(sa.sum(axis=0), [Float32(15), 18])


def test_grad_sum_axis_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2), requires_grad=True)
    var sa = a[0:6:2]
    assert_true(not sa.is_contiguous())
    var loss = sa.sum(axis=0).sum()
    var grads = loss.gradient([a])
    assert_allclose(grads[0], [Float32(1), 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])


def test_softmax_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 100, 100, 3, 4, 100, 100], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var out = sa.softmax()
    assert_allclose(out.sum(axis=1), [Float32(1), 1], tol=1e-5)
    var row1 = out[1:2]
    assert_true(row1.to_list()[0] < Float32(0.4))


def test_argmax_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 4, 5, 0, 2, 3, 0, 9], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    assert_allclose(sa.argmax(axis=1), [Float32(1), 1])


def test_argmax_global_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 4, 5, 0, 2, 3, 0, 9], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    assert_allclose(sa.argmax(), [Float32(1)])


def test_argmax_axis0_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 4, 5, 0, 2, 3, 0, 9], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    assert_allclose(sa.argmax(axis=0), [Float32(1), 0])


def test_argmax_keepdim_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 4, 5, 0, 2, 3, 0, 9], (4, 2))
    var sa = a[0:4:2]
    assert_true(not sa.is_contiguous())
    var out = sa.argmax(axis=1, keepdim=True)
    assert_true(out.shape(0) == 2 and out.shape(1) == 1)
    assert_allclose(out, [Float32(1), 1])


def test_cross_entropy_strided_logits() raises:
    var device = Device()
    var logits = Tensor(device, [Float32(0), 0, 100, 0, 0, 0, 0, 0], (4, 2))
    var sl = logits[0:4:2]
    assert_true(not sl.is_contiguous())
    var labels = Tensor.full(device, (2, 2), Float32(0.5))
    assert_close(sl.cross_entropy(labels), Float32(0.693147), tol=1e-3)


def test_reshape_strided() raises:
    var device = Device()
    var a = Tensor(device, [Float32(1), 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], (6, 2))
    var sa = a[0:6:2]  # [[1,2],[5,6],[9,10]]
    assert_true(not sa.is_contiguous())
    var r = sa.reshape((1, 6))
    assert_allclose(r, [Float32(1), 2, 5, 6, 9, 10])


def main() raises:
    comptime assert has_accelerator(), "GPU required to run non-contiguous tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
