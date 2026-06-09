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

def test_grad_add_strided() raises:
    var device = Device()
    var a_data: List[Float32] = [1, 2, 3, 4, 5, 6, 7, 8]
    var a = Tensor(device, a_data, (4, 2), requires_grad=True)
    var b = Tensor.ones(device, (2, 2))
    var c = (a[0:4:2] + b).sum()
    var grads = c.gradient([a])
    assert_allclose(grads[0], [Float32(1), 1, 0, 0, 1, 1, 0, 0])


def main() raises:
    comptime assert has_accelerator(), "GPU required to run non-contiguous tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
