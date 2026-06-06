from std.sys import has_accelerator
from std.testing import TestSuite, assert_true

from mograd import Tensor, Device
from mograd.data import mnist


def test_mnist_shapes() raises:
    var ctx = Device()
    var data = mnist(ctx)
    assert_true(data.x_train.shape(0) == 60000 and data.x_train.shape(1) == 784)
    assert_true(data.y_train.shape(0) == 60000)
    assert_true(data.x_test.shape(0) == 10000 and data.x_test.shape(1) == 784)
    assert_true(data.y_test.shape(0) == 10000)


def test_mnist_images_in_unit_range() raises:
    var ctx = Device()
    var data = mnist(ctx)
    var sample = data.x_train[0:1].to_list()
    for v in sample:
        assert_true(v >= 0.0 and v <= 1.0)


def test_mnist_labels_are_integer_0_to_9() raises:
    var ctx = Device()
    var data = mnist(ctx)
    var labels = data.y_train[0:100].to_list()
    for l in labels:
        var li = Int(l)
        assert_true(Float32(li) == l)
        assert_true(li >= 0 and li <= 9)


def test_mnist_slice_shape() raises:
    var ctx = Device()
    var data = mnist(ctx)
    var xb = data.x_train[0:32]
    assert_true(xb.shape(0) == 32 and xb.shape(1) == 784)
    var yb = data.y_train[0:32]
    assert_true(yb.shape(0) == 32)


def main() raises:
    comptime assert has_accelerator(), "GPU required to run data tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
