from std.python import Python
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true

from mograd import Tensor, Device
from mograd.data import mnist, tiny_shakespeare

# TODO: Ensure prior data is removed.


def _fresh_cache_dir() raises -> String:
    var tempfile = Python.import_module("tempfile")
    return String(tempfile.mkdtemp())


def test_mnist_shapes() raises:
    var device = Device()
    var data = mnist(device)
    assert_true(data.x_train.shape(0) == 60000 and data.x_train.shape(1) == 784)
    assert_true(data.y_train.shape(0) == 60000)
    assert_true(data.x_test.shape(0) == 10000 and data.x_test.shape(1) == 784)
    assert_true(data.y_test.shape(0) == 10000)


def test_mnist_images_in_unit_range() raises:
    var device = Device()
    var data = mnist(device)
    var sample = data.x_train[0:1].to_list()
    for v in sample:
        assert_true(v >= 0.0 and v <= 1.0)


def test_mnist_labels_are_integer_0_to_9() raises:
    var device = Device()
    var data = mnist(device)
    var labels = data.y_train[0:100].to_list()
    for l in labels:
        var li = Int(l)
        assert_true(Float32(li) == l)
        assert_true(li >= 0 and li <= 9)


def test_mnist_slice_shape() raises:
    var device = Device()
    var data = mnist(device)
    var xb = data.x_train[0:32]
    assert_true(xb.shape(0) == 32 and xb.shape(1) == 784)
    var yb = data.y_train[0:32]
    assert_true(yb.shape(0) == 32)


def test_tiny_shakespeare_shapes() raises:
    var device = Device()
    var data = tiny_shakespeare(device, cache_dir=_fresh_cache_dir())
    assert_true(data.vocab_size == 65)
    assert_true(data.data.shape(0) == data.train_size + data.val_size)
    assert_true(data.train_size > data.val_size)


def test_tiny_shakespeare_decodes_known_prefix() raises:
    var device = Device()
    var cache_dir = _fresh_cache_dir()
    var data = tiny_shakespeare(device, cache_dir=cache_dir)
    var ids = data.data[0:13].to_list[DType.int64]()
    assert_true(data.decode(ids) == "First Citizen")


def main() raises:
    comptime assert has_accelerator(), "GPU required to run data tests"
    TestSuite.discover_tests[__functions_in_module()]().run()
