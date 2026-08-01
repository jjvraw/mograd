from std.testing import assert_equal, assert_true, TestSuite

from mograd import Tensor, Device


def test_stats_count_kernels_and_runs() raises:
    var device = Device()
    var a = Tensor.randn(device, (64, 64))
    var b = Tensor.randn(device, (64, 64), seed=7)
    _ = ((a @ b).relu() + a).sum().item()
    assert_equal(device.stats[].runs, 1)
    assert_equal(device.stats[].kernels, 6)
    assert_true(device.stats[].gpu_ns > 0)
    assert_true(device.stats[].dispatch_ns > 0)
    assert_true(device.stats[].bytes_moved > 0)


def test_stats_accumulate_across_runs() raises:
    var device = Device()
    var a = Tensor.randn(device, (32, 32))
    _ = a.relu().sum().item()
    var kernels_after_first = device.stats[].kernels
    _ = (a + a).sum().item()
    assert_equal(device.stats[].runs, 2)
    # The RANDN result is cached from the first run, so the second run only
    # dispatches ADD and SUM.
    assert_equal(device.stats[].kernels, kernels_after_first + 2)


def test_stats_shared_across_device_copies() raises:
    var device = Device()
    var copied = device.copy()
    var a = Tensor.randn(device, (16, 16))
    _ = a.sum().item()
    assert_equal(copied.stats[].runs, device.stats[].runs)
    assert_true(copied.stats[].kernels > 0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
