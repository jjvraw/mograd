from std.testing import TestSuite, assert_equal, assert_true, assert_false

from mograd import Device
from mograd.buffer import Buffer, AnyBuffer
from mograd.layout import Layout
from mograd.memory import (
    memory_stats,
    pool_enabled,
    round_class,
    scratch_take,
    take_storage,
)

# ===-------------------------------------------------------------------===#
# Size classes
# ===-------------------------------------------------------------------===#


def test_round_class() raises:
    assert_equal(round_class(0), 512)
    assert_equal(round_class(1), 512)
    assert_equal(round_class(512), 512)
    assert_equal(round_class(513), 576)
    assert_equal(round_class(640), 640)
    assert_equal(round_class(1000), 1024)
    assert_equal(round_class(1024), 1024)
    assert_equal(round_class(1025), 1152)
    assert_equal(round_class(4096), 4096)
    assert_equal(round_class(5000), 5120)
    assert_equal(round_class(100_000), 106_496)
    assert_equal(round_class(1 << 20), 1 << 20)
    assert_equal(round_class((1 << 20) + 1), (1 << 20) + (1 << 17))


# ===-------------------------------------------------------------------===#
# Reuse
# ===-------------------------------------------------------------------===#


def _alloc_and_drop_buffer(device: Device, numel: Int) raises -> Int:
    """Allocates a float32 buffer, returns its pointer; the buffer dies here."""
    var b = Buffer[DType.float32].empty(device, numel)
    return Int(b.data_ptr())


def test_reuse_returns_same_pointer() raises:
    var device = Device()
    var p1 = _alloc_and_drop_buffer(device, 1000)
    var b2 = Buffer[DType.float32].empty(device, 1000)
    if pool_enabled():
        assert_equal(Int(b2.data_ptr()), p1)


def _take_and_drop_storage(device: Device, nbytes: Int) raises -> Int:
    var s = take_storage(device.ctx, nbytes)
    return Int(s.bytes.unsafe_ptr())


def test_size_class_sharing() raises:
    var device = Device()
    # 1000 B and 990 B both round to the 1024 B class.
    var p1 = _take_and_drop_storage(device, 1000)
    var s2 = take_storage(device.ctx, 990)
    if pool_enabled():
        assert_equal(Int(s2.bytes.unsafe_ptr()), p1)


def _make_view(device: Device) raises -> AnyBuffer:
    """Returns a view of a fresh buffer; the original Buffer dies here."""
    var b = Buffer[DType.float32].empty(device, 256)
    var any = AnyBuffer(b^)
    return any.view(Layout(16, 16))


def _drop(var any: AnyBuffer):
    pass


def test_view_keeps_storage_alive() raises:
    var device = Device()
    var v = _make_view(device)
    # The view's layout has base_offset 0, so its pointer is the storage base.
    var p1 = Int(v.data_ptr())
    # The view still co-owns the storage: a same-class alloc must not get it.
    var other = Buffer[DType.float32].empty(device, 256)
    if pool_enabled():
        assert_true(Int(other.data_ptr()) != p1)
    _drop(v^)
    var reused = Buffer[DType.float32].empty(device, 256)
    if pool_enabled():
        assert_equal(Int(reused.data_ptr()), p1)
    # Referencing `other` here also keeps it alive through `reused`'s
    # allocation, so its block cannot have been what `reused` received.
    assert_true(Int(other.data_ptr()) != Int(reused.data_ptr()))


def _scratch_fill_and_drop(device: Device) raises -> Int:
    var s = scratch_take[DType.float32](device.ctx, 128)
    s.enqueue_fill(3.5)
    device.ctx.synchronize()
    with s.view.map_to_host() as host:
        assert_equal(host[0], 3.5)
        assert_equal(host[127], 3.5)
    return Int(s.unsafe_ptr())


def test_scratch_take_reuse() raises:
    var device = Device()
    var p1 = _scratch_fill_and_drop(device)
    var s2 = scratch_take[DType.float32](device.ctx, 128)
    if pool_enabled():
        assert_equal(Int(s2.unsafe_ptr()), p1)


def test_layout_buffer_contents() raises:
    var device = Device()
    var layout = Layout(4, 6)
    var strides = layout.strides_buffer(device.ctx)
    device.ctx.synchronize()
    with strides.view.map_to_host() as host:
        for i in range(layout.rank()):
            assert_equal(host[i], Int64(layout.stride(i)))


# ===-------------------------------------------------------------------===#
# Cache control and stats
# ===-------------------------------------------------------------------===#


def test_empty_cache_drops_reserved() raises:
    var device = Device()
    _ = _alloc_and_drop_buffer(device, 4096)
    if not pool_enabled():
        return
    var stats = device.memory_stats()
    assert_true(stats.reserved_bytes > stats.allocated_bytes)
    device.empty_cache()
    stats = device.memory_stats()
    assert_equal(stats.reserved_bytes, stats.allocated_bytes)
    # The cache is empty, so the next allocation is a real one.
    var misses = stats.miss_count
    var b = Buffer[DType.float32].empty(device, 4096)
    assert_equal(device.memory_stats().miss_count, misses + 1)


def _alloc_two_and_drop(device: Device) raises -> Int:
    var b1 = Buffer[DType.float32].empty(device, 1024)
    var b2 = Buffer[DType.float32].empty(device, 1024)
    # Both alive at once: two distinct blocks.
    return Int(b1.data_ptr()) + Int(b2.data_ptr())


def test_stats_counts() raises:
    var device = Device()
    if not pool_enabled():
        return
    var t0 = device.memory_stats()
    _ = _alloc_two_and_drop(device)  # two misses in the 4096 B class
    var stats = device.memory_stats()
    assert_equal(stats.miss_count, t0.miss_count + 2)
    assert_equal(stats.free_count, t0.free_count + 2)
    assert_equal(stats.allocated_bytes, t0.allocated_bytes)
    assert_true(stats.peak_allocated >= t0.allocated_bytes + 8192)
    assert_true(stats.reserved_bytes >= 8192)
    var b3 = Buffer[DType.float32].empty(device, 1024)
    stats = device.memory_stats()
    assert_equal(stats.hit_count, t0.hit_count + 1)
    assert_equal(stats.allocated_bytes, t0.allocated_bytes + 4096)
    # Also keeps b3 alive through the stats reads above.
    assert_true(Int(b3.data_ptr()) != 0)


# ===-------------------------------------------------------------------===#
# Teardown
# ===-------------------------------------------------------------------===#


def test_device_teardown_loop() raises:
    # Regression tripwire for the ~600s poisoned-mutex stall: a Device dying
    # with pooled buffers released but unsynchronized must not stall the next
    # context's first allocation (BUG_REPORT_asyncrt_first_dispatch_stall.md).
    # The suite's own runtime bounds this: any stall is an obvious hang.
    for _ in range(5):
        var device = Device()
        var b = Buffer[DType.float32].full(device, 1.5, 1024)
        assert_equal(b.item(), 1.5)


def _make_device_and_buffer() raises -> Buffer[DType.float32]:
    """Returns a pooled buffer whose Device dies here."""
    var device = Device()
    return Buffer[DType.float32].full(device, 2.0, 64)


def test_buffer_outlives_device() raises:
    var b = _make_device_and_buffer()
    # The buffer co-owns the teardown guard: storage and context stay valid
    # until the last pooled buffer dies.
    assert_equal(b.item(), 2.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
