from std.testing import TestSuite, assert_equal, assert_true, assert_false

from mograd import Device
from mograd.buffer import Buffer, AnyBuffer
from mograd.layout import Layout

# ===-------------------------------------------------------------------===#
# Size classes
# ===-------------------------------------------------------------------===#


def _alloc_and_drop_bytes(device: Device, nbytes: Int) raises -> Int:
    """Allocates a uint8 buffer of nbytes, returns its pointer; dies here."""
    var b = Buffer[DType.uint8].empty(device, nbytes)
    return Int(b.data_ptr())


def test_size_classes_round_up() raises:
    # Requests within one power-of-2/8 class share a block; a request that
    # crosses into the next class does not. (round_class(1000) == 1024,
    # round_class(1025) == 1152.)
    var device = Device()
    if not device.pool_enabled():
        return
    var p_1000 = _alloc_and_drop_bytes(device, 1000)
    var b_1024 = Buffer[DType.uint8].empty(device, 1024)
    assert_equal(Int(b_1024.data_ptr()), p_1000)
    var b_1025 = Buffer[DType.uint8].empty(device, 1025)
    assert_true(Int(b_1025.data_ptr()) != p_1000)


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
    if device.pool_enabled():
        assert_equal(Int(b2.data_ptr()), p1)


def test_size_class_sharing() raises:
    var device = Device()
    # 1000 B and 990 B both round to the 1024 B class.
    var p1 = _alloc_and_drop_bytes(device, 1000)
    var b2 = Buffer[DType.uint8].empty(device, 990)
    if device.pool_enabled():
        assert_equal(Int(b2.data_ptr()), p1)


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
    if device.pool_enabled():
        assert_true(Int(other.data_ptr()) != p1)
    _drop(v^)
    var reused = Buffer[DType.float32].empty(device, 256)
    if device.pool_enabled():
        assert_equal(Int(reused.data_ptr()), p1)
    # Referencing `other` here also keeps it alive through `reused`'s
    # allocation, so its block cannot have been what `reused` received.
    assert_true(Int(other.data_ptr()) != Int(reused.data_ptr()))


def test_layout_buffer_contents() raises:
    var device = Device()
    var layout = Layout(4, 6)
    var strides = layout.strides_buffer(device.ctx)
    device.ctx.synchronize()
    with strides.view.map_to_host() as host:
        for i in range(layout.rank()):
            assert_equal(host[i], Int64(layout.stride(i)))
    device.ctx.synchronize()


# ===-------------------------------------------------------------------===#
# Cache control and stats
# ===-------------------------------------------------------------------===#


def test_empty_cache_drops_reserved() raises:
    var device = Device()
    _ = _alloc_and_drop_buffer(device, 4096)
    if not device.pool_enabled():
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
    if not device.pool_enabled():
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
    # A Device dying with pooled buffers released must not break the next
    # Device's first allocation.
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
