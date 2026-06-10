from std.testing import TestSuite, assert_equal, assert_true, assert_false

from layout.int_tuple import IntTuple, crd2idx 

from mograd.layout import Layout

# TODO: Negative tests

def test_row_major_contiguous_1d() raises:
    var l = Layout(3)
    assert_equal(len(l), 1)
    assert_equal(l.shape, IntTuple(3))
    assert_equal(l.strides, IntTuple(1))
    assert_equal(l.rank, 1)

def test_row_major_contiguous_2d() raises:
    var l = Layout(3, 4)
    assert_equal(len(l), 2)
    assert_equal(l.shape, IntTuple(3, 4))
    assert_equal(l.strides, IntTuple(4, 1))
    assert_equal(l.rank, 2)


def test_strides_1d() raises:
    var l = Layout(1024)

    for i in range(l.numel()):
        assert_equal(i, crd2idx(i, l.shape, l.strides))

def test_strides_2d() raises:
    var l = Layout(64, 102)

    assert_true(l.is_contiguous())
    for row in range(64):
        for col in range(102):
            var expected = row * 102 + col
            var actual = crd2idx(IntTuple(row, col), l.shape, l.strides)
            assert_equal(expected, actual)

def test_bounds() raises:
    var l = Layout(64, 102)

    assert_equal(l.stride(0), 102)
    assert_equal(l.stride(-2), 102)

    assert_equal(l.stride(1), 1)
    assert_equal(l.stride(-1), 1)

def test_permute_2d() raises:
    var l = Layout(3, 4)

    l = l.permute(1, 0)

    assert_equal(l.rank, 2)
    assert_equal(l.shape, IntTuple(4, 3))
    assert_equal(l.strides, IntTuple(1, 4))
    assert_equal(l.base_offset, 0)
    assert_false(l.is_contiguous())

def test_permute_2d_offsets() raises:
    var original = Layout(3, 4)
    var permuted = original

    permuted = permuted.permute(1, 0)

    for row in range(3):
        for col in range(4):
            var original_offset = crd2idx(
                IntTuple(row, col),
                original.shape,
                original.strides,
            )

            var permuted_offset = crd2idx(
                IntTuple(col, row),
                permuted.shape,
                permuted.strides,
            )

            assert_equal(original_offset, permuted_offset)

    assert_false(permuted.is_contiguous())

def test_permute_negative_axes_2d() raises:
    var l = Layout(3, 4)

    l = l.permute(-1, -2)

    assert_equal(l.shape, IntTuple(4, 3))
    assert_equal(l.strides, IntTuple(1, 4))
    assert_false(l.is_contiguous())

def test_view_1d_to_2d() raises:
    var l = Layout(12)

    var v = l.view(3, 4)

    assert_equal(v.rank, 2)
    assert_equal(v.shape, IntTuple(3, 4))
    assert_equal(v.strides, IntTuple(4, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())

def test_view_2d_to_1d() raises:
    var l = Layout(3, 4)

    var v = l.view(12)

    assert_equal(v.rank, 1)
    assert_equal(v.shape, IntTuple(12))
    assert_equal(v.strides, IntTuple(1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())

def test_view_2d_to_2d() raises:
    var l = Layout(3, 4)

    var v = l.view(2, 6)

    assert_equal(v.rank, 2)
    assert_equal(v.shape, IntTuple(2, 6))
    assert_equal(v.strides, IntTuple(6, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())

def test_view_preserves_linear_offsets() raises:
    var l = Layout(3, 4)
    var v = l.view(2, 6)

    for i in range(l.numel()):
        var old_coord = IntTuple(i // 4, i % 4)
        var new_coord = IntTuple(i // 6, i % 6)

        var old_offset = crd2idx(old_coord, l.shape, l.strides)
        var new_offset = crd2idx(new_coord, v.shape, v.strides)

        assert_equal(old_offset, new_offset)

def test_slice_1d_simple() raises:
    var l = Layout(10)

    var s = l[2:7:1]

    assert_equal(s.rank, 1)
    assert_equal(s.shape, IntTuple(5))
    assert_equal(s.strides, IntTuple(1))
    assert_equal(s.base_offset, 2)


def test_slice_1d_step() raises:
    var l = Layout(10)

    var s = l[2:9:2]

    assert_equal(s.rank, 1)
    assert_equal(s.shape, IntTuple(4))
    assert_equal(s.strides, IntTuple(2))
    assert_equal(s.base_offset, 2)


def test_slice_1d_negative_step() raises:
    var l = Layout(10)

    var s = l[8:2:-2]

    assert_equal(s.rank, 1)
    assert_equal(s.shape, IntTuple(3))
    assert_equal(s.strides, IntTuple(-2))
    assert_equal(s.base_offset, 8)


def test_slice_1d_reverse() raises:
    var l = Layout(10)

    var s = l[::-1]

    assert_equal(s.rank, 1)
    assert_equal(s.shape, IntTuple(10))
    assert_equal(s.strides, IntTuple(-1))
    assert_equal(s.base_offset, 9)

def test_view_infers_last_dim() raises:
    var l = Layout(25088)

    var v = l.view(32, -1)

    assert_equal(v.rank, 2)
    assert_equal(v.shape, IntTuple(32, 784))
    assert_equal(v.strides, IntTuple(784, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())

def test_view_infers_first_dim() raises:
    var l = Layout(25088)

    var v = l.view(-1, 784)

    assert_equal(v.rank, 2)
    assert_equal(v.shape, IntTuple(32, 784))
    assert_equal(v.strides, IntTuple(784, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())

def test_view_inferred_shape_preserves_linear_offsets() raises:
    var l = Layout(25088)
    var v = l.view(32, -1)

    for i in range(l.numel()):
        var old_coord = IntTuple(i)
        var new_coord = IntTuple(i // 784, i % 784)

        var old_offset = crd2idx(old_coord, l.shape, l.strides)
        var new_offset = crd2idx(new_coord, v.shape, v.strides)

        assert_equal(old_offset, new_offset)

def test_view_rejects_multiple_inferred_dims() raises:
    var l = Layout(12)

    try:
        _ = l.view(-1, -1)
        assert_true(False)
    except:
        assert_true(True)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
