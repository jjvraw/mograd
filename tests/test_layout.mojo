from std.testing import TestSuite, assert_equal, assert_true, assert_false, assert_raises

from layout.int_tuple import IntTuple, crd2idx

from mograd.layout import Layout

# TODO: Negative tests


def test_row_major_contiguous_1d() raises:
    var l = Layout(3)
    assert_equal(l.rank(), 1)
    assert_equal(l.shape(), IntTuple(3))
    assert_equal(l.stride(), IntTuple(1))


def test_row_major_contiguous_2d() raises:
    var l = Layout(3, 4)
    assert_equal(l.rank(), 2)
    assert_equal(l.shape(), IntTuple(3, 4))
    assert_equal(l.stride(), IntTuple(4, 1))
    assert_equal(l.rank(), 2)


def test_stride_1d() raises:
    var l = Layout(1024)

    for i in range(l.numel()):
        assert_equal(i, crd2idx(i, l.shape(), l.stride()))


def test_stride_2d() raises:
    var l = Layout(64, 102)

    assert_true(l.is_contiguous())
    for row in range(64):
        for col in range(102):
            var expected = row * 102 + col
            var actual = crd2idx(IntTuple(row, col), l.shape(), l.stride())
            assert_equal(expected, actual)


def test_bounds() raises:
    var l = Layout(64, 102)

    assert_equal(l.stride(0), 102)
    assert_equal(l.stride(-2), 102)

    assert_equal(l.stride(1), 1)


def test_permute_2d() raises:
    var l = Layout(3, 4)

    l = l.permute(1, 0)

    assert_equal(l.rank(), 2)
    assert_equal(l.shape(), IntTuple(4, 3))
    assert_equal(l.stride(), IntTuple(1, 4))
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
                original.shape(),
                original.stride(),
            )

            var permuted_offset = crd2idx(
                IntTuple(col, row),
                permuted.shape(),
                permuted.stride(),
            )

            assert_equal(original_offset, permuted_offset)

    assert_false(permuted.is_contiguous())


def test_permute_negative_axes_2d() raises:
    var l = Layout(3, 4)

    l = l.permute(-1, -2)

    assert_equal(l.shape(), IntTuple(4, 3))
    assert_equal(l.stride(), IntTuple(1, 4))
    assert_false(l.is_contiguous())


def test_permutation_of_contiguous_identity() raises:
    var l = Layout(3, 4)

    var order = l.permutation_of_contiguous()

    assert_true(order)
    assert_equal(order.value(), IntTuple(0, 1))


def test_permutation_of_contiguous_2d_transpose() raises:
    var l = Layout(3, 4).permute(1, 0)

    var order = l.permutation_of_contiguous()

    assert_true(order)
    assert_equal(order.value(), IntTuple(1, 0))


def test_permutation_of_contiguous_swaps_last_two_axes_only() raises:
    var l = Layout(2, 3, 4).permute(0, 2, 1)

    var order = l.permutation_of_contiguous()

    assert_true(order)
    assert_equal(order.value(), IntTuple(0, 2, 1))


def test_permutation_of_contiguous_via_transpose_2d() raises:
    var l = Layout(3, 4).transpose()

    var order = l.permutation_of_contiguous()

    assert_true(order)
    assert_equal(order.value(), IntTuple(1, 0))


def test_permutation_of_contiguous_via_transpose_3d_batched() raises:
    var l = Layout(2, 3, 4).transpose()

    var order = l.permutation_of_contiguous()

    assert_true(order)
    assert_equal(order.value(), IntTuple(0, 2, 1))


def test_permutation_of_contiguous_via_transpose_3d_explicit_dims() raises:
    # Explicit, non-default dims: swap the first two axes, leave the last untouched.
    var l = Layout(2, 3, 4).transpose(0, 1)

    var order = l.permutation_of_contiguous()

    assert_true(order)
    assert_equal(order.value(), IntTuple(1, 0, 2))


def test_permutation_of_contiguous_via_transpose_negative_dims() raises:
    var l = Layout(2, 3, 4).transpose(-3, -1)

    var order = l.permutation_of_contiguous()

    assert_true(order)
    assert_equal(order.value(), IntTuple(2, 1, 0))


def test_permutation_of_contiguous_rejects_broadcast() raises:
    var l = Layout(3, 4).expand_axis(0, 5)

    assert_false(l.permutation_of_contiguous())


def test_permutation_of_contiguous_rejects_slice() raises:
    var l = Layout(10)[0:10:2]

    assert_false(l.permutation_of_contiguous())


def test_view_1d_to_2d() raises:
    var l = Layout(12)

    var v = l.view(3, 4)

    assert_equal(v.rank(), 2)
    assert_equal(v.shape(), IntTuple(3, 4))
    assert_equal(v.stride(), IntTuple(4, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())


def test_view_2d_to_1d() raises:
    var l = Layout(3, 4)

    var v = l.view(12)

    assert_equal(v.rank(), 1)
    assert_equal(v.shape(), IntTuple(12))
    assert_equal(v.stride(), IntTuple(1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())


def test_view_2d_to_2d() raises:
    var l = Layout(3, 4)

    var v = l.view(2, 6)

    assert_equal(v.rank(), 2)
    assert_equal(v.shape(), IntTuple(2, 6))
    assert_equal(v.stride(), IntTuple(6, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())


def test_view_preserves_linear_offsets() raises:
    var l = Layout(3, 4)
    var v = l.view(2, 6)

    for i in range(l.numel()):
        var old_coord = IntTuple(i // 4, i % 4)
        var new_coord = IntTuple(i // 6, i % 6)

        var old_offset = crd2idx(old_coord, l.shape(), l.stride())
        var new_offset = crd2idx(new_coord, v.shape(), v.stride())

        assert_equal(old_offset, new_offset)


def test_slice_1d_simple() raises:
    var l = Layout(10)

    var s = l[2:7:1]

    assert_equal(s.rank(), 1)
    assert_equal(s.shape(), IntTuple(5))
    assert_equal(s.stride(), IntTuple(1))
    assert_equal(s.base_offset, 2)


def test_slice_1d_step() raises:
    var l = Layout(10)

    var s = l[2:9:2]

    assert_equal(s.rank(), 1)
    assert_equal(s.shape(), IntTuple(4))
    assert_equal(s.stride(), IntTuple(2))
    assert_equal(s.base_offset, 2)


def test_slice_1d_negative_step() raises:
    var l = Layout(10)

    var s = l[8:2:-2]

    assert_equal(s.rank(), 1)
    assert_equal(s.shape(), IntTuple(3))
    assert_equal(s.stride(), IntTuple(-2))
    assert_equal(s.base_offset, 8)


def test_slice_1d_reverse() raises:
    var l = Layout(10)

    var s = l[::-1]

    assert_equal(s.rank(), 1)
    assert_equal(s.shape(), IntTuple(10))
    assert_equal(s.stride(), IntTuple(-1))
    assert_equal(s.base_offset, 9)


def test_view_infers_last_dim() raises:
    var l = Layout(25088)

    var v = l.view(32, -1)

    assert_equal(v.rank(), 2)
    assert_equal(v.shape(), IntTuple(32, 784))
    assert_equal(v.stride(), IntTuple(784, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())


def test_view_infers_first_dim() raises:
    var l = Layout(25088)

    var v = l.view(-1, 784)

    assert_equal(v.rank(), 2)
    assert_equal(v.shape(), IntTuple(32, 784))
    assert_equal(v.stride(), IntTuple(784, 1))
    assert_equal(v.base_offset, l.base_offset)
    assert_equal(v.numel(), l.numel())
    assert_true(v.is_contiguous())


def test_view_inferred_shape_preserves_linear_offsets() raises:
    var l = Layout(25088)
    var v = l.view(32, -1)

    for i in range(l.numel()):
        var old_coord = IntTuple(i)
        var new_coord = IntTuple(i // 784, i % 784)

        var old_offset = crd2idx(old_coord, l.shape(), l.stride())
        var new_offset = crd2idx(new_coord, v.shape(), v.stride())

        assert_equal(old_offset, new_offset)


def test_view_rejects_multiple_inferred_dims() raises:
    var l = Layout(12)

    try:
        _ = l.view(-1, -1)
        assert_true(False)
    except:
        assert_true(True)


def test_unsqueeze_at_0() raises:
    var l = Layout(3, 4)
    var u = l.unsqueeze(0)
    assert_equal(u.rank(), 3)
    assert_equal(u.shape(), IntTuple(1, 3, 4))
    assert_equal(u.stride(), IntTuple(0, 4, 1))


def test_unsqueeze_at_end() raises:
    var l = Layout(3, 4)
    var u = l.unsqueeze(2)
    assert_equal(u.rank(), 3)
    assert_equal(u.shape(), IntTuple(3, 4, 1))
    assert_equal(u.stride(), IntTuple(4, 1, 0))


def test_unsqueeze_negative_index() raises:
    var l = Layout(3, 4)
    var u = l.unsqueeze(-1)
    assert_equal(u.shape(), IntTuple(3, 4, 1))


def test_squeeze_at_0() raises:
    var l = Layout(1, 3, 4)
    var s = l.squeeze(0)
    assert_equal(s.rank(), 2)
    assert_equal(s.shape(), IntTuple(3, 4))
    assert_equal(s.stride(), IntTuple(4, 1))


def test_squeeze_negative_index() raises:
    var l = Layout(3, 4, 1)
    var s = l.squeeze(-1)
    assert_equal(s.shape(), IntTuple(3, 4))


def test_squeeze_all() raises:
    var l = Layout(1, 3, 1, 4, 1)
    var s = l.squeeze_all()
    assert_equal(s.rank(), 2)
    assert_equal(s.shape(), IntTuple(3, 4))


def test_squeeze_all_preserves_non_ones() raises:
    var l = Layout(2, 1, 3, 1, 4)
    var s = l.squeeze_all()
    assert_equal(s.shape(), IntTuple(2, 3, 4))


def test_unsqueeze_squeeze_roundtrip() raises:
    var l = Layout(3, 4)
    var u = l.unsqueeze(1)
    var s = u.squeeze(1)
    assert_equal(s.shape(), l.shape())
    assert_equal(s.stride(), l.stride())


def test_squeeze_rejects_non_one_dim() raises:
    var l = Layout(3, 4)
    with assert_raises():
        _ = l.squeeze(0)


def test_expand_stretches_size_one_axis() raises:
    var l = Layout(3, 1)
    var e = l.expand(3, 4)
    assert_equal(e.rank(), 2)
    assert_equal(e.shape(), IntTuple(3, 4))
    assert_equal(e.stride(), IntTuple(l.stride(0), 0))


def test_expand_keeps_matching_axis() raises:
    var l = Layout(3, 1)
    var e = l.expand(3, 4)
    assert_equal(e.stride(0), l.stride(0))


def test_expand_negative_one_keeps_size() raises:
    var l = Layout(3, 1)
    var e = l.expand(-1, 4)
    assert_equal(e.shape(), IntTuple(3, 4))


def test_expand_rejects_rank_mismatch() raises:
    var l = Layout(3, 1)
    with assert_raises():
        _ = l.expand(3, 4, 5)


def test_expand_rejects_non_one_axis_mismatch() raises:
    var l = Layout(3, 4)
    with assert_raises():
        _ = l.expand(3, 5)


def test_slice_axis_middle_of_rank3() raises:
    var l = Layout(2, 10, 4)
    var s = l.slice_axis(1, 3, 7)
    assert_equal(s.rank(), 3)
    assert_equal(s.shape(), IntTuple(2, 4, 4))
    assert_equal(s.stride(), l.stride())
    assert_equal(s.base_offset, 3 * l.stride(1))


def test_slice_axis_negative_axis() raises:
    var l = Layout(2, 3, 10)
    var s = l.slice_axis(-1, 2, 6)
    assert_equal(s.shape(), IntTuple(2, 3, 4))


def test_slice_axis_other_axes_untouched() raises:
    var l = Layout(5, 10)
    var s = l.slice_axis(1, 2, 5)
    assert_equal(s.shape(0), 5)
    assert_equal(s.stride(0), l.stride(0))


def test_layout_concat_axis0() raises:
    var a = Layout(2, 3)
    var b = Layout(4, 3)
    var c = Layout.concat([a, b], 0)
    assert_equal(c.shape(), IntTuple(6, 3))
    assert_equal(c.stride(), Layout.row_major_strides(IntTuple(6, 3)))


def test_layout_concat_axis1() raises:
    var a = Layout(2, 3)
    var b = Layout(2, 5)
    var c = Layout.concat([a, b], 1)
    assert_equal(c.shape(), IntTuple(2, 8))


def test_layout_concat_rejects_rank_mismatch() raises:
    var a = Layout(2, 3)
    var b = Layout(2, 3, 1)
    with assert_raises():
        _ = Layout.concat([a, b], 0)


def test_layout_concat_rejects_other_axis_mismatch() raises:
    var a = Layout(2, 3)
    var b = Layout(4, 5)
    with assert_raises():
        _ = Layout.concat([a, b], 0)


def test_flatten_all() raises:
    var l = Layout(2, 3, 4)
    var f = l.flatten()
    assert_equal(f.rank(), 1)
    assert_equal(f.shape(), IntTuple(24))


def test_flatten_partial() raises:
    var l = Layout(2, 3, 4)
    var f = l.flatten(1, 2)
    assert_equal(f.rank(), 2)
    assert_equal(f.shape(), IntTuple(2, 12))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
