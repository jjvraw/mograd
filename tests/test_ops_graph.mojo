from std.testing import TestSuite, assert_true, assert_false, assert_equal, assert_raises

from mograd.op import OpType, concat
from mograd.grad import concat_grad
from mograd.testing import leaf


def test_matmul_materializes_noncollapsible_batch_dims() raises:
    # transpose(1, 2) scrambles B/H's relative order, so it must be wrapped in CONTIGUOUS.
    var a = leaf((2, 2, 2, 2)).transpose(1, 2)
    var b = leaf((2, 2, 2, 2))
    var result = a.matmul(b)
    assert_true(result.src(0).op_type() == OpType.CONTIGUOUS)
    assert_true(result.src(0).src(0).op_type() == OpType.TRANSPOSE)
    assert_true(result.src(1).op_type() == OpType.BUFFER)


def test_matmul_does_not_materialize_collapsible_transpose() raises:
    # transpose(-2, -1) only touches the last two dims, so the batch axis stays as-is.
    var a = leaf((2, 3, 4))
    var b = leaf((2, 5, 4)).transpose()
    var result = a.matmul(b)
    assert_true(result.src(1).op_type() == OpType.TRANSPOSE)


def test_concat_axis0_shape() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    var result = concat([a, b], 0)
    assert_equal(result.shape(0), 6)
    assert_equal(result.shape(1), 3)
    assert_true(result.op_type() == OpType.CONCAT)


def test_concat_axis1_shape() raises:
    var a = leaf((2, 3))
    var b = leaf((2, 5))
    var result = concat([a, b], 1)
    assert_equal(result.shape(0), 2)
    assert_equal(result.shape(1), 8)


def test_concat_preserves_srcs_order() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    var result = concat([a, b], 0)
    assert_equal(len(result.srcs()), 2)
    assert_true(result.src(0) == a)
    assert_true(result.src(1) == b)


def test_concat_preserves_dtype() raises:
    var a = leaf((2, 3), DType.float32)
    var b = leaf((4, 3), DType.float32)
    var result = concat([a, b], 0)
    assert_true(result.dtype() == DType.float32)


def test_concat_rejects_rank_mismatch() raises:
    var a = leaf((2, 3))
    var b = leaf((2, 3, 1))
    with assert_raises():
        _ = concat([a, b], 0)


def test_concat_rejects_other_axis_mismatch() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 5))
    with assert_raises():
        _ = concat([a, b], 0)


def test_concat_grad_splits_along_axis() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    var node = concat([a, b], 0)
    var upstream = leaf((6, 3))
    var grads = concat_grad(node, upstream)
    assert_equal(len(grads), 2)
    assert_true(grads[0].value().op_type() == OpType.SLICE)
    assert_true(grads[1].value().op_type() == OpType.SLICE)
    assert_equal(grads[0].value().shape(0), 2)
    assert_equal(grads[0].value().shape(1), 3)
    assert_equal(grads[1].value().shape(0), 4)
    assert_equal(grads[1].value().shape(1), 3)


def test_concat_grad_splits_along_axis1() raises:
    var a = leaf((2, 3))
    var b = leaf((2, 5))
    var node = concat([a, b], 1)
    var upstream = leaf((2, 8))
    var grads = concat_grad(node, upstream)
    assert_equal(grads[0].value().shape(1), 3)
    assert_equal(grads[1].value().shape(1), 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
