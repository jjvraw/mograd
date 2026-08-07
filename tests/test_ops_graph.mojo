from std.testing import TestSuite, assert_true, assert_false, assert_equal, assert_raises

from mograd.op import OpType, concat
from mograd.grad import Grad, concat_grad
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


def test_cast_grad_reaches_pre_cast_source() raises:
    # A float->float cast is differentiable: backprop must not stop at it, and
    # the gradient arrives in the source dtype.
    var x = leaf((2, 3), DType.float16)
    var y = x.cast(DType.float32)
    var loss = y * leaf((2, 3), DType.float32)
    var grads = Grad.compute(loss, leaf((2, 3), DType.float32), [x])
    assert_true(Bool(grads[0]))
    assert_true(grads[0].value().dtype() == DType.float16)


def test_cast_to_same_dtype_folds_at_construction() raises:
    # Decidable from the arguments alone, so no CAST node is built at all.
    var x = leaf((2, 3), DType.float32)
    assert_true(x.cast(DType.float32) == x)


def test_reshape_to_same_layout_folds_at_construction() raises:
    var x = leaf((2, 3), DType.float32)
    assert_true(x.reshape((2, 3)) == x)
    assert_true(x.reshape((3, 2)) != x)


def test_contiguous_on_contiguous_folds_at_construction() raises:
    var x = leaf((2, 3), DType.float32)
    assert_true(x.contiguous() == x)


def test_mixed_dtype_op_still_grads_the_float_source() raises:
    # `add_grad` hands the upstream to every source without inspecting dtypes;
    # the float source gets its gradient and the integer subgraph is never
    # entered (integer targets themselves are rejected at `Tensor.gradient`).
    var f = leaf((2, 3), DType.float32)
    var i = leaf((2, 3), DType.int64)
    var grads = Grad.compute(f + i, leaf((2, 3), DType.float32), [f])
    assert_true(Bool(grads[0]))


def test_grad_does_not_propagate_through_integer_node() raises:
    # An integer-valued node has no derivative even when it is the root and gets
    # the seed gradient directly.
    var x = leaf((4,), DType.float32)
    var grads = Grad.compute(x.cast(DType.int64), leaf((4,), DType.int64), [x])
    assert_false(Bool(grads[0]))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
