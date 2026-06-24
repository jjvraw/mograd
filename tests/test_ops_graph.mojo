from std.testing import TestSuite, assert_true, assert_false, assert_equal

from mograd.op import OpType
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


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
