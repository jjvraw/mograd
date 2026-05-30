from std.testing import TestSuite, assert_true, assert_false

from mograd.op import OpType
from mograd.pattern_matcher import Pat
from mograd.testing import leaf

# ===-------------------------------------------------------------------===#
# Pat.matches
# ===-------------------------------------------------------------------===#


def test_wildcard_matches_any_op() raises:
    var a = leaf((2, 3))
    var mm = a.matmul(leaf((3, 4)))
    assert_true(Pat().matches(a))
    assert_true(Pat().matches(mm))


def test_op_type_match() raises:
    var mm = leaf((2, 3)).matmul(leaf((3, 4)))
    assert_true(Pat(OpType.MATMUL).matches(mm))


def test_op_type_no_match() raises:
    var mm = leaf((2, 3)).matmul(leaf((3, 4)))
    assert_false(Pat(OpType.TRANSPOSE).matches(mm))


def test_nested_src_match() raises:
    var b = leaf((4, 3))
    var graph = leaf((2, 3)).matmul(b.transpose())
    assert_true(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]).matches(graph))


def test_nested_src_no_match_wrong_src() raises:
    var graph = leaf((2, 3)).matmul(leaf((3, 4)))
    assert_false(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]).matches(graph))


def test_src_count_mismatch() raises:
    var mm = leaf((2, 3)).matmul(leaf((3, 4)))
    assert_false(Pat(OpType.MATMUL, [Pat()]).matches(mm))


def test_leaf_has_no_srcs() raises:
    assert_false(Pat(OpType.BUFFER, [Pat()]).matches(leaf((4, 4))))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
