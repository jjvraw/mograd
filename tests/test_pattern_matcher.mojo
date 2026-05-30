from std.testing import TestSuite

from mograd.op import OpType
from mograd.pattern_matcher import Pat
from mograd.testing import leaf

# ===-------------------------------------------------------------------===#
# Pat.matches
# ===-------------------------------------------------------------------===#


def test_wildcard_matches_any_op() raises:
    var a = leaf((2, 3))
    var mm = a.matmul(leaf((3, 4)))
    if not Pat().matches(a):
        raise Error("wildcard should match leaf")
    if not Pat().matches(mm):
        raise Error("wildcard should match matmul")


def test_op_type_match() raises:
    var mm = leaf((2, 3)).matmul(leaf((3, 4)))
    if not Pat(OpType.MATMUL).matches(mm):
        raise Error("MATMUL pat should match matmul node")


def test_op_type_no_match() raises:
    var mm = leaf((2, 3)).matmul(leaf((3, 4)))
    if Pat(OpType.TRANSPOSE).matches(mm):
        raise Error("TRANSPOSE pat should not match matmul node")


def test_nested_src_match() raises:
    var b = leaf((4, 3))
    var graph = leaf((2, 3)).matmul(b.transpose())
    var pat = Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)])
    if not pat.matches(graph):
        raise Error("MATMUL(_, TRANSPOSE) should match matmul(a, b.transpose())")


def test_nested_src_no_match_wrong_src() raises:
    var graph = leaf((2, 3)).matmul(leaf((3, 4)))
    var pat = Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)])
    if pat.matches(graph):
        raise Error("MATMUL(_, TRANSPOSE) should not match matmul(a, b) with no transpose")


def test_src_count_mismatch() raises:
    var a = leaf((2, 3))
    # MATMUL has 2 srcs; pattern requires 1 — should not match
    var mm = a.matmul(leaf((3, 4)))
    var pat = Pat(OpType.MATMUL, [Pat()])
    if pat.matches(mm):
        raise Error("pattern with 1 src should not match node with 2 srcs")


def test_leaf_has_no_srcs() raises:
    var a = leaf((4, 4))
    var pat = Pat(OpType.BUFFER, [Pat()])
    if pat.matches(a):
        raise Error("BUFFER node has no srcs, pattern with 1 src should not match")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
