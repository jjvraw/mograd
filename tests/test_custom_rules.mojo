from std.testing import TestSuite, assert_equal, assert_true

from mograd import Device
from mograd.buffer import AnyBuffer
from mograd.op import AttrVal, Op, OpRef, OpType
from mograd.pattern_matcher import Rule, Pat
from mograd.scheduler import BoundExecFn, SchedulerRules
from mograd.runtime.gpu.rewrites import GPU_REWRITES
from mograd.simplify import Simplifier, RewriteFn
from mograd.testing import leaf, assert_graph


# Shared dummy kernel, only IR-level tests here.
def dummy(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    raise Error("dummy")


comptime NO_REWRITES = List[Rule[RewriteFn]]


def test_compound_rule_emits_fuse_node() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))

    var extra = SchedulerRules()
    var result = Simplifier(NO_REWRITES()).run(x + y, [Rule(Pat(OpType.ADD, [Pat(), Pat()]), dummy)], extra)

    assert_graph(result, Pat(OpType("__fuse_0")))


def test_compound_rule_extracts_wildcard_leaves() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))

    var extra = SchedulerRules()
    var result = Simplifier(NO_REWRITES()).run(x + y, [Rule(Pat(OpType.ADD, [Pat(), Pat()]), dummy)], extra)

    assert_equal(len(result.srcs()), 2)
    assert_true(result.src(0).op_type() == OpType.BUFFER)
    assert_true(result.src(1).op_type() == OpType.BUFFER)


def test_compound_rule_preserves_layout() raises:
    var x = leaf((4, 5))
    var y = leaf((4, 5))

    var extra = SchedulerRules()
    var result = Simplifier(NO_REWRITES()).run(x + y, [Rule(Pat(OpType.ADD, [Pat(), Pat()]), dummy)], extra)

    assert_equal(result.shape(0), 4)
    assert_equal(result.shape(1), 5)


def test_compound_rule_preserves_dtype() raises:
    var x = leaf((2, 3), DType.float32)
    var y = leaf((2, 3), DType.float32)

    var extra = SchedulerRules()
    var result = Simplifier(NO_REWRITES()).run(x + y, [Rule(Pat(OpType.ADD, [Pat(), Pat()]), dummy)], extra)

    assert_true(result.dtype() == DType.float32)


def test_compound_rule_registers_exec_in_extra_sched() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))

    var extra = SchedulerRules()
    _ = Simplifier(NO_REWRITES()).run(x + y, [Rule(Pat(OpType.ADD, [Pat(), Pat()]), dummy)], extra)

    assert_equal(len(extra), 1)
    assert_equal(extra[0].pat.op_type._name, "__fuse_0")


def test_nonmatching_compound_rule_leaves_graph_unchanged() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))

    var extra = SchedulerRules()
    var result = Simplifier(NO_REWRITES()).run(x + y, [Rule(Pat(OpType.MUL, [Pat(), Pat()]), dummy)], extra)

    assert_graph(result, Pat(OpType.ADD, [Pat(), Pat()]))
    assert_equal(len(extra), 0)


def test_two_compound_rules_produce_distinct_fuse_types() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))
    var rules: SchedulerRules = [
        Rule(Pat(OpType.ADD, [Pat(), Pat()]), dummy),
        Rule(Pat(OpType.MUL, [Pat(), Pat()]), dummy),
    ]

    var extra_add = SchedulerRules()
    var fused_add = Simplifier(NO_REWRITES()).run(x + y, rules.copy(), extra_add)

    var extra_mul = SchedulerRules()
    var fused_mul = Simplifier(NO_REWRITES()).run(x * y, rules.copy(), extra_mul)

    assert_graph(fused_add, Pat(OpType("__fuse_0")))
    assert_graph(fused_mul, Pat(OpType("__fuse_1")))


def test_same_rule_matched_twice_registers_exec_once() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))
    var z = leaf((2, 3))

    var extra = SchedulerRules()
    _ = Simplifier(NO_REWRITES()).run((x + y) + z, [Rule(Pat(OpType.ADD, [Pat(), Pat()]), dummy)], extra)

    assert_equal(len(extra), 1)


def test_nested_compound_rule_extracts_all_leaves() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))
    var z = leaf((2, 3))

    var extra = SchedulerRules()
    var result = Simplifier(NO_REWRITES()).run(
        (x + y) * z,
        [Rule(Pat(OpType.MUL, [Pat(OpType.ADD, [Pat(), Pat()]), Pat()]), dummy)],
        extra,
    )

    assert_graph(result, Pat(OpType("__fuse_0")))
    assert_equal(len(result.srcs()), 3)


def test_nested_compound_rule_intermediate_node_removed() raises:
    var x = leaf((2, 3))
    var y = leaf((2, 3))
    var z = leaf((2, 3))

    var extra = SchedulerRules()
    var result = Simplifier(NO_REWRITES()).run(
        (x + y) * z,
        [Rule(Pat(OpType.MUL, [Pat(OpType.ADD, [Pat(), Pat()]), Pat()]), dummy)],
        extra,
    )

    for i in range(len(result.srcs())):
        assert_true(result.src(i).op_type() == OpType.BUFFER)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_compound_rule_matches_after_canonicalization() raises:
    # A user pattern written against the canonical SCALE spelling must also
    # catch a graph spelled MUL(x, FULL(v)). The canonicalization rewrites
    # the node, and the fusion matcher gets another look within the same
    # per-node fixed point.
    var x = leaf((2, 3))
    var y = leaf((2, 3))
    var c = OpRef(Op(OpType.FULL, x.layout(), x.dtype(), [], {"value": AttrVal(Float32(2.0))}))

    var extra = SchedulerRules()
    var result = Simplifier(GPU_REWRITES()).run(
        (x + y) * c,
        [Rule(Pat(OpType.SCALE, [Pat(OpType.ADD, [Pat(), Pat()])]), dummy)],
        extra,
    )

    assert_graph(result, Pat(OpType("__fuse_0")))
