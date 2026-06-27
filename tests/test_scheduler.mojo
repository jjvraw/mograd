from std.sys import has_accelerator
from std.testing import TestSuite, assert_equal, assert_true

from mograd import Device
from mograd.buffer import AnyBuffer
from mograd.layout import Layout
from mograd.op import Op, OpRef, OpType, AttrVal
from mograd.pattern_matcher import Rule, Pat
from mograd.scheduler import BoundExecFn, Scheduler, SchedulerRules
from mograd.testing import leaf

# ===-------------------------------------------------------------------===#
# Helpers
# ===-------------------------------------------------------------------===#

comptime FAKE_SPLIT = OpType("FAKE_SPLIT")


def fake_split(node: OpRef, inputs: List[AnyBuffer], device: Device) raises -> List[AnyBuffer]:
    """Returns two buffers filled with 1.0 and 2.0 respectively."""
    var a = AnyBuffer.create(DType.float32, device, 4, fill=1.0)
    var b = AnyBuffer.create(DType.float32, device, 4, fill=2.0)
    return [a^, b^]


def make_scheduler(device: Device) -> Scheduler:
    return Scheduler([Rule(Pat(FAKE_SPLIT), fake_split)])


def make_graph(device: Device) raises -> Tuple[OpRef, OpRef, OpRef]:
    """Returns (split, get0, get1) where split is FAKE_SPLIT and gets are GETTUPLE."""
    var x = leaf(Layout(4))
    x.op().buf = Optional[AnyBuffer](AnyBuffer.create(DType.float32, device, 4))
    var split = OpRef(Op(FAKE_SPLIT, Layout(4), DType.float32, [x]))
    var get0 = OpRef(Op(OpType.GETTUPLE, Layout(4), DType.float32, [split], {"index": 1}))
    var get1 = OpRef(Op(OpType.GETTUPLE, Layout(4), DType.float32, [split], {"index": 0}))
    return (split, get0, get1)


# ===-------------------------------------------------------------------===#
# Tests
# ===-------------------------------------------------------------------===#


def test_gettuple_index1_picks_second_output() raises:
    # graph[1] = GETTUPLE(split, index=1) → second output (fill=2.0)
    var device = Device()
    var graph = make_graph(device)
    var result = make_scheduler(device).run(graph[1], device)
    var vals = result.to_list[DType.float32](Layout(4))
    for v in vals:
        assert_equal(v, Float32(2.0))


def test_gettuple_index0_picks_first_output() raises:
    # graph[2] = GETTUPLE(split, index=0) → first output (fill=1.0)
    var device = Device()
    var graph = make_graph(device)
    var result = make_scheduler(device).run(graph[2], device)
    var vals = result.to_list[DType.float32](Layout(4))
    for v in vals:
        assert_equal(v, Float32(1.0))


def test_gettuple_run_many_returns_both_outputs() raises:
    var device = Device()
    var graph = make_graph(device)
    var results = make_scheduler(device).run_many([graph[1], graph[2]], device)
    assert_equal(len(results), 2)
    assert_equal(results[0].to_list[DType.float32](Layout(4))[0], Float32(2.0))
    assert_equal(results[1].to_list[DType.float32](Layout(4))[0], Float32(1.0))


def test_gettuple_outputs_have_correct_element_count() raises:
    var device = Device()
    var graph = make_graph(device)
    var results = make_scheduler(device).run_many([graph[1], graph[2]], device)
    assert_equal(len(results[0].to_list[DType.float32](Layout(4))), 4)
    assert_equal(len(results[1].to_list[DType.float32](Layout(4))), 4)


def main() raises:
    comptime assert has_accelerator(), "GPU required"
    TestSuite.discover_tests[__functions_in_module()]().run()
