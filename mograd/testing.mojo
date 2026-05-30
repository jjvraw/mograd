from std.math import abs

from mograd.op import Op, OpRef, OpType
from mograd.pattern_matcher import Pat, Rule
from mograd.shape import Shape
from mograd.simplify import Simplifier, RewriteFn
from mograd.tensor import Tensor


def assert_allclose(
    actual: Tensor,
    expected: List[Float32],
    tol: Float32 = 1e-5,
) raises:
    var vals = actual.to_list()
    if len(vals) != len(expected):
        raise Error("size mismatch: tensor has " + String(len(vals)) + " elements, expected " + String(len(expected)))
    for i in range(len(expected)):
        if abs(vals[i] - expected[i]) >= tol:
            raise Error(
                "mismatch at ["
                + String(i)
                + "]: got "
                + String(vals[i])
                + ", expected "
                + String(expected[i])
                + " (tol="
                + String(tol)
                + ")"
            )


def assert_allclose(
    actual: Tensor,
    expected: Tensor,
    tol: Float32 = 1e-5,
) raises:
    assert_allclose(actual, expected.to_list(), tol=tol)


def assert_close(actual: Tensor, expected: Float32, tol: Float32 = 1e-5) raises:
    var val = actual.item()
    if abs(val - expected) >= tol:
        raise Error(
            "scalar mismatch: got " + String(val) + ", expected " + String(expected) + " (tol=" + String(tol) + ")"
        )


def leaf(shape: Shape, dtype: DType = DType.float32) -> OpRef:
    return OpRef(Op(OpType.BUFFER, shape, dtype, []))


def assert_graph(node: OpRef, pat: Pat) raises:
    if not pat.matches(node):
        raise Error(
            "graph mismatch: expected pattern rooted at '" + pat.op_type._name + "', got '" + String(node) + "'"
        )


def assert_rewrites_to[
    rules: List[Rule[RewriteFn]]
](graph: OpRef, expected: Pat,) raises:
    var result = Simplifier[rules].run(graph)
    assert_graph(result, expected)
