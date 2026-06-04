from std.math import abs

from mograd.op import Op, OpRef, OpType
from mograd.pattern_matcher import Pat, Rule
from mograd.shape import Shape
from mograd.simplify import Simplifier, RewriteFn
from mograd.tensor import Tensor


def _assert_allclose_impl[
    dtype: DType
](vals: List[Scalar[dtype]], expected: List[Scalar[dtype]], tol: Scalar[dtype],) raises:
    if len(vals) != len(expected):
        raise Error("size mismatch: tensor has " + String(len(vals)) + " elements, expected " + String(len(expected)))
    for i in range(len(expected)):
        if abs(vals[i] - expected[i]) > tol:
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


def assert_allclose[
    dtype: DType
](
    actual: Tensor[dtype],
    expected: List[Scalar[dtype]],
    tol: Scalar[dtype] = 1e-5,
) raises where dtype.is_floating_point():
    _assert_allclose_impl[dtype](actual.to_list(), expected, tol)


def assert_allclose[
    dtype: DType
](actual: Tensor[dtype], expected: List[Scalar[dtype]], tol: Scalar[dtype] = 0,) raises where dtype.is_integral():
    _assert_allclose_impl[dtype](actual.to_list(), expected, tol)


def assert_allclose[
    dtype: DType
](actual: Tensor[dtype], expected: Tensor[dtype], tol: Scalar[dtype] = 1e-5,) raises where dtype.is_floating_point():
    _assert_allclose_impl[dtype](actual.to_list(), expected.to_list(), tol)


def _assert_close_impl[
    dtype: DType
](val: Scalar[dtype], expected: Scalar[dtype], tol: Scalar[dtype],) raises:
    if abs(val - expected) > tol:
        raise Error(
            "scalar mismatch: got " + String(val) + ", expected " + String(expected) + " (tol=" + String(tol) + ")"
        )


def assert_close[
    dtype: DType
](actual: Tensor[dtype], expected: Scalar[dtype], tol: Scalar[dtype] = 1e-5,) raises where dtype.is_floating_point():
    _assert_close_impl[dtype](actual.item(), expected, tol)


def assert_close[
    dtype: DType
](actual: Tensor[dtype], expected: Scalar[dtype], tol: Scalar[dtype] = 0,) raises where dtype.is_integral():
    _assert_close_impl[dtype](actual.item(), expected, tol)


def leaf(shape: Shape, dtype: DType = DType.float32) -> OpRef:
    return OpRef(Op(OpType.BUFFER, shape, dtype, []))


def assert_graph(node: OpRef, pat: Pat) raises:
    if not pat.matches(node):
        raise Error(
            "graph mismatch: expected pattern rooted at '" + pat.op_type._name + "', got '" + String(node) + "'"
        )


def assert_rewrites_to[rules: List[Rule[RewriteFn]]](graph: OpRef, expected: Pat) raises:
    var result = Simplifier[rules].run(graph)
    assert_graph(result, expected)
