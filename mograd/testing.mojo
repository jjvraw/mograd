from std.math import abs

from mograd.op import Op, OpRef, OpType
from mograd.pattern_matcher import Pat, Rule
from mograd.layout import Layout
from mograd.simplify import Simplifier, RewriteFn
from mograd.tensor import Tensor
from mograd.buffer import AnyBuffer, BufferArm


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
    dtype: DType, //
](
    actual: Tensor,
    expected: List[Scalar[dtype]],
    tol: Scalar[dtype] = Scalar[dtype](Float64(1e-5) if dtype.is_floating_point() else Float64(0)),
) raises:
    _assert_allclose_impl[dtype](actual.to_list[dtype](), expected, tol)


def assert_allclose(actual: AnyBuffer, actual_layout: Layout, expected: Tensor, tol: Float64 = 1e-5) raises:
    """Compare an eagerly-evaluated AnyBuffer (from Tensor.value(simplifier=False))
    against a lazy Tensor, element-wise within `tol`.  Useful for fused-vs-unfused checks."""
    if actual.dtype() != expected.dtype:
        raise Error("dtype mismatch: " + String(actual.dtype()) + " vs " + String(expected.dtype))
    comptime for k in range(AnyBuffer.BufVariant.Ts.length):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if actual.dtype() == d:
            _assert_allclose_impl[d](actual.to_list[d](actual_layout), expected.to_list[d](), Scalar[d](tol))
            return
    raise Error("Unsupported dtype: " + String(actual.dtype()))


def assert_allclose(actual: Tensor, expected: Tensor, tol: Float64 = 1e-5) raises:
    if actual.dtype != expected.dtype:
        raise Error("dtype mismatch: " + String(actual.dtype) + " vs " + String(expected.dtype))
    comptime for k in range(AnyBuffer.BufVariant.Ts.length):
        comptime T = AnyBuffer.BufVariant.Ts[k]
        comptime assert conforms_to(T, BufferArm)
        comptime d = T.node_dtype
        if actual.dtype == d:
            _assert_allclose_impl[d](actual.to_list[d](), expected.to_list[d](), Scalar[d](tol))
            return
    raise Error("Unsupported dtype: " + String(actual.dtype))


def _assert_close_impl[
    dtype: DType
](val: Scalar[dtype], expected: Scalar[dtype], tol: Scalar[dtype],) raises:
    if abs(val - expected) > tol:
        raise Error(
            "scalar mismatch: got " + String(val) + ", expected " + String(expected) + " (tol=" + String(tol) + ")"
        )


def assert_close[
    dtype: DType
](actual: Tensor, expected: Scalar[dtype], tol: Scalar[dtype] = 1e-5,) raises where dtype.is_floating_point():
    _assert_close_impl[dtype](actual.item[dtype](), expected, tol)


def assert_close[
    dtype: DType
](actual: Tensor, expected: Scalar[dtype], tol: Scalar[dtype] = 0,) raises where dtype.is_integral():
    _assert_close_impl[dtype](actual.item[dtype](), expected, tol)


def leaf(shape: Layout, dtype: DType = DType.float32) -> OpRef:
    return OpRef(Op(OpType.BUFFER, shape, dtype, []))


def assert_graph(node: OpRef, pat: Pat) raises:
    if not pat.matches(node):
        raise Error(
            "graph mismatch: expected pattern rooted at '" + pat.op_type._name + "', got '" + String(node) + "'"
        )


def assert_rewrites_to(var rules: List[Rule[RewriteFn]], graph: OpRef, expected: Pat) raises:
    var result = Simplifier(rules^).run(graph)
    assert_graph(result, expected)
