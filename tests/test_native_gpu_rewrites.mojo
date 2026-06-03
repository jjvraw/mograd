from std.testing import TestSuite, assert_true, assert_false, assert_equal

from mograd.op import OpType, AnyOpRef, OpRef, NodeOps
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime.native.gpu.rewrites import fuse_matmul_transpose, MATMUL_T
from mograd.simplify import RewriteFn
from mograd.testing import leaf, assert_rewrites_to
from mograd.runtime.native.gpu.rewrites import NATIVE_GPU_REWRITES
from mograd.simplify import Simplifier

# ===-------------------------------------------------------------------===#
# MATMUL_T rewrite
# ===-------------------------------------------------------------------===#


def test_fuse_matmul_transpose() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    # MATMUL(A, TRANSPOSE(B)) -> MATMUL_T(A, B)
    assert_rewrites_to[NATIVE_GPU_REWRITES](
        a.matmul(b.transpose()),
        Pat(MATMUL_T, [Pat(), Pat()]),
    )


def test_fuse_matmul_transpose_removes_transpose_node() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    var result = a.matmul(b.transpose())
    var rewritten = Simplifier[NATIVE_GPU_REWRITES].run(AnyOpRef(result))
    # Unwrap to float32 to inspect srcs
    var r = rewritten.unsafe_get[OpRef[DType.float32]]()
    assert_false(r.src(1).op_type() == OpType.TRANSPOSE)


def test_plain_matmul_not_rewritten() raises:
    var a = leaf((2, 3))
    var b = leaf((3, 4))
    # MATMUL(A, B) with no TRANSPOSE must not be touched
    assert_rewrites_to[NATIVE_GPU_REWRITES](
        a.matmul(b),
        Pat(OpType.MATMUL, [Pat(), Pat()]),
    )


def test_rewrite_preserves_shape() raises:
    var a = leaf((5, 7))
    var b = leaf((3, 7))
    var result = Simplifier[NATIVE_GPU_REWRITES].run(AnyOpRef(a.matmul(b.transpose())))
    var r = result.unsafe_get[OpRef[DType.float32]]()
    assert_equal(r.shape(0), 5)
    assert_equal(r.shape(1), 3)


def test_rewrite_preserves_dtype() raises:
    var a = leaf[DType.float32]((2, 4))
    var b = leaf[DType.float32]((6, 4))
    var result = Simplifier[NATIVE_GPU_REWRITES].run(AnyOpRef(a.matmul(b.transpose())))
    assert_true(result.isa[OpRef[DType.float32]]())


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
