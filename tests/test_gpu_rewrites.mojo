from std.testing import TestSuite, assert_true, assert_false, assert_equal

from mograd.op import OpType, OpRef
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime.gpu.rewrites import GPU_REWRITES, fuse_matmul_transpose, MATMUL_T, MEAN
from mograd.simplify import RewriteFn
from mograd.testing import leaf, assert_rewrites_to
from mograd.simplify import Simplifier

# ===-------------------------------------------------------------------===#
# MATMUL_T rewrite
# ===-------------------------------------------------------------------===#


def test_fuse_matmul_transpose() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    # MATMUL(A, TRANSPOSE(B)) -> MATMUL_T(A, B)
    assert_rewrites_to(
        GPU_REWRITES(),
        a.matmul(b.transpose()),
        Pat(MATMUL_T, [Pat(), Pat()]),
    )


def test_fuse_matmul_transpose_removes_transpose_node() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    var rewritten = Simplifier(GPU_REWRITES()).run(a.matmul(b.transpose()))
    assert_false(rewritten.src(1).op_type() == OpType.TRANSPOSE)


def test_plain_matmul_not_rewritten() raises:
    var a = leaf((2, 3))
    var b = leaf((3, 4))
    # MATMUL(A, B) with no TRANSPOSE must not be touched
    assert_rewrites_to(
        GPU_REWRITES(),
        a.matmul(b),
        Pat(OpType.MATMUL, [Pat(), Pat()]),
    )


def test_rewrite_preserves_shape() raises:
    var a = leaf((5, 7))
    var b = leaf((3, 7))
    var result = Simplifier(GPU_REWRITES()).run(a.matmul(b.transpose()))
    assert_equal(result.shape(0), 5)
    assert_equal(result.shape(1), 3)


def test_rewrite_preserves_dtype() raises:
    var a = leaf((2, 4), DType.float32)
    var b = leaf((6, 4), DType.float32)
    var result = Simplifier(GPU_REWRITES()).run(a.matmul(b.transpose()))
    assert_true(result.dtype() == DType.float32)


# ===-------------------------------------------------------------------===#
# MEAN rewrite (fused sum + scale)
# ===-------------------------------------------------------------------===#


def test_fuse_sum_scale_full_reduce() raises:
    var a = leaf((4,))
    # SCALE(SUM(a), 1/4) -> MEAN(a)
    assert_rewrites_to(
        GPU_REWRITES(),
        a.sum().scale(0.25),
        Pat(MEAN, [Pat()]),
    )


def test_fuse_sum_scale_axis() raises:
    var a = leaf((2, 3))
    # SCALE(SUM(a, axis=1), 1/3) -> MEAN(a)
    assert_rewrites_to(
        GPU_REWRITES(),
        a.sum(1).scale(1.0 / 3.0),
        Pat(MEAN, [Pat()]),
    )


def test_fuse_sum_scale_removes_sum_node() raises:
    var a = leaf((4,))
    var rewritten = Simplifier(GPU_REWRITES()).run(a.sum().scale(0.25))
    assert_false(rewritten.src(0).op_type() == OpType.SUM)


def test_non_reciprocal_scale_not_rewritten() raises:
    var a = leaf((4,))
    # 5.0 != 1/4, this is not a mean - must not fuse into MEAN
    assert_rewrites_to(
        GPU_REWRITES(),
        a.sum().scale(5.0),
        Pat(OpType.SCALE, [Pat(OpType.SUM)]),
    )


def test_plain_sum_not_rewritten() raises:
    var a = leaf((4,))
    # SUM with no surrounding SCALE must not be touched
    assert_rewrites_to(
        GPU_REWRITES(),
        a.sum(),
        Pat(OpType.SUM, [Pat()]),
    )


def test_mean_rewrite_preserves_shape() raises:
    var a = leaf((2, 3))
    var result = Simplifier(GPU_REWRITES()).run(a.sum(1).scale(1.0 / 3.0))
    assert_equal(result.shape(0), 2)


def test_mean_rewrite_preserves_dtype() raises:
    var a = leaf((4,), DType.float32)
    var result = Simplifier(GPU_REWRITES()).run(a.sum().scale(0.25))
    assert_true(result.dtype() == DType.float32)


def test_mean_rewrite_carries_axis_attr() raises:
    var a = leaf((2, 3))
    var result = Simplifier(GPU_REWRITES()).run(a.sum(1).scale(1.0 / 3.0))
    assert_true("axis" in result.attrs())
    assert_equal(result.attr_int("axis"), 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
