from std.math import sqrt, abs
from std.sys import has_accelerator
from std.testing import TestSuite, assert_true, assert_false, assert_equal
from std.utils.numerics import neg_inf

from mograd import Device, Tensor
from mograd.op import AttrVal, Op, OpRef, OpType
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime.gpu.rewrites import GPU_REWRITES, fuse_matmul_transpose, MATMUL_BT, MATMUL_BIAS_BT, MEAN, FLASH_ATTN
from mograd.simplify import RewriteFn
from mograd.testing import leaf, assert_rewrites_to, assert_allclose
from mograd.simplify import Simplifier
import mograd.nn as nn

# ===-------------------------------------------------------------------===#
# MATMUL_BT rewrite
# ===-------------------------------------------------------------------===#


def test_fuse_matmul_transpose() raises:
    var a = leaf((2, 3))
    var b = leaf((4, 3))
    # MATMUL(A, TRANSPOSE(B)) -> MATMUL_BT(A, B)
    assert_rewrites_to(
        GPU_REWRITES(),
        a.matmul(b.transpose()),
        Pat(MATMUL_BT, [Pat(), Pat()]),
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
# MATMUL_BIAS_BT rewrite (fused matmul_bt + broadcast bias add)
# ===-------------------------------------------------------------------===#


def test_fuse_matmul_bias() raises:
    var a = leaf((2, 4))
    var w = leaf((3, 4))
    var bias = leaf((3,))
    # ADD(MATMUL_BT(A, W), EXPAND(bias)) -> MATMUL_BIAS_BT(A, W, bias)
    assert_rewrites_to(
        GPU_REWRITES(),
        a.matmul(w.transpose()) + bias.expand(2, 3),
        Pat(MATMUL_BIAS_BT, [Pat(), Pat(), Pat()]),
    )


def test_fuse_matmul_bias_through_identity_reshape() raises:
    # Linear stays rank-generic via a reshape that is a no-op for 2D inputs;
    # it must fold away at construction or it breaks this pattern.
    var a = leaf((2, 4))
    var w = leaf((3, 4))
    var bias = leaf((3,))
    assert_rewrites_to(
        GPU_REWRITES(),
        a.matmul(w.transpose()).reshape((2, 3)) + bias.expand(2, 3),
        Pat(MATMUL_BIAS_BT, [Pat(), Pat(), Pat()]),
    )


def test_fuse_matmul_bias_keeps_original_leaves() raises:
    var a = leaf((2, 4))
    var w = leaf((3, 4))
    var bias = leaf((3,))
    var rewritten = Simplifier(GPU_REWRITES()).run(a.matmul(w.transpose()) + bias.expand(2, 3))
    assert_true(rewritten.src(0) == a)
    assert_true(rewritten.src(1) == w)
    assert_true(rewritten.src(2) == bias)


def test_fuse_matmul_bias_removes_add_expand_nodes() raises:
    var a = leaf((2, 4))
    var w = leaf((3, 4))
    var bias = leaf((3,))
    var rewritten = Simplifier(GPU_REWRITES()).run(a.matmul(w.transpose()) + bias.expand(2, 3))
    assert_true(rewritten.op_type() == MATMUL_BIAS_BT)
    assert_false(rewritten.op_type() == OpType.ADD)


def test_fuse_matmul_bias_preserves_shape() raises:
    var a = leaf((2, 4))
    var w = leaf((3, 4))
    var bias = leaf((3,))
    var result = Simplifier(GPU_REWRITES()).run(a.matmul(w.transpose()) + bias.expand(2, 3))
    assert_equal(result.shape(0), 2)
    assert_equal(result.shape(1), 3)


def test_fuse_matmul_bias_preserves_dtype() raises:
    var a = leaf((2, 4), DType.float32)
    var w = leaf((3, 4), DType.float32)
    var bias = leaf((3,), DType.float32)
    var result = Simplifier(GPU_REWRITES()).run(a.matmul(w.transpose()) + bias.expand(2, 3))
    assert_true(result.dtype() == DType.float32)


def test_matmul_t_plus_non_expand_not_rewritten() raises:
    var a = leaf((2, 4))
    var w = leaf((3, 4))
    var other = leaf((2, 3))
    # ADD(MATMUL_BT(A, W), other) with no EXPAND must not fuse into MATMUL_BIAS_BT
    assert_rewrites_to(
        GPU_REWRITES(),
        a.matmul(w.transpose()) + other,
        Pat(OpType.ADD, [Pat(MATMUL_BT), Pat()]),
    )


def test_plain_add_not_rewritten() raises:
    var a = leaf((4,))
    var b = leaf((4,))
    # ADD with no MATMUL_BT operand must not be touched
    assert_rewrites_to(
        GPU_REWRITES(),
        a + b,
        Pat(OpType.ADD, [Pat(), Pat()]),
    )


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


# ===-------------------------------------------------------------------===#
# FLASH_ATTN rewrite
# ===-------------------------------------------------------------------===#


def _make_attn() raises -> OpRef:
    # B=2, S=4, H=2, D=8
    # BSHD leaves, transposed to BHSD for attention
    var q = leaf((2, 4, 2, 8))
    var k = leaf((2, 4, 2, 8))
    var v = leaf((2, 4, 2, 8))
    var Q = q.transpose(1, 2)
    var K = k.transpose(1, 2)
    var V = v.transpose(1, 2)
    var scores = Q.matmul(K.transpose(-2, -1))  # (2, 2, 4, 4)
    var sqrt_d = OpRef(
        Op(
            OpType.FULL,
            scores.layout().as_contiguous(),
            scores.dtype(),
            [],
            {"value": AttrVal(sqrt(Float32(8)))},
        )
    )
    var mask = leaf((2, 2, 4, 4))
    return (scores / sqrt_d + mask).softmax().matmul(V)


def _flash_attn_node(rewritten: OpRef) -> OpRef:
    # fuse_flash_attention returns GETTUPLE(FLASH_ATTN, 0). Unwrap to get FLASH_ATTN.
    return rewritten.src(0)


def test_fuse_flash_attention() raises:
    assert_rewrites_to(
        GPU_REWRITES(),
        _make_attn(),
        Pat(OpType.GETTUPLE, [Pat(FLASH_ATTN)]),
    )


def test_flash_attention_leaves_are_bshd() raises:
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_attn()))
    # shape(1) should be S=4 (BSHD), not H=2 (BHSD)
    assert_equal(flash.src(0).shape(1), 4)
    assert_equal(flash.src(1).shape(1), 4)
    assert_equal(flash.src(2).shape(1), 4)


def test_flash_attention_ordering_wins_over_matmul_bt() raises:
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_attn()))
    assert_true(flash.op_type() == FLASH_ATTN)
    assert_false(flash.op_type() == MATMUL_BT)


def test_flash_attention_not_rewritten_without_transpose_v() raises:
    # V has no TRANSPOSE wrapper, pattern doesn't match, stays as MATMUL
    var q = leaf((2, 4, 2, 8))
    var k = leaf((2, 4, 2, 8))
    var v = leaf((2, 2, 4, 8))  # already BHSD, no transpose
    var Q = q.transpose(1, 2)
    var K = k.transpose(1, 2)
    var scores = Q.matmul(K.transpose(-2, -1))
    var scale = leaf((2, 2, 4, 4))
    var mask = leaf((2, 2, 4, 4))
    var weights = (scores / scale + mask).softmax()
    assert_rewrites_to(
        GPU_REWRITES(),
        weights.matmul(v),
        Pat(OpType.MATMUL, [Pat(), Pat()]),
    )


def test_flash_attention_preserves_shape() raises:
    var rewritten = Simplifier(GPU_REWRITES()).run(_make_attn())
    assert_equal(rewritten.shape(0), 2)
    assert_equal(rewritten.shape(1), 2)
    assert_equal(rewritten.shape(2), 4)
    assert_equal(rewritten.shape(3), 8)


def test_flash_attention_preserves_dtype() raises:
    var rewritten = Simplifier(GPU_REWRITES()).run(_make_attn())
    assert_true(rewritten.dtype() == DType.float32)


def test_flash_attention_carries_scale_attr() raises:
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_attn()))
    assert_true("scale" in flash.attrs())


def test_flash_attention_scale_is_inv_sqrt_d_head() raises:
    # _make_attn uses d_head=8, so scale should be 1/sqrt(8)
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_attn()))
    var scale = flash.attrs()["scale"][Float32]
    var expected = Float32(1.0) / sqrt(Float32(8.0))
    assert_true(abs(scale - expected) < Float32(1e-6))


# ===-------------------------------------------------------------------===#
# scaled_dot_product_attention fusion
# ===-------------------------------------------------------------------===#


def _make_sdpa(is_causal: Bool) raises -> OpRef:
    # Mirror the model pattern: BSHD leaves transposed to BHSD.
    # fuse_flash_attention peels the TRANSPOSE to recover the contiguous BSHD
    # tensor the kernel needs, so inputs must carry the transpose wrapper.
    var B = 2
    var S = 4
    var H = 2
    var Dh = 8
    var Q = leaf((B, S, H, Dh)).transpose(1, 2)  # BHSD
    var K = leaf((B, S, H, Dh)).transpose(1, 2)
    var V = leaf((B, S, H, Dh)).transpose(1, 2)
    return Q.scaled_dot_product_attention(K, V, None, is_causal)


def test_sdpa_causal_fuses_to_flash_attn() raises:
    assert_rewrites_to(
        GPU_REWRITES(),
        _make_sdpa(True),
        Pat(OpType.GETTUPLE, [Pat(FLASH_ATTN)]),
    )


def test_sdpa_non_causal_with_explicit_mask_fuses_to_flash_attn() raises:
    var B = 2
    var S = 4
    var H = 2
    var Dh = 8
    var Q = leaf((B, S, H, Dh)).transpose(1, 2)
    var K = leaf((B, S, H, Dh)).transpose(1, 2)
    var V = leaf((B, S, H, Dh)).transpose(1, 2)
    var mask = leaf((B, H, S, S))  # explicit attention bias, not a causal triu
    var out = Q.scaled_dot_product_attention(K, V, mask, False)
    assert_rewrites_to(GPU_REWRITES(), out, Pat(OpType.GETTUPLE, [Pat(FLASH_ATTN)]))


def test_sdpa_causal_is_causal_attr_true() raises:
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_sdpa(True)))
    assert_true(flash.attrs()["is_causal"][Bool])


def test_sdpa_explicit_mask_is_causal_attr_false() raises:
    var B = 2
    var S = 4
    var H = 2
    var Dh = 8
    var Q = leaf((B, S, H, Dh)).transpose(1, 2)
    var K = leaf((B, S, H, Dh)).transpose(1, 2)
    var V = leaf((B, S, H, Dh)).transpose(1, 2)
    var mask = leaf((B, H, S, S))
    var out = Q.scaled_dot_product_attention(K, V, mask, False)
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(out))
    assert_false(flash.attrs()["is_causal"][Bool])


def test_sdpa_causal_leaves_are_bshd() raises:
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_sdpa(True)))
    # shape(1) should be S=4 (BSHD), not H=2 (BHSD)
    assert_equal(flash.src(0).shape(1), 4)
    assert_equal(flash.src(1).shape(1), 4)
    assert_equal(flash.src(2).shape(1), 4)


def test_sdpa_maskless_fuses_to_flash_attn() raises:
    assert_rewrites_to(
        GPU_REWRITES(),
        _make_sdpa(False),
        Pat(OpType.GETTUPLE, [Pat(FLASH_ATTN)]),
    )


def test_sdpa_maskless_attrs_no_causal_no_bias() raises:
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_sdpa(False)))
    assert_false(flash.attrs()["is_causal"][Bool])
    assert_false(flash.attrs()["has_bias"][Bool])


def test_sdpa_causal_has_bias_attr_false() raises:
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(_make_sdpa(True)))
    assert_false(flash.attrs()["has_bias"][Bool])


def test_sdpa_explicit_mask_has_bias_attr_true() raises:
    var Q = leaf((2, 4, 2, 8)).transpose(1, 2)
    var K = leaf((2, 4, 2, 8)).transpose(1, 2)
    var V = leaf((2, 4, 2, 8)).transpose(1, 2)
    var mask = leaf((2, 2, 4, 4))
    var out = Q.scaled_dot_product_attention(K, V, mask, False)
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(out))
    assert_true(flash.attrs()["has_bias"][Bool])


def test_sdpa_custom_scale_carries_scale_attr() raises:
    # The rewrite must carry the user's scale, not recompute 1/sqrt(d_head).
    var Q = leaf((2, 4, 2, 8)).transpose(1, 2)
    var K = leaf((2, 4, 2, 8)).transpose(1, 2)
    var V = leaf((2, 4, 2, 8)).transpose(1, 2)
    var out = Q.scaled_dot_product_attention(K, V, None, False, Float32(0.5))
    var flash = _flash_attn_node(Simplifier(GPU_REWRITES()).run(out))
    assert_true(abs(flash.attrs()["scale"][Float32] - Float32(0.5)) < Float32(1e-6))


def test_div_by_non_constant_divisor_not_rewritten() raises:
    # A runtime-tensor divisor has no statically-known scale to bake into the
    # kernel, so the pattern must not fuse.
    var q = leaf((2, 4, 2, 8))
    var k = leaf((2, 4, 2, 8))
    var v = leaf((2, 4, 2, 8))
    var Q = q.transpose(1, 2)
    var K = k.transpose(1, 2)
    var V = v.transpose(1, 2)
    var scores = Q.matmul(K.transpose(-2, -1))
    var divisor = leaf((2, 2, 4, 4))
    var mask = leaf((2, 2, 4, 4))
    assert_rewrites_to(
        GPU_REWRITES(),
        (scores / divisor + mask).softmax().matmul(V),
        Pat(OpType.MATMUL, [Pat(), Pat()]),
    )


def test_mul_by_full_canonicalizes_to_scale() raises:
    var x = leaf((2, 3))
    var c = OpRef(Op(OpType.FULL, x.layout(), x.dtype(), [], {"value": AttrVal(Float32(2.5))}))
    var out = Simplifier(GPU_REWRITES()).run(x * c)
    assert_true(out.op_type() == OpType.SCALE)
    assert_true(abs(out.attrs()["scalar"][Float32] - Float32(2.5)) < Float32(1e-6))
    assert_true(out.src(0) == x)


def test_full_times_x_canonicalizes_to_scale() raises:
    var x = leaf((2, 3))
    var c = OpRef(Op(OpType.FULL, x.layout(), x.dtype(), [], {"value": AttrVal(Float32(4.0))}))
    var out = Simplifier(GPU_REWRITES()).run(c * x)
    assert_true(out.op_type() == OpType.SCALE)
    assert_true(out.src(0) == x)


def test_div_by_full_canonicalizes_to_scale() raises:
    var x = leaf((2, 3))
    var c = OpRef(Op(OpType.FULL, x.layout(), x.dtype(), [], {"value": AttrVal(Float32(4.0))}))
    var out = Simplifier(GPU_REWRITES()).run(x / c)
    assert_true(out.op_type() == OpType.SCALE)
    assert_true(abs(out.attrs()["scalar"][Float32] - Float32(0.25)) < Float32(1e-6))


def test_div_by_zero_full_not_canonicalized() raises:
    var x = leaf((2, 3))
    var c = OpRef(Op(OpType.FULL, x.layout(), x.dtype(), [], {"value": AttrVal(Float32(0))}))
    var out = Simplifier(GPU_REWRITES()).run(x / c)
    assert_true(out.op_type() == OpType.DIV)


def test_div_by_full_then_flash_attention_fuses() raises:
    # The canonicalization and the fusion chain within one simplifier pass:
    # DIV(scores, FULL) becomes SCALE when the DIV node is visited, and the
    # outer matmul then fuses against the canonical spelling.
    assert_rewrites_to(
        GPU_REWRITES(),
        _make_attn(),
        Pat(OpType.GETTUPLE, [Pat(FLASH_ATTN)]),
    )


def test_rule_fallthrough_none_does_not_shadow() raises:
    # A rule whose function declines (returns None) must not shadow later
    # rules matching the same node. The flash-attention rule matches any
    # MATMUL(SOFTMAX, x) but declines non-attention graphs, and the
    # matmul-transpose rule must still fire on the same node.
    var a = leaf((4, 4))
    var b = leaf((4, 4))
    var out = a.softmax().matmul(b.transpose(0, 1))
    assert_rewrites_to(
        GPU_REWRITES(),
        out,
        Pat(MATMUL_BT, [Pat(), Pat()]),
    )


def test_rewritten_node_rematches_to_fixpoint() raises:
    # mean spelled as sum(x) / FULL(n): the DIV canonicalizes to
    # SCALE(SUM, 1/n), and the fresh SCALE node must itself be re-matched so
    # sum-scale fusion can chain to MEAN within the same pass.
    var a = leaf((4,))
    var n = OpRef(Op(OpType.FULL, a.sum().layout(), a.dtype(), [], {"value": AttrVal(Float32(4.0))}))
    assert_rewrites_to(
        GPU_REWRITES(),
        a.sum() / n,
        Pat(MEAN, [Pat()]),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
