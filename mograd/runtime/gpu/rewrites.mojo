from std.math import abs, sqrt

from mograd.layout import Layout
from mograd.pattern_matcher import Pat, Rule
from mograd.simplify import RewriteFn
from mograd.op import AttrVal, Op, OpRef, OpType

# ===-------------------------------------------------------------------===#
# GPU-specific rewrites
# ===-------------------------------------------------------------------===#


def GPU_REWRITES() -> List[Rule[RewriteFn]]:
    return [
        # Flash Attention (handles MATMUL(SOFTMAX, V) and extracts Q/K/V from
        # any CONTIGUOUS/TRANSPOSE wrappers on the inner Q@K^T matmul directly,
        # so no CONTIGUOUS(TRANSPOSE) rule for matmul_transpose is needed here)
        Rule(Pat(OpType.MATMUL, [Pat(OpType.SOFTMAX), Pat()]), fuse_flash_attention),
        # Matmul(A, B.T)
        Rule(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]), fuse_matmul_transpose),
        # Matmul(A, B.T) + Bias
        Rule(Pat(OpType.ADD, [Pat(MATMUL_BT), Pat(OpType.EXPAND)]), fuse_matmul_bias),
        # Canonicalize x */÷ constant-splat to SCALE so downstream patterns
        # (flash attention, sum-scale) only ever match one spelling.
        Rule(Pat(OpType.MUL, [Pat(), Pat()]), canonicalize_mul_full),
        Rule(Pat(OpType.DIV, [Pat(), Pat()]), canonicalize_div_full),
        # Scale(Sum(x)), or mean
        Rule(Pat(OpType.SCALE, [Pat(OpType.SUM)]), fuse_sum_scale),
        # LayerNorm
        Rule(
            Pat(OpType.ADD, [Pat(OpType.MUL, [Pat(), Pat(OpType.EXPAND)]), Pat(OpType.EXPAND)]),
            fuse_layer_norm,
        ),
    ]


# ===-------------------------------------------------------------------===#
# GPU-specific OpTypes
# ===-------------------------------------------------------------------===#

comptime MATMUL_BT = OpType("MATMUL_BT")
comptime MATMUL_BIAS_BT = OpType("MATMUL_BIAS_BT")
comptime MEAN = OpType("MEAN")
comptime LAYER_NORM = OpType("LAYER_NORM")
comptime LAYER_NORM_GRAD = OpType("LAYER_NORM_GRAD")
comptime FLASH_ATTN = OpType("FLASH_ATTN")
comptime FLASH_ATTN_GRAD = OpType("FLASH_ATTN_GRAD")

# ===-------------------------------------------------------------------===#
# GPU-specific rewrite methods
# ===-------------------------------------------------------------------===#


def fuse_flash_attention(node: OpRef) raises -> Optional[OpRef]:
    # Strip CONTIGUOUS then one movement op to recover the underlying tensor.
    def peel(n: OpRef) -> OpRef:
        var m = n
        if m.op_type() == OpType.CONTIGUOUS:
            var inner = m.src(0)
            m = inner^
        if m.op_type() == OpType.TRANSPOSE or m.op_type() == OpType.RESHAPE or m.op_type() == OpType.VIEW:
            return m.src(0)
        return m

    def ensure_4d(n: OpRef) raises -> Optional[OpRef]:
        var rank = n.layout().rank()
        if rank == 4:
            return n
        if rank == 3:
            var s = n.layout()
            return n.reshape(Layout(s.shape(0), s.shape(1), 1, s.shape(2)))
        return None

    var softmax = node.src(0)
    var V = node.src(1)

    # V must go through a movement op so we can recover BSHD.
    var V_inner = V
    if V_inner.op_type() == OpType.CONTIGUOUS:
        var inner = V_inner.src(0)
        V_inner = inner^
    if (
        V_inner.op_type() != OpType.TRANSPOSE
        and V_inner.op_type() != OpType.RESHAPE
        and V_inner.op_type() != OpType.VIEW
    ):
        return None

    var V_bshd_opt = ensure_4d(peel(V))
    if not V_bshd_opt:
        return None
    var V_bshd = V_bshd_opt.value()

    # softmax(scores + mask) fuses with a causal or additive-bias mask.
    # Bare softmax(scores) fuses mask-free: has_bias=False with a 1-element
    # placeholder mask input the kernels never read.
    var pre_softmax = softmax.src(0)
    var has_mask = pre_softmax.op_type() == OpType.ADD
    var mask: OpRef
    if has_mask:
        mask = pre_softmax.src(1)
    else:
        mask = OpRef(Op(OpType.FULL, Layout(1, 1, 1, 1), node.dtype(), [], {"value": AttrVal(Float32(0))}))

    # Scores must be SCALE(QK) with a static scalar. DIV-by-splat spellings
    # were canonicalized to SCALE before this pattern runs.
    # A DIV that survives has a runtime divisor and correctly does not fuse.
    var scores_node = pre_softmax.src(0) if has_mask else pre_softmax
    if scores_node.op_type() != OpType.SCALE:
        return None
    var qk_node = scores_node.src(0)

    # Accept either MATMUL_BT(A, K_bhsd) or MATMUL(A, CONTIGUOUS(TRANSPOSE(K_bhsd)))
    # (the latter arises when op.mojo wraps non-collapsible K in CONTIGUOUS before
    # the MATMUL_BT rewrite fires, e.g. K_bhsd = TRANSPOSE(RESHAPE(x), 1, 2)).
    # TODO: Should we always just work on the simplified/"canonical" version?
    var Q_raw: OpRef
    var K_raw: OpRef
    if qk_node.op_type() == MATMUL_BT:
        Q_raw = qk_node.src(0)
        K_raw = qk_node.src(1)
    elif qk_node.op_type() == OpType.MATMUL:
        var rhs = qk_node.src(1)
        if rhs.op_type() == OpType.CONTIGUOUS:
            var inner = rhs.src(0)  # strip CONTIGUOUS → TRANSPOSE(K_bhsd, -2, -1)
            rhs = inner^
        if rhs.op_type() != OpType.TRANSPOSE:
            return None
        # rhs.src(0) is K_bhsd (e.g. TRANSPOSE(RESHAPE(x), 1, 2)).
        # peel() will then strip that TRANSPOSE to reach the raw BSHD tensor.
        Q_raw = qk_node.src(0)
        K_raw = rhs.src(0)
    else:
        return None

    var Q_bshd_opt = ensure_4d(peel(Q_raw))
    var K_bshd_opt = ensure_4d(peel(K_raw))
    if not Q_bshd_opt or not K_bshd_opt:
        return None
    var Q_bshd = Q_bshd_opt.value()
    var K_bshd = K_bshd_opt.value()

    var scale = scores_node.attrs()["scalar"][Float32]

    var is_causal = has_mask and (
        mask.op_type() == OpType.TRIU and mask.attrs()["diagonal"][Int] == 1 and mask.src(0).op_type() == OpType.FULL
    )
    var has_bias = has_mask and not is_causal
    var flash = OpRef(
        Op(
            FLASH_ATTN,
            node.layout(),
            node.dtype(),
            [Q_bshd, K_bshd, V_bshd, mask],
            {"scale": scale, "is_causal": is_causal, "has_bias": has_bias},
        )
    )
    # FLASH_ATTN is multi-output: output 0 = O (BHSD), output 1 = LSE (BHS).
    # Expose O to downstream consumers via GETTUPLE so the backward can access LSE.
    return OpRef(Op(OpType.GETTUPLE, node.layout(), node.dtype(), [flash], {"index": 0}))


def fuse_matmul_transpose(node: OpRef) raises -> Optional[OpRef]:
    var A = node.src(0)
    var B = node.src(1).src(0)
    return OpRef(Op(MATMUL_BT, node.layout(), node.dtype(), [A, B]))


def fuse_matmul_bias(node: OpRef) raises -> Optional[OpRef]:
    var mm = node.src(0)
    var A = mm.src(0)
    var B = mm.src(1)
    var bias = node.src(1).src(0)
    return OpRef(Op(MATMUL_BIAS_BT, node.layout(), node.dtype(), [A, B, bias]))


def canonicalize_mul_full(node: OpRef) raises -> Optional[OpRef]:
    # MUL(x, FULL(v)) and MUL(FULL(v), x) are SCALE(x, v) spelled through a
    # materialized splat. One canonical spelling keeps every downstream
    # pattern matcher (and the kernels behind them) to a single case.
    var a = node.src(0)
    var b = node.src(1)
    if b.op_type() == OpType.FULL:
        return Optional(a.scale(b.attrs()["value"][Float32]))
    if a.op_type() == OpType.FULL:
        return Optional(b.scale(a.attrs()["value"][Float32]))
    return None


def canonicalize_div_full(node: OpRef) raises -> Optional[OpRef]:
    # DIV(x, FULL(v)) is SCALE(x, 1/v). A zero-valued or runtime-tensor
    # divisor has no static reciprocal and stays a DIV.
    var divisor = node.src(1)
    if divisor.op_type() != OpType.FULL:
        return None
    var v = divisor.attrs()["value"][Float32]
    if v == 0:
        return None
    return Optional(node.src(0).scale(Float32(1.0) / v))


def fuse_sum_scale(node: OpRef) raises -> Optional[OpRef]:
    # scale(sum(x)) is only literally "mean" when scalar == 1/N.
    var sum_node = node.src(0)
    var src = sum_node.src(0)
    var scalar = node.attrs()["scalar"][Float32]

    var n: Int
    if "axis" in sum_node.attrs():
        var ax = src.layout().normalise_dim(sum_node.attr_int("axis"))
        n = src.layout().shape(ax)
    else:
        n = src.layout().numel()

    var expected = Float32(1.0) / Float32(n)
    if abs(scalar - expected) > Float32(1e-6):
        return None

    var attrs = sum_node.attrs_copy()
    attrs["scalar"] = node.attrs()["scalar"]
    return OpRef(Op(MEAN, node.layout(), node.dtype(), [src], attrs^))


def fuse_layer_norm(node: OpRef) raises -> Optional[OpRef]:
    # Pattern (after MEAN rewrite has fired):
    # ADD(MUL(DIV(diff, SQRT(ADD(EXPAND(MEAN(diff*diff,-1)), FULL(eps)))), EXPAND(γ)), EXPAND(β))
    # where diff = SUB(x, EXPAND(MEAN(x, -1)))
    var mul_node = node.src(0)
    var beta_expand = node.src(1)
    if mul_node.op_type() != OpType.MUL:
        return None
    if beta_expand.op_type() != OpType.EXPAND:
        return None

    var x_norm = mul_node.src(0)
    var gamma_expand = mul_node.src(1)
    if x_norm.op_type() != OpType.DIV:
        return None
    if gamma_expand.op_type() != OpType.EXPAND:
        return None

    var diff = x_norm.src(0)
    var denom = x_norm.src(1)
    # diff = x - mean(x) = ADD(x, NEG(EXPAND(MEAN(x))))
    if diff.op_type() != OpType.ADD:
        return None
    if denom.op_type() != OpType.SQRT:
        return None

    var var_eps = denom.src(0)
    if var_eps.op_type() != OpType.ADD:
        return None

    var var_expand = var_eps.src(0)
    var eps_full = var_eps.src(1)
    if var_expand.op_type() != OpType.EXPAND:
        return None
    if eps_full.op_type() != OpType.FULL:
        return None

    var mean_var = var_expand.src(0)
    if mean_var.op_type() != MEAN:
        return None
    if "axis" not in mean_var.attrs():
        return None

    var diff_sq = mean_var.src(0)
    if diff_sq.op_type() != OpType.MUL:
        return None
    if diff_sq.src(0) != diff or diff_sq.src(1) != diff:
        return None

    # diff = ADD(x, NEG(EXPAND(MEAN(x, axis))))
    var x = diff.src(0)
    var neg_mean = diff.src(1)
    if neg_mean.op_type() != OpType.NEG:
        return None
    var mean_x_expand = neg_mean.src(0)
    if mean_x_expand.op_type() != OpType.EXPAND:
        return None
    var mean_x = mean_x_expand.src(0)
    if mean_x.op_type() != MEAN:
        return None
    if mean_x.src(0) != x:
        return None

    var axis = mean_var.attr_int("axis")
    if mean_x.attr_int("axis") != axis:
        return None

    var eps = eps_full.attrs()["value"][Float32]
    var gamma = gamma_expand.src(0)
    var beta = beta_expand.src(0)

    return OpRef(Op(LAYER_NORM, node.layout(), node.dtype(), [x, gamma, beta], {"eps": eps, "axis": axis}))
