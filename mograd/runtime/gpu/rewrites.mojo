from std.math import abs

from mograd.simplify import RewriteFn
from mograd.op import Op, OpRef, OpType

# ===-------------------------------------------------------------------===#
# GPU-specific rewrites
# ===-------------------------------------------------------------------===#


def GPU_REWRITES() -> List[Rule[RewriteFn]]:
    return [
        Rule(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]), fuse_matmul_transpose),
        Rule(Pat(OpType.SCALE, [Pat(OpType.SUM)]), fuse_sum_scale),
        Rule(Pat(OpType.ADD, [Pat(MATMUL_BT), Pat(OpType.EXPAND)]), fuse_matmul_bias),
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

# ===-------------------------------------------------------------------===#
# GPU-specific rewrite methods
# ===-------------------------------------------------------------------===#


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
