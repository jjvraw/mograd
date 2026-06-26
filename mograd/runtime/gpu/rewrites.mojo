from std.math import abs

from mograd.simplify import RewriteFn
from mograd.op import Op, OpRef, OpType

# ===-------------------------------------------------------------------===#
# GPU-specific rewrites
# ===-------------------------------------------------------------------===#


def GPU_REWRITES() -> List[Rule[RewriteFn]]:
    return [
        Rule(Pat(OpType.MATMUL, [Pat(OpType.TRANSPOSE), Pat()]), fuse_transpose_matmul),
        Rule(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]), fuse_matmul_transpose),
        Rule(Pat(OpType.SCALE, [Pat(OpType.SUM)]), fuse_sum_scale),
        Rule(Pat(OpType.ADD, [Pat(MATMUL_BT), Pat(OpType.EXPAND)]), fuse_matmul_bias),
    ]


# ===-------------------------------------------------------------------===#
# GPU-specific OpTypes
# ===-------------------------------------------------------------------===#

comptime MATMUL_AT = OpType("MATMUL_AT")
comptime MATMUL_BT = OpType("MATMUL_BT")
comptime MATMUL_BIAS_BT = OpType("MATMUL_BIAS_BT")
comptime MEAN = OpType("MEAN")

# ===-------------------------------------------------------------------===#
# GPU-specific rewrite methods
# ===-------------------------------------------------------------------===#


def fuse_transpose_matmul(node: OpRef) raises -> Optional[OpRef]:
    var A = node.src(0).src(0)
    var B = node.src(1)
    return OpRef(Op(MATMUL_AT, node.layout(), node.dtype(), [A, B]))


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
