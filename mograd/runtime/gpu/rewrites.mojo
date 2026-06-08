from mograd.simplify import RewriteFn
from mograd.op import Op, OpRef, OpType

# ===-------------------------------------------------------------------===#
# GPU-specific rewrites
# ===-------------------------------------------------------------------===#


def GPU_REWRITES() -> List[Rule[RewriteFn]]:
    return [Rule(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]), fuse_matmul_transpose)]


# ===-------------------------------------------------------------------===#
# GPU-specific OpTypes
# ===-------------------------------------------------------------------===#

comptime MATMUL_T = OpType("MATMUL_T")

# ===-------------------------------------------------------------------===#
# GPU-specific rewrite methods
# ===-------------------------------------------------------------------===#


def fuse_matmul_transpose(node: OpRef) raises -> Optional[OpRef]:
    var A = node.src(0)
    var B = node.src(1).src(0)
    return OpRef(Op(MATMUL_T, node.layout(), node.dtype(), [A, B]))
