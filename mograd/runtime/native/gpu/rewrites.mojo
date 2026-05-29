from mograd.op import Op, OpRef, OpType

# ===-------------------------------------------------------------------===#
# GPU-specific OpTypes
# ===-------------------------------------------------------------------===#

comptime MATMUL_T = OpType("MATMUL_T")

# ===-------------------------------------------------------------------===#
# GPU-speicifc rewrites
# ===-------------------------------------------------------------------===#


def fuse_matmul_transpose(node: OpRef) raises -> Optional[OpRef]:
    var A = node.srcs()[0]
    var B = node.srcs()[1].srcs()[0]
    return OpRef(Op(MATMUL_T, node.shape(), node.dtype(), [A, B]))
