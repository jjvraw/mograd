from mograd.simplify import RewriteFn
from mograd.op import OpRef, OpType, AnyOpRef

# ===-------------------------------------------------------------------===#
# GPU-specific rewrites
# ===-------------------------------------------------------------------===#

comptime NATIVE_GPU_REWRITES: List[Rule[RewriteFn]] = [
    Rule(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]), fuse_matmul_transpose)
]

# ===-------------------------------------------------------------------===#
# GPU-specific OpTypes
# ===-------------------------------------------------------------------===#

comptime MATMUL_T = OpType("MATMUL_T")

# ===-------------------------------------------------------------------===#
# GPU-specific rewrite methods
# ===-------------------------------------------------------------------===#


def fuse_matmul_transpose(node: AnyOpRef) raises -> Optional[AnyOpRef]:
    var n = node.unsafe_get[OpRef[DType.float32]]()
    var A = n.src(0)
    var B = n.src(1).src(0)
    return AnyOpRef(OpRef[DType.float32](MATMUL_T, n.shape(), [A, B]))
