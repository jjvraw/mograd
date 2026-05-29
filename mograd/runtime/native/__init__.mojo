from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime import Runtime
from mograd.simplify import Simplifier, RewriteFn
from mograd.runtime.native.gpu import GPURuntime
from mograd.runtime.native.gpu.rewrites import fuse_matmul_transpose, MATMUL_T

# ===-------------------------------------------------------------------===#
# NativeRuntime
# ===-------------------------------------------------------------------===#


struct NativeRuntime(Runtime):
    @staticmethod
    def run(root: OpRef, ctx: Optional[DeviceContext]) raises -> Buffer:
        if ctx:
            var simplified = Simplifier[
                [Rule(Pat(OpType.MATMUL, [Pat(), Pat(OpType.TRANSPOSE)]), fuse_matmul_transpose)]
            ].run(root)
            return GPURuntime.run(simplified, ctx)
        else:
            raise Error("Provide context, CPU backend not supported.")
