from std.gpu.host import DeviceContext

from mograd.op import OpRef, OpType
from mograd.buffer import Buffer
from mograd.pattern_matcher import Rule, Pat
from mograd.runtime import Runtime
from mograd.simplify import Simplifier, RewriteFn
from mograd.runtime.native.gpu import GPURuntime
from mograd.runtime.native.gpu.rewrites import NATIVE_GPU_REWRITES

# ===-------------------------------------------------------------------===#
# NativeRuntime
# ===-------------------------------------------------------------------===#


struct NativeRuntime(Runtime):
    @staticmethod
    def run(root: OpRef, ctx: Optional[DeviceContext]) raises -> Buffer:
        if ctx:
            var simplified = Simplifier[NATIVE_GPU_REWRITES].run(root)
            return GPURuntime.run(simplified, ctx)
        else:
            raise Error("Provide context, CPU backend not supported.")
