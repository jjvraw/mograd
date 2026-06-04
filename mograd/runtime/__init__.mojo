from std.gpu.host import DeviceContext

from mograd.op import OpRef
from mograd.buffer import AnyBuffer

from .native import GPURuntime

# ===-------------------------------------------------------------------===#
# Runtime
# ===-------------------------------------------------------------------===#


trait Runtime:
    @staticmethod
    def run(root: OpRef, ctx: Optional[DeviceContext]) raises -> AnyBuffer:
        ...
