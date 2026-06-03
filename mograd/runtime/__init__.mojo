from std.gpu.host import DeviceContext

from mograd.op import AnyOpRef
from mograd.buffer import AnyBuffer

from .native import GPURuntime

# ===-------------------------------------------------------------------===#
# Runtime
# ===-------------------------------------------------------------------===#


trait Runtime:
    @staticmethod
    def run(root: AnyOpRef, ctx: Optional[DeviceContext]) raises -> AnyBuffer:
        ...
