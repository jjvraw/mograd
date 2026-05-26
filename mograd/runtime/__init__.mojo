from std.gpu.host import DeviceContext

from mograd.op import OpRef
from mograd.buffer import Buffer

from .native import NativeRuntime

# ===-------------------------------------------------------------------===#
# Runtime
# ===-------------------------------------------------------------------===#


trait Runtime:
    @staticmethod
    def run(root: OpRef, ctx: Optional[DeviceContext]) raises -> Buffer:
        ...
