from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.ffi import OwnedDLHandle
from std.os.env import getenv

# ===-------------------------------------------------------------------===#
# Device
# ===-------------------------------------------------------------------===#


struct CtxKeeper(Movable):
    """Synchronizes the context when its last reference dies, so it is never
    destroyed with work still enqueued."""

    var ctx: DeviceContext

    def __init__(out self, ctx: DeviceContext):
        self.ctx = ctx.copy()

    def __del__(deinit self):
        try:
            self.ctx.synchronize()
        except:
            pass


def device(out device: Device) raises:
    device = Device()


@fieldwise_init
struct Device(Copyable, ImplicitlyCopyable, Movable):
    var ctx: DeviceContext
    var handle: ArcPointer[OwnedDLHandle]
    var keeper: ArcPointer[CtxKeeper]

    # ===-------------------------------------------------------------------===#
    # Lifecycle
    # ===-------------------------------------------------------------------===#

    def __init__(out self) raises:
        self.ctx = DeviceContext()

        var p = getenv("MOGRAD_SO")
        if not p:
            raise Error("MOGRAD_SO not set: run `pixi run build-gpu` first")

        self.handle = ArcPointer(OwnedDLHandle(p))
        self.keeper = ArcPointer(CtxKeeper(self.ctx))

    def __init__(out self, *, copy: Self):
        self.ctx = copy.ctx.copy()
        self.handle = copy.handle.copy()
        self.keeper = copy.keeper.copy()

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx^
        self.handle = move.handle^
        self.keeper = move.keeper^
