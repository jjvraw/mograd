from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.ffi import OwnedDLHandle
from std.os.env import getenv

from mograd.debug import DEBUG, DebugStats, Tracer

# ===-------------------------------------------------------------------===#
# Device
# ===-------------------------------------------------------------------===#


def device(out device: Device) raises:
    device = Device()


@fieldwise_init
struct Device(Copyable, ImplicitlyCopyable, Movable):
    var ctx: DeviceContext
    var handle: ArcPointer[OwnedDLHandle]
    var stats: ArcPointer[DebugStats]

    # ===-------------------------------------------------------------------===#
    # Lifecycle
    # ===-------------------------------------------------------------------===#

    def __init__(out self) raises:
        self.ctx = DeviceContext()

        var p = getenv("MOGRAD_SO")
        if not p:
            raise Error("MOGRAD_SO not set: run `pixi run build-gpu` first")

        self.handle = ArcPointer(OwnedDLHandle(p))
        self.stats = ArcPointer(DebugStats())

    def __init__(out self, *, copy: Self):
        self.ctx = copy.ctx.copy()
        self.handle = copy.handle.copy()
        self.stats = copy.stats.copy()

    def __init__(out self, *, deinit take: Self):
        self.ctx = take.ctx^
        self.handle = take.handle^
        self.stats = take.stats^

    def get_function[T: TrivialRegisterPassable](self, var name: String) -> T:
        """Looks up an exported kernel in the mograd GPU library."""
        comptime if DEBUG >= 3:
            print("  ↳ " + Tracer.rpad(name, Tracer.KERNEL_COL), end="")
        elif DEBUG >= 2:
            print("  ↳", name)
        return self.handle[].get_function[T](name^)
