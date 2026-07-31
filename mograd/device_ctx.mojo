from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.ffi import OwnedDLHandle
from std.os.env import getenv

# ===-------------------------------------------------------------------===#
# Device
# ===-------------------------------------------------------------------===#


def device(out device: Device) raises:
    device = Device()


@fieldwise_init
struct Device(Copyable, ImplicitlyCopyable, Movable):
    var ctx: DeviceContext
    var handle: ArcPointer[OwnedDLHandle]

    # ===-------------------------------------------------------------------===#
    # Lifecycle
    # ===-------------------------------------------------------------------===#

    def __init__(out self) raises:
        self.ctx = DeviceContext()

        var p = getenv("MOGRAD_SO")
        if not p:
            raise Error("MOGRAD_SO not set: run `pixi run build-gpu` first")

        self.handle = ArcPointer(OwnedDLHandle(p))

    def __init__(out self, *, copy: Self):
        self.ctx = copy.ctx.copy()
        self.handle = copy.handle.copy()

    def __init__(out self, *, deinit take: Self):
        self.ctx = take.ctx^
        self.handle = take.handle^

    def get_function[T: TrivialRegisterPassable](self, var name: String) -> T:
        """Looks up an exported kernel in the mograd GPU library."""
        return self.handle[].get_function[T](name^)
