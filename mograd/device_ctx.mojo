from std.memory import ArcPointer
from max.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.ffi import OwnedDLHandle
from std.os.env import getenv

from mograd.debug import DEBUG, DebugStats, Tracer
from mograd.memory import (
    DeviceMemGuard,
    PoolStats,
    empty_cache as _pool_empty_cache,
    memory_stats as _pool_memory_stats,
    memory_summary as _pool_memory_summary,
    pool_enabled as _pool_enabled,
)

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
    var rng: ArcPointer[UInt64]

    var mem: ArcPointer[DeviceMemGuard]
    """Memory-pool teardown guard, shared by all copies of this Device
    and co-owned by every pooled Buffer.

    The guard keeps its own DeviceContext copy, so it stays correct regardless 
    of field destruction order here.
    """

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
        self.rng = ArcPointer(UInt64(0))
        self.mem = ArcPointer(DeviceMemGuard(self.ctx))

    def __init__(out self, *, copy: Self):
        self.ctx = copy.ctx.copy()
        self.handle = copy.handle.copy()
        self.stats = copy.stats.copy()
        self.rng = copy.rng.copy()
        self.mem = copy.mem.copy()

    def __init__(out self, *, deinit move: Self):
        self.ctx = move.ctx^
        self.handle = move.handle^
        self.stats = move.stats^
        self.rng = move.rng^
        self.mem = move.mem^

    # ===-------------------------------------------------------------------===#
    # Memory pool
    # ===-------------------------------------------------------------------===#

    def empty_cache(self) raises:
        """Releases all cached device-memory blocks back to the runtime
        allocator (PyTorch `torch.cuda.empty_cache` analog)."""
        _pool_empty_cache(self.ctx)

    def memory_stats(self) raises -> PoolStats:
        return _pool_memory_stats(self.ctx)

    def memory_summary(self) raises -> String:
        return _pool_memory_summary(self.ctx)

    def pool_enabled(self) raises -> Bool:
        return _pool_enabled()

    # ===-------------------------------------------------------------------===#
    # RNG stream
    # ===-------------------------------------------------------------------===#

    def manual_seed(self, seed: Int):
        """Resets this device's RNG stream. Unseeded random factories draw
        from the stream, so runs replay identically after the same reset.
        All copies of this Device share the stream."""
        self.rng[] = UInt64(seed & 0x7FFFFFFFFFFFFFFF)

    def next_seed(self) -> Int:
        """Draws the next per-op seed from this device's RNG stream (splitmix64).
        Non-negative, so seeds pass losslessly through Int op attrs."""
        var x = self.rng[]
        self.rng[] = x + 0x9E3779B97F4A7C15
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
        x = (x ^ (x >> 27)) * 0x94D049BB133111EB
        return Int((x ^ (x >> 31)) & 0x7FFFFFFFFFFFFFFF)

    def get_function[T: TrivialRegisterPassable](self, var name: String) raises -> T:
        """Looks up an exported kernel in the mograd GPU library."""
        comptime if DEBUG >= 3:
            print("  ↳ " + Tracer.rpad(name, Tracer.KERNEL_COL), end="")
        elif DEBUG >= 2:
            print("  ↳", name)
        return self.handle[]._get_function[T](cstr_name=name.as_c_string_slice())
