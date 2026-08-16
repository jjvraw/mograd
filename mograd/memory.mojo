"""Caching device-memory allocator.

The 2026-07-10 MAX nightly dropped the runtime's caching allocator,
https://github.com/modular/modular/issues/6815, every `enqueue_create_buffer`
/destroy pair costs a real driver allocation. We therefore manage
our own caching buffer.

Every device allocation is a pop from a size-keyed free list, and every
deallocation is a push back onto it. Driver allocations is only realised 
when a list is empty, and memory is only returned on `empty_cache()`.
This is somewhat inspired by PyTorch's CUDACachingAllocator.

Allocations are rounded up to size classes (power-of-2 intervals split into
`DIVISIONS` subdivisions, <=12.5% waste) and cached in per-context free lists
of untyped byte buffers. Typed views are handed out via
`create_sub_buffer[dtype](0, numel)`. Cached blocks are retained until
`empty_cache` or an allocation failure (which empties the cache and retries
once). Disable with MOGRAD_POOL=0.

State is process-global (`_Global`) and keyed by context handle so the same
pool serves `Buffer` storage in the main binary and kernel/layout scratch
inside libmograd_gpu.so, which only receives a `DeviceContext`. Each entry
keeps a copy of its context, pinning the handle address so the pointer key is
unambiguous. This module is compiled into both the binary and the kernel
library: after changing state layout here, rebuild libmograd_gpu.so.

NOTE: There is currently no synchronization as all current mograd work for
a context is enqueued on a single in-order stream: work enqueued after
a take cannot run before the earlier-enqueued work that last touched the
memory. If mograd ever runs multiple streams per context, frees must record
an event (`Storage.__del__`) that takes wait on (`take_storage`), PyTorch's
`record_stream` analog.

Teardown discipline: releasing device buffers and then destroying their
DeviceContext leaves the enqueued releases unsynchronized, which poisons a
per-device AsyncRT mutex on the 2026-07 runtime and stalls the next context's
first allocation. `DeviceMemGuard` therefore purges this context's pool state
and synchronizes before the context handle can drop. It is co-owned by `Device`
and by every pooled `Storage` ticket, so the purge runs only after the last
pooled buffer is gone, a Tensor outliving its Device stays safe.
"""

from std.bit import prev_power_of_two
from std.ffi import _Global
from max.gpu.host import DeviceBuffer, DeviceContext
from std.memory import ArcPointer
from std.os.env import getenv
from std.sys import size_of


# ===-------------------------------------------------------------------===#
# Size classes
# ===-------------------------------------------------------------------===#


comptime MIN_CLASS = 512
comptime DIVISIONS = 8


def round_class(nbytes: Int) -> Int:
    """Rounds a byte count up to its allocation class.

    Classes are multiples of `prev_power_of_two(nbytes) // DIVISIONS`, so the
    rounded size overshoots by at most 1/DIVISIONS; every class is a multiple
    of 64 >= MIN_CLASS, which satisfies alignment for all dtypes at offset 0.
    """
    if nbytes <= MIN_CLASS:
        return MIN_CLASS
    var p = prev_power_of_two(nbytes)
    if p == nbytes:
        return nbytes
    var step = p // DIVISIONS
    return ((nbytes + step - 1) // step) * step


# ===-------------------------------------------------------------------===#
# Stats
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct PoolStats(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Per-context allocator counters. Byte figures are class (rounded) sizes."""

    var allocated_bytes: Int
    """Bytes currently handed out to live Storage tickets."""
    var reserved_bytes: Int
    """Allocated bytes plus bytes cached in free lists."""
    var peak_allocated: Int
    var peak_reserved: Int
    var alloc_count: Int
    """Real driver allocations."""
    var hit_count: Int
    var miss_count: Int
    var free_count: Int
    """Returns of storage to the free lists."""

    def __init__(out self):
        self.allocated_bytes = 0
        self.reserved_bytes = 0
        self.peak_allocated = 0
        self.peak_reserved = 0
        self.alloc_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.free_count = 0


# ===-------------------------------------------------------------------===#
# Global state
# ===-------------------------------------------------------------------===#


struct _ContextPool(Movable):
    var keeper: DeviceContext
    var free: Dict[Int, List[DeviceBuffer[DType.uint8]]]
    var stats: PoolStats

    def __init__(out self, keeper: DeviceContext):
        self.keeper = keeper.copy()
        self.free = {}
        self.stats = PoolStats()


struct _MemoryState(Movable):
    var keys: List[Int]
    var pools: List[_ContextPool]
    var enabled: Bool

    def __init__(out self):
        self.keys = []
        self.pools = []
        self.enabled = getenv("MOGRAD_POOL") != "0"

    def _index_of(self, key: Int) -> Int:
        for i in range(len(self.keys)):
            if self.keys[i] == key:
                return i
        return -1

    def _ensure(mut self, ctx: DeviceContext, key: Int) raises -> Int:
        var i = self._index_of(key)
        if i >= 0:
            return i
        self.keys.append(key)
        self.pools.append(_ContextPool(ctx))
        return len(self.keys) - 1


def _memory_state_init() -> _MemoryState:
    return _MemoryState()


comptime _STATE = _Global["MOGRAD_MEMORY_POOL", _memory_state_init]


@always_inline
def _ctx_key(ctx: DeviceContext) -> Int:
    return Int(ctx._handle.value())


def pool_enabled() raises -> Bool:
    return _STATE.get_or_create_ptr()[].enabled


# ===-------------------------------------------------------------------===#
# Teardown guard
# ===-------------------------------------------------------------------===#


struct DeviceMemGuard(Movable):
    """Drains this context's pool state before the context handle can drop.

    Held via ArcPointer by `Device` and by every pooled `Storage`, so the
    purge+synchronize runs when the last of {Device copies, outstanding pooled
    buffers} dies. Owns its own DeviceContext copy, making it independent of
    field destruction order in `Device`.
    """

    var ctx: DeviceContext

    def __init__(out self, ctx: DeviceContext):
        self.ctx = ctx.copy()

    def __deinit__(deinit self):
        try:
            purge(self.ctx)
            self.ctx.synchronize()
        except:
            # Never propagate from teardown.
            # At process exit the driver may already be shutting down.
            pass


# ===-------------------------------------------------------------------===#
# Storage
# ===-------------------------------------------------------------------===#


struct Storage(Movable):
    """Owns one pooled allocation, recycles it into the free lists on teardown.

    Also the owner of record for any typed sub-buffer views: those are not
    guaranteed to retain their parent, so `bytes` must stay alive for the
    lifetime of every view handed out over it.
    """

    var bytes: DeviceBuffer[DType.uint8]
    var class_bytes: Int
    var key: Int
    var pooled: Bool
    var guard: Optional[ArcPointer[DeviceMemGuard]]

    def __init__(
        out self,
        var bytes: DeviceBuffer[DType.uint8],
        class_bytes: Int,
        key: Int,
        pooled: Bool,
        guard: Optional[ArcPointer[DeviceMemGuard]],
    ):
        self.bytes = bytes^
        self.class_bytes = class_bytes
        self.key = key
        self.pooled = pooled
        self.guard = guard.copy()

    def __deinit__(deinit self):
        var class_bytes = self.class_bytes
        var key = self.key
        var pooled = self.pooled
        # Hold the guard through the release: fields unreferenced in a
        # deinit body die immediately, which would run the guard's purge
        # before this block re-enters the pool.
        var guard = self.guard^
        try:
            _release(self.bytes^, class_bytes, key, pooled)
        except:
            pass
        _ = guard^


def take_storage(
    ctx: DeviceContext,
    nbytes: Int,
    guard: Optional[ArcPointer[DeviceMemGuard]] = None,
) raises -> Storage:
    """Returns storage of at least `nbytes`, reusing a cached block when one
    of matching class exists for this context."""
    var st = _STATE.get_or_create_ptr()
    if not st[].enabled:
        var raw = ctx.enqueue_create_buffer[DType.uint8](nbytes)
        return Storage(raw^, nbytes, 0, False, guard.copy())

    var cls = round_class(nbytes)
    var key = _ctx_key(ctx)
    var i = st[]._ensure(ctx, key)

    var cached = Optional[DeviceBuffer[DType.uint8]](None)
    if cls in st[].pools[i].free:
        ref lst = st[].pools[i].free[cls]
        if len(lst) > 0:
            cached = lst.pop()

    var bytes: DeviceBuffer[DType.uint8]
    if cached:
        bytes = cached.take()
        st[].pools[i].stats.hit_count += 1
    else:
        bytes = _driver_alloc(ctx, cls)
        # _driver_alloc may have emptied the cache; the entry itself survives.
        st[].pools[i].stats.miss_count += 1
        st[].pools[i].stats.alloc_count += 1
        st[].pools[i].stats.reserved_bytes += cls

    ref stats = st[].pools[i].stats
    stats.allocated_bytes += cls
    if stats.allocated_bytes > stats.peak_allocated:
        stats.peak_allocated = stats.allocated_bytes
    if stats.reserved_bytes > stats.peak_reserved:
        stats.peak_reserved = stats.reserved_bytes
    return Storage(bytes^, cls, key, True, guard.copy())


def _driver_alloc(ctx: DeviceContext, cls: Int) raises -> DeviceBuffer[DType.uint8]:
    try:
        return ctx.enqueue_create_buffer[DType.uint8](cls)
    except:
        # Out of memory: release every cached block, wait for the enqueued
        # releases to actually land in the driver, then retry once.
        empty_cache(ctx)
        ctx.synchronize()
        try:
            return ctx.enqueue_create_buffer[DType.uint8](cls)
        except:
            raise Error("mograd: out of device memory allocating " + _fmt_bytes(cls) + "\n" + memory_summary(ctx))


def _release(var bytes: DeviceBuffer[DType.uint8], class_bytes: Int, key: Int, pooled: Bool) raises:
    if not pooled:
        return
    var st = _STATE.get_or_create_ptr()
    var i = st[]._index_of(key)
    if i < 0:
        # Context already purged: plain enqueued release.
        return
    ref pool = st[].pools[i]
    if class_bytes not in pool.free:
        pool.free[class_bytes] = List[DeviceBuffer[DType.uint8]]()
    pool.free[class_bytes].append(bytes^)
    pool.stats.allocated_bytes -= class_bytes
    pool.stats.free_count += 1


# ===-------------------------------------------------------------------===#
# Scratch
# ===-------------------------------------------------------------------===#


struct ScratchBuf[dtype: DType](Movable):
    """A pool-backed typed device buffer for kernel temporaries and layout
    uploads; returns its storage to the pool when destroyed."""

    var view: DeviceBuffer[Self.dtype]
    var _storage: Storage

    def __init__(out self, var view: DeviceBuffer[Self.dtype], var storage: Storage):
        self.view = view^
        self._storage = storage^

    def unsafe_ptr(mut self) -> Pointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.view.unsafe_ptr().as_unsafe_any_origin()

    def enqueue_fill(self, value: Scalar[Self.dtype]) raises:
        self.view.enqueue_fill(value)

    def __len__(self) -> Int:
        return len(self.view)


def scratch_take[dtype: DType](ctx: DeviceContext, numel: Int) raises -> ScratchBuf[dtype]:
    """Returns a temporary device buffer of `numel` elements, reusing a pooled
    allocation when one of matching class exists for this context."""
    var storage = take_storage(ctx, numel * size_of[Scalar[dtype]]())
    var view = storage.bytes.create_sub_buffer[dtype](0, numel)
    return ScratchBuf[dtype](view^, storage^)


# ===-------------------------------------------------------------------===#
# Maintenance
# ===-------------------------------------------------------------------===#


def empty_cache(ctx: DeviceContext) raises:
    """Releases all cached (unallocated) blocks for `ctx` back to the runtime
    allocator. The context's pool entry itself survives."""
    var st = _STATE.get_or_create_ptr()
    var i = st[]._index_of(_ctx_key(ctx))
    if i < 0:
        return
    ref pool = st[].pools[i]
    pool.free = {}
    pool.stats.reserved_bytes = pool.stats.allocated_bytes


def purge(ctx: DeviceContext) raises:
    """Removes the pool entry (keeper reference included) for `ctx`. The
    caller synchronizes the context afterwards so the enqueued releases drain
    before the context handle drops."""
    var st = _STATE.get_or_create_ptr()
    var i = st[]._index_of(_ctx_key(ctx))
    if i < 0:
        return
    _ = st[].keys.pop(i)
    _ = st[].pools.pop(i)


def memory_stats(ctx: DeviceContext) raises -> PoolStats:
    var st = _STATE.get_or_create_ptr()
    var i = st[]._index_of(_ctx_key(ctx))
    if i < 0:
        return PoolStats()
    return st[].pools[i].stats


def memory_summary(ctx: DeviceContext) raises -> String:
    var stats = memory_stats(ctx)
    var s = String("mograd memory pool:\n")
    s += "  allocated: " + _fmt_bytes(stats.allocated_bytes)
    s += "  (peak " + _fmt_bytes(stats.peak_allocated) + ")\n"
    s += "  reserved:  " + _fmt_bytes(stats.reserved_bytes)
    s += "  (peak " + _fmt_bytes(stats.peak_reserved) + ")\n"
    s += "  counters:  " + String(stats.alloc_count) + " driver allocs | "
    s += String(stats.hit_count) + " hits | "
    s += String(stats.miss_count) + " misses | "
    s += String(stats.free_count) + " frees\n"

    var st = _STATE.get_or_create_ptr()
    var i = st[]._index_of(_ctx_key(ctx))
    if i < 0:
        return s
    var classes = 0
    var blocks = 0
    var census = String("")
    for entry in st[].pools[i].free.items():
        if len(entry.value) == 0:
            continue
        classes += 1
        blocks += len(entry.value)
        census += "    " + _fmt_bytes(entry.key) + " x " + String(len(entry.value)) + "\n"
    s += "  free lists (" + String(classes) + " classes, " + String(blocks) + " blocks):\n"
    s += census
    return s


def _fmt_bytes(n: Int) -> String:
    if n < 1024:
        return String(n) + " B"
    var unit = String("KiB")
    var div = 1 << 10
    if n >= (1 << 30):
        unit = "GiB"
        div = 1 << 30
    elif n >= (1 << 20):
        unit = "MiB"
        div = 1 << 20
    var tenths = (n * 10) // div
    return String(tenths // 10) + "." + String(tenths % 10) + " " + unit
