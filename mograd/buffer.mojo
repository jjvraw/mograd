from std.algorithm import vectorize
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.sys import simd_width_of, size_of
from std.utils import Variant

from mograd.layout import Layout
from mograd.memory import Storage, take_storage
from mograd import Device


# ===-------------------------------------------------------------------===#
# BufferArm
# ===-------------------------------------------------------------------===#


trait BufferArm(Copyable, Movable):
    comptime node_dtype: DType

    def rawptr(ref self, with_offset: Bool = True) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
        ...


# ===-------------------------------------------------------------------===#
# Buffer
# ===-------------------------------------------------------------------===#


struct Buffer[dtype: DType](BufferArm, Copyable):
    comptime node_dtype = Self.dtype
    var ptr: ArcPointer[DeviceBuffer[Self.dtype]]
    # Pool ticket: while any copy of this Buffer (or a view of it) is alive
    # the underlying allocation is held here; when the last one dies the
    # allocation recycles into the memory pool instead of returning to the
    # runtime allocator. None for buffers wrapping external DeviceBuffers.
    var storage: Optional[ArcPointer[Storage]]
    var base_offset: Int

    def __init__(
        out self,
        var buf: DeviceBuffer[Self.dtype],
        size: Int,
    ):
        self.ptr = ArcPointer(buf^)
        self.storage = None
        self.base_offset = 0

    def __init__(
        out self,
        ptr: ArcPointer[DeviceBuffer[Self.dtype]],
        size: Int,
        base_offset: Int,
        storage: Optional[ArcPointer[Storage]] = None,
    ):
        self.ptr = ptr
        self.storage = storage.copy()
        self.base_offset = base_offset

    def buf(ref self) -> ref[self.ptr[]] DeviceBuffer[Self.dtype]:
        return self.ptr[]

    def data_ptr(ref self, with_offset: Bool = True) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        offset = self.base_offset if with_offset else 0
        return (self.buf().unsafe_ptr() + offset).as_unsafe_any_origin()

    def rawptr(ref self, with_offset: Bool = True) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
        return self.data_ptr(with_offset).bitcast[NoneType]()

    def item(self) raises -> Scalar[Self.dtype]:
        with self.buf().map_to_host() as host:
            return (host.unsafe_ptr() + self.base_offset)[0]

    def to_list(self, layout: Layout) raises -> List[Scalar[Self.dtype]]:
        var result = List[Scalar[Self.dtype]]()
        var inner = layout.inner_sizes()
        with self.buf().map_to_host() as host:
            var base = host.unsafe_ptr() + self.base_offset
            for i in range(layout.numel()):
                var off = 0
                var rem = i
                for d in range(layout.rank()):
                    var idx = rem // inner.value(d)
                    rem %= inner.value(d)
                    off += idx * layout._strides.value(d)
                result.append(base[off])
        return result^

    @staticmethod
    def empty(device: Device, numel: Int) raises -> Self:
        if numel == 0:
            var dev_buf = device.ctx.enqueue_create_buffer[Self.dtype](0)
            return Self(dev_buf^, 0)
        # Allocate raw bytes (pool first, runtime allocator on miss) and hand
        # out a typed view over the allocation. Contents are undefined, same
        # as a fresh enqueue_create_buffer.
        var storage = take_storage(device.ctx, numel * size_of[Scalar[Self.dtype]](), guard=device.mem)
        var view = storage.bytes.create_sub_buffer[Self.dtype](0, numel)
        return Self(ArcPointer(view^), numel, 0, storage=ArcPointer(storage^))

    @staticmethod
    def full(device: Device, value: Scalar[Self.dtype], numel: Int) raises -> Self:
        var buf = Self.empty(device, numel)
        buf.buf().enqueue_fill(value)
        return buf^

    @staticmethod
    def from_data[S: DType = Self.dtype, /](device: Device, data: List[Scalar[S]]) raises -> Self:
        var size = len(data)
        var host_buf = device.ctx.enqueue_create_host_buffer[Self.dtype](size)
        var hostptr = host_buf.unsafe_ptr()
        var src = data.unsafe_ptr()

        def fill[width: Int](i: Int) {read src, read hostptr}:
            hostptr.store(i, src.load[width=width](i).cast[Self.dtype]())

        vectorize[simd_width_of[Self.dtype]()](size, fill)
        var buf = Self.empty(device, size)
        device.ctx.enqueue_copy(dst_buf=buf.buf(), src_buf=host_buf)
        # Drain the queue before host_buf dies and before this context can be
        # torn down: destroying a DeviceContext with unsynchronized enqueued
        # work poisons the per-device AsyncRT mutex on the 2026-07 runtime and
        # stalls the next context's first allocation for ~600s (see
        # BUG_REPORT_asyncrt_first_dispatch_stall.md).
        device.ctx.synchronize()
        return buf^


# ===-------------------------------------------------------------------===#
# AnyBuffer
# ===-------------------------------------------------------------------===#


struct AnyBuffer(Copyable, Movable):
    comptime BufVariant = Variant[Buffer[DType.float16], Buffer[DType.float32], Buffer[DType.int64]]
    var _buf: Self.BufVariant

    @implicit
    def __init__[dtype: DType](out self, var buf: Buffer[dtype]):
        self._buf = Self.BufVariant(buf^)

    def __init__(out self, *, copy: Self):
        self._buf = copy._buf.copy()

    def unsafe_get[dtype: DType](ref self) -> ref[self._buf] Buffer[dtype]:
        return self._buf.unsafe_get[Buffer[dtype]]()

    @staticmethod
    def supports(d: DType) -> Bool:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            if d == T.node_dtype:
                return True
        return False

    def dtype(self) raises -> DType:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            comptime d = T.node_dtype
            if self._buf.isa[T]():
                return d
        raise Error("Unsupported dtype")

    def size_bytes(self) raises -> Int:
        """Physical size of the underlying device allocation."""
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            comptime d = T.node_dtype
            if self._buf.isa[T]():
                return len(self.unsafe_get[d]().buf()) * size_of[Scalar[d]]()
        raise Error("Unsupported dtype")

    def item[dtype: DType](self) raises -> Scalar[dtype]:
        return self.unsafe_get[dtype]().item()

    def to_list[dtype: DType](self, layout: Layout) raises -> List[Scalar[dtype]]:
        return self.unsafe_get[dtype]().to_list(layout)

    def data_ptr(ref self, with_offset: Bool = True) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            if self._buf.isa[T]():
                return trait_downcast[BufferArm](self._buf.unsafe_get[T]()).rawptr(with_offset)
        raise Error("Unsupported dtype")

    def view(ref self, layout: Layout) raises -> AnyBuffer:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            comptime d = T.node_dtype

            if self._buf.isa[T]():
                ref src = self.unsafe_get[d]()
                return AnyBuffer(Buffer[d](src.ptr.copy(), layout.numel(), layout.base_offset, src.storage))
        raise Error("Unsupported dtype")

    @staticmethod
    def create(dtype: DType, device: Device, numel: Int, fill: Optional[Float64] = None) raises -> AnyBuffer:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            comptime d = T.node_dtype
            if dtype == d:
                if fill:
                    return AnyBuffer(Buffer[d].full(device, Scalar[d](fill.value()), numel))
                else:
                    return AnyBuffer(Buffer[d].empty(device, numel))
        raise Error("Unsupported dtype: " + String(dtype))
