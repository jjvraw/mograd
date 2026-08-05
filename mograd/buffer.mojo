from std.algorithm import vectorize
from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.sys import simd_width_of, size_of
from std.utils import Variant

from mograd.layout import Layout
from mograd import Device


# ===-------------------------------------------------------------------===#
# BufferArm
# ===-------------------------------------------------------------------===#


trait BufferArm(Copyable, Movable):
    comptime node_dtype: DType

    def raw_ptr(ref self, with_offset: Bool = True) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
        ...


# ===-------------------------------------------------------------------===#
# Buffer
# ===-------------------------------------------------------------------===#


struct Buffer[dtype: DType](BufferArm, Copyable):
    comptime node_dtype = Self.dtype
    var _ptr: ArcPointer[DeviceBuffer[Self.dtype]]
    var base_offset: Int

    def __init__(
        out self,
        var buf: DeviceBuffer[Self.dtype],
        size: Int,
    ):
        self._ptr = ArcPointer(buf^)
        self.base_offset = 0

    def __init__(
        out self,
        ptr: ArcPointer[DeviceBuffer[Self.dtype]],
        size: Int,
        base_offset: Int,
    ):
        self._ptr = ptr
        self.base_offset = base_offset

    def buf(ref self) -> ref[self._ptr[]] DeviceBuffer[Self.dtype]:
        return self._ptr[]

    def data_ptr(ref self, with_offset: Bool = True) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        offset = self.base_offset if with_offset else 0
        return (self.buf().unsafe_ptr() + offset).as_unsafe_any_origin()

    def raw_ptr(ref self, with_offset: Bool = True) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
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
        var dev_buf = device.ctx.enqueue_create_buffer[Self.dtype](numel)
        return Self(dev_buf^, numel)

    @staticmethod
    def full(device: Device, value: Scalar[Self.dtype], numel: Int) raises -> Self:
        var dev_buf = device.ctx.enqueue_create_buffer[Self.dtype](numel)
        dev_buf.enqueue_fill(value)
        return Self(dev_buf^, numel)

    @staticmethod
    def from_data[S: DType = Self.dtype, /](device: Device, data: List[Scalar[S]]) raises -> Self:
        var size = len(data)
        var host_buf = device.ctx.enqueue_create_host_buffer[Self.dtype](size)
        var host_ptr = host_buf.unsafe_ptr()
        var src = data.unsafe_ptr()

        def fill[width: Int](i: Int) {read src, read host_ptr}:
            host_ptr.store(i, src.load[width=width](i).cast[Self.dtype]())

        vectorize[simd_width_of[Self.dtype]()](size, fill)
        var dev_buf = device.ctx.enqueue_create_buffer[Self.dtype](size)
        device.ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return Self(dev_buf^, size)


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
                return trait_downcast[BufferArm](self._buf.unsafe_get[T]()).raw_ptr(with_offset)
        raise Error("Unsupported dtype")

    def view(ref self, layout: Layout) raises -> AnyBuffer:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            comptime d = T.node_dtype

            if self._buf.isa[T]():
                ref src = self.unsafe_get[d]()
                return AnyBuffer(Buffer[d](src._ptr.copy(), layout.numel(), layout.base_offset))
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
                    var dev_buf = device.ctx.enqueue_create_buffer[d](numel)
                    return AnyBuffer(Buffer[d](dev_buf^, numel))
        raise Error("Unsupported dtype: " + String(dtype))
