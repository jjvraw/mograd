from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.utils import Variant

from mograd.layout import Layout
from mograd import Device


# ===-------------------------------------------------------------------===#
# BufferArm
# ===-------------------------------------------------------------------===#


trait BufferArm(Copyable, Movable):
    comptime node_dtype: DType

    def raw_ptr(ref self) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
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

    def data_ptr(ref self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.buf().unsafe_ptr() + self.base_offset

    def raw_ptr(ref self) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
        return self.data_ptr().bitcast[NoneType]()

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
    def from_data(
        device: Device,
        data: List[Scalar[Self.dtype]],
    ) raises -> Self:
        var size = len(data)
        var host_buf = device.ctx.enqueue_create_host_buffer[Self.dtype](size)
        var host_ptr = host_buf.unsafe_ptr()
        for i in range(size):
            host_ptr[i] = data[i]
        var dev_buf = device.ctx.enqueue_create_buffer[Self.dtype](size)
        device.ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return Self(dev_buf^, size)


# ===-------------------------------------------------------------------===#
# AnyBuffer
# ===-------------------------------------------------------------------===#


struct AnyBuffer(Copyable, Movable):
    comptime BufVariant = Variant[Buffer[DType.float32], Buffer[DType.int64]]
    var _buf: Self.BufVariant

    @implicit
    def __init__[dtype: DType](out self, var buf: Buffer[dtype]):
        self._buf = Self.BufVariant(buf^)

    def __init__(out self, *, copy: Self):
        self._buf = copy._buf.copy()

    def unsafe_get[dtype: DType](ref self) -> ref[self._buf] Buffer[dtype]:
        return self._buf.unsafe_get[Buffer[dtype]]()

    def data_ptr(ref self) raises -> UnsafePointer[NoneType, MutAnyOrigin]:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            if self._buf.isa[T]():
                return trait_downcast[BufferArm](self._buf.unsafe_get[T]()).raw_ptr()
        raise Error("AnyBuffer.data_ptr: unknown variant")

    def view(ref self, layout: Layout) raises -> AnyBuffer:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            comptime d = T.node_dtype

            if self._buf.isa[T]():
                ref src = self.unsafe_get[d]()
                return AnyBuffer(Buffer[d](src._ptr.copy(), layout.numel(), layout.base_offset))

        raise Error("unsupported dtype")

    @staticmethod
    def empty(dtype: DType, device: Device, numel: Int) raises -> AnyBuffer:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            comptime d = T.node_dtype
            if dtype == d:
                var dev_buf = device.ctx.enqueue_create_buffer[d](numel)
                return AnyBuffer(Buffer[d](dev_buf^, numel))
        raise Error("unsupported dtype: " + String(dtype))
