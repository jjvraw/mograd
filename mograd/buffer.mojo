from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer
from std.utils import Variant

from mograd.shape import Shape


# ===-------------------------------------------------------------------===#
# BufferArm
# ===-------------------------------------------------------------------===#


trait BufferArm(Copyable, Movable, Writable):
    comptime node_dtype: DType

    def get_shape(ref self) -> Shape:
        ...

    def get_size(ref self) -> Int:
        ...


# ===-------------------------------------------------------------------===#
# Buffer
# ===-------------------------------------------------------------------===#


struct Buffer[dtype: DType](BufferArm, Copyable, Movable, Writable):
    comptime node_dtype = Self.dtype
    var _ptr: ArcPointer[DeviceBuffer[Self.dtype]]
    var shape: Shape
    var strides: Shape
    var base_offset: Int
    var size: Int

    def __init__(
        out self,
        var buf: DeviceBuffer[Self.dtype],
        shape: Shape,
        size: Int,
    ):
        self._ptr = ArcPointer(buf^)
        self.shape = shape
        self.strides = shape.strides()
        self.base_offset = 0
        self.size = size

    def __init__(
        out self,
        ptr: ArcPointer[DeviceBuffer[Self.dtype]],
        shape: Shape,
        strides: Shape,
        base_offset: Int,
    ):
        self._ptr = ptr
        self.shape = shape
        self.strides = strides
        self.base_offset = base_offset
        self.size = shape.numel()

    def buf(ref self) -> ref[self._ptr[]] DeviceBuffer[Self.dtype]:
        return self._ptr[]

    def data_ptr(ref self) -> UnsafePointer[Scalar[Self.dtype], MutAnyOrigin]:
        return self.buf().unsafe_ptr() + self.base_offset

    def get_shape(ref self) -> Shape:
        return self.shape

    def get_size(ref self) -> Int:
        return self.size

    def broadcast_to(self, new_shape: Shape) -> Self:
        var old_rank = len(self.shape)
        var new_rank = len(new_shape)
        var rank_diff = new_rank - old_rank
        var new_strides_list = List[Int]()
        for i in range(new_rank):
            if i < rank_diff:
                new_strides_list.append(0)
            else:
                var old_i = i - rank_diff
                if self.shape[old_i] == 1 and new_shape[i] > 1:
                    new_strides_list.append(0)
                else:
                    new_strides_list.append(self.strides[old_i])
        return Self(self._ptr, new_shape, Shape(new_strides_list), self.base_offset)

    @staticmethod
    def empty(ctx: DeviceContext, shape: Shape) raises -> Self:
        var size = shape.numel()
        var dev_buf = ctx.enqueue_create_buffer[Self.dtype](size)
        return Self(dev_buf^, shape, size)

    @staticmethod
    def ones(ctx: DeviceContext, shape: Shape) raises -> Self:
        var size = shape.numel()
        var dev_buf = ctx.enqueue_create_buffer[Self.dtype](size)
        dev_buf.enqueue_fill(1.0)
        return Self(dev_buf^, shape, size)

    @staticmethod
    def from_data(
        ctx: DeviceContext,
        data: List[Scalar[Self.dtype]],
        shape: Shape,
    ) raises -> Self:
        var size = len(data)
        var host_buf = ctx.enqueue_create_host_buffer[Self.dtype](size)
        var host_ptr = host_buf.unsafe_ptr()
        for i in range(size):
            host_ptr[i] = data[i]
        var dev_buf = ctx.enqueue_create_buffer[Self.dtype](size)
        ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return Self(dev_buf^, shape, size)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Buffer(shape=")
        self.shape.write_to(writer)
        writer.write(", size=", String(self.size), ")")


# ===-------------------------------------------------------------------===#
# AnyBuffer
# ===-------------------------------------------------------------------===#


struct AnyBuffer(Copyable, Movable, Writable):
    comptime BufVariant = Variant[Buffer[DType.float32], Buffer[DType.int64]]
    var _buf: Self.BufVariant

    @implicit
    def __init__[dtype: DType](out self, var buf: Buffer[dtype]):
        self._buf = Self.BufVariant(buf^)

    def __init__(out self, *, copy: Self):
        self._buf = copy._buf.copy()

    def isa[dtype: DType](self) -> Bool:
        return self._buf.isa[Buffer[dtype]]()

    def unsafe_get[dtype: DType](ref self) -> ref[self._buf] Buffer[dtype]:
        return self._buf.unsafe_get[Buffer[dtype]]()

    def dtype(self) -> DType:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            if self._buf.isa[T]():
                return T.node_dtype
        return DType.invalid

    def shape(ref self) -> Shape:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            if self._buf.isa[T]():
                return trait_downcast[BufferArm](self._buf.unsafe_get[T]()).get_shape()
        return Shape()

    def size(ref self) -> Int:
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, BufferArm)
            if self._buf.isa[T]():
                return trait_downcast[BufferArm](self._buf.unsafe_get[T]()).get_size()
        return 0

    def write_to(self, mut writer: Some[Writer]):
        comptime for k in range(Self.BufVariant.Ts.size):
            comptime T = Self.BufVariant.Ts[k]
            comptime assert conforms_to(T, Writable)
            if self._buf.isa[T]():
                trait_downcast[Writable](self._buf.unsafe_get[T]()).write_to(writer)
                return
