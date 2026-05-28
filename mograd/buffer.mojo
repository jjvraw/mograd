from std.memory import ArcPointer
from std.gpu.host import DeviceContext, DeviceBuffer

from mograd.shape import Shape


# ===-------------------------------------------------------------------===#
# Buffer
# ===-------------------------------------------------------------------===#


struct Buffer(Copyable, Movable, Writable):
    var _ptr: ArcPointer[DeviceBuffer[DType.float32]]
    var shape: Shape
    var size: Int

    def __init__(
        out self,
        var buf: DeviceBuffer[DType.float32],
        shape: Shape,
        size: Int,
    ):
        self._ptr = ArcPointer(buf^)
        self.shape = shape
        self.size = size

    def buf(ref self) -> ref[self._ptr[]] DeviceBuffer[DType.float32]:
        return self._ptr[]

    @staticmethod
    def empty(ctx: DeviceContext, shape: Shape) raises -> Self:
        var size = shape.numel()
        var dev_buf = ctx.enqueue_create_buffer[DType.float32](size)
        return Self(dev_buf^, shape, size)

    @staticmethod
    def ones(ctx: DeviceContext, shape: Shape) raises -> Self:
        var size = shape.numel()
        var dev_buf = ctx.enqueue_create_buffer[DType.float32](size)
        dev_buf.enqueue_fill(1.0)
        return Self(dev_buf^, shape, size)

    @staticmethod
    def from_data(
        ctx: DeviceContext,
        data: List[Float32],
        shape: Shape,
    ) raises -> Self:
        var size = len(data)
        var host_buf = ctx.enqueue_create_host_buffer[DType.float32](size)
        var host_ptr = host_buf.unsafe_ptr()
        for i in range(size):
            host_ptr[i] = data[i]
        var dev_buf = ctx.enqueue_create_buffer[DType.float32](size)
        ctx.enqueue_copy(dst_buf=dev_buf, src_buf=host_buf)
        return Self(dev_buf^, shape, size)

    def write_to(self, mut writer: Some[Writer]):
        writer.write("Buffer(shape=")
        self.shape.write_to(writer)
        writer.write(", size=", String(self.size), ")")
