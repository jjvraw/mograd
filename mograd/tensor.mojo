from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from mograd import Device
from mograd.op import AttrVal, Op, OpRef, OpType
from mograd.shape import Shape
from mograd.buffer import Buffer
from mograd.runtime import NativeRuntime
from mograd.grad import Grad

# ===-------------------------------------------------------------------===#
# Tensor
# ===-------------------------------------------------------------------===#


struct Tensor[dtype: DType = DType.float32](Copyable, ImplicitlyCopyable, Movable, Writable):
    var op: OpRef
    var requires_grad: Bool
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    var _grad: ArcPointer[Optional[Tensor[Self.dtype]]]
    var ctx: Optional[Device]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(
        out self,
        ctx: Optional[Device],
        var op: OpRef,
        requires_grad: Bool = False,
    ):
        self.ctx = ctx
        self.op = op^
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor[Self.dtype]](None))

    def __init__(
        out self,
        ctx: Device,
        data: List[Scalar[Self.dtype]],
        shape: Shape,
        requires_grad: Bool = False,
    ) raises:
        var b = Buffer.from_data(ctx, data, shape)
        self.ctx = ctx
        self.op = OpRef(Op(OpType.BUFFER, shape, self.dtype, [], b^))
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor[self.dtype]](None))

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def empty(
        ctx: Device,
        shape: Shape,
        requires_grad: Bool = False,
    ) raises -> Self:
        var b = Buffer[Self.dtype].empty(ctx, shape)
        return Tensor[Self.dtype](ctx, OpRef(Op(OpType.BUFFER, shape, Self.dtype, [], b^)), requires_grad)

    @staticmethod
    def ones(
        ctx: Device,
        shape: Shape,
        requires_grad: Bool = False,
    ) raises -> Self:
        var b = Buffer[Self.dtype].ones(ctx, shape)
        return Tensor[Self.dtype](ctx, OpRef(Op(OpType.BUFFER, shape, Self.dtype, [], b^)), requires_grad)

    @staticmethod
    def uniform(
        ctx: Device,
        shape: Shape,
        low: Float32 = 0.0,
        high: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Self where Self.dtype.is_floating_point():
        attrs: Dict[String, AttrVal] = {"low": low, "high": high, "seed": Float32(seed)}
        return Tensor[Self.dtype](ctx, OpRef(Op(OpType.UNIFORM, shape, Self.dtype, [], attrs=attrs^)), requires_grad)

    @staticmethod
    def randn(
        ctx: Device,
        shape: Shape,
        mean: Float32 = 0.0,
        std: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Self where Self.dtype.is_floating_point():
        attrs: Dict[String, AttrVal] = {"mean": mean, "std": std, "seed": Float32(seed)}
        return Tensor[Self.dtype](ctx, OpRef(Op(OpType.RANDN, shape, Self.dtype, [], attrs^)), requires_grad)

    @staticmethod
    def disk(
        ctx: Device,
        path: String,
        shape: Shape,
    ) -> Self:
        return Tensor[Self.dtype](ctx, OpRef(Op(OpType.DISK, shape, Self.dtype, [], {"path": path})))

    @staticmethod
    def full(
        ctx: Device,
        shape: Shape,
        fill_value: Float32,
        requires_grad: Bool = False,
    ) -> Self:
        return Tensor[Self.dtype](
            ctx, OpRef(Op(OpType.FULL, shape, Self.dtype, [], {"value": fill_value})), requires_grad
        )

    @staticmethod
    def ones_like[
        other_dtype: DType
    ](other: Tensor[other_dtype], requires_grad: Bool = False) raises -> Tensor[other_dtype]:
        return Tensor[other_dtype].ones(other.ctx.value(), other.op.shape(), requires_grad)

    @staticmethod
    def from_buffer(ctx: Device, var buf: Buffer[Self.dtype]) -> Self:
        var shape = buf.shape
        return Tensor[Self.dtype](ctx, OpRef(Op(OpType.BUFFER, shape, Self.dtype, [], buf^)))

    # ===-------------------------------------------------------------------===#
    # Layout transformative operations
    # ===-------------------------------------------------------------------===#

    def shape(self) -> Shape:
        return self.op.shape()

    def shape(self, idx: Int) -> Int:
        return self.op.shape(idx)

    def reshape(self, shape: Shape) -> Self:
        return Self(self.ctx, self.op.reshape(shape), self.requires_grad)

    def transpose(self) -> Self:
        return Self(self.ctx, self.op.transpose(), self.requires_grad)

    def slice(self, start: Int, end: Int) -> Self:
        return Self(self.ctx, self.op.slice(start, end), self.requires_grad)

    def __getitem__(self, s: Slice) -> Self:
        var start = s.start.value() if s.start else 0
        var end = s.end.value() if s.end else self.op.shape(0)
        return self.slice(start, end)

    def cast[out_dtype: DType](self) -> Tensor[out_dtype]:
        return Tensor[out_dtype](self.ctx, self.op.cast(out_dtype), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Pointwise operations
    # ===-------------------------------------------------------------------===#

    def __add__(self, other: Self) -> Self:
        return self.add(other)

    def __add__(self, scalar: Float32) -> Self:
        return self.add(scalar)

    def __radd__(self, scalar: Float32) -> Self:
        return self.add(scalar)

    def add(self, other: Self) -> Self:
        return Self(self.ctx, self.op + other.op, self.requires_grad or other.requires_grad)

    def add(self, scalar: Float32) -> Self:
        var attrs: Dict[String, AttrVal] = {"value": AttrVal(scalar)}
        return Self(
            self.ctx,
            self.op + OpRef(Op(OpType.FULL, self.op.shape(), self.dtype, [], attrs=attrs^)),
            self.requires_grad,
        )

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __neg__(self) -> Self:
        return Self(self.ctx, -self.op, self.requires_grad)

    def neg(self) -> Self:
        return Self(self.ctx, self.op.neg(), self.requires_grad)

    def __mul__(self, other: Self) -> Self:
        return self.mul(other)

    def mul(self, other: Self) -> Self:
        return Self(self.ctx, self.op * other.op, self.requires_grad or other.requires_grad)

    def __mul__(self, scalar: Scalar[Self.dtype]) -> Self:
        return self.scale(scalar)

    def __rmul__(self, scalar: Scalar[Self.dtype]) -> Self:
        return self.scale(scalar)

    def scale(self, scalar: Scalar[Self.dtype]) -> Self:
        return Self(self.ctx, self.op.scale(scalar), self.requires_grad)

    def __truediv__(self, other: Self) -> Self:
        return Self(self.ctx, self.op / other.op, self.requires_grad or other.requires_grad)

    def exp(self) -> Self:
        return Self(self.ctx, self.op.exp(), self.requires_grad)

    def relu(self) -> Self:
        return Self(self.ctx, self.op.relu(), self.requires_grad)

    def log(self) -> Self:
        return Self(self.ctx, self.op.log(), self.requires_grad)

    def softmax(self) -> Self:
        return Self(self.ctx, self.op.softmax(), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self) -> Self:
        return Self(self.ctx, self.op.sum(), self.requires_grad)

    def mean(self) -> Self:
        return self.sum() * (1.0 / Scalar[Self.dtype](self.op.shape().numel()))

    def argmax(self) -> Self:
        return Self(self.ctx, self.op.argmax(), self.requires_grad)

    def one_hot[
        out_dtype: DType = DType.int64
    ](self, num_classes: Int) -> Tensor[out_dtype] where out_dtype.is_integral():
        return Tensor[out_dtype](self.ctx, self.op.one_hot(num_classes, out_dtype), False)

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def __matmul__(self, other: Self) -> Self:
        return self.matmul(other)

    def matmul(self, other: Self) -> Self:
        return Self(self.ctx, self.op.matmul(other.op), self.requires_grad or other.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Comparison operations
    # ===-------------------------------------------------------------------===#

    def eq(self, other: Self) -> Self:
        return Self(self.ctx, self.op.eq(other.op), False)

    def __eq__(self, other: Self) -> Self:
        return self.eq(other)

    # ===-------------------------------------------------------------------===#
    # Loss
    # ===-------------------------------------------------------------------===#

    def cross_entropy(self, labels: Self) -> Self where Self.dtype.is_floating_point():
        return Self(self.ctx, self.op.cross_entropy(labels.op), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Autograd
    # ===-------------------------------------------------------------------===#

    def gradient(
        mut self,
        targets: List[Tensor[Self.dtype]],
        var gradient: Optional[Tensor[Self.dtype]] = None,
    ) raises -> List[Tensor[Self.dtype]] where Self.dtype.is_floating_point():
        var initial_grad_op: OpRef
        if gradient:
            initial_grad_op = gradient.take().op
        else:
            initial_grad_op = Tensor[Self.dtype].ones_like(self).op

        var target_ops = List[OpRef]()
        for t in targets:
            target_ops.append(t.op)

        var grads = Grad[Self.dtype].compute(self.op, initial_grad_op, target_ops)

        var result = List[Tensor[Self.dtype]]()
        for i in range(len(grads)):
            if grads[i]:
                result.append(Tensor[Self.dtype](self.ctx, grads[i].value()))
            else:
                if not self.ctx:
                    raise Error("gradient requires a device context")
                result.append(Tensor[Self.dtype].empty(self.ctx.value(), targets[i].op.shape()))
        return result^

    # ===-------------------------------------------------------------------===#
    # Device I/O
    # ===-------------------------------------------------------------------===#

    def value(self) raises -> Buffer[Self.dtype]:
        var result = NativeRuntime.run(self.op, self.ctx)
        return result.unsafe_get[Self.dtype]().copy()

    def item(self) raises -> Scalar[Self.dtype]:
        var buf = self.value()
        var result: Scalar[Self.dtype]
        with buf.buf().map_to_host() as host:
            result = (host.unsafe_ptr() + buf.base_offset)[0]
        return result

    def to_list(self) raises -> List[Scalar[Self.dtype]]:
        var result = List[Scalar[Self.dtype]]()
        var buf = self.value()
        with buf.buf().map_to_host() as host:
            var ptr = host.unsafe_ptr() + buf.base_offset
            for i in range(buf.size):
                result.append(ptr[i])
        return result^

    # ===-------------------------------------------------------------------===#
    # Display
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        self.op.write_to(writer)
