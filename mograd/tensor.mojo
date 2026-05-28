from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from mograd.op import AttrVal, Op, OpRef, OpType
from mograd.buffer import Buffer
from mograd.runtime.native import NativeRuntime
from mograd.grad import Grad

# ===-------------------------------------------------------------------===#
# Tensor
# ===-------------------------------------------------------------------===#


struct Tensor(Copyable, ImplicitlyCopyable, Movable, Writable):
    var ctx: Optional[DeviceContext]
    var op: OpRef
    var requires_grad: Bool
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    var _grad: ArcPointer[Optional[Tensor]]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(
        out self,
        ctx: Optional[DeviceContext],
        var op: OpRef,
        requires_grad: Bool = False,
    ):
        self.ctx = ctx
        self.op = op^
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor](None))

    def __init__(
        out self,
        ctx: DeviceContext,
        data: List[Float32],
        shape: List[Int],
        requires_grad: Bool = False,
    ) raises:
        var b = Buffer.from_data(ctx, data, shape.copy())
        self.ctx = ctx
        self.op = OpRef(Op(OpType.BUFFER, shape.copy(), DType.float32, [], b^))
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor](None))

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def empty(
        ctx: DeviceContext,
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) raises -> Tensor:
        var b = Buffer.empty(ctx, shape.copy())
        return Tensor(ctx, OpRef(Op(OpType.BUFFER, shape.copy(), dtype, [], b^)), requires_grad)

    @staticmethod
    def ones(
        ctx: DeviceContext,
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) raises -> Tensor:
        var b = Buffer.ones(ctx, shape.copy())
        return Tensor(ctx, OpRef(Op(OpType.BUFFER, shape.copy(), dtype, [], b^)), requires_grad)

    @staticmethod
    def uniform(
        ctx: DeviceContext,
        shape: List[Int],
        low: Float32 = 0.0,
        high: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Tensor:
        return Tensor(
            ctx,
            OpRef(
                Op(
                    OpType.UNIFORM,
                    shape.copy(),
                    DType.float32,
                    [],
                    {"low": low, "high": high, "seed": Float32(seed)},
                )
            ),
            requires_grad,
        )

    @staticmethod
    def randn(
        ctx: DeviceContext,
        shape: List[Int],
        mean: Float32 = 0.0,
        std: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Tensor:
        var srcs: List[OpRef] = []
        var attrs: Dict[String, AttrVal] = {
            "mean": AttrVal(mean),
            "std": AttrVal(std),
            "seed": AttrVal(Float32(seed)),
        }
        return Tensor(
            Optional[DeviceContext](ctx),
            OpRef(Op(OpType.RANDN, shape.copy(), DType.float32, srcs^, attrs^)),
            requires_grad,
        )

    @staticmethod
    def disk(
        ctx: DeviceContext,
        path: String,
        var shape: List[Int],
        dtype: DType = DType.float32,
    ) -> Tensor:
        return Tensor(ctx, OpRef(Op(OpType.DISK, shape^, dtype, [], {"path": path})))

    @staticmethod
    def full(
        ctx: DeviceContext,
        shape: List[Int],
        fill_value: Float32,
        requires_grad: Bool = False,
    ) -> Tensor:
        return Tensor(
            ctx, OpRef(Op(OpType.FULL, shape.copy(), DType.float32, [], {"value": fill_value})), requires_grad
        )

    @staticmethod
    def ones_like(other: Tensor, requires_grad: Bool = False) raises -> Tensor:
        return Tensor.ones(other.ctx.value(), other.op.shape().copy(), other.op.dtype(), requires_grad)

    @staticmethod
    def from_buffer(ctx: DeviceContext, var buf: Buffer) -> Tensor:
        return Tensor(ctx, OpRef(Op(OpType.BUFFER, buf.shape.copy(), DType.float32, [], buf^)))

    # ===-------------------------------------------------------------------===#
    # Layout transformative operations
    # ===-------------------------------------------------------------------===#

    def shape(self) -> List[Int]:
        return self.op.shape().copy()

    def reshape(self, var shape: List[Int]) -> Self:
        return Tensor(self.ctx, self.op.reshape(shape^), self.requires_grad)

    def transpose(self) -> Self:
        return Tensor(self.ctx, self.op.transpose(), self.requires_grad)

    def slice(self, start: Int, end: Int) -> Self:
        return Tensor(self.ctx, self.op.slice(start, end), self.requires_grad)

    def __getitem__(self, s: Slice) -> Self:
        var start = s.start.value() if s.start else 0
        var end = s.end.value() if s.end else self.op.shape()[0]
        return self.slice(start, end)

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
        return Tensor(self.ctx, self.op + other.op, self.requires_grad or other.requires_grad)

    def add(self, scalar: Float32) -> Self:
        var scalar_op = OpRef(Op(OpType.FULL, self.op.shape().copy(), DType.float32, [], {"value": scalar}))
        return Tensor(self.ctx, self.op + scalar_op, self.requires_grad)

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __neg__(self) -> Self:
        return Tensor(self.ctx, -self.op, self.requires_grad)

    def neg(self) -> Self:
        return Tensor(self.ctx, self.op.neg(), self.requires_grad)

    def __mul__(self, other: Self) -> Self:
        return self.mul(other)

    def mul(self, other: Self) -> Self:
        return Tensor(self.ctx, self.op * other.op, self.requires_grad or other.requires_grad)

    def __mul__(self, scalar: Float32) -> Self:
        return self.scale(scalar)

    def __rmul__(self, scalar: Float32) -> Self:
        return self.scale(scalar)

    def scale(self, scalar: Float32) -> Self:
        return Tensor(self.ctx, self.op.scale(scalar), self.requires_grad)

    def __truediv__(self, other: Self) -> Self:
        return Tensor(self.ctx, self.op / other.op, self.requires_grad or other.requires_grad)

    def exp(self) -> Self:
        return Tensor(self.ctx, self.op.exp(), self.requires_grad)

    def relu(self) -> Self:
        return Tensor(self.ctx, self.op.relu(), self.requires_grad)

    def log(self) -> Self:
        return Tensor(self.ctx, self.op.log(), self.requires_grad)

    def softmax(self) -> Self:
        return Tensor(self.ctx, self.op.softmax(), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self) -> Self:
        return Tensor(self.ctx, self.op.sum(), self.requires_grad)

    def mean(self) -> Self:
        var n = 1
        for d in self.op.shape():
            n *= d
        return self.sum() * (1.0 / Float32(n))

    def argmax(self) -> Self:
        return Tensor(self.ctx, self.op.argmax(), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def __matmul__(self, other: Self) -> Self:
        return self.matmul(other)

    def matmul(self, other: Self) -> Self:
        return Tensor(self.ctx, self.op.matmul(other.op), self.requires_grad or other.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Comparison operations
    # ===-------------------------------------------------------------------===#

    def eq(self, other: Self) -> Self:
        return Tensor(self.ctx, self.op.eq(other.op), False)

    def __eq__(self, other: Self) -> Self:
        return self.eq(other)

    # ===-------------------------------------------------------------------===#
    # Loss
    # ===-------------------------------------------------------------------===#

    def cross_entropy(self, labels: Self) -> Self:
        return Tensor(self.ctx, self.op.cross_entropy(labels.op), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Autograd
    # ===-------------------------------------------------------------------===#

    def gradient(mut self, *targets: Tensor, var gradient: Optional[Tensor] = None) raises -> List[Tensor]:
        var initial_grad: OpRef
        if gradient:
            initial_grad = gradient.take().op
        else:
            initial_grad = Self.ones_like(self).op
        target_ops: List[OpRef] = [t.op for t in targets]

        var grads = Grad.compute(self.op, initial_grad, target_ops)

        var result = List[Tensor]()
        for i in range(len(grads)):
            if grads[i]:
                result.append(Tensor(self.ctx, grads[i].value()))
            else:
                if not self.ctx:
                    raise Error("gradient requires a device context")
                result.append(
                    Self.empty(
                        self.ctx.value(),
                        targets[i].op.shape().copy(),
                        targets[i].op.dtype(),
                    )
                )
        return result^

    def gradient(
        mut self, targets: List[Tensor], var gradient: Optional[Tensor] = None
    ) raises -> List[Tensor]:
        var initial_grad: OpRef
        if gradient:
            initial_grad = gradient.take().op
        else:
            initial_grad = Self.ones_like(self).op
        var target_ops: List[OpRef] = [t.op for t in targets]

        var grads = Grad.compute(self.op, initial_grad, target_ops)

        var result = List[Tensor]()
        for i in range(len(grads)):
            if grads[i]:
                result.append(Tensor(self.ctx, grads[i].value()))
            else:
                if not self.ctx:
                    raise Error("gradient requires a device context")
                result.append(
                    Self.empty(
                        self.ctx.value(),
                        targets[i].op.shape().copy(),
                        targets[i].op.dtype(),
                    )
                )
        return result^

    # ===-------------------------------------------------------------------===#
    # Device I/O
    # ===-------------------------------------------------------------------===#

    def value(self) raises -> Buffer:
        return NativeRuntime.run(self.op, self.ctx)

    def item(self) raises -> Float32:
        var buf = self.value()
        var result = Float32(0)
        with buf.buf().map_to_host() as host:
            result = host.unsafe_ptr()[0]
        return result

    def to_list(self) raises -> List[Float32]:
        var result = List[Float32]()
        var buf = self.value()
        with buf.buf().map_to_host() as host:
            for i in range(buf.size):
                result.append(host.unsafe_ptr()[i])
        return result^

    # ===-------------------------------------------------------------------===#
    # Display
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def _str_op(op: OpRef) -> String:
        var op_name = op.op().__str__()

        if len(op.srcs()) == 0:
            return op_name

        var result = op_name + "("
        for i in range(len(op.srcs())):
            result = result + Tensor._str_op(op.srcs()[i])
            if i < len(op.srcs()) - 1:
                result = result + ", "
        result = result + ")"
        return result

    def write_to(self, mut writer: Some[Writer]):
        self.op.write_to(writer)
