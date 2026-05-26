from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from mograd.op import Op, OpRef, OpType
from mograd.buffer import Buffer
from mograd.runtime.native import NativeRuntime
from mograd.grad import Grad

# ===-------------------------------------------------------------------===#
# Tensor
# ===-------------------------------------------------------------------===#


struct Tensor(Copyable, Movable, Writable):
    var ctx: Optional[DeviceContext]
    var op: OpRef
    var requires_grad: Bool
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    var _grad: ArcPointer[Optional[Tensor]]

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
        var srcs: List[OpRef] = []
        var buf = Optional[Buffer](b^)
        self.ctx = ctx
        self.op = OpRef(Op(OpType.BUFFER, shape.copy(), DType.float32, srcs^, buf^))
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor](None))

    @staticmethod
    def empty(
        ctx: DeviceContext,
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) raises -> Tensor:
        var b = Buffer.empty(ctx, shape.copy())
        var srcs: List[OpRef] = []
        var buf = Optional[Buffer](b^)
        return Tensor(Optional[DeviceContext](ctx), OpRef(Op(OpType.BUFFER, shape.copy(), dtype, srcs^, buf^)), requires_grad)

    @staticmethod
    def ones(
        ctx: DeviceContext,
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) raises -> Tensor:
        var b = Buffer.ones(ctx, shape.copy())
        var srcs: List[OpRef] = []
        var buf = Optional[Buffer](b^)
        return Tensor(Optional[DeviceContext](ctx), OpRef(Op(OpType.BUFFER, shape.copy(), dtype, srcs^, buf^)), requires_grad)

    @staticmethod
    def ones_like(other: Tensor, requires_grad: Bool = False) raises -> Tensor:
        if not other.ctx:
            raise Error("ones_like requires a device context")
        return Tensor.ones(other.ctx.value(), other.op.shape().copy(), other.op.dtype(), requires_grad)

    def __add__(self, other: Self) -> Self:
        return self.add(other)

    def __mul__(self, other: Self) -> Self:
        return self.mul(other)

    def add(self, other: Self) -> Self:
        return Tensor(
            self.ctx,
            self.op + other.op,
            self.requires_grad or other.requires_grad,
        )

    def mul(self, other: Self) -> Self:
        return Tensor(
            self.ctx,
            self.op * other.op,
            self.requires_grad or other.requires_grad,
        )

    def gradient(
        mut self, *targets: Tensor, var gradient: Optional[Tensor] = None
    ) raises -> List[Tensor]:
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
                    Self.empty(self.ctx.value(), targets[i].op.shape().copy(), targets[i].op.dtype())
                )
        return result^

    def value(self) raises -> Buffer:
        return NativeRuntime.run(self.op, self.ctx)

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
