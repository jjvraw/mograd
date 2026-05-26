from std.memory import ArcPointer

from mograd.op import Op, OpRef, OpType
from mograd.grad import Grad

# ===-------------------------------------------------------------------===#
# Tensor
# ===-------------------------------------------------------------------===#


# TODO: Make factory methods for tensor constructors.
struct Tensor(Copyable, Movable, Writable):
    var op: OpRef
    var requires_grad: Bool
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    var _grad: ArcPointer[Optional[Tensor]]

    def __init__(out self, var op: OpRef, requires_grad: Bool = False):
        self.op = op^
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor](None))

    @implicit
    def __init__(out self, var op: Op, requires_grad: Bool = False):
        self.op = OpRef(op^)
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor](None))

    @staticmethod
    def ones_like(other: Tensor, requires_grad: Bool = False) -> Tensor:
        var srcs: List[OpRef] = []
        return Tensor(
            OpRef(
                Op(
                    OpType.ONES,
                    other.op.shape().copy(),
                    other.op.dtype(),
                    srcs^,
                )
            ),
            requires_grad,
        )

    @staticmethod
    def empty(
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) -> Tensor:
        var srcs: List[OpRef] = []
        return Tensor(
            OpRef(Op(OpType.BUFFER, shape.copy(), dtype, srcs^)), requires_grad
        )

    def __add__(self, other: Self) -> Self:
        return self.add(other)

    def __mul__(self, other: Self) -> Self:
        return self.mul(other)

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
                result.append(Tensor(grads[i].value()))
            else:
                result.append(
                    Self.empty(
                        targets[i].op.shape().copy(), targets[i].op.dtype()
                    )
                )
        return result^

    def add(self, other: Self) -> Self:
        return Tensor(
            self.op + other.op,
            self.requires_grad or other.requires_grad,
        )

    def mul(self, other: Self) -> Self:
        return Tensor(
            self.op * other.op,
            self.requires_grad or other.requires_grad,
        )

    # TODO: Clean up
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
