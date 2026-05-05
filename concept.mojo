from std.memory import ArcPointer


# struct MulOP:
#     @staticmethod
#     def forward() -> Op:
#         ...


@fieldwise_init
struct OpType(Copyable, Equatable, ImplicitlyCopyable, Movable):
    var _value: Int
    comptime BUFFER = OpType(2)
    comptime ONES = OpType(3)
    comptime ADD = OpType(5)
    comptime MUL = OpType(7)


@fieldwise_init
struct Op(Copyable, Movable):
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    comptime OpPointer = UnsafePointer[Self, MutExternalOrigin]

    var op_type: OpType
    var shape: List[Int]
    var dtype: DType
    var srcs: List[ArcPointer[Op]]
    var grad: Optional[Self.OpPointer]

    def __init__(
        out self,
        op_type: OpType,
        var shape: List[Int],
        dtype: DType,
        var srcs: List[ArcPointer[Op]],
    ):
        self.op_type = op_type
        self.shape = shape^
        self.dtype = dtype
        self.srcs = srcs^
        self.grad = None

    def __del__(deinit self):
        if self.grad:
            self.grad.value().destroy_pointee()
            self.grad.value().free()

    @staticmethod
    def mul(lhs: ArcPointer[Op], rhs: ArcPointer[Op]) -> ArcPointer[Op]:
        var srcs = List[ArcPointer[Op]]()
        srcs.append(lhs)
        srcs.append(rhs)
        return ArcPointer(
            Op(OpType.MUL, lhs[].shape.copy(), lhs[].dtype, srcs^)
        )

    @staticmethod
    def add(lhs: ArcPointer[Op], rhs: ArcPointer[Op]) -> ArcPointer[Op]:
        var srcs = List[ArcPointer[Op]]()
        srcs.append(lhs)
        srcs.append(rhs)
        return ArcPointer(
            Op(OpType.ADD, lhs[].shape.copy(), lhs[].dtype, srcs^)
        )

    def set_grad(mut self, var grad: Op):
        if self.grad:
            self.grad.value().destroy_pointee()
            self.grad.value().free()
        var ptr = alloc[Op](1)
        ptr.init_pointee_move(grad^)
        self.grad = ptr

    def accum_grad(mut self, grad: ArcPointer[Op]):
        if self.grad:
            var srcs = List[ArcPointer[Op]]()
            srcs.append(ArcPointer(self.grad.value()[].copy()))
            srcs.append(grad)
            var accumulated = ArcPointer(
                Op(OpType.ADD, self.shape.copy(), self.dtype, srcs^)
            )
            self.set_grad(accumulated[].copy())
        else:
            self.set_grad(grad[].copy())


struct Tensor:
    var op: ArcPointer[Op]
    var requires_grad: Bool

    def __init__(out self, var op: ArcPointer[Op], requires_grad: Bool = False):
        self.op = op^
        self.requires_grad = requires_grad

    @implicit
    def __init__(out self, var op: Op, requires_grad: Bool = False):
        self.op = ArcPointer(op^)
        self.requires_grad = requires_grad

    @staticmethod
    def empty(
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) -> Tensor:
        s = shape.copy()
        srcs = List[ArcPointer[Op]]()
        op = ArcPointer(Op(OpType.BUFFER, s^, dtype, srcs^))
        return Tensor(op, requires_grad)

    def __add__(self, other: Self) -> Self:
        return Tensor(
            Op.add(self.op, other.op), self.requires_grad or other.requires_grad
        )

    def __mul__(self, other: Self) -> Self:
        return Tensor(
            Op.mul(self.op, other.op), self.requires_grad or other.requires_grad
        )

    def backward(mut self) raises -> None:
        self.op[].set_grad(
            Op(
                OpType.ONES,
                self.op[].shape.copy(),
                self.op[].dtype,
                List[ArcPointer[Op]](),
            )
        )

        var topo = toposort(self)

        for i in reversed(range(len(topo))):
            var node = topo[i]
            if not node[].grad:
                continue
            var upstream = ArcPointer(node[].grad.value()[].copy())

            if node[].op_type == OpType.MUL:
                node[].srcs[0][].accum_grad(Op.mul(node[].srcs[1], upstream))
                node[].srcs[1][].accum_grad(Op.mul(node[].srcs[0], upstream))
            elif node[].op_type == OpType.ADD:
                for j in range(len(node[].srcs)):
                    node[].srcs[j][].accum_grad(upstream)

    def grad(self) -> Optional[ArcPointer[Op]]:
        if self.op[].grad:
            return ArcPointer(self.op[].grad.value()[].copy())
        return None

    @staticmethod
    def _str_op(op: ArcPointer[Op]) -> String:
        var op_name: String
        if op[].op_type == OpType.BUFFER:
            op_name = "BUFFER"
        elif op[].op_type == OpType.ADD:
            op_name = "ADD"
        elif op[].op_type == OpType.MUL:
            op_name = "MUL"
        elif op[].op_type == OpType.ONES:
            op_name = "ONES"
        else:
            op_name = "?"

        if len(op[].srcs) == 0:
            return op_name

        var result = op_name + "("
        for i in range(len(op[].srcs)):
            result = result + Tensor._str_op(op[].srcs[i])
            if i < len(op[].srcs) - 1:
                result = result + ", "
        result = result + ")"
        return result

    def __str__(self) -> String:
        return Tensor._str_op(self.op)


def _dfs(
    node: ArcPointer[Op],
    mut visited: List[Int],
    mut result: List[ArcPointer[Op]],
):
    var addr = Int(node.unsafe_ptr())
    for i in range(len(visited)):
        if visited[i] == addr:
            return
    visited.append(addr)
    for i in range(len(node[].srcs)):
        _dfs(node[].srcs[i], visited, result)
    result.append(node)


def toposort(root: Tensor) -> List[ArcPointer[Op]]:
    var visited = List[Int]()
    var result = List[ArcPointer[Op]]()
    _dfs(root.op, visited, result)
    return result^


def main() raises:
    var x = Tensor.empty([2, 3], requires_grad=True)
    var w = Tensor.empty([2, 3], requires_grad=True)
    var out = x * w + w

    print(out.__str__())
    out.backward()

    if g := x.grad():
        print("x.grad:", Tensor(g.value()).__str__())
    if g := w.grad():
        print("w.grad:", Tensor(g.value()).__str__())
