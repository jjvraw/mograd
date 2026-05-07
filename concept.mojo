from std.memory import ArcPointer


trait OpImpl:
    @staticmethod
    def node(lhs: ArcPointer[Op], rhs: ArcPointer[Op]) -> ArcPointer[Op]:
        ...

    @staticmethod
    def grad(
        mut ctx: Grad,
        node: ArcPointer[Op],
        upstream: ArcPointer[Op],
    ) raises:
        ...


struct MulOp(OpImpl):
    @staticmethod
    def node(lhs: ArcPointer[Op], rhs: ArcPointer[Op]) -> ArcPointer[Op]:
        var srcs = List[ArcPointer[Op]]()
        srcs.append(lhs)
        srcs.append(rhs)
        return ArcPointer(
            Op(OpType.MUL, lhs[].shape.copy(), lhs[].dtype, srcs^)
        )

    @staticmethod
    def grad(
        mut ctx: Grad,
        node: ArcPointer[Op],
        upstream: ArcPointer[Op],
    ) raises:
        var s0 = node[].srcs[0]
        var s1 = node[].srcs[1]
        ctx.accum_into(s0, MulOp.node(s1, upstream))
        ctx.accum_into(s1, MulOp.node(s0, upstream))


struct AddOp(OpImpl):
    @staticmethod
    def node(lhs: ArcPointer[Op], rhs: ArcPointer[Op]) -> ArcPointer[Op]:
        var srcs = List[ArcPointer[Op]]()
        srcs.append(lhs)
        srcs.append(rhs)
        return ArcPointer(
            Op(OpType.ADD, lhs[].shape.copy(), lhs[].dtype, srcs^)
        )

    @staticmethod
    def grad(
        mut ctx: Grad,
        node: ArcPointer[Op],
        upstream: ArcPointer[Op],
    ) raises:
        for j in range(len(node[].srcs)):
            ctx.accum_into(node[].srcs[j], upstream)


@fieldwise_init
struct OpType(Copyable, Equatable, ImplicitlyCopyable, Movable):
    var _value: Int
    var _name: String
    comptime BUFFER = OpType(2, "BUFFER")
    comptime ONES = OpType(3, "ONES")
    comptime ADD = OpType(5, "ADD")
    comptime MUL = OpType(7, "MUL")


struct Op(Copyable, Movable, Writable):
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    var op_type: OpType
    var shape: List[Int]
    var dtype: DType
    var srcs: List[ArcPointer[Op]]

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

    def __str__(self) -> String:
        return self.op_type._name


struct Grad:
    comptime RULES = (
        (OpType.MUL._value, MulOp.grad),
        (OpType.ADD._value, AddOp.grad),
    )
    var grad_map: Dict[Int, ArcPointer[Op]]

    def __init__(out self):
        self.grad_map = Dict[Int, ArcPointer[Op]]()

    @staticmethod
    def compute(
        root: ArcPointer[Op],
        initial_grad: ArcPointer[Op],
        target_ops: List[ArcPointer[Op]],
    ) raises -> List[Optional[ArcPointer[Op]]]:
        var self = Grad()
        self.grad_map[Int(root.unsafe_ptr())] = initial_grad

        var topo = Self.toposort(root)

        for i in reversed(range(len(topo))):
            var node = topo[i]
            var addr = Int(node.unsafe_ptr())

            if addr not in self.grad_map:
                continue

            var upstream = self.grad_map[addr]
            Self.backward_dispatch(self, node, upstream)

        var result = List[Optional[ArcPointer[Op]]]()
        for i in range(len(target_ops)):
            var taddr = Int(target_ops[i].unsafe_ptr())
            if taddr in self.grad_map:
                result.append(self.grad_map[taddr])
            else:
                result.append(None)
        return result^

    def accum_into(
        mut self,
        op: ArcPointer[Op],
        grad: ArcPointer[Op],
    ) raises:
        var addr = Int(op.unsafe_ptr())
        if addr in self.grad_map:
            self.grad_map[addr] = AddOp.node(self.grad_map[addr], grad)
        else:
            self.grad_map[addr] = grad

    def backward_dispatch(
        mut self,
        node: ArcPointer[Op],
        upstream: ArcPointer[Op],
    ) raises:
        var pat = node[].op_type._value
        comptime for i in range(len(Self.RULES)):
            if pat == Self.RULES[i][0]:
                Self.RULES[i][1](self, node, upstream)
                return

    @staticmethod
    def toposort(
        root: ArcPointer[Op],
    ) -> List[ArcPointer[Op]]:
        var visited = Dict[Int, Bool]()
        var result = List[ArcPointer[Op]]()
        Self._dfs(root, visited, result)
        return result^

    @staticmethod
    def _dfs(
        node: ArcPointer[Op],
        mut visited: Dict[Int, Bool],
        mut result: List[ArcPointer[Op]],
    ):
        var addr = Int(node.unsafe_ptr())
        if addr in visited:
            return
        visited[addr] = True
        for i in range(len(node[].srcs)):
            Self._dfs(node[].srcs[i], visited, result)
        result.append(node)


struct Tensor(Copyable, Movable):
    var op: ArcPointer[Op]
    var requires_grad: Bool
    var _grad: ArcPointer[Optional[Tensor]]

    def __init__(out self, var op: ArcPointer[Op], requires_grad: Bool = False):
        self.op = op^
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor](None))

    @implicit
    def __init__(out self, var op: Op, requires_grad: Bool = False):
        self.op = ArcPointer(op^)
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor](None))

    @staticmethod
    def ones_like(other: Tensor, requires_grad: Bool = False) -> Tensor:
        srcs = List[ArcPointer[Op]]()
        op = Op(OpType.ONES, other.op[].shape.copy(), other.op[].dtype, srcs^)
        return Tensor(ArcPointer(op^), requires_grad)

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
            AddOp.node(self.op, other.op),
            self.requires_grad or other.requires_grad,
        )

    def __mul__(self, other: Self) -> Self:
        return Tensor(
            MulOp.node(self.op, other.op),
            self.requires_grad or other.requires_grad,
        )

    def gradient(
        mut self, *targets: Tensor, var gradient: Optional[Tensor] = None
    ) raises -> List[Tensor]:
        var initial_grad: Tensor
        if gradient:
            initial_grad = gradient.take()
        else:
            initial_grad = Self.ones_like(self)
        target_ops: List[ArcPointer[Op]] = [t.op for t in targets]

        var grads = Grad.compute(self.op, initial_grad.op, target_ops)

        var result = List[Tensor]()
        for i in range(len(grads)):
            if grads[i]:
                result.append(Tensor(grads[i].value()))
            else:
                result.append(
                    Self.empty(
                        targets[i].op[].shape.copy(), targets[i].op[].dtype
                    )
                )
        return result^

    @staticmethod
    def _str_op(op: ArcPointer[Op]) -> String:
        var op_name = op[].__str__()

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


def main() raises:
    var x = Tensor.empty([2, 3], requires_grad=True)
    var w = Tensor.empty([2, 3], requires_grad=True)
    var out = x * w + w

    print(out.__str__())
    list = out.gradient(x, w)

    for t in list:
        print(t.__str__())
