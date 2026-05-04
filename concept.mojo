from std.memory import ArcPointer


@fieldwise_init
struct OpType(Copyable, Equatable, ImplicitlyCopyable, Movable):
    var _value: Int
    comptime BUFFER = OpType(2)
    comptime ONES = OpType(3)
    comptime ADD = OpType(5)
    comptime MUL = OpType(7)


@fieldwise_init
struct Op(Copyable, Movable):
    var op_type: OpType
    var shape: List[Int]
    var dtype: DType
    var srcs: List[ArcPointer[Op]]


@fieldwise_init
struct TensorInner(Copyable, Movable):
    var op: ArcPointer[Op]
    var grad: Optional[ArcPointer[Op]]
    var requires_grad: Bool
    var parents: List[ArcPointer[TensorInner]]


@fieldwise_init
struct Tensor(Copyable, Movable):
    var _inner: ArcPointer[TensorInner]

    @implicit
    def __init__(out self, op: ArcPointer[Op]):
        self._inner = ArcPointer(
            TensorInner(
                op,
                Optional[ArcPointer[Op]](None),
                False,
                List[ArcPointer[TensorInner]](),
            )
        )

    @staticmethod
    def empty(
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) -> Tensor:
        var op = ArcPointer(
            Op(OpType.BUFFER, shape.copy(), dtype, List[ArcPointer[Op]]())
        )
        return Tensor(
            ArcPointer(
                TensorInner(
                    op,
                    Optional[ArcPointer[Op]](None),
                    requires_grad,
                    List[ArcPointer[TensorInner]](),
                )
            )
        )

    def __add__(self, other: Self) -> Self:
        var srcs = List[ArcPointer[Op]]()
        srcs.append(self._inner[].op)
        srcs.append(other._inner[].op)
        var op = ArcPointer(
            Op(
                OpType.ADD,
                self._inner[].op[].shape.copy(),
                self._inner[].op[].dtype,
                srcs^,
            )
        )
        var parents = List[ArcPointer[TensorInner]]()
        parents.append(self._inner)
        parents.append(other._inner)
        return Tensor(
            ArcPointer(
                TensorInner(
                    op,
                    Optional[ArcPointer[Op]](None),
                    self._inner[].requires_grad or other._inner[].requires_grad,
                    parents^,
                )
            )
        )

    def __mul__(self, other: Self) -> Self:
        var srcs = List[ArcPointer[Op]]()
        srcs.append(self._inner[].op)
        srcs.append(other._inner[].op)
        var op = ArcPointer(
            Op(
                OpType.MUL,
                self._inner[].op[].shape.copy(),
                self._inner[].op[].dtype,
                srcs^,
            )
        )
        var parents = List[ArcPointer[TensorInner]]()
        parents.append(self._inner)
        parents.append(other._inner)
        return Tensor(
            ArcPointer(
                TensorInner(
                    op,
                    Optional[ArcPointer[Op]](None),
                    self._inner[].requires_grad or other._inner[].requires_grad,
                    parents^,
                )
            )
        )

    def backward(mut self) raises -> None:
        self._inner[].grad = Optional(
            ArcPointer(
                Op(
                    OpType.ONES,
                    self._inner[].op[].shape.copy(),
                    self._inner[].op[].dtype,
                    List[ArcPointer[Op]](),
                )
            )
        )

        var topo = toposort(self)

        for i in range(len(topo) - 1, -1, -1):
            var node = topo[i]
            if not node[].grad:
                continue
            var upstream = Tensor(node[].grad.value())

            if node[].op[].op_type == OpType.MUL:
                # d/d(p0) = upstream * p1
                # d/d(p1) = upstream * p0
                var grad0 = upstream * Tensor(node[].parents[1][].op)
                var grad1 = upstream * Tensor(node[].parents[0][].op)
                if node[].parents[0][].requires_grad:
                    if node[].parents[0][].grad:
                        node[].parents[0][].grad = Optional(
                            (Tensor(node[].parents[0][].grad.value()) + grad0)
                            ._inner[]
                            .op
                        )
                    else:
                        node[].parents[0][].grad = Optional(grad0._inner[].op)
                if node[].parents[1][].requires_grad:
                    if node[].parents[1][].grad:
                        node[].parents[1][].grad = Optional(
                            (Tensor(node[].parents[1][].grad.value()) + grad1)
                            ._inner[]
                            .op
                        )
                    else:
                        node[].parents[1][].grad = Optional(grad1._inner[].op)

            elif node[].op[].op_type == OpType.ADD:
                for j in range(len(node[].parents)):
                    if node[].parents[j][].requires_grad:
                        if node[].parents[j][].grad:
                            node[].parents[j][].grad = Optional(
                                (
                                    Tensor(node[].parents[j][].grad.value())
                                    + upstream
                                )
                                ._inner[]
                                .op
                            )
                        else:
                            node[].parents[j][].grad = Optional(
                                upstream._inner[].op
                            )

            node[].parents = List[ArcPointer[TensorInner]]()

    def __str__(self) -> String:
        var op_name: String
        if self._inner[].op[].op_type == OpType.BUFFER:
            op_name = "BUFFER"
        elif self._inner[].op[].op_type == OpType.ADD:
            op_name = "ADD"
        elif self._inner[].op[].op_type == OpType.MUL:
            op_name = "MUL"
        elif self._inner[].op[].op_type == OpType.ONES:
            op_name = "ONES"
        else:
            op_name = "?"

        if len(self._inner[].op[].srcs) == 0:
            return op_name

        var result = op_name + "("
        for i in range(len(self._inner[].op[].srcs)):
            var child = Tensor(self._inner[].op[].srcs[i]).__str__()
            result = result + child
            if i < len(self._inner[].op[].srcs) - 1:
                result = result + ", "
        result = result + ")"
        return result


def _dfs(
    node: ArcPointer[TensorInner],
    mut visited: List[Int],
    mut result: List[ArcPointer[TensorInner]],
):
    var addr = Int(node.unsafe_ptr())
    for i in range(len(visited)):
        if visited[i] == addr:
            return
    visited.append(addr)
    for i in range(len(node[].parents)):
        _dfs(node[].parents[i], visited, result)
    result.append(node)


def toposort(root: Tensor) -> List[ArcPointer[TensorInner]]:
    var visited = List[Int]()
    var result = List[ArcPointer[TensorInner]]()
    _dfs(root._inner, visited, result)
    return result^


def main() raises:
    var x = Tensor.empty([2, 3], requires_grad=True)
    var w = Tensor.empty([2, 3], requires_grad=True)
    var out = x * w + w

    print(out.__str__())
    out.backward()

    if x._inner[].grad:
        print("x.grad:", Tensor(x._inner[].grad.value()).__str__())
    if w._inner[].grad:
        print("w.grad:", Tensor(w._inner[].grad.value()).__str__())
