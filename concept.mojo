from std.memory import ArcPointer


# struct MulOP:
#     @staticmethod
#     def forward() -> Op:
#         ...


@fieldwise_init
struct OpType(Copyable, Equatable, ImplicitlyCopyable, Movable):
    var _value: Int
    comptime NONE = OpType(2)
    comptime BUFFER = OpType(3)
    comptime ONES = OpType(5)
    comptime ADD = OpType(7)
    comptime MUL = OpType(11)


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

    def set_grad(mut self, var grad: Op):
        if self.grad:
            self.grad.value().destroy_pointee()
            self.grad.value().free()
        var ptr = alloc[Op](1)
        ptr.init_pointee_move(grad^)
        self.grad = ptr

    def __del__(deinit self):
        if self.grad:
            self.grad.value().destroy_pointee()
            self.grad.value().free()


struct Tensor:
    var op: ArcPointer[Op]
    var requires_grad: Bool

    def __init__(out self, var op: ArcPointer[Op], requires_grad: Bool=False):
        self.op = op^
        self.requires_grad = requires_grad

    @implicit
    def __init__(out self, var op: Op, requires_grad: Bool=False):
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
        var srcs = List[ArcPointer[Op]]()
        srcs.append(self.op)
        srcs.append(other.op)
        var op = ArcPointer(
            Op(
                OpType.ADD,
                self.op[].shape.copy(),
                self.op[].dtype,
                srcs^,
            )
        )
        return Tensor(op, self.requires_grad or other.requires_grad)

    def __mul__(self, other: Self) -> Self:
        var srcs = List[ArcPointer[Op]]()
        srcs.append(self.op)
        srcs.append(other.op)
        var op = ArcPointer(
            Op(
                OpType.MUL,
                self.op[].shape.copy(),
                self.op[].dtype,
                srcs^,
            )
        )
        return Tensor(op, self.requires_grad or other.requires_grad)

    def backward(mut self) raises -> None:
        var ones_shape = self.op[].shape.copy()
        var ones_srcs = List[ArcPointer[Op]]()
        self.op[].set_grad(
            Op(OpType.ONES, ones_shape^, self.op[].dtype, ones_srcs^)
        )

        var topo = toposort(self)

        for i in reversed(range(len(topo))):
            var node = topo[i]
            if not node[].grad:
                continue
            var upstream = Tensor(ArcPointer(node[].grad.value()[].copy()))

            if node[].op_type == OpType.MUL:
                var grad0 = upstream * Tensor(node[].srcs[1])
                var grad1 = upstream * Tensor(node[].srcs[0])
                if node[].srcs[0][].grad:
                    node[].srcs[0][].set_grad(
                        (Tensor(ArcPointer(node[].srcs[0][].grad.value()[].copy())) + grad0).op[].copy()
                    )
                else:
                    node[].srcs[0][].set_grad(grad0.op[].copy())
                if node[].srcs[1][].grad:
                    node[].srcs[1][].set_grad(
                        (Tensor(ArcPointer(node[].srcs[1][].grad.value()[].copy())) + grad1).op[].copy()
                    )
                else:
                    node[].srcs[1][].set_grad(grad1.op[].copy())

            elif node[].op_type == OpType.ADD:
                for j in range(len(node[].srcs)):
                    if node[].srcs[j][].grad:
                        node[].srcs[j][].set_grad(
                            (Tensor(ArcPointer(node[].srcs[j][].grad.value()[].copy())) + upstream).op[].copy()
                        )
                    else:
                        node[].srcs[j][].set_grad(upstream.op[].copy())

    def __str__(self) -> String:
        var op_name: String
        if self.op[].op_type == OpType.BUFFER:
            op_name = "BUFFER"
        elif self.op[].op_type == OpType.ADD:
            op_name = "ADD"
        elif self.op[].op_type == OpType.MUL:
            op_name = "MUL"
        elif self.op[].op_type == OpType.ONES:
            op_name = "ONES"
        else:
            op_name = "?"

        if len(self.op[].srcs) == 0:
            return op_name

        var result = op_name + "("
        for i in range(len(self.op[].srcs)):
            var child = Tensor(self.op[].srcs[i], False).__str__()
            result = result + child
            if i < len(self.op[].srcs) - 1:
                result = result + ", "
        result = result + ")"
        return result


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

    if x.op[].grad:
        print("x.grad:", Tensor(ArcPointer(x.op[].grad.value()[].copy())).__str__())
    if w.op[].grad:
        print("w.grad:", Tensor(ArcPointer(w.op[].grad.value()[].copy())).__str__())
