from std.memory import ArcPointer

# ===-------------------------------------------------------------------===#
# Op
# ===-------------------------------------------------------------------===#


# TODO: Lets rather have a op registry that is defined and built at comptime.
#       Main motivation here is for custom rewrites.
@fieldwise_init
struct OpType(Copyable, Equatable, ImplicitlyCopyable, Movable):
    var _value: Int
    var _name: String
    comptime BUFFER = OpType(2, "BUFFER")
    comptime ONES = OpType(3, "ONES")
    comptime ADD = OpType(5, "ADD")
    comptime MUL = OpType(7, "MUL")


struct Op(Copyable, Movable, Writable):
    var op_type: OpType
    var shape: List[Int]
    var dtype: DType
    var srcs: List[OpRef]

    def __init__(
        out self,
        op_type: OpType,
        var shape: List[Int],
        dtype: DType,
        var srcs: List[OpRef],
    ):
        self.op_type = op_type
        self.shape = shape^
        self.dtype = dtype
        self.srcs = srcs^

    def __str__(self) -> String:
        return self.op_type._name

struct OpRef(Copyable, Movable, ImplicitlyCopyable):
    var _ptr: ArcPointer[Op]

    def __init__(out self, var op: Op):
        self._ptr = ArcPointer(op^)

    def op(ref self) -> ref [self._ptr[]] Op:
        return self._ptr[]

    def id(self) -> Int:
        return Int(self._ptr.unsafe_ptr())

    def __add__(self, rhs: OpRef) -> OpRef:
        return OpRef(Op(OpType.ADD, self.op().shape.copy(), self.op().dtype, [self, rhs]))

    def __mul__(self, rhs: OpRef) -> OpRef:
        return OpRef(Op(OpType.MUL, self.op().shape.copy(), self.op().dtype, [self, rhs]))



# ===-------------------------------------------------------------------===#
# PatternMatcher
# ===-------------------------------------------------------------------===#

comptime RuleFn = def(OpRef, OpRef) raises thin -> List[
   OpRef 
]


@fieldwise_init
struct Rule(Copyable, Movable):
    var pat: Pat
    var func: RuleFn


struct Pat(Copyable, Movable):
    var op_type: OpType
    var srcs: List[Pat]  # empty = match any srcs

    def __init__(out self, op_type: OpType):
        self.op_type = op_type
        self.srcs = List[Pat]()

    def matches(self, node: OpRef) -> Bool:
        if node.op().op_type != self.op_type:
            return False
        if len(self.srcs) == 0:
            return True
        if len(self.srcs) != len(node.op().srcs):
            return False
        for i in range(len(self.srcs)):
            if not self.srcs[i].matches(node.op().srcs[i]):
                return False
        return True


def build_rule_table[
    rules: List[Rule]
    ]()
 -> Dict[Int, List[Rule]]:
    var d = Dict[Int, List[Rule]]()
    comptime for rule in rules:
        key = rule.pat.op_type._value
        r = materialize[rule]()
        d.setdefault(key, List[Rule]()).append(r^)
    return d^


struct PatternMatcher[rules: List[Rule]]:
    var rule_table: Dict[Int, List[Rule]]

    def __init__(out self):
        # TODO: Is the below possible? Maybe `global_constant()` eventually?
        # https://mojolang.org/docs/manual/metaprogramming/materialization/
        # https://github.com/modular/modular/issues/6505
        # comptime ct_table = build_rule_table(Self.rules)
        # self.rule_table = materialize[ct_table]()
        self.rule_table = build_rule_table[Self.rules]()

    def rewrite(
        self,
        node: OpRef,
        upstream: OpRef,
    ) raises -> Optional[List[OpRef]]:
        var matches = self.rule_table.get(node.op().op_type._value)
        if matches:
            for rule in matches.value():
                if rule.pat.matches(node):
                    return rule.func(node, upstream)
        return None


# ===-------------------------------------------------------------------===#
# Grad
# ===-------------------------------------------------------------------===#


def mul_grad(node: OpRef, upstream: OpRef) -> List[OpRef]:
    return [node.op().srcs[1] * upstream, node.op().srcs[0] * upstream]


def add_grad(node: OpRef, upstream: OpRef) -> List[OpRef]:
    return [upstream] * len(node.op().srcs)


struct Grad:
    var grad_map: Dict[Int, OpRef]

    def __init__(out self):
        self.grad_map = Dict[Int, OpRef]()

    @staticmethod
    def compute(
        root: OpRef,
        initial_grad: OpRef,
        target_ops: List[OpRef],
    ) raises -> List[Optional[OpRef]]:
        var grad = Grad()
        grad.grad_map[root.id()] = initial_grad

        var pm = PatternMatcher[
            [
                Rule(Pat(OpType.MUL), mul_grad),
                Rule(Pat(OpType.ADD), add_grad),
            ]
        ]()

        var topo = Self.toposort(root)
        for i in reversed(range(len(topo))):
            var node = topo[i]
            var addr = node.id()
            var upstream = grad.grad_map.get(addr)
            if not upstream:
                continue

            var up = upstream.value()
            var src_grads = pm.rewrite(node, up)
            if src_grads:
                ref sg = src_grads.value()
                for j in range(len(node.op().srcs)):
                    grad.accum(node.op().srcs[j], sg[j])

        var result = List[Optional[OpRef]]()
        for i in range(len(target_ops)):
            result.append(grad.grad_map.get(target_ops[i].id()))
        return result^

    def accum(mut self, op: OpRef, g: OpRef) raises:
        var addr = op.id()
        self.grad_map[addr] = self.grad_map[addr] + g if addr in self.grad_map else g

    @staticmethod
    def toposort(root: OpRef) -> List[OpRef]:
        var visited = Dict[Int, Bool]()
        var result = List[OpRef]()
        Self._dfs(root, visited, result)
        return result^

    @staticmethod
    def _dfs(
        node: OpRef,
        mut visited: Dict[Int, Bool],
        mut result: List[OpRef],
    ):
        var addr = node.id()
        if addr in visited:
            return
        visited[addr] = True
        for i in range(len(node.op().srcs)):
            Self._dfs(node.op().srcs[i], visited, result)
        result.append(node)


# ===-------------------------------------------------------------------===#
# Tensor
# ===-------------------------------------------------------------------===#


# TODO: Make factory methods for tensor constructors.
struct Tensor(Copyable, Movable):
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
        return Tensor(OpRef(Op(OpType.ONES, other.op.op().shape.copy(), other.op.op().dtype, srcs^)), requires_grad)


    @staticmethod
    def empty(
        shape: List[Int],
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) -> Tensor:
        var srcs: List[OpRef] = []
        return Tensor(OpRef(Op(OpType.BUFFER, shape.copy(), dtype, srcs^)), requires_grad)


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
                        targets[i].op.op().shape.copy(), targets[i].op.op().dtype
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

    @staticmethod
    def _str_op(op: OpRef) -> String:
        var op_name = op.op().__str__()

        if len(op.op().srcs) == 0:
            return op_name

        var result = op_name + "("
        for i in range(len(op.op().srcs)):
            result = result + Tensor._str_op(op.op().srcs[i])
            if i < len(op.op().srcs) - 1:
                result = result + ", "
        result = result + ")"
        return result

    def __str__(self) -> String:
        return Tensor._str_op(self.op)


# ===-------------------------------------------------------------------===#
# Main
# ===-------------------------------------------------------------------===#


def main() raises:
    var x = Tensor.empty([2, 3], requires_grad=True)
    var w = Tensor.empty([2, 3], requires_grad=True)
    var out = x * w + w

    print(out.__str__())
    var grads = out.gradient(x, w)

    for t in grads:
        print(t.__str__())
