from std.memory import ArcPointer
from std.hashlib.hasher import Hasher

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

struct OpRef(Copyable, Movable, ImplicitlyCopyable,  KeyElement):
    var _ptr: ArcPointer[Op]

    def __init__(out self, var op: Op):
        self._ptr = ArcPointer(op^)

    def __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(Int(self._ptr.unsafe_ptr()))

    def __eq__(self, other: Self) -> Bool:
        return self._ptr.unsafe_ptr() == other._ptr.unsafe_ptr()

    def __ne__(self, other: Self) -> Bool:
        return self._ptr.unsafe_ptr() != other._ptr.unsafe_ptr()

    def op(ref self) -> ref [self._ptr[]] Op:
        return self._ptr[]

    def shape(ref self) -> ref [self._ptr[].shape] List[Int]:
        return self._ptr[].shape

    def dtype(ref self) -> ref [self._ptr[].dtype] DType:
        return self._ptr[].dtype

    def op_type(ref self) -> ref [self._ptr[].op_type] OpType:
        return self._ptr[].op_type

    def srcs(ref self) -> ref [self._ptr[].srcs] List[OpRef]:
        return self._ptr[].srcs

    def __add__(self, rhs: OpRef) -> OpRef:
        return OpRef(Op(OpType.ADD, self.shape().copy(), self.dtype(), [self, rhs]))

    def __mul__(self, rhs: OpRef) -> OpRef:
        return OpRef(Op(OpType.MUL, self.shape().copy(), self.dtype(), [self, rhs]))



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
        if node.op_type() != self.op_type:
            return False
        if len(self.srcs) == 0:
            return True
        if len(self.srcs) != len(node.srcs()):
            return False
        for i in range(len(self.srcs)):
            if not self.srcs[i].matches(node.srcs()[i]):
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
        var matches = self.rule_table.get(node.op_type()._value)
        if matches:
            for rule in matches.value():
                if rule.pat.matches(node):
                    return rule.func(node, upstream)
        return None


# ===-------------------------------------------------------------------===#
# Grad
# ===-------------------------------------------------------------------===#


def mul_grad(node: OpRef, upstream: OpRef) -> List[OpRef]:
    return [node.srcs()[1] * upstream, node.srcs()[0] * upstream]


def add_grad(node: OpRef, upstream: OpRef) -> List[OpRef]:
    return [upstream] * len(node.srcs())


struct Grad:
    var grad_map: Dict[OpRef, OpRef]

    def __init__(out self):
        self.grad_map = Dict[OpRef, OpRef]()

    @staticmethod
    def compute(
        root: OpRef,
        initial_grad: OpRef,
        target_ops: List[OpRef],
    ) raises -> List[Optional[OpRef]]:
        var grad = Grad()
        grad.grad_map[root] = initial_grad

        var pm = PatternMatcher[
            [
                Rule(Pat(OpType.MUL), mul_grad),
                Rule(Pat(OpType.ADD), add_grad),
            ]
        ]()

        var topo = Self.toposort(root)
        for i in reversed(range(len(topo))):
            var node = topo[i]
            var upstream = grad.grad_map.get(node)
            if not upstream:
                continue

            var up = upstream.value()
            var src_grads = pm.rewrite(node, up)
            if src_grads:
                ref sg = src_grads.value()
                for j in range(len(node.srcs())):
                    grad.accum(node.srcs()[j], sg[j])

        var result = List[Optional[OpRef]]()
        for i in range(len(target_ops)):
            result.append(grad.grad_map.get(target_ops[i]))
        return result^

    def accum(mut self, op: OpRef, g: OpRef) raises:
        var existing = self.grad_map.get(op)
        self.grad_map[op] = existing.value() + g if existing else g

    @staticmethod
    def toposort(root: OpRef) -> List[OpRef]:
        var visited = Dict[OpRef, Bool]()
        var result = List[OpRef]()
        Self._dfs(root, visited, result)
        return result^

    @staticmethod
    def _dfs(
        node: OpRef,
        mut visited: Dict[OpRef, Bool],
        mut result: List[OpRef],
    ):
        if node in visited:
            return
        visited[node] = True
        for i in range(len(node.srcs())):
            Self._dfs(node.srcs()[i], visited, result)
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
        return Tensor(OpRef(Op(OpType.ONES, other.op.shape().copy(), other.op.dtype(), srcs^)), requires_grad)


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
