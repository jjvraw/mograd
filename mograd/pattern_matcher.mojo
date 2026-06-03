from mograd.op import OpRef, OpType, AnyOpRef, NodeOps

# ===-------------------------------------------------------------------===#
# PatternMatcher
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct Rule[F: TrivialRegisterPassable](Copyable, ImplicitlyDestructible, Movable):
    var pat: Pat
    var func: Self.F


# TODO: Clean up wildcard semantics
struct Pat(Copyable, ImplicitlyCopyable, Movable):
    var op_type: OpType
    var srcs: List[Pat]
    var _any_op: Bool

    def __init__(out self):
        self.op_type = OpType("")
        self.srcs = List[Pat]()
        self._any_op = True

    def __init__(out self, op_type: OpType):
        self.op_type = op_type
        self.srcs = List[Pat]()
        self._any_op = False

    def __init__(out self, op_type: OpType, var srcs: List[Pat]):
        self.op_type = op_type
        self.srcs = srcs^
        self._any_op = False

    def __init__(out self, *, copy: Self):
        self.op_type = copy.op_type
        self.srcs = copy.srcs.copy()
        self._any_op = copy._any_op

    def matches(self, node: AnyOpRef) -> Bool:
        comptime for i in range(AnyOpRef.Ts.size):
            comptime T = AnyOpRef.Ts[i]
            if node.isa[T]():
                comptime assert conforms_to(T, NodeOps)
                ref typed = node.unsafe_get[T]()
                if not self._any_op and typed.op_type() != self.op_type:
                    return False
                if len(self.srcs) == 0:
                    return True
                if len(self.srcs) != typed.srcs_count():
                    return False
                for j in range(len(self.srcs)):
                    if not self.srcs[j].matches(typed.get_src(j)):
                        return False
                return True
        return False


def build_rule_table[F: TrivialRegisterPassable, rules: List[Rule[F]]]() -> Dict[OpType, List[Rule[F]]]:
    var d = Dict[OpType, List[Rule[F]]]()
    comptime for rule in rules:
        key = rule.pat.op_type
        r = materialize[rule]()
        d.setdefault(key, List[Rule[F]]()).append(r^)
    return d^


struct PatternMatcher[F: TrivialRegisterPassable, rules: List[Rule[F]]]:
    var rule_table: Dict[OpType, List[Rule[Self.F]]]

    def __init__(out self):
        # TODO: Is the below possible? Maybe `global_constant()` somewhere eventually?
        # https://mojolang.org/docs/manual/metaprogramming/materialization/
        # https://github.com/modular/modular/issues/6505
        # comptime ct_table = build_rule_table[Self.F, Self.rules]()
        # self.rule_table = materialize[ct_table]()
        self.rule_table = build_rule_table[Self.F, Self.rules]()

    def match(self, node: AnyOpRef) -> Optional[Self.F]:
        comptime for i in range(AnyOpRef.Ts.size):
            comptime T = AnyOpRef.Ts[i]
            if node.isa[T]():
                ref typed = node.unsafe_get[T]()
                comptime assert conforms_to(T, NodeOps)
                var matches = self.rule_table.get(typed.op_type())
                if matches:
                    for rule in matches.value():
                        if rule.pat.matches(node):
                            return rule.func
                return None
        return None


# ===-------------------------------------------------------------------===#
# GraphUtils
# ===-------------------------------------------------------------------===#


struct GraphUtils:
    @staticmethod
    def toposort(root: AnyOpRef) -> List[AnyOpRef]:
        var visited = Dict[AnyOpRef, Bool]()
        var result = List[AnyOpRef]()
        Self._dfs_any(root, visited, result)
        return result^

    @staticmethod
    def _dfs_any(
        node: AnyOpRef,
        mut visited: Dict[AnyOpRef, Bool],
        mut result: List[AnyOpRef],
    ):
        if node in visited:
            return
        visited[node] = True
        comptime for i in range(AnyOpRef.Ts.size):
            comptime T = AnyOpRef.Ts[i]
            if node.isa[T]():
                comptime assert conforms_to(T, NodeOps)
                ref n = trait_downcast[NodeOps](node.unsafe_get[T]())
                for j in range(n.srcs_count()):
                    Self._dfs_any(n.get_src(j), visited, result)
        result.append(node)

    @staticmethod
    def toposort[dtype: DType](root: OpRef[dtype]) -> List[OpRef[dtype]]:
        var visited = Dict[OpRef[dtype], Bool]()
        var result = List[OpRef[dtype]]()
        Self._dfs[dtype](root, visited, result)
        return result^

    @staticmethod
    def _dfs[
        dtype: DType
    ](node: OpRef[dtype], mut visited: Dict[OpRef[dtype], Bool], mut result: List[OpRef[dtype]],):
        if node in visited:
            return
        visited[node] = True
        for i in range(len(node.srcs())):
            if node.srcs()[i].isa[OpRef[dtype]]():
                Self._dfs[dtype](node.srcs()[i].unsafe_get[OpRef[dtype]](), visited, result)
        result.append(node)
