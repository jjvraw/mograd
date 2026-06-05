from mograd.op import OpRef, OpType

# ===-------------------------------------------------------------------===#
# PatternMatcher
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct Rule[F: TrivialRegisterPassable](Copyable, ImplicitlyDestructible, Movable):
    var pat: Pat
    var func: Self.F


# TODO Clean up wildcard semantics
struct Pat(Copyable, ImplicitlyCopyable, Movable):
    var op_type: OpType
    var srcs: List[Pat]
    var _any_op: Bool
    var fp_only: Bool

    def __init__(out self):
        self.op_type = OpType("")
        self.srcs = List[Pat]()
        self._any_op = True
        self.fp_only = False

    def __init__(out self, op_type: OpType, fp_only: Bool = False):
        self.op_type = op_type
        self.srcs = List[Pat]()
        self._any_op = False
        self.fp_only = fp_only

    def __init__(out self, op_type: OpType, var srcs: List[Pat]):
        self.op_type = op_type
        self.srcs = srcs^
        self._any_op = False
        self.fp_only = False

    def __init__(out self, *, copy: Self):
        self.op_type = copy.op_type
        self.srcs = copy.srcs.copy()
        self._any_op = copy._any_op
        self.fp_only = copy.fp_only

    def matches(self, node: OpRef) -> Bool:
        if not self._any_op and node.op_type() != self.op_type:
            return False
        if self.fp_only and not node.dtype().is_floating_point():
            return False
        if len(self.srcs) == 0:
            return True
        if len(self.srcs) != len(node.srcs()):
            return False
        for i in range(len(self.srcs)):
            if not self.srcs[i].matches(node.srcs()[i]):
                return False
        return True


struct PatternMatcher[F: TrivialRegisterPassable]:
    var rule_table: Dict[OpType, List[Rule[Self.F]]]

    def __init__(out self, ref rules: List[Rule[Self.F]]):
        self.rule_table = Dict[OpType, List[Rule[Self.F]]]()
        for i in range(len(rules)):
            var key = rules[i].pat.op_type
            self.rule_table.setdefault(key, List[Rule[Self.F]]()).append(rules[i].copy())

    def match(self, node: OpRef) -> Optional[Self.F]:
        var matches = self.rule_table.get(node.op_type())
        if matches:
            for rule in matches.value():
                if rule.pat.matches(node):
                    return rule.func
        return None


# ===-------------------------------------------------------------------===#
# GraphUtils
# ===-------------------------------------------------------------------===#


struct GraphUtils:
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
