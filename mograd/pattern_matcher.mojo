from mograd.op import OpRef, OpType

# ===-------------------------------------------------------------------===#
# PatternMatcher
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct Rule[F: TrivialRegisterPassable](Copyable, ImplicitlyDeletable, Movable):
    var pat: Pat
    var func: Self.F


# TODO Clean up wildcard semantics
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

    def matches(self, node: OpRef) -> Bool:
        if not self._any_op and node.op_type() != self.op_type:
            return False
        if len(self.srcs) == 0:
            return True
        if len(self.srcs) != len(node.srcs()):
            return False
        for i in range(len(self.srcs)):
            if not self.srcs[i].matches(node.srcs()[i]):
                return False
        return True

    def is_compound(self) -> Bool:
        for src in self.srcs:
            if not src._any_op:
                return True
            if src.is_compound():
                return True
        return False


struct PatternMatcher[F: TrivialRegisterPassable]:
    var rule_table: Dict[OpType, List[Rule[Self.F]]]

    def __init__(out self, ref rules: List[Rule[Self.F]]):
        self.rule_table = Dict[OpType, List[Rule[Self.F]]]()
        for rule in rules:
            var key = rule.pat.op_type
            self.rule_table.setdefault(key, List[Rule[Self.F]]()).append(rule.copy())

    def match(self, node: OpRef) -> Optional[Rule[Self.F]]:
        var matches = self.rule_table.get(node.op_type())
        if matches:
            for rule in matches.value():
                if rule.pat.matches(node):
                    return rule.copy()
        return None


# ===-------------------------------------------------------------------===#
# GraphUtils
# ===-------------------------------------------------------------------===#


def extract_wildcard_srcs(pat: Pat, node: OpRef) -> List[OpRef]:
    if pat._any_op:
        return [node]
    var result = List[OpRef]()
    for i in range(len(pat.srcs)):
        result += extract_wildcard_srcs(pat.srcs[i], node.src(i))
    return result^


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
