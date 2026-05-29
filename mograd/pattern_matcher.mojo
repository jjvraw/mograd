from mograd.op import OpRef, OpType

# ===-------------------------------------------------------------------===#
# PatternMatcher
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct Rule[F: TrivialRegisterPassable](Copyable, ImplicitlyDestructible, Movable):
    var pat: Pat
    var func: Self.F


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
        # comptime ct_table = build_rule_table(Self.rules)
        # self.rule_table = materialize[ct_table]()
        self.rule_table = build_rule_table[Self.F, Self.rules]()

    def match(self, node: OpRef) -> Optional[Self.F]:
        var matches = self.rule_table.get(node.op_type())
        if matches:
            for rule in matches.value():
                if rule.pat.matches(node):
                    return rule.func
        return None


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
