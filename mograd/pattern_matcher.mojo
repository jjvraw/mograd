from mograd.op import OpRef, OpType

# ===-------------------------------------------------------------------===#
# PatternMatcher
# ===-------------------------------------------------------------------===#

comptime RuleFn = def(OpRef, OpRef) raises thin -> List[OpRef]


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


def build_rule_table[rules: List[Rule]]() -> Dict[OpType, List[Rule]]:
    var d = Dict[OpType, List[Rule]]()
    comptime for rule in rules:
        key = rule.pat.op_type
        r = materialize[rule]()
        d.setdefault(key, List[Rule]()).append(r^)
    return d^


struct PatternMatcher[rules: List[Rule]]:
    var rule_table: Dict[OpType, List[Rule]]

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
        var matches = self.rule_table.get(node.op_type())
        if matches:
            for rule in matches.value():
                if rule.pat.matches(node):
                    return rule.func(node, upstream)
        return None
