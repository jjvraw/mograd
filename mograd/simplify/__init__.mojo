from mograd.op import Op, OpRef, OpType, AttrVal
from mograd.pattern_matcher import PatternMatcher, Rule, Pat, GraphUtils, extract_wildcard_srcs
from mograd.scheduler import BoundExecFn, SchedulerRules

# ===-------------------------------------------------------------------===#
# Simplifier
# ===-------------------------------------------------------------------===#

comptime RewriteFn = def(node: OpRef) thin raises -> Optional[OpRef]


struct Simplifier:
    var rules: List[Rule[RewriteFn]]

    def __init__(out self, var rules: List[Rule[RewriteFn]]):
        self.rules = rules^

    def run(self, root: OpRef) raises -> OpRef:
        var extra = SchedulerRules()
        return self.run(root, SchedulerRules(), extra)

    def run(self, root: OpRef, var extern_rules: SchedulerRules, mut extra_sched: SchedulerRules) raises -> OpRef:
        var fusion_pm = PatternMatcher[BoundExecFn](extern_rules)
        var subst = Dict[OpRef, OpRef]()
        var topo = GraphUtils.toposort(root)

        for node in topo:
            # Per-node fixed point: matchers run in priority of user rules.
            var node_r = _apply_subst(node, subst)
            var seen = Dict[OpRef, Bool]()
            while True:
                if node_r in seen:
                    raise Error("Simplifier: rewrite rules cycled without converging")
                seen[node_r] = True

                # User compound patterns produce opaque __fuse_N nodes that
                # are never rewritten further.
                var fmatch = fusion_pm.match(node_r)
                if fmatch:
                    var leaves = extract_wildcard_srcs(fmatch.value().pat, node_r)
                    for j in range(len(extern_rules)):
                        if extern_rules[j].pat.matches(node_r):
                            var stype = OpType("__fuse_" + String(j))
                            node_r = OpRef(Op(stype, node_r.layout(), node_r.dtype(), leaves^))
                            var already = False
                            for sched in extra_sched:
                                if sched.pat.op_type == stype:
                                    already = True
                                    break
                            if not already:
                                extra_sched.append(Rule(Pat(stype), extern_rules[j].func))
                            break
                    break

                var progressed = False
                for rule in self.rules:
                    if not rule.pat.matches(node_r):
                        continue
                    var rewritten = rule.func(node_r)
                    if rewritten:
                        node_r = rewritten.value()
                        progressed = True
                        break
                if not progressed:
                    break

            if node_r != node:
                subst[node] = node_r

        var result = subst.get(root)
        return result.value() if result else root


def _apply_subst(node: OpRef, subst: Dict[OpRef, OpRef]) raises -> OpRef:
    if len(node.srcs()) == 0:
        return node
    var new_srcs = List[OpRef]()
    var changed = False
    for i in range(len(node.srcs())):
        var src = node.srcs()[i]
        var rep = subst.get(src)
        if rep:
            new_srcs.append(rep.value())
            changed = True
        else:
            new_srcs.append(src)
    if not changed:
        return node
    return OpRef(Op(node.op_type(), node.layout(), node.dtype(), new_srcs^, node.attrs_copy()))
