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
        var rewrite_pm = PatternMatcher[RewriteFn](self.rules)
        var fusion_pm = PatternMatcher[BoundExecFn](extern_rules)
        var subst = Dict[OpRef, OpRef]()
        var topo = GraphUtils.toposort(root)

        for i in range(len(topo)):
            var node = topo[i]
            var node_s = _apply_subst(node, subst)

            # Fusion rules first, user compound patterns take priority over GPU rewrites
            var fmatch = fusion_pm.match(node_s)
            if fmatch:
                var leaves = extract_wildcard_srcs(fmatch.value().pat, node_s)
                for j in range(len(extern_rules)):
                    if extern_rules[j].pat.matches(node_s):
                        var stype = OpType("__fuse_" + String(j))
                        subst[node] = OpRef(Op(stype, node_s.layout(), node_s.dtype(), leaves^))
                        var already = False
                        for k in range(len(extra_sched)):
                            if extra_sched[k].pat.op_type == stype:
                                already = True
                                break
                        if not already:
                            extra_sched.append(Rule(Pat(stype), extern_rules[j].func))
                        break
                continue

            var rmatch = rewrite_pm.match(node_s)
            if rmatch:
                var rewritten = rmatch.value().func(node_s)
                if rewritten:
                    subst[node] = rewritten.value()
                    continue

            if node_s != node:
                subst[node] = node_s

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
