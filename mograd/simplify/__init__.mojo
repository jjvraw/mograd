from mograd.op import Op, OpRef, OpType, AttrVal
from mograd.pattern_matcher import PatternMatcher, Rule, Pat, GraphUtils

# ===-------------------------------------------------------------------===#
# Simplifier
# ===-------------------------------------------------------------------===#

comptime RewriteFn = def(node: OpRef) thin raises -> Optional[OpRef]


struct Simplifier[rules: List[Rule[RewriteFn]]]:
    @staticmethod
    def run(root: OpRef) raises -> OpRef:
        var pm = PatternMatcher[RewriteFn, Self.rules]()
        var subst = Dict[OpRef, OpRef]()
        var topo = GraphUtils.toposort(root)

        for i in range(len(topo)):
            var node = topo[i]
            var node_s = _apply_subst(node, subst)
            var rule = pm.match(node_s)
            if rule:
                var rewritten = rule.value()(node_s)
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
    return OpRef(Op(node.op_type(), node.shape(), node.dtype(), new_srcs^, node.attrs_copy()))
