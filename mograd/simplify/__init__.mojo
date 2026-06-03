from mograd.op import Op, OpRef, OpType, AnyOpRef, NodeOps, AttrVal
from mograd.pattern_matcher import PatternMatcher, Rule, Pat, GraphUtils

# ===-------------------------------------------------------------------===#
# Simplifier
# ===-------------------------------------------------------------------===#

comptime RewriteFn = def(node: AnyOpRef) thin raises -> Optional[AnyOpRef]


struct Simplifier[rules: List[Rule[RewriteFn]]]:
    @staticmethod
    def run(root: AnyOpRef) raises -> AnyOpRef:
        var pm = PatternMatcher[RewriteFn, Self.rules]()
        var subst = Dict[AnyOpRef, AnyOpRef]()
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


def _apply_subst(node: AnyOpRef, subst: Dict[AnyOpRef, AnyOpRef]) raises -> AnyOpRef:
    # Explicit isa dispatch: T.dtype not accessible via comptime-for TypeList bound
    if node.isa[OpRef[DType.float32]]():
        return AnyOpRef(_apply_subst_typed[DType.float32](node.unsafe_get[OpRef[DType.float32]](), subst))
    elif node.isa[OpRef[DType.int64]]():
        return AnyOpRef(_apply_subst_typed[DType.int64](node.unsafe_get[OpRef[DType.int64]](), subst))
    return node


def _apply_subst_typed[
    dtype: DType
](node: OpRef[dtype], subst: Dict[AnyOpRef, AnyOpRef],) raises -> OpRef[dtype]:
    if len(node.srcs()) == 0:
        return node
    var new_srcs = List[AnyOpRef]()
    var changed = False
    for i in range(len(node.srcs())):
        var src = node.srcs()[i]
        if src.isa[OpRef[dtype]]():
            var rep = subst.get(src)
            if rep:
                new_srcs.append(AnyOpRef(rep.value().unsafe_get[OpRef[dtype]]()))
                changed = True
                continue
        new_srcs.append(src)
    if not changed:
        return node
    return OpRef[dtype](node.op_type(), node.shape(), new_srcs^, node.attrs().copy())
