from mograd.op import OpRef, OpType
from mograd.pattern_matcher import PatternMatcher, Rule, Pat

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
