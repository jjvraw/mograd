from mograd.op import Op, OpRef, OpType
from mograd.pattern_matcher import PatternMatcher, Rule, Pat, GraphUtils

# ===-------------------------------------------------------------------===#
# Grad
# ===-------------------------------------------------------------------===#

comptime GradFn = def(node: OpRef, upstream: OpRef) thin raises -> List[OpRef]


struct Grad[dtype: DType] where dtype.is_floating_point():
    var grad_map: Dict[OpRef, OpRef]

    def __init__(out self):
        self.grad_map = Dict[OpRef, OpRef]()

    @staticmethod
    def compute(
        root: OpRef,
        initial_grad: OpRef,
        target_ops: List[OpRef],
    ) raises -> List[Optional[OpRef]]:
        var grad = Grad[Self.dtype]()
        grad.grad_map[root] = initial_grad

        var pm = PatternMatcher[GradFn](
            [
                Rule(Pat(OpType.MUL), mul_grad),
                Rule(Pat(OpType.ADD), add_grad),
                Rule(Pat(OpType.RELU), relu_grad),
                Rule(Pat(OpType.SOFTMAX), softmax_grad),
                Rule(Pat(OpType.EXP), exp_grad),
                Rule(Pat(OpType.LOG), log_grad),
                Rule(Pat(OpType.NEG), neg_grad),
                Rule(Pat(OpType.DIV), div_grad),
                Rule(Pat(OpType.SUM), sum_grad),
                Rule(Pat(OpType.RESHAPE), reshape_grad),
                Rule(Pat(OpType.VIEW), view_grad),
                Rule(Pat(OpType.SLICE), slice_grad),
                Rule(Pat(OpType.MATMUL), matmul_grad),
                Rule(Pat(OpType.CROSS_ENTROPY), cross_entropy_grad),
                Rule(Pat(OpType.SCALE), scale_grad),
                Rule(Pat(OpType.TRANSPOSE), transpose_grad),
            ]
        )

        var topo = GraphUtils.toposort(root)
        for i in reversed(range(len(topo))):
            var node = topo[i]
            var upstream = grad.grad_map.get(node)
            if not upstream:
                continue

            var up = upstream.value()
            var rule = pm.match(node)
            if rule:
                var src_grads = rule.value()(node, up)
                for j in range(len(node.srcs())):
                    grad.accum(node.srcs()[j], src_grads[j])

        var result = List[Optional[OpRef]]()
        for i in range(len(target_ops)):
            result.append(grad.grad_map.get(target_ops[i]))
        return result^

    def accum(mut self, op: OpRef, g: OpRef) raises:
        var existing = self.grad_map.get(op)
        self.grad_map[op] = existing.value() + g if existing else g


# ===-------------------------------------------------------------------===#
# Grad functions
# ===-------------------------------------------------------------------===#


def mul_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [node.src(1) * upstream, node.src(0) * upstream]


def add_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [upstream] * len(node.srcs())


def relu_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.RELU_GRAD, node.layout(), node.dtype(), [node.src(0), upstream]))]


def exp_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [node * upstream]


def log_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [upstream / node.src(0)]


def neg_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [-upstream]


def div_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var a = node.src(0)
    var b = node.src(1)
    return [upstream / b, -(upstream * a / (b * b))]


def sum_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.BROADCAST, node.src(0).layout().as_contiguous(), node.dtype(), [upstream]))]


def matmul_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var a = node.src(0)
    var b = node.src(1)
    return [upstream.matmul(b.transpose()), a.transpose().matmul(upstream)]


def transpose_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [upstream.transpose()]


def reshape_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.RESHAPE, node.src(0).layout(), node.dtype(), [upstream]))]


def view_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.VIEW, node.src(0).layout(), node.dtype(), [upstream]))]


def slice_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.SLICE_GRAD, node.src(0).layout().as_contiguous(), node.dtype(), [upstream, node]))]


def scale_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var scalar = Scalar(node.attrs()["scalar"][Float32])
    return [upstream.scale(scalar)]


def cross_entropy_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var logits = node.src(0)
    var labels = node.src(1)
    var grad_logits = OpRef(Op(OpType.CROSS_ENTROPY_GRAD, logits.layout(), logits.dtype(), [logits, labels, upstream]))
    var dummy = OpRef(Op(OpType.BROADCAST, labels.layout(), labels.dtype(), [upstream]))
    return [grad_logits, dummy]


def softmax_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.SOFTMAX_GRAD, node.layout(), node.dtype(), [node, upstream]))]
