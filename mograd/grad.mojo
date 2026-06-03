from mograd.op import Op, OpRef, OpType, AnyOpRef
from mograd.pattern_matcher import PatternMatcher, Rule, Pat, GraphUtils

# ===-------------------------------------------------------------------===#
# Grad
# ===-------------------------------------------------------------------===#

comptime GradFn[dtype: DType] where dtype.is_floating_point() = def(
    node: OpRef[dtype], upstream: OpRef[dtype]
) thin raises -> List[OpRef[dtype]]


struct Grad[dtype: DType] where dtype.is_floating_point():
    var grad_map: Dict[OpRef[Self.dtype], OpRef[Self.dtype]]

    def __init__(out self):
        self.grad_map = Dict[OpRef[Self.dtype], OpRef[Self.dtype]]()

    @staticmethod
    def compute(
        root: OpRef[Self.dtype],
        initial_grad: OpRef[Self.dtype],
        target_ops: List[OpRef[Self.dtype]],
    ) raises -> List[Optional[OpRef[Self.dtype]]]:
        var grad = Grad[Self.dtype]()
        grad.grad_map[root] = initial_grad

        var pm = PatternMatcher[
            GradFn[Self.dtype],
            [
                Rule(Pat(OpType.MUL), mul_grad[Self.dtype]),
                Rule(Pat(OpType.ADD), add_grad[Self.dtype]),
                Rule(Pat(OpType.RELU), relu_grad[Self.dtype]),
                Rule(Pat(OpType.SOFTMAX), softmax_grad[Self.dtype]),
                Rule(Pat(OpType.EXP), exp_grad[Self.dtype]),
                Rule(Pat(OpType.LOG), log_grad[Self.dtype]),
                Rule(Pat(OpType.NEG), neg_grad[Self.dtype]),
                Rule(Pat(OpType.DIV), div_grad[Self.dtype]),
                Rule(Pat(OpType.SUM), sum_grad[Self.dtype]),
                Rule(Pat(OpType.RESHAPE), reshape_grad[Self.dtype]),
                Rule(Pat(OpType.MATMUL), matmul_grad[Self.dtype]),
                Rule(Pat(OpType.CROSS_ENTROPY), cross_entropy_grad[Self.dtype]),
                Rule(Pat(OpType.SCALE), scale_grad[Self.dtype]),
                Rule(Pat(OpType.TRANSPOSE), transpose_grad[Self.dtype]),
            ],
        ]()

        var topo = GraphUtils.toposort[Self.dtype](root)
        for i in reversed(range(len(topo))):
            var node = topo[i]
            var upstream = grad.grad_map.get(node)
            if not upstream:
                continue

            var up = upstream.value()
            var rule = pm.match(AnyOpRef(node))
            if rule:
                var src_grads = rule.value()(node, up)
                for j in range(len(node.srcs())):
                    if j < len(node.srcs()) and node.srcs()[j].isa[OpRef[Self.dtype]]():
                        grad.accum(node.srcs()[j].unsafe_get[OpRef[Self.dtype]](), src_grads[j])

        var result = List[Optional[OpRef[Self.dtype]]]()
        for i in range(len(target_ops)):
            result.append(grad.grad_map.get(target_ops[i]))
        return result^

    def accum(mut self, op: OpRef[Self.dtype], g: OpRef[Self.dtype]) raises:
        var existing = self.grad_map.get(op)
        self.grad_map[op] = existing.value() + g if existing else g


# ===-------------------------------------------------------------------===#
# Grad functions
# ===-------------------------------------------------------------------===#


def mul_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [node.src(1) * upstream, node.src(0) * upstream]


def add_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [upstream] * len(node.srcs())


def relu_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [OpRef[dtype](OpType.RELU_GRAD, node.shape(), [node.src(0), upstream])]


def exp_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [node * upstream]


def log_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [upstream / node.src(0)]


def neg_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [-upstream]


def div_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    var a = node.src(0)
    var b = node.src(1)
    return [upstream / b, -(upstream * a / (b * b))]


def sum_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [OpRef[dtype](OpType.BROADCAST, node.src(0).shape(), [upstream])]


def matmul_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    var a = node.src(0)
    var b = node.src(1)
    return [upstream.matmul(b.transpose()), a.transpose().matmul(upstream)]


def transpose_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [upstream.transpose()]


def reshape_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [OpRef[dtype](OpType.RESHAPE, node.src(0).shape(), [upstream])]


def scale_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    var scalar = Scalar[dtype](node.attrs()["scalar"][Float32])
    return [upstream.scale(scalar)]


def cross_entropy_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    var logits = node.src(0)
    var labels = node.srcs()[1]  # AnyOpRef & integral
    var grad_logits = OpRef[dtype](OpType.CROSS_ENTROPY_GRAD, logits.shape(), [logits, labels, upstream])
    var dummy = OpRef[dtype](OpType.BROADCAST, logits.shape(), [upstream])
    return [grad_logits, dummy]


def softmax_grad[dtype: DType](node: OpRef[dtype], upstream: OpRef[dtype]) raises -> List[OpRef[dtype]]:
    return [OpRef[dtype](OpType.SOFTMAX_GRAD, node.shape(), [node, upstream])]
