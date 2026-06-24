from layout.int_tuple import IntTuple

from mograd.layout import Layout
from mograd.op import Op, OpRef, OpType
from mograd.pattern_matcher import PatternMatcher, Rule, Pat, GraphUtils

# ===-------------------------------------------------------------------===#
# Grad
# ===-------------------------------------------------------------------===#

comptime GradFn = def(node: OpRef, upstream: OpRef) thin raises -> List[OpRef]


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
                Rule(Pat(OpType.GATHER), gather_grad),
                Rule(Pat(OpType.SCATTER_ADD), scatter_add_grad),
                Rule(Pat(OpType.SQUEEZE), squeeze_grad),
                Rule(Pat(OpType.UNSQUEEZE), unsqueeze_grad),
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
                var src_grads = rule.value().func(node, up)
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
    return [OpRef(Op(OpType.RELU_GRAD, node.layout().as_contiguous(), node.dtype(), [node.src(0), upstream]))]


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
    var in_layout = node.src(0).layout().as_contiguous()
    if "axis" in node.attrs():
        var ax = node.attr_int("axis")
        var up_layout = upstream.layout()
        var bcast_layout: Layout
        if up_layout.rank() == in_layout.rank():
            # keepdim=True: upstream has size-1 at axis, expand it with stride 0
            var new_strides = IntTuple()
            for i in range(in_layout.rank()):
                new_strides.append(IntTuple(0 if i == ax else up_layout._strides.value(i)))
            bcast_layout = Layout(in_layout.rank(), in_layout.shape(), new_strides, 0)
        else:
            # keepdim=False: upstream is missing the axis dim, insert it with stride 0
            bcast_layout = up_layout.expand_axis(ax, in_layout.shape(ax))
        return [OpRef(Op(OpType.EXPAND, bcast_layout, node.dtype(), [upstream]))]
    # Full reduce: all strides zero so every output element reads upstream[0]
    var zero_strides = IntTuple()
    for _ in range(in_layout.rank()):
        zero_strides.append(IntTuple(0))
    var bcast_layout = Layout(in_layout.rank(), in_layout.shape(), zero_strides, 0)
    return [OpRef(Op(OpType.EXPAND, bcast_layout, node.dtype(), [upstream]))]


def matmul_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var a = node.src(0)
    var b = node.src(1)
    return [upstream.matmul(b.transpose()), a.transpose().matmul(upstream)]


def transpose_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [upstream.transpose()]


def reshape_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.RESHAPE, upstream.layout().view(node.src(0).layout().shape()), node.dtype(), [upstream]))]


def view_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.VIEW, node.src(0).layout(), node.dtype(), [upstream]))]


def slice_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [
        OpRef(Op(OpType.SLICE_GRAD, node.src(0).layout().as_contiguous(), node.dtype(), [upstream.contiguous(), node]))
    ]


def scale_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var scalar = Scalar(node.attrs()["scalar"][Float32])
    return [upstream.scale(scalar)]


def cross_entropy_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var logits = node.src(0)
    var labels = node.src(1)
    var grad_logits = OpRef(
        Op(OpType.CROSS_ENTROPY_GRAD, logits.layout().as_contiguous(), logits.dtype(), [logits, labels, upstream])
    )
    var dummy = OpRef(Op(OpType.EXPAND, labels.layout(), labels.dtype(), [upstream]))
    return [grad_logits, dummy]


def softmax_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    return [OpRef(Op(OpType.SOFTMAX_GRAD, node.layout().as_contiguous(), node.dtype(), [node, upstream.contiguous()]))]


def gather_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var src = node.src(0)
    var indices = node.src(1)
    var grad_src = upstream.scatter_add(indices, src.shape(0))
    var dummy = OpRef(Op(OpType.EXPAND, indices.layout(), indices.dtype(), [upstream]))
    return [grad_src, dummy]


def scatter_add_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var indices = node.src(0)
    var dummy = OpRef(Op(OpType.EXPAND, indices.layout(), indices.dtype(), [upstream]))
    var grad_values = upstream.gather(indices)
    return [dummy, grad_values]


def squeeze_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    if "dim" in node.attrs():
        var dim = node.attr_int("dim")
        return [upstream.unsqueeze(dim)]
    else:
        # squeeze_all: reconstruct original shape by finding where dims were size 1
        var in_layout = node.src(0).layout()
        var expanded = upstream
        for i in range(in_layout.rank()):
            if in_layout.shape(i) == 1:
                expanded = expanded.unsqueeze(i)
        return [expanded]


def unsqueeze_grad(node: OpRef, upstream: OpRef) raises -> List[OpRef]:
    var dim = node.attr_int("dim")
    return [upstream.squeeze(dim)]
