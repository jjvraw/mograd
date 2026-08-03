from layout.int_tuple import IntTuple

from mograd.layout import Layout
from mograd.op import Op, OpRef, OpType
from mograd.pattern_matcher import PatternMatcher, Rule, Pat, GraphUtils
from mograd.runtime.gpu.rewrites import LAYER_NORM, LAYER_NORM_GRAD, FLASH_ATTN, FLASH_ATTN_GRAD

# ===-------------------------------------------------------------------===#
# Grad
# ===-------------------------------------------------------------------===#

comptime GradFn = def(node: OpRef, upstream: OpRef) thin raises -> List[Optional[OpRef]]


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
                Rule(Pat(OpType.SQRT), sqrt_grad),
                Rule(Pat(OpType.LOG), log_grad),
                Rule(Pat(OpType.NEG), neg_grad),
                Rule(Pat(OpType.DIV), div_grad),
                Rule(Pat(OpType.SUM), sum_grad),
                Rule(Pat(OpType.CONTIGUOUS), contiguous_grad),
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
                Rule(Pat(OpType.TRIU), triu_grad),
                Rule(Pat(OpType.EXPAND), expand_grad),
                Rule(Pat(OpType.CONCAT), concat_grad),
                Rule(Pat(OpType.GETTUPLE), gettuple_grad),
                Rule(Pat(LAYER_NORM), layer_norm_grad),
                Rule(Pat(FLASH_ATTN), flash_attn_grad),
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
                    # None marks a non-differentiable input (indices, labels, masks):
                    # no gradient edge is created, so backprop never enters that subgraph
                    if src_grads[j]:
                        grad.accum(node.srcs()[j], src_grads[j].value())

        var result = List[Optional[OpRef]]()
        for ref target in target_ops:
            result.append(grad.grad_map.get(target))
        return result^

    def accum(mut self, op: OpRef, g: OpRef) raises:
        var existing = self.grad_map.get(op)
        self.grad_map[op] = existing.value() + g if existing else g


# ===-------------------------------------------------------------------===#
# Grad functions
# ===-------------------------------------------------------------------===#


def mul_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [node.src(1) * upstream, node.src(0) * upstream]


def add_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [Optional(upstream)] * len(node.srcs())


def relu_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [OpRef(Op(OpType.RELU_GRAD, node.layout().as_contiguous(), node.dtype(), [node.src(0), upstream]))]


def exp_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [node * upstream]


def sqrt_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [(upstream / node).scale(0.5)]


def log_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [upstream / node.src(0)]


def neg_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [-upstream]


def div_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var a = node.src(0)
    var b = node.src(1)
    return [upstream / b, -(upstream * a / (b * b))]


def sum_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
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


def matmul_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var a = node.src(0)
    var b = node.src(1)
    return [upstream.matmul(b.transpose()), a.transpose().matmul(upstream)]


def transpose_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [upstream.transpose(node.attr_int("dim0"), node.attr_int("dim1"))]


def contiguous_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [upstream]


def reshape_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [upstream.reshape(node.src(0).layout())]


def view_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [OpRef(Op(OpType.VIEW, node.src(0).layout(), node.dtype(), [upstream]))]


def slice_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [
        OpRef(Op(OpType.SLICE_GRAD, node.src(0).layout().as_contiguous(), node.dtype(), [upstream.contiguous(), node]))
    ]


def concat_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var ax = node.attr_int("axis")
    var grads = List[Optional[OpRef]]()
    var offset = 0
    for i in range(len(node.srcs())):
        var size = node.src(i).layout().shape(ax)
        grads.append(upstream.slice_axis(ax, offset, offset + size))
        offset += size
    return grads^


def scale_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var scalar = Scalar(node.attrs()["scalar"][Float32])
    return [upstream.scale(scalar)]


def cross_entropy_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var logits = node.src(0)
    var labels = node.src(1)
    var grad_logits = OpRef(
        Op(OpType.CROSS_ENTROPY_GRAD, logits.layout().as_contiguous(), logits.dtype(), [logits, labels, upstream])
    )
    return [grad_logits, None]


def softmax_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    return [OpRef(Op(OpType.SOFTMAX_GRAD, node.layout().as_contiguous(), node.dtype(), [node, upstream.contiguous()]))]


def gather_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var src = node.src(0)
    var indices = node.src(1)
    var grad_src = upstream.scatter_add(indices, src.shape(0))
    return [grad_src, None]


def scatter_add_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var indices = node.src(0)
    var grad_values = upstream.gather(indices)
    return [None, grad_values]


def squeeze_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
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


def unsqueeze_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var dim = node.attr_int("dim")
    return [upstream.squeeze(dim)]


def triu_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var diagonal = node.attr_int("diagonal")
    return [upstream.triu(diagonal)]


def expand_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    var src_layout = node.src(0).layout()
    var out_layout = node.layout()
    var rank_diff = out_layout.rank() - src_layout.rank()
    var grad = upstream
    for _ in range(rank_diff):
        grad = grad.sum(0)
    for i in range(src_layout.rank()):
        if src_layout.shape(i) != out_layout.shape(i + rank_diff):
            grad = grad.sum(i, keepdim=True)
    return [grad]


def gettuple_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    # node = GETTUPLE(src, index=i). Pass upstream through to the multi-output src.
    return [upstream]


def layer_norm_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    # Emit a LAYER_NORM_GRAD node whose kernel returns [dx, dgamma, dbeta],
    # then select each output with GETTUPLE.
    var x = node.src(0)
    var gamma = node.src(1)
    var attrs = node.attrs_copy()
    var bwd = OpRef(Op(LAYER_NORM_GRAD, x.layout().as_contiguous(), x.dtype(), [upstream, x, gamma], attrs^))
    var dx = OpRef(Op(OpType.GETTUPLE, x.layout().as_contiguous(), x.dtype(), [bwd], {"index": 0}))
    var dgamma = OpRef(Op(OpType.GETTUPLE, gamma.layout().as_contiguous(), gamma.dtype(), [bwd], {"index": 1}))
    var dbeta = OpRef(Op(OpType.GETTUPLE, gamma.layout().as_contiguous(), gamma.dtype(), [bwd], {"index": 2}))
    return [dx, dgamma, dbeta]


def flash_attn_grad(node: OpRef, upstream: OpRef) raises -> List[Optional[OpRef]]:
    # node = FLASH_ATTN(Q, K, V, mask), multi-output: output 0 = O (BHSD), output 1 = LSE (BHS).
    # upstream = dO accumulated via gettuple_grad from GETTUPLE(node, 0).
    # FLASH_ATTN_GRAD takes [dO, O, Q, K, V, mask, LSE] and returns [dQ, dK, dV].
    var Q = node.src(0)
    var K = node.src(1)
    var V = node.src(2)
    var mask = node.src(3)

    # Extract O and LSE from the forward multi-output node.
    var O = OpRef(Op(OpType.GETTUPLE, node.layout(), node.dtype(), [node], {"index": 0}))
    # LSE layout: (B, H, S) row-major, float32.  Q is BSHD → B=0,S=1,H=2.
    var B = Q.layout().shape(0)
    var S = Q.layout().shape(1)
    var H = Q.layout().shape(2)
    var lse_shape = IntTuple()
    lse_shape.append(IntTuple(B))
    lse_shape.append(IntTuple(H))
    lse_shape.append(IntTuple(S))
    var lse_strides = IntTuple()
    lse_strides.append(IntTuple(H * S))
    lse_strides.append(IntTuple(S))
    lse_strides.append(IntTuple(1))
    var lse_layout = Layout(3, lse_shape, lse_strides, 0)
    var LSE = OpRef(Op(OpType.GETTUPLE, lse_layout, DType.float32, [node], {"index": 1}))

    var attrs = node.attrs_copy()
    # Upstream (dO) may be a zero-stride broadcast EXPAND.
    # The FLASH_ATTN_GRAD kernel reads it with raw pointer offsets and would read garbage memory past
    # the 1-element scalar buffer without materialisation first.
    var dO = upstream.contiguous()
    var bwd = OpRef(Op(FLASH_ATTN_GRAD, Q.layout().as_contiguous(), Q.dtype(), [dO, O, Q, K, V, mask, LSE], attrs^))
    var dQ = OpRef(Op(OpType.GETTUPLE, Q.layout().as_contiguous(), Q.dtype(), [bwd], {"index": 0}))
    var dK = OpRef(Op(OpType.GETTUPLE, K.layout().as_contiguous(), K.dtype(), [bwd], {"index": 1}))
    var dV = OpRef(Op(OpType.GETTUPLE, V.layout().as_contiguous(), V.dtype(), [bwd], {"index": 2}))
    return [dQ, dK, dV, None]
