from std.math import sqrt
from std.utils.numerics import neg_inf
from std.memory import ArcPointer
from std.hashlib.hasher import Hasher
from std.sys.info import has_apple_gpu_accelerator
from std.utils import Variant

from layout import IntTuple

from mograd.buffer import Buffer, AnyBuffer
from mograd.layout import Layout
from mograd.pattern_matcher import GraphUtils


# ===-------------------------------------------------------------------===#
# OpType
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct OpType(Copyable, ImplicitlyCopyable, KeyElement, Movable):
    var _name: String

    comptime SINK = OpType("SINK")
    """Void bundling node for evaluating multiple targets in one simplify/schedule pass."""

    comptime GETTUPLE = OpType("GETTUPLE")
    """Selects the i-th buffer from a multi-output node (index stored in attrs["index"])."""

    # ===-------------------------------------------------------------------===#
    # Leaf ops
    # ===-------------------------------------------------------------------===#

    comptime BUFFER = OpType("BUFFER")
    comptime ONES = OpType("ONES")
    comptime FULL = OpType("FULL")
    comptime UNIFORM = OpType("UNIFORM")
    comptime RANDN = OpType("RANDN")
    comptime RANDINT = OpType("RANDINT")
    comptime DISK = OpType("DISK")

    # ===-------------------------------------------------------------------===#
    # Elementwise ops
    # ===-------------------------------------------------------------------===#

    comptime ADD = OpType("ADD")
    comptime MUL = OpType("MUL")
    comptime DIV = OpType("DIV")
    comptime NEG = OpType("NEG")
    comptime SCALE = OpType("SCALE")
    comptime EXP = OpType("EXP")
    comptime SQRT = OpType("SQRT")
    comptime LOG = OpType("LOG")
    comptime RELU = OpType("RELU")
    comptime RELU_GRAD = OpType("RELU_GRAD")
    comptime SOFTMAX = OpType("SOFTMAX")
    comptime SOFTMAX_GRAD = OpType("SOFTMAX_GRAD")
    comptime EQ = OpType("EQ")

    # ===-------------------------------------------------------------------===#
    # Reduction ops
    # ===-------------------------------------------------------------------===#

    comptime SUM = OpType("SUM")
    comptime ARGMAX = OpType("ARGMAX")

    # ===-------------------------------------------------------------------===#
    # Layout ops
    # ===-------------------------------------------------------------------===#

    comptime RESHAPE = OpType("RESHAPE")
    comptime VIEW = OpType("VIEW")
    comptime TRANSPOSE = OpType("TRANSPOSE")
    comptime SLICE = OpType("SLICE")
    comptime SLICE_GRAD = OpType("SLICE_GRAD")
    comptime CONCAT = OpType("CONCAT")
    comptime CONTIGUOUS = OpType("CONTIGUOUS")
    comptime EXPAND = OpType("EXPAND")
    comptime ONE_HOT = OpType("ONE_HOT")
    comptime CAST = OpType("CAST")
    comptime GATHER = OpType("GATHER")
    comptime SCATTER_ADD = OpType("SCATTER_ADD")
    comptime SQUEEZE = OpType("SQUEEZE")
    comptime UNSQUEEZE = OpType("UNSQUEEZE")
    comptime TRIU = OpType("TRIU")

    # ===-------------------------------------------------------------------===#
    # Contraction ops
    # ===-------------------------------------------------------------------===#

    comptime MATMUL = OpType("MATMUL")

    # ===-------------------------------------------------------------------===#
    # Loss ops
    # ===-------------------------------------------------------------------===#

    comptime CROSS_ENTROPY = OpType("CROSS_ENTROPY")
    comptime CROSS_ENTROPY_GRAD = OpType("CROSS_ENTROPY_GRAD")

    # ===-------------------------------------------------------------------===#
    # Trait methods
    # ===-------------------------------------------------------------------===#

    def __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self._name)

    def __eq__(self, other: Self) -> Bool:
        return self._name == other._name

    def __ne__(self, other: Self) -> Bool:
        return self._name != other._name


# ===-------------------------------------------------------------------===#
# Op
# ===-------------------------------------------------------------------===#


comptime Attrs = Dict[String, AttrVal]
# TODO: Casting rules
comptime AttrVal = Variant[Int, Float32, Bool, String]


struct Op(Copyable, Movable, Writable):
    var op_type: OpType
    var layout: Layout
    var dtype: DType
    var srcs: List[OpRef]
    var buf: List[AnyBuffer]
    var attrs: Attrs

    def __init__(
        out self,
        op_type: OpType,
        shape: Layout,
        dtype: DType,
        var srcs: List[OpRef],
    ):
        self.op_type = op_type
        self.layout = shape
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = List[AnyBuffer]()
        self.attrs = {}

    def __init__(
        out self,
        op_type: OpType,
        shape: Layout,
        dtype: DType,
        var srcs: List[OpRef],
        var buf: AnyBuffer,
    ):
        self.op_type = op_type
        self.layout = shape
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = [buf^]
        self.attrs = {}

    def __init__(
        out self,
        op_type: OpType,
        shape: Layout,
        dtype: DType,
        var srcs: List[OpRef],
        var attrs: Attrs,
    ):
        self.op_type = op_type
        self.layout = shape
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = List[AnyBuffer]()
        self.attrs = attrs^

    def __str__(self) -> String:
        return self.op_type._name

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.op_type._name)


# ===-------------------------------------------------------------------===#
# OpRef
# ===-------------------------------------------------------------------===#


struct OpRef(Copyable, ImplicitlyCopyable, KeyElement, Movable, Writable):
    var _ptr: ArcPointer[Op]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @implicit
    def __init__(out self, var op: Op):
        self._ptr = ArcPointer(op^)

    # ===-------------------------------------------------------------------===#
    # Accessors
    # ===-------------------------------------------------------------------===#

    def op(ref self) -> ref[self._ptr[]] Op:
        return self._ptr[]

    def layout(ref self) -> ref[self._ptr[].layout] Layout:
        return self._ptr[].layout

    def shape(ref self) -> IntTuple:
        return self._ptr[].layout.shape()

    def shape(ref self, i: Int) -> Int:
        return self._ptr[].layout.shape(i)

    def numel(ref self) -> Int:
        return self._ptr[].layout.numel()

    def dtype(ref self) -> ref[self._ptr[].dtype] DType:
        return self._ptr[].dtype

    def op_type(ref self) -> ref[self._ptr[].op_type] OpType:
        return self._ptr[].op_type

    def srcs(ref self) -> ref[self._ptr[].srcs] List[OpRef]:
        return self._ptr[].srcs

    def src(ref self, i: Int) -> ref[self._ptr[].srcs[i]] OpRef:
        return self._ptr[].srcs[i]

    def attrs(ref self) -> ref[self._ptr[].attrs] Attrs:
        return self._ptr[].attrs

    def attr[T: DType = DType.float32](ref self, key: String) raises -> Scalar[T]:
        return self._ptr[].attrs[key][Float32].cast[T]()

    def attr_int(ref self, key: String) raises -> Int:
        return self._ptr[].attrs[key][Int]

    def attrs_copy(self) -> Attrs:
        return self._ptr[].attrs.copy()

    # ===-------------------------------------------------------------------===#
    # Elementwise operations
    # ===-------------------------------------------------------------------===#

    def __add__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.ADD, self.layout().as_contiguous(), self.dtype(), [self, rhs]))

    def add(self, rhs: OpRef) -> Self:
        return self + rhs

    def __mul__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.MUL, self.layout().as_contiguous(), self.dtype(), [self, rhs]))

    def __truediv__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.DIV, self.layout().as_contiguous(), self.dtype(), [self, rhs]))

    def __neg__(self) -> Self:
        return Self(Op(OpType.NEG, self.layout().as_contiguous(), self.dtype(), [self]))

    def neg(self) -> Self:
        return Self(Op(OpType.NEG, self.layout().as_contiguous(), self.dtype(), [self]))

    def scale(self, scalar: Float32) -> Self:
        return Self(Op(OpType.SCALE, self.layout().as_contiguous(), self.dtype(), [self], attrs={"scalar": scalar}))

    def exp(self) -> Self:
        return Self(Op(OpType.EXP, self.layout().as_contiguous(), self.dtype(), [self]))

    def sqrt(self) -> Self:
        return Self(Op(OpType.SQRT, self.layout().as_contiguous(), self.dtype(), [self]))

    def log(self) -> Self:
        return Self(Op(OpType.LOG, self.layout().as_contiguous(), self.dtype(), [self]))

    def relu(self) -> Self:
        return Self(Op(OpType.RELU, self.layout().as_contiguous(), self.dtype(), [self]))

    def softmax(self) -> Self:
        return Self(Op(OpType.SOFTMAX, self.layout().as_contiguous(), self.dtype(), [self]))

    def eq(self, other: OpRef) -> Self:
        return Self(Op(OpType.EQ, self.layout().as_contiguous(), self.dtype(), [self, other]))

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> Self:
        if not axis:
            return Self(Op(OpType.SUM, (1,), self.dtype(), [self]))
        var ax = axis.value()
        var out_layout = self.layout().reduce_output_shape(ax, keepdim).as_contiguous()
        attrs: Attrs = {"axis": ax}
        return Self(Op(OpType.SUM, out_layout, self.dtype(), [self], attrs=attrs^))

    def argmax(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> Self:
        if not axis:
            return Self(Op(OpType.ARGMAX, (1,), self.dtype(), [self]))
        var ax = axis.value()
        var out_layout = self.layout().reduce_output_shape(ax, keepdim)
        attrs: Attrs = {"axis": ax}
        return Self(Op(OpType.ARGMAX, out_layout, self.dtype(), [self], attrs=attrs^))

    # ===-------------------------------------------------------------------===#
    # Shape operations
    # ===-------------------------------------------------------------------===#

    def reshape(self, shape: Layout) raises -> Self:
        try:
            return Self(Op(OpType.RESHAPE, self.layout().view(shape.shape()), self.dtype(), [self]))
        except:
            var src = self.contiguous()
            return Self(Op(OpType.RESHAPE, src.layout().view(shape.shape()), self.dtype(), [src]))

    def view(self, shape: Layout) raises -> Self:
        assert self.layout().is_contiguous(), "Tensor.view requires contiguous layout."
        return Self(Op(OpType.VIEW, self.layout().view(shape.shape()), self.dtype(), [self]))

    def flatten(self, start_dim: Int = 0, end_dim: Int = -1) raises -> Self:
        try:
            var flattened_layout = self.layout().flatten(start_dim, end_dim)
            return Self(Op(OpType.RESHAPE, flattened_layout, self.dtype(), [self]))
        except:
            var src = self.contiguous()
            var flattened_layout = src.layout().flatten(start_dim, end_dim)
            return Self(Op(OpType.RESHAPE, flattened_layout, self.dtype(), [src]))

    def one_hot(self, var num_classes: Int, out_dtype: DType) -> OpRef:
        n = Float32(num_classes)
        return OpRef(Op(OpType.ONE_HOT, (self.shape(0), num_classes), out_dtype, [self], attrs={"num_classes": n}))

    def cast(self, out_dtype: DType) -> OpRef:
        return OpRef(Op(OpType.CAST, self.layout().as_contiguous(), out_dtype, [self]))

    def gather(self, indices: OpRef) raises -> OpRef:
        if self.layout().rank() != 2:
            raise Error("gather expects a 2D source tensor [num_rows, row_size]")
        var out_shape = indices.shape()
        out_shape.append(IntTuple(self.shape(1)))
        var out_layout = Layout(len(out_shape), out_shape, Layout.row_major_strides(out_shape), 0)
        return OpRef(Op(OpType.GATHER, out_layout, self.dtype(), [self, indices]))

    def scatter_add(self, indices: OpRef, num_rows: Int) raises -> OpRef:
        if self.layout().rank() != indices.layout().rank() + 1:
            raise Error("scatter_add expects values shaped [*indices.shape, row_size]")
        var row_size = self.shape(self.layout().rank() - 1)
        comptime if has_apple_gpu_accelerator():
            if self.dtype() != DType.float32:
                raise Error("scatter_add: only float32 is supported on Apple GPU (no native non-f32 atomic add)")
        return OpRef(Op(OpType.SCATTER_ADD, Layout(num_rows, row_size), self.dtype(), [indices, self]))

    def transpose(self, dim0: Int = -2, dim1: Int = -1) raises -> Self:
        var rank = self.layout().rank()
        var d0 = dim0 if dim0 >= 0 else rank + dim0
        var d1 = dim1 if dim1 >= 0 else rank + dim1
        attrs: Attrs = {"dim0": d0, "dim1": d1}
        return Self(Op(OpType.TRANSPOSE, self.layout().transpose(dim0, dim1), self.dtype(), [self], attrs^))

    def zeros_like(self) -> Self:
        return Self(Op(OpType.FULL, self.layout().as_contiguous(), self.dtype(), [], {"value": AttrVal(Float32(0.0))}))

    def contiguous(self) -> Self:
        if self.layout().is_contiguous():
            return self
        return Self(Op(OpType.CONTIGUOUS, self.layout().as_contiguous(), self.dtype(), [self]))

    def expand(self, *shape: Int) raises -> Self:
        return self.expand(IntTuple(*shape))

    def expand(self, shape: IntTuple) raises -> Self:
        return Self(Op(OpType.EXPAND, self.layout().expand(shape), self.dtype(), [self]))

    def slice(self, start: Int, stop: Int, step: Int = 1) raises -> Self:
        return self.slice_axis(0, start, stop, step)

    def slice_axis(self, axis: Int, start: Int, stop: Int, step: Int = 1) raises -> Self:
        return Self(Op(OpType.SLICE, self.layout().slice_axis(axis, start, stop, step), self.dtype(), [self], {}))

    def squeeze(self, dim: Optional[Int] = None) raises -> Self:
        if dim:
            var d = self.layout().normalise_dim(dim.value())
            var new_layout = self.layout().squeeze(d)
            attrs: Attrs = {"dim": d}
            return Self(Op(OpType.SQUEEZE, new_layout, self.dtype(), [self], attrs=attrs^))
        else:
            return Self(Op(OpType.SQUEEZE, self.layout().squeeze_all(), self.dtype(), [self]))

    def unsqueeze(self, dim: Int) raises -> Self:
        var new_layout = self.layout().unsqueeze(dim)
        attrs: Attrs = {"dim": dim}
        return Self(Op(OpType.UNSQUEEZE, new_layout, self.dtype(), [self], attrs=attrs^))

    def triu(self, diagonal: Int = 0) raises -> Self:
        if self.layout().rank() < 2:
            raise Error("triu requires a tensor of rank >= 2")
        attrs: Attrs = {"diagonal": diagonal}
        return Self(Op(OpType.TRIU, self.layout().as_contiguous(), self.dtype(), [self], attrs=attrs^))

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def matmul(self, rhs: OpRef) raises -> Self:
        var la = self.layout()
        var lb = rhs.layout()
        var rank = la.rank()

        if rank != lb.rank():
            raise Error(t"matmul: rank mismatch {String(rank)} vs {String(lb.rank())}")
        if rank < 2:
            raise Error("matmul requires tensors of rank >= 2")

        var out_shape = IntTuple()
        for i in range(rank - 2):
            var a_dim = la.shape(i)
            var b_dim = lb.shape(i)
            if a_dim != b_dim:
                raise Error(t"matmul: batch dim {String(i)} mismatch {String(a_dim)} vs {String(b_dim)}")
            out_shape.append(IntTuple(a_dim))
        out_shape.append(IntTuple(self.shape(rank - 2)))
        out_shape.append(IntTuple(rhs.shape(rank - 1)))

        var out_layout = Layout(rank, out_shape, Layout.row_major_strides(out_shape), 0)

        # Batch dims that don't nest into a single flat stride can't be expressed
        # as a strided view for the matmul kernels, so materialise first.
        var a_src = self if la.batch_dims_collapsible() else self.contiguous()
        var b_src = rhs if lb.batch_dims_collapsible() else rhs.contiguous()

        return Self(Op(OpType.MATMUL, out_layout, self.dtype(), [a_src, b_src]))

    def scaled_dot_product_attention(
        self,
        key: Self,
        value: Self,
        attn_mask: Optional[Self],
        is_causal: Bool = False,
        scale: Optional[Float32] = None,
    ) raises -> Self:
        if is_causal and attn_mask:
            raise Error("scaled_dot_product_attention: attn_mask cannot be combined with is_causal=True")
        # Promote 3D (B, T, D) → 4D (B, 1, T, D) so fusion sees the expected BHTD layout.
        var rank = self.layout().rank()
        var q = (
            self.reshape(Layout(self.layout().shape(0), 1, self.layout().shape(1), self.layout().shape(2))) if rank
            == 3 else self
        )
        var k = (
            key.reshape(Layout(key.layout().shape(0), 1, key.layout().shape(1), key.layout().shape(2))) if rank
            == 3 else key
        )
        var v = (
            value.reshape(Layout(value.layout().shape(0), 1, value.layout().shape(1), value.layout().shape(2))) if rank
            == 3 else value
        )

        var scale_factor = scale.value() if scale else Float32(1.0) / sqrt(Float32(q.layout().size(-1)))
        var scores = q.matmul(k.transpose(-2, -1)).scale(scale_factor)
        var out: Self
        if is_causal:
            var causal_mask = Self(
                Op(
                    OpType.FULL,
                    scores.layout().as_contiguous(),
                    scores.dtype(),
                    [],
                    {"value": AttrVal(neg_inf[DType.float32]())},
                )
            ).triu(1)
            out = scores.add(causal_mask).softmax().matmul(v)
        elif attn_mask:
            out = scores.add(attn_mask.value()).softmax().matmul(v)
        else:
            out = scores.softmax().matmul(v)

        # Squeeze back to 3D if input was 3D.
        if rank == 3:
            var s = out.layout()
            return out.reshape(Layout(s.shape(0), s.shape(2), s.shape(3)))
        return out

    # ===-------------------------------------------------------------------===#
    # Loss operations
    # ===-------------------------------------------------------------------===#

    def cross_entropy(self, labels: Self) -> Self:
        return Self(Op(OpType.CROSS_ENTROPY, (1,), self.dtype(), [self, labels]))

    # ===-------------------------------------------------------------------===#
    # Trait methods
    # ===-------------------------------------------------------------------===#

    def __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(Int(self._ptr.unsafe_ptr()))

    def __eq__(self, other: Self) -> Bool:
        return self._ptr.unsafe_ptr() == other._ptr.unsafe_ptr()

    def __ne__(self, other: Self) -> Bool:
        return self._ptr.unsafe_ptr() != other._ptr.unsafe_ptr()

    def write_to(self, mut writer: Some[Writer]):
        var cache = Dict[Self, _NodeEntry]()
        var topo = GraphUtils.toposort(self)
        for i in range(len(topo)):
            for j in range(len(topo[i].srcs())):
                var s = topo[i].src(j)
                var e = cache.get(s)
                if e:
                    var entry = e.value()
                    entry.refs += 1
                    cache[s] = entry
                else:
                    cache[s] = _NodeEntry(1, -1, False)
        var next_alias = 0
        self._render(writer, 0, cache, next_alias)

    def _render(self, mut writer: Some[Writer], indent: Int, mut cache: Dict[Self, _NodeEntry], mut next_alias: Int):
        var pad = String("")
        for _ in range(indent * 2):
            pad += " "

        var e = cache.get(self)
        if e and e.value().printed:
            writer.write(pad, "x", e.value().idx)
            return

        var entry = e.value() if e else _NodeEntry(0, -1, False)
        if entry.refs > 1 and entry.idx == -1:
            entry.idx = next_alias
            next_alias += 1
        entry.printed = True
        cache[self] = entry

        var prefix = "x" + String(entry.idx) + ":=" if entry.refs > 1 else ""
        writer.write(pad, prefix, "Op(", self.op_type()._name, ", ", self.dtype(), ", shape=(")
        for i in range(self.layout().rank()):
            if i > 0:
                writer.write(",")
            writer.write(self.layout().shape(i))
        writer.write(")")
        if len(self.srcs()) == 0:
            writer.write(", src=())")
            return
        writer.write(", src=(")
        for i in range(len(self.srcs())):
            if i > 0:
                writer.write(",")
            writer.write("\n")
            self.src(i)._render(writer, indent + 1, cache, next_alias)
        writer.write("\n", pad, "))")


def concat(var tensors: List[OpRef], axis: Int) raises -> OpRef:
    var layouts = List[Layout]()
    for i in range(len(tensors)):
        layouts.append(tensors[i].layout())
    var out_layout = Layout.concat(layouts, axis)
    var ax = tensors[0].layout().normalise_dim(axis)
    var dtype = tensors[0].dtype()
    attrs: Attrs = {"axis": ax}
    return OpRef(Op(OpType.CONCAT, out_layout, dtype, tensors^, attrs^))


def sink(var targets: List[OpRef]) -> OpRef:
    """Bundles `targets` into one void SINK node so they can be simplified
    and scheduled together. The SINK's own layout/dtype are never used;
    after running, read back `result.srcs()[i]` for each target's outcome.
    """
    return OpRef(Op(OpType.SINK, Layout(1), DType.float32, targets^))


@fieldwise_init
struct _NodeEntry(Copyable, ImplicitlyCopyable, Movable):
    var refs: Int
    var idx: Int  # -1 until first render, then sequential alias number
    var printed: Bool
