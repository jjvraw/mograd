from std.memory import ArcPointer
from std.hashlib.hasher import Hasher
from std.utils import Variant

from layout import IntTuple

from mograd.buffer import Buffer, AnyBuffer
from mograd.layout import Layout


# ===-------------------------------------------------------------------===#
# OpType
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct OpType(Copyable, ImplicitlyCopyable, KeyElement, Movable):
    var _name: String

    # ===-------------------------------------------------------------------===#
    # Leaf ops
    # ===-------------------------------------------------------------------===#

    comptime BUFFER = OpType("BUFFER")
    comptime ONES = OpType("ONES")
    comptime FULL = OpType("FULL")
    comptime UNIFORM = OpType("UNIFORM")
    comptime RANDN = OpType("RANDN")
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
    comptime CONTIGUOUS = OpType("CONTIGUOUS")
    comptime EXPAND = OpType("EXPAND")
    comptime ONE_HOT = OpType("ONE_HOT")
    comptime CAST = OpType("CAST")

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
comptime AttrVal = Variant[Int, Float32, Bool, String]


struct Op(Copyable, Movable, Writable):
    var op_type: OpType
    var layout: Layout
    var dtype: DType
    var srcs: List[OpRef]
    var buf: Optional[AnyBuffer]
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
        self.buf = None
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
        self.buf = buf^
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
        self.buf = None
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

    def shape(ref self, i: Int) -> Int:
        return self._ptr[].layout.shape(i)

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

    def __mul__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.MUL, self.layout().as_contiguous(), self.dtype(), [self, rhs]))

    def __truediv__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.DIV, self.layout().as_contiguous(), self.dtype(), [self, rhs]))

    def __neg__(self) -> Self:
        return Self(Op(OpType.NEG, self.layout().as_contiguous(), self.dtype(), [self]))

    def neg(self) -> Self:
        return Self(Op(OpType.NEG, self.layout().as_contiguous(), self.dtype(), [self]))

    def scale(self, scalar: Scalar) -> Self:
        return Self(Op(OpType.SCALE, self.layout().as_contiguous(), self.dtype(), [self], attrs={"scalar": scalar}))

    def exp(self) -> Self:
        return Self(Op(OpType.EXP, self.layout().as_contiguous(), self.dtype(), [self]))

    def log(self) -> Self:
        return Self(Op(OpType.LOG, self.layout().as_contiguous(), self.dtype(), [self]))

    def relu(self) -> Self:
        return Self(Op(OpType.RELU, self.layout().as_contiguous(), self.dtype(), [self]))

    def softmax(self) -> Self:
        return Self(Op(OpType.SOFTMAX, self.layout().as_contiguous(), self.dtype(), [self.contiguous()]))

    def eq(self, other: OpRef) -> Self:
        return Self(Op(OpType.EQ, self.layout().as_contiguous(), self.dtype(), [self, other]))

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> Self:
        if not axis:
            return Self(Op(OpType.SUM, (1,), self.dtype(), [self]))
        var ax = axis.value()
        var out_layout = self.layout().reduce_output_shape(ax, keepdim)
        attrs: Attrs = {"axis": ax}
        return Self(Op(OpType.SUM, out_layout, self.dtype(), [self.contiguous()], attrs=attrs^))

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
        var src = self.contiguous()
        return Self(Op(OpType.RESHAPE, src.layout().view(shape.shape()), self.dtype(), [src]))

    def view(self, shape: Layout) raises -> Self:
        assert self.layout().is_contiguous(), "Tensor.view requires contiguous layout."
        return Self(Op(OpType.VIEW, self.layout().view(shape.shape()), self.dtype(), [self]))

    def one_hot(self, var num_classes: Int, out_dtype: DType) -> OpRef:
        n = Float32(num_classes)
        return OpRef(Op(OpType.ONE_HOT, (self.shape(0), num_classes), out_dtype, [self], attrs={"num_classes": n}))

    def cast(self, out_dtype: DType) -> OpRef:
        return OpRef(Op(OpType.CAST, self.layout().as_contiguous(), out_dtype, [self]))

    def transpose(self) -> Self:
        # TODO: use self.layout().transpose() once the transpose kernel handles non-contiguous layouts
        return Self(Op(OpType.TRANSPOSE, (self.shape(1), self.shape(0)), self.dtype(), [self.contiguous()]))

    def contiguous(self) -> Self:
        if self.layout().is_contiguous():
            return self
        return Self(Op(OpType.CONTIGUOUS, self.layout().as_contiguous(), self.dtype(), [self]))

    def expand(self, layout: Layout) -> Self:
        return Self(Op(OpType.EXPAND, layout, self.dtype(), [self]))

    def slice(self, start: Int, stop: Int, step: Int = 1) raises -> Self:
        return Self(Op(OpType.SLICE, self.layout()[start:stop:step], self.dtype(), [self], {}))

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def matmul(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.MATMUL, (self.shape(0), rhs.shape(1)), self.dtype(), [self, rhs]))

    # ===-------------------------------------------------------------------===#
    # Loss operations
    # ===-------------------------------------------------------------------===#

    def cross_entropy(self, labels: Self) -> Self:
        return Self(Op(OpType.CROSS_ENTROPY, (1,), self.dtype(), [self.contiguous(), labels]))

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
        writer.write("TODO")

    #     self._write_indented(writer, 0)
    #
    # def _write_indented(self, mut writer: Some[Writer], indent: Int):
    #     var pad = String(" ") * indent
    #     writer.write(pad + self.op_type()._name + "(shape=")
    #     self.layout().write_to(writer)
    #     writer.write(", dtype=" + String(self.dtype())
    #     if len(self.srcs()) == 0:
    #         writer.write(")")
    #         return
    #     writer.write(", srcs=(\n")
    #     for i in range(len(self.srcs())):
    #         (self.srcs()[i], writer, indent + 4)
    #         writer.write("\n")
    #     writer.write(pad + "))")
    #
