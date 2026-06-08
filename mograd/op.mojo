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
    # Pointwise ops
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
    comptime TRANSPOSE = OpType("TRANSPOSE")
    comptime SLICE = OpType("SLICE")
    comptime BROADCAST = OpType("BROADCAST")
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


comptime AttrVal = Variant[Float32, String]


struct Op(Copyable, Movable, Writable):
    var op_type: OpType
    var layout: Layout
    var dtype: DType
    var srcs: List[OpRef]
    var buf: Optional[AnyBuffer]
    var attrs: Dict[String, AttrVal]

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
        var attrs: Dict[String, AttrVal],
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
        return self._ptr[].layout.shape[i].value()

    def dtype(ref self) -> ref[self._ptr[].dtype] DType:
        return self._ptr[].dtype

    def op_type(ref self) -> ref[self._ptr[].op_type] OpType:
        return self._ptr[].op_type

    def srcs(ref self) -> ref[self._ptr[].srcs] List[OpRef]:
        return self._ptr[].srcs

    def src(ref self, i: Int) -> ref[self._ptr[].srcs[i]] OpRef:
        return self._ptr[].srcs[i]

    def attrs(ref self) -> ref[self._ptr[].attrs] Dict[String, AttrVal]:
        return self._ptr[].attrs

    def attrs_copy(self) -> Dict[String, AttrVal]:
        return self._ptr[].attrs.copy()

    # ===-------------------------------------------------------------------===#
    # Pointwise operations
    # ===-------------------------------------------------------------------===#

    def __add__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.ADD, self.layout(), self.dtype(), [self, rhs]))

    def __mul__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.MUL, self.layout(), self.dtype(), [self, rhs]))

    def __truediv__(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.DIV, self.layout(), self.dtype(), [self, rhs]))

    def __neg__(self) -> Self:
        return Self(Op(OpType.NEG, self.layout(), self.dtype(), [self]))

    def neg(self) -> Self:
        return Self(Op(OpType.NEG, self.layout(), self.dtype(), [self]))

    def scale(self, scalar: Scalar) -> Self:
        var attrs: Dict[String, AttrVal] = {"scalar": AttrVal(scalar)}
        return Self(Op(OpType.SCALE, self.layout(), self.dtype(), [self], attrs=attrs^))

    def exp(self) -> Self:
        return Self(Op(OpType.EXP, self.layout(), self.dtype(), [self]))

    def log(self) -> Self:
        return Self(Op(OpType.LOG, self.layout(), self.dtype(), [self]))

    def relu(self) -> Self:
        return Self(Op(OpType.RELU, self.layout(), self.dtype(), [self]))

    def softmax(self) -> Self:
        return Self(Op(OpType.SOFTMAX, self.layout(), self.dtype(), [self]))

    def eq(self, other: OpRef) -> Self:
        return Self(Op(OpType.EQ, self.layout(), self.dtype(), [self, other]))

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self) -> Self:
        return Self(Op(OpType.SUM, (1,), self.dtype(), [self]))

    def argmax(self) -> Self:
        return Self(Op(OpType.ARGMAX, (self.shape(0),), self.dtype(), [self]))

    # ===-------------------------------------------------------------------===#
    # Shape operations
    # ===-------------------------------------------------------------------===#

    def reshape(self, shape: Layout) raises -> Self:
        return Self(Op(OpType.RESHAPE, self.layout().view(shape.shape), self.dtype(), [self]))

    def one_hot(self, num_classes: Int, out_dtype: DType) -> OpRef:
        var attrs: Dict[String, AttrVal] = {"num_classes": AttrVal(Float32(num_classes))}
        return OpRef(Op(OpType.ONE_HOT, (self.shape(0), num_classes), out_dtype, [self], attrs=attrs^))

    def cast(self, out_dtype: DType) -> OpRef:
        return OpRef(Op(OpType.CAST, self.layout(), out_dtype, [self]))

    def transpose(self) -> Self:
        return Self(Op(OpType.TRANSPOSE, (self.shape(1), self.shape(0)), self.dtype(), [self]))

    def slice(self, start: Int, stop: Int, step: Int = 1) raises -> Self:
        var attrs: Dict[String, AttrVal] = {
            "start": AttrVal(Float32(start)),
            "stop": AttrVal(Float32(stop)),
            "step": AttrVal(Float32(step)),
        }

        return Self(Op(OpType.SLICE, self.layout()[start:stop:step], self.dtype(), [self], attrs=attrs^))

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def matmul(self, rhs: OpRef) -> Self:
        return Self(Op(OpType.MATMUL, (self.shape(0), rhs.shape(1)), self.dtype(), [self, rhs]))

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
