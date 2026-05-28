from std.memory import ArcPointer
from std.hashlib.hasher import Hasher
from std.utils import Variant

from mograd.buffer import Buffer
from mograd.shape import Shape

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
    comptime SUM_GRAD = OpType("SUM_GRAD")
    comptime ARGMAX = OpType("ARGMAX")

    # ===-------------------------------------------------------------------===#
    # Shape ops
    # ===-------------------------------------------------------------------===#

    comptime RESHAPE = OpType("RESHAPE")
    comptime TRANSPOSE = OpType("TRANSPOSE")
    comptime SLICE = OpType("SLICE")

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
    var shape: Shape
    var dtype: DType
    var srcs: List[OpRef]
    var buf: Optional[Buffer]
    var attrs: Dict[String, AttrVal]

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        dtype: DType,
        var srcs: List[OpRef],
    ):
        self.op_type = op_type
        self.shape = shape
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = Optional[Buffer](None)
        self.attrs = {}

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        dtype: DType,
        var srcs: List[OpRef],
        var buf: Optional[Buffer],
    ):
        self.op_type = op_type
        self.shape = shape
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = buf^
        self.attrs = {}

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        dtype: DType,
        var srcs: List[OpRef],
        var attrs: Dict[String, AttrVal],
    ):
        self.op_type = op_type
        self.shape = shape
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = Optional[Buffer](None)
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

    def __init__(out self, var op: Op):
        self._ptr = ArcPointer(op^)

    # ===-------------------------------------------------------------------===#
    # Accessors
    # ===-------------------------------------------------------------------===#

    def op(ref self) -> ref[self._ptr[]] Op:
        return self._ptr[]

    def shape(ref self) -> ref[self._ptr[].shape] Shape:
        return self._ptr[].shape

    def dtype(ref self) -> ref[self._ptr[].dtype] DType:
        return self._ptr[].dtype

    def op_type(ref self) -> ref[self._ptr[].op_type] OpType:
        return self._ptr[].op_type

    def srcs(ref self) -> ref[self._ptr[].srcs] List[OpRef]:
        return self._ptr[].srcs

    def attrs(ref self) -> ref[self._ptr[].attrs] Dict[String, AttrVal]:
        return self._ptr[].attrs

    # ===-------------------------------------------------------------------===#
    # Pointwise operations
    # ===-------------------------------------------------------------------===#

    def __add__(self, rhs: OpRef) -> OpRef:
        return OpRef(Op(OpType.ADD, self.shape(), self.dtype(), [self, rhs]))

    def __mul__(self, rhs: OpRef) -> OpRef:
        return OpRef(Op(OpType.MUL, self.shape(), self.dtype(), [self, rhs]))

    def __truediv__(self, rhs: OpRef) -> OpRef:
        return OpRef(Op(OpType.DIV, self.shape(), self.dtype(), [self, rhs]))

    def __neg__(self) -> OpRef:
        return OpRef(Op(OpType.NEG, self.shape(), self.dtype(), [self]))

    def neg(self) -> OpRef:
        return OpRef(Op(OpType.NEG, self.shape(), self.dtype(), [self]))

    def scale(self, scalar: Float32) -> OpRef:
        var attrs: Dict[String, AttrVal] = {"scalar": AttrVal(scalar)}
        return OpRef(Op(OpType.SCALE, self.shape(), self.dtype(), [self], attrs^))

    def exp(self) -> OpRef:
        return OpRef(Op(OpType.EXP, self.shape(), self.dtype(), [self]))

    def log(self) -> OpRef:
        return OpRef(Op(OpType.LOG, self.shape(), self.dtype(), [self]))

    def relu(self) -> OpRef:
        return OpRef(Op(OpType.RELU, self.shape(), self.dtype(), [self]))

    def softmax(self) -> OpRef:
        return OpRef(Op(OpType.SOFTMAX, self.shape(), self.dtype(), [self]))

    def eq(self, other: OpRef) -> OpRef:
        return OpRef(Op(OpType.EQ, self.shape(), self.dtype(), [self, other]))

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self) -> OpRef:
        return OpRef(Op(OpType.SUM, (1,), self.dtype(), [self]))

    def argmax(self) -> OpRef:
        return OpRef(Op(OpType.ARGMAX, (self.shape()[0],), self.dtype(), [self]))

    # ===-------------------------------------------------------------------===#
    # Shape operations
    # ===-------------------------------------------------------------------===#

    def reshape(self, shape: Shape) -> OpRef:
        var s = shape
        var total = self.shape().numel()
        var known = 1
        var neg_idx = -1
        for i in range(len(s)):
            if s[i] == -1:
                neg_idx = i
            else:
                known *= s[i]
        if neg_idx >= 0:
            s[neg_idx] = total // known
        return OpRef(Op(OpType.RESHAPE, s, self.dtype(), [self]))

    def transpose(self) -> OpRef:
        return OpRef(
            Op(
                OpType.TRANSPOSE,
                (self.shape()[1], self.shape()[0]),
                self.dtype(),
                [self],
            )
        )

    def slice(self, start: Int, end: Int) -> OpRef:
        var new_shape = Shape(end - start)
        var attrs: Dict[String, AttrVal] = {
            "start": AttrVal(Float32(start)),
            "end": AttrVal(Float32(end)),
        }
        return OpRef(Op(OpType.SLICE, new_shape, self.dtype(), [self], attrs^))

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def matmul(self, rhs: OpRef) -> OpRef:
        return OpRef(
            Op(
                OpType.MATMUL,
                (self.shape()[0], rhs.shape()[1]),
                self.dtype(),
                [self, rhs],
            )
        )

    # ===-------------------------------------------------------------------===#
    # Loss operations
    # ===-------------------------------------------------------------------===#

    def cross_entropy(self, labels: OpRef) -> OpRef:
        return OpRef(Op(OpType.CROSS_ENTROPY, (1,), self.dtype(), [self, labels]))

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
        self._write_indented(writer, 0)

    def _write_indented(self, mut writer: Some[Writer], indent: Int):
        var pad = String(" ") * indent
        writer.write(pad + self.op_type()._name + "(shape=")
        self.shape().write_to(writer)
        writer.write(", dtype=" + String(self.dtype()))
        if len(self.srcs()) == 0:
            writer.write(")")
        else:
            writer.write(", srcs=(\n")
            for i in range(len(self.srcs())):
                self.srcs()[i]._write_indented(writer, indent + 4)
                writer.write("\n")
            writer.write(pad + "))")
