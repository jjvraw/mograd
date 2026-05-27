from std.memory import ArcPointer
from std.hashlib.hasher import Hasher

from mograd.buffer import Buffer

# ===-------------------------------------------------------------------===#
# OpType
# ===-------------------------------------------------------------------===#


# TODO: Lets rather have a op registry that is defined and built at comptime.
#       Main motivation here is for custom rewrites.
@fieldwise_init
struct OpType(Copyable, ImplicitlyCopyable, KeyElement, Movable):
    var _value: Int
    var _name: String
    comptime BUFFER = OpType(1, "BUFFER")
    comptime ONES = OpType(2, "ONES")
    comptime ADD = OpType(3, "ADD")
    comptime MUL = OpType(4, "MUL")
    comptime RELU = OpType(5, "RELU")
    comptime RELU_GRAD = OpType(6, "RELU_GRAD")
    comptime SOFTMAX = OpType(7, "SOFTMAX")
    comptime SOFTMAX_GRAD = OpType(8, "SOFTMAX_GRAD")
    comptime EXP = OpType(9, "EXP")
    comptime LOG = OpType(10, "LOG")
    comptime NEG = OpType(11, "NEG")
    comptime DIV = OpType(12, "DIV")
    comptime SUM = OpType(13, "SUM")
    comptime SUM_GRAD = OpType(14, "SUM_GRAD")
    comptime RESHAPE = OpType(15, "RESHAPE")
    comptime MATMUL = OpType(16, "MATMUL")
    comptime TRANSPOSE = OpType(17, "TRANSPOSE")
    comptime UNIFORM = OpType(18, "UNIFORM")
    comptime DISK = OpType(19, "DISK")

    def __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(self._value)

    def __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    def __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# ===-------------------------------------------------------------------===#
# Op
# ===-------------------------------------------------------------------===#


struct Op(Copyable, Movable, Writable):
    var op_type: OpType
    var shape: List[Int]
    var dtype: DType
    var srcs: List[OpRef]
    var buf: Optional[Buffer]
    var attrs: List[Float32]
    var str_attrs: List[String]

    def __init__(
        out self,
        op_type: OpType,
        var shape: List[Int],
        dtype: DType,
        var srcs: List[OpRef],
    ):
        self.op_type = op_type
        self.shape = shape^
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = Optional[Buffer](None)
        self.attrs = []
        self.str_attrs = []

    def __init__(
        out self,
        op_type: OpType,
        var shape: List[Int],
        dtype: DType,
        var srcs: List[OpRef],
        var buf: Optional[Buffer],
    ):
        self.op_type = op_type
        self.shape = shape^
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = buf^
        self.attrs = []
        self.str_attrs = []

    def __init__(
        out self,
        op_type: OpType,
        var shape: List[Int],
        dtype: DType,
        var srcs: List[OpRef],
        var attrs: List[Float32],
    ):
        self.op_type = op_type
        self.shape = shape^
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = Optional[Buffer](None)
        self.attrs = attrs^
        self.str_attrs = []

    def __init__(
        out self,
        op_type: OpType,
        var shape: List[Int],
        dtype: DType,
        var srcs: List[OpRef],
        var str_attrs: List[String],
    ):
        self.op_type = op_type
        self.shape = shape^
        self.dtype = dtype
        self.srcs = srcs^
        self.buf = Optional[Buffer](None)
        self.attrs = []
        self.str_attrs = str_attrs^

    def __str__(self) -> String:
        return self.op_type._name

    def write_to(self, mut writer: Some[Writer]):
        writer.write(self.op_type._name)


# ===-------------------------------------------------------------------===#
# OpRef
# ===-------------------------------------------------------------------===#


struct OpRef(Copyable, ImplicitlyCopyable, KeyElement, Movable, Writable):
    var _ptr: ArcPointer[Op]

    def __init__(out self, var op: Op):
        self._ptr = ArcPointer(op^)

    def __hash__[H: Hasher](self, mut hasher: H):
        hasher.update(Int(self._ptr.unsafe_ptr()))

    def __eq__(self, other: Self) -> Bool:
        return self._ptr.unsafe_ptr() == other._ptr.unsafe_ptr()

    def __ne__(self, other: Self) -> Bool:
        return self._ptr.unsafe_ptr() != other._ptr.unsafe_ptr()

    def op(ref self) -> ref[self._ptr[]] Op:
        return self._ptr[]

    def shape(ref self) -> ref[self._ptr[].shape] List[Int]:
        return self._ptr[].shape

    def dtype(ref self) -> ref[self._ptr[].dtype] DType:
        return self._ptr[].dtype

    def op_type(ref self) -> ref[self._ptr[].op_type] OpType:
        return self._ptr[].op_type

    def srcs(ref self) -> ref[self._ptr[].srcs] List[OpRef]:
        return self._ptr[].srcs

    def attrs(ref self) -> ref[self._ptr[].attrs] List[Float32]:
        return self._ptr[].attrs

    def str_attrs(ref self) -> ref[self._ptr[].str_attrs] List[String]:
        return self._ptr[].str_attrs

    def __add__(self, rhs: OpRef) -> OpRef:
        return OpRef(
            Op(OpType.ADD, self.shape().copy(), self.dtype(), [self, rhs])
        )

    def __mul__(self, rhs: OpRef) -> OpRef:
        return OpRef(
            Op(OpType.MUL, self.shape().copy(), self.dtype(), [self, rhs])
        )

    def __truediv__(self, rhs: OpRef) -> OpRef:
        return OpRef(
            Op(OpType.DIV, self.shape().copy(), self.dtype(), [self, rhs])
        )

    def __neg__(self) -> OpRef:
        return OpRef(Op(OpType.NEG, self.shape().copy(), self.dtype(), [self]))

    def relu(self) -> OpRef:
        return OpRef(Op(OpType.RELU, self.shape().copy(), self.dtype(), [self]))

    def softmax(self) -> OpRef:
        return OpRef(
            Op(OpType.SOFTMAX, self.shape().copy(), self.dtype(), [self])
        )

    def exp(self) -> OpRef:
        return OpRef(Op(OpType.EXP, self.shape().copy(), self.dtype(), [self]))

    def log(self) -> OpRef:
        return OpRef(Op(OpType.LOG, self.shape().copy(), self.dtype(), [self]))

    def neg(self) -> OpRef:
        return OpRef(Op(OpType.NEG, self.shape().copy(), self.dtype(), [self]))

    def sum(self) -> OpRef:
        return OpRef(Op(OpType.SUM, [1], self.dtype(), [self]))

    def reshape(self, var new_shape: List[Int]) -> OpRef:
        var neg_idx = -1
        var known_product = 1
        var total = 1
        for d in self.shape():
            total *= d
        for i in range(len(new_shape)):
            if new_shape[i] == -1:
                neg_idx = i
            else:
                known_product *= new_shape[i]
        if neg_idx >= 0:
            new_shape[neg_idx] = total // known_product
        return OpRef(Op(OpType.RESHAPE, new_shape^, self.dtype(), [self]))

    def matmul(self, rhs: OpRef) -> OpRef:
        return OpRef(
            Op(
                OpType.MATMUL,
                [self.shape()[0], rhs.shape()[1]],
                self.dtype(),
                [self, rhs],
            )
        )

    def transpose(self) -> OpRef:
        return OpRef(
            Op(
                OpType.TRANSPOSE,
                [self.shape()[1], self.shape()[0]],
                self.dtype(),
                [self],
            )
        )

    def write_to(self, mut writer: Some[Writer]):
        self._write_indented(writer, 0)

    def _write_indented(self, mut writer: Some[Writer], indent: Int):
        var pad = String(" ") * indent
        writer.write(pad + self.op_type()._name + "(shape=[")
        for i in range(len(self.shape())):
            writer.write(String(self.shape()[i]))
            if i < len(self.shape()) - 1:
                writer.write(", ")
        writer.write("], dtype=" + String(self.dtype()))
        if len(self.srcs()) == 0:
            writer.write(")")
        else:
            writer.write(", srcs=(\n")
            for i in range(len(self.srcs())):
                self.srcs()[i]._write_indented(writer, indent + 4)
                writer.write("\n")
            writer.write(pad + "))")
