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
    comptime ARGMAX = OpType("ARGMAX")

    # ===-------------------------------------------------------------------===#
    # Shape ops
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


struct Op[dtype: DType](Copyable, Movable, Writable):
    var op_type: OpType
    var shape: Shape
    var srcs: List[AnyOpRef]
    var buf: Optional[Buffer[Self.dtype]]
    var attrs: Dict[String, AttrVal]

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        var srcs: List[AnyOpRef],
    ):
        self.op_type = op_type
        self.shape = shape
        self.srcs = srcs^
        self.buf = None
        self.attrs = {}

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        var srcs: List[AnyOpRef],
        var buf: Buffer[Self.dtype],
    ):
        self.op_type = op_type
        self.shape = shape
        self.srcs = srcs^
        self.buf = Optional[Buffer[Self.dtype]](buf^)
        self.attrs = {}

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        var srcs: List[AnyOpRef],
        var attrs: Dict[String, AttrVal],
    ):
        self.op_type = op_type
        self.shape = shape
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


comptime AnyOpRef = Variant[OpRef[DType.float32], OpRef[DType.int64]]


trait NodeOps(Movable):
    def op_type(ref self) -> OpType:
        ...

    def srcs_count(ref self) -> Int:
        ...

    def get_src(self, i: Int) -> AnyOpRef:
        ...

    def shape_val(self) -> Shape:
        ...

    def attrs_copy(self) -> Dict[String, AttrVal]:
        ...


trait OpPrinter:
    def _write_indented(self, mut writer: Some[Writer], indent: Int):
        ...


trait HasDtype:
    comptime node_dtype: DType


struct OpRef[dtype: DType](Copyable, HasDtype, ImplicitlyCopyable, KeyElement, Movable, NodeOps, OpPrinter, Writable):
    comptime node_dtype: DType = Self.dtype
    var _ptr: ArcPointer[Op[Self.dtype]]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @implicit
    def __init__(out self, var op: Op[Self.dtype]):
        self._ptr = ArcPointer(op^)

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        var srcs: List[AnyOpRef],
        var attrs: Optional[Dict[String, AttrVal]] = None,
    ):
        var owned_attrs = attrs.take() if attrs else {}
        self._ptr = ArcPointer(Op[Self.dtype](op_type, shape, srcs^, owned_attrs^))

    def __init__(
        out self,
        op_type: OpType,
        shape: Shape,
        var srcs: List[AnyOpRef],
        var buf: Buffer[Self.dtype],
    ):
        self._ptr = ArcPointer(Op[Self.dtype](op_type, shape, srcs^, buf^))

    # ===-------------------------------------------------------------------===#
    # Accessors
    # ===-------------------------------------------------------------------===#

    def op(ref self) -> ref[self._ptr[]] Op[Self.dtype]:
        return self._ptr[]

    def shape(ref self) -> ref[self._ptr[].shape] Shape:
        return self._ptr[].shape

    def shape(ref self, idx: Int) -> Int:
        return self._ptr[].shape[idx]

    def op_type(ref self) -> OpType:
        return self._ptr[].op_type

    def srcs(ref self) -> ref[self._ptr[].srcs] List[AnyOpRef]:
        return self._ptr[].srcs

    def src(ref self, i: Int) -> OpRef[Self.dtype]:
        return self._ptr[].srcs[i].unsafe_get[OpRef[Self.dtype]]()

    def srcs_count(ref self) -> Int:
        return len(self._ptr[].srcs)

    def get_src(self, i: Int) -> AnyOpRef:
        return self._ptr[].srcs[i]

    def attrs(ref self) -> ref[self._ptr[].attrs] Dict[String, AttrVal]:
        return self._ptr[].attrs

    def shape_val(self) -> Shape:
        return self._ptr[].shape

    def attrs_copy(self) -> Dict[String, AttrVal]:
        return self._ptr[].attrs.copy()

    # ===-------------------------------------------------------------------===#
    # Pointwise operations
    # ===-------------------------------------------------------------------===#

    def __add__(self, rhs: OpRef) -> Self:
        return Self(OpType.ADD, self.shape(), [self, rhs])

    def __mul__(self, rhs: OpRef) -> Self:
        return Self(OpType.MUL, self.shape(), [self, rhs])

    def __truediv__(self, rhs: OpRef) -> Self:
        return Self(OpType.DIV, self.shape(), [self, rhs])

    def __neg__(self) -> Self:
        return Self(OpType.NEG, self.shape(), [self])

    def neg(self) -> Self:
        return Self(OpType.NEG, self.shape(), [self])

    def scale(self, scalar: Scalar[Self.dtype]) -> Self:
        var attrs: Dict[String, AttrVal] = {"scalar": AttrVal(scalar)}
        return Self(OpType.SCALE, self.shape(), [self], attrs=attrs^)

    def exp(self) -> Self:
        return Self(OpType.EXP, self.shape(), [self])

    def log(self) -> Self:
        return Self(OpType.LOG, self.shape(), [self])

    def relu(self) -> Self:
        return Self(OpType.RELU, self.shape(), [self])

    def softmax(self) -> Self:
        return Self(OpType.SOFTMAX, self.shape(), [self])

    def eq(self, other: OpRef) -> Self:
        return Self(OpType.EQ, self.shape(), [self, other])

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self) -> Self:
        return Self(OpType.SUM, (1,), [self])

    def argmax(self) -> Self:
        return Self(OpType.ARGMAX, (self.shape(0),), [self])

    # ===-------------------------------------------------------------------===#
    # Shape operations
    # ===-------------------------------------------------------------------===#

    def reshape(self, shape: Shape) -> Self:
        # TODO: move below to infer_size or equivalent.
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
        return Self(OpType.RESHAPE, s, [self])

    def one_hot[
        out_dtype: DType = DType.int64
    ](self, num_classes: Int) -> OpRef[out_dtype] where out_dtype.is_integral():
        var attrs: Dict[String, AttrVal] = {"num_classes": AttrVal(Float32(num_classes))}
        return OpRef[out_dtype](
            OpType.ONE_HOT,
            (self.shape(0), num_classes),
            [self],
            attrs=attrs^,
        )

    def cast[out_dtype: DType](self) -> OpRef[out_dtype]:
        return OpRef[out_dtype](OpType.CAST, self.shape(), [self])

    def transpose(self) -> Self:
        return Self(OpType.TRANSPOSE, (self.shape(1), self.shape(0)), [self])

    def slice(self, start: Int, end: Int) -> Self:
        # TODO: Use shape slice
        var src = self.shape()
        var new_dims = List[Int]()
        new_dims.append(end - start)
        for i in range(1, len(src)):
            new_dims.append(src[i])
        var new_shape = Shape(new_dims)
        var attrs: Dict[String, AttrVal] = {
            "start": AttrVal(Float32(start)),
            "end": AttrVal(Float32(end)),
        }
        return Self(OpType.SLICE, new_shape, [self], attrs=attrs^)

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def matmul(self, rhs: OpRef) -> Self:
        return Self(OpType.MATMUL, (self.shape(0), rhs.shape(1)), [self, rhs])

    # ===-------------------------------------------------------------------===#
    # Loss operations
    # ===-------------------------------------------------------------------===#

    def cross_entropy(self, labels: Self) -> Self where Self.dtype.is_floating_point():
        return Self(OpType.CROSS_ENTROPY, (1,), [self, labels])

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
        writer.write(", dtype=" + String(self.dtype))
        if len(self.srcs()) == 0:
            writer.write(")")
            return
        writer.write(", srcs=(\n")
        for i in range(len(self.srcs())):
            _write_any_op_ref(self.srcs()[i], writer, indent + 4)
            writer.write("\n")
        writer.write(pad + "))")


def _write_any_op_ref(op: AnyOpRef, mut writer: Some[Writer], indent: Int):
    comptime for i in range(AnyOpRef.Ts.size):
        comptime T = AnyOpRef.Ts[i]
        if op.isa[T]():
            comptime assert conforms_to(T, OpPrinter)
            trait_downcast[OpPrinter](op.unsafe_get[T]())._write_indented(writer, indent)
            return
