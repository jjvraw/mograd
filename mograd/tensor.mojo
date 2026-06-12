from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from layout import IntTuple

from mograd import Device
from mograd.op import AttrVal, Op, OpRef, OpType
from mograd.layout import Layout
from mograd.buffer import Buffer
from mograd.runtime import NativeRuntime
from mograd.scheduler import SchedulerRules
from mograd.grad import Grad

# ===-------------------------------------------------------------------===#
# Tensor
# ===-------------------------------------------------------------------===#


struct Tensor[dtype: DType = DType.float32](Copyable, ImplicitlyCopyable, Movable, Writable):
    var op: OpRef
    var requires_grad: Bool
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    var _grad: ArcPointer[Optional[Tensor[Self.dtype]]]
    var device: Optional[Device]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(
        out self,
        device: Optional[Device],
        var op: OpRef,
        requires_grad: Bool = False,
    ):
        self.device = device
        self.op = op^
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor[Self.dtype]](None))

    def __init__(
        out self,
        device: Device,
        data: List[Scalar[Self.dtype]],
        shape: Layout,
        requires_grad: Bool = False,
    ) raises:
        var b = Buffer.from_data(device, data)
        self.device = device
        self.op = OpRef(Op(OpType.BUFFER, shape, self.dtype, [], b^))
        self.requires_grad = requires_grad
        self._grad = ArcPointer(Optional[Tensor[self.dtype]](None))

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def empty(
        device: Device,
        shape: Layout,
        requires_grad: Bool = False,
    ) raises -> Self:
        var b = Buffer[Self.dtype].empty(device, shape.numel())
        return Tensor[Self.dtype](device, OpRef(Op(OpType.BUFFER, shape, Self.dtype, [], b^)), requires_grad)

    @staticmethod
    def ones(
        device: Device,
        shape: Layout,
        requires_grad: Bool = False,
    ) raises -> Self:
        return Self.full(device, Scalar[Self.dtype](1.0), shape, requires_grad)

    @staticmethod
    def full(
        device: Device,
        value: Scalar[Self.dtype],
        shape: Layout,
        requires_grad: Bool = False,
    ) raises -> Self:
        var b = Buffer[Self.dtype].full(device, value, shape.numel())
        return Tensor[Self.dtype](device, OpRef(Op(OpType.BUFFER, shape, Self.dtype, [], b^)), requires_grad)

    @staticmethod
    def uniform(
        device: Device,
        shape: Layout,
        low: Float32 = 0.0,
        high: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Self where Self.dtype.is_floating_point():
        attrs: Dict[String, AttrVal] = {"low": low, "high": high, "seed": Float32(seed)}
        return Tensor[Self.dtype](device, OpRef(Op(OpType.UNIFORM, shape, Self.dtype, [], attrs=attrs^)), requires_grad)

    @staticmethod
    def randn(
        device: Device,
        shape: Layout,
        mean: Float32 = 0.0,
        std: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Self where Self.dtype.is_floating_point():
        attrs: Dict[String, AttrVal] = {"mean": mean, "std": std, "seed": Float32(seed)}
        return Tensor[Self.dtype](device, OpRef(Op(OpType.RANDN, shape, Self.dtype, [], attrs^)), requires_grad)

    @staticmethod
    def disk(
        device: Device,
        path: String,
        shape: Layout,
    ) -> Self:
        return Tensor[Self.dtype](device, OpRef(Op(OpType.DISK, shape, Self.dtype, [], {"path": path})))

    @staticmethod
    def full(
        device: Device,
        shape: Layout,
        fill_value: Float32,
        requires_grad: Bool = False,
    ) -> Self:
        return Tensor[Self.dtype](
            device, OpRef(Op(OpType.FULL, shape, Self.dtype, [], {"value": fill_value})), requires_grad
        )

    @staticmethod
    def ones_like[
        other_dtype: DType
    ](other: Tensor[other_dtype], requires_grad: Bool = False) raises -> Tensor[other_dtype]:
        return Tensor[other_dtype].ones(other.device.value(), other.op.layout(), requires_grad)

    @staticmethod
    def from_buffer(device: Device, layout: Layout, var buf: Buffer[Self.dtype]) -> Self:
        return Tensor[Self.dtype](
            device,
            OpRef(Op(OpType.BUFFER, layout, Self.dtype, [], buf^)),
        )

    # ===-------------------------------------------------------------------===#
    # Materialisation / Device I/O
    # ===-------------------------------------------------------------------===#

    def value(self, rules: Optional[SchedulerRules] = None) raises -> Buffer[Self.dtype]:
        var result = NativeRuntime.run(self.op, self.device)
        return result.unsafe_get[Self.dtype]().copy()

    def item(self, rules: Optional[SchedulerRules] = None) raises -> Scalar[Self.dtype]:
        var buf = self.value(rules)
        var result: Scalar[Self.dtype]
        with buf.buf().map_to_host() as host:
            result = (host.unsafe_ptr() + buf.base_offset)[0]
        return result

    def to_list(self, rules: Optional[SchedulerRules] = None) raises -> List[Scalar[Self.dtype]]:
        var result = List[Scalar[Self.dtype]]()
        var buf = self.value(rules)
        var layout = self.op.layout()
        var inner = layout.inner_sizes()
        with buf.buf().map_to_host() as host:
            var base = host.unsafe_ptr() + buf.base_offset
            for i in range(layout.numel()):
                var off = 0
                var rem = i
                for d in range(layout.rank()):
                    var idx = rem // inner.value(d)
                    rem %= inner.value(d)
                    off += idx * layout._strides.value(d)
                result.append(base[off])
        return result^

    # ===-------------------------------------------------------------------===#
    # Layout
    # ===-------------------------------------------------------------------===#

    def is_contiguous(self) -> Bool:
        return self.op.layout().is_contiguous()

    def shape(self) -> Layout:
        return self.op.layout().copy()

    def shape(self, idx: Int) -> Int:
        return self.op.shape(idx)

    def numel(self) -> Int:
        return self.op.layout().numel()

    def stride(self, axis: Int) raises -> Int:
        return self.op.layout().stride(axis)

    def stride(self) -> IntTuple:
        return self.op.layout().stride()

    # ===-------------------------------------------------------------------===#
    # Layout transformative operations
    # ===-------------------------------------------------------------------===#

    def contiguous(self) -> Self:
        return Self(self.device, self.op.contiguous(), self.requires_grad)

    def reshape(self, shape: Layout) raises -> Self:
        """Returns a new tensor with the given shape, copying the underlying data.

        Args:
            shape: The desried output layout.

        Returns:
            A new tensor with the specified shape.

        Raises:
            Error: If the number of elements in `shape` does not match the original.
        """
        return Self(self.device, self.op.reshape(shape), self.requires_grad)

    def view(self, shape: Layout) raises -> Self:
        """Returns a new tensor with the fiven shape, sharing the underlying data.

        Args:
            shape: The desired output layout.

        Returns:
            A new tensor with the specified shape.

        Raises:
            Error: If the tensor is not contiguous or if the number of elements in `shape`
            does match the original.
        """
        return Self(self.device, self.op.view(shape), self.requires_grad)

    def transpose(self) raises -> Self:
        return Self(self.device, self.op.transpose(), self.requires_grad)

    def __getitem__(self, s: Slice) raises -> Self:
        var dim = self.op.layout().shape(0)
        var start, stop, step = s.indices(dim)
        return Self(self.device, self.op.slice(start, stop, step), self.requires_grad)

    def cast[out_dtype: DType](self) -> Tensor[out_dtype]:
        return Tensor[out_dtype](self.device, self.op.cast(out_dtype), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Elementwise operations
    # ===-------------------------------------------------------------------===#

    def __add__(self, other: Self) -> Self:
        return self.add(other)

    def __add__(self, scalar: Float32) -> Self:
        return self.add(scalar)

    def __radd__(self, scalar: Float32) -> Self:
        return self.add(scalar)

    def add(self, other: Self) -> Self:
        return Self(self.device, self.op + other.op, self.requires_grad or other.requires_grad)

    def add(self, scalar: Float32) -> Self:
        var attrs: Dict[String, AttrVal] = {"value": AttrVal(scalar)}
        return Self(
            self.device,
            self.op + OpRef(Op(OpType.FULL, self.op.layout(), self.dtype, [], attrs=attrs^)),
            self.requires_grad,
        )

    def __sub__(self, other: Self) -> Self:
        return self + (-other)

    def __neg__(self) -> Self:
        return Self(self.device, -self.op, self.requires_grad)

    def neg(self) -> Self:
        return Self(self.device, self.op.neg(), self.requires_grad)

    def __mul__(self, other: Self) -> Self:
        return self.mul(other)

    def mul(self, other: Self) -> Self:
        return Self(self.device, self.op * other.op, self.requires_grad or other.requires_grad)

    def __mul__(self, scalar: Scalar[Self.dtype]) -> Self:
        return self.scale(scalar)

    def __rmul__(self, scalar: Scalar[Self.dtype]) -> Self:
        return self.scale(scalar)

    def scale(self, scalar: Scalar[Self.dtype]) -> Self:
        return Self(self.device, self.op.scale(scalar), self.requires_grad)

    def __truediv__(self, other: Self) -> Self:
        return Self(self.device, self.op / other.op, self.requires_grad or other.requires_grad)

    def exp(self) -> Self:
        return Self(self.device, self.op.exp(), self.requires_grad)

    def relu(self) -> Self:
        return Self(self.device, self.op.relu(), self.requires_grad)

    def log(self) -> Self:
        return Self(self.device, self.op.log(), self.requires_grad)

    def softmax(self) -> Self:
        return Self(self.device, self.op.softmax(), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Reduction operations
    # ===-------------------------------------------------------------------===#

    def sum(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> Self:
        """Reduces the tensor by summing elements along a specified axis.

        Args:
            axis: The axis to sum over. If `None`, sums elements and returns a scalar tensor.
            keepdim: If `True`, the reduced axis is retained as a dimension of size 1,
                     preserving the tensor's rank.

        Returns:
            A tensor containing the sum. If `axis` is `None`, a scalar tensor.

        Raises:
            If axis is out of bounds for the tensor's rank.
        """
        return Self(self.device, self.op.sum(axis, keepdim), self.requires_grad)

    def mean(self) raises -> Self:
        return self.sum() * (1.0 / Scalar[Self.dtype](self.op.layout().numel()))

    def argmax(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> Self:
        """Returns the indices of the maximum values of a tensor across a dimension.

        Args:
            axis: The axis to reduce over. If `None`, returns the index of the maximum
                  element across all elements in row-major order.
            keepdim: If `True`, the reduced axis is retained as a dimension of size 1,
                     preserving the tensor's rank.

        Returns:
            A tensor containing the indices of the maximum values. If `axis` is `None`,
            a scalar tensor.

        Raises:
            If axis is out of bounds for the tensor's rank.
        """
        return Self(self.device, self.op.argmax(axis, keepdim), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Indexing & Encoding
    # ===-------------------------------------------------------------------===#

    def one_hot[
        out_dtype: DType = DType.int64
    ](self, num_classes: Int) -> Tensor[out_dtype] where out_dtype.is_integral():
        return Tensor[out_dtype](self.device, self.op.one_hot(num_classes, out_dtype), False)

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def __matmul__(self, other: Self) -> Self:
        return self.matmul(other)

    def matmul(self, other: Self) -> Self:
        return Self(self.device, self.op.matmul(other.op), self.requires_grad or other.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Comparison operations
    # ===-------------------------------------------------------------------===#

    def eq(self, other: Self) -> Self:
        return Self(self.device, self.op.eq(other.op), False)

    def __eq__(self, other: Self) -> Self:
        return self.eq(other)

    # ===-------------------------------------------------------------------===#
    # Loss
    # ===-------------------------------------------------------------------===#

    def cross_entropy(self, labels: Self) -> Self where Self.dtype.is_floating_point():
        return Self(self.device, self.op.cross_entropy(labels.op), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Autograd
    # ===-------------------------------------------------------------------===#

    def gradient(
        mut self,
        targets: List[Tensor[Self.dtype]],
        var gradient: Optional[Tensor[Self.dtype]] = None,
    ) raises -> List[Tensor[Self.dtype]] where Self.dtype.is_floating_point():
        var initial_grad_op: OpRef
        if gradient:
            initial_grad_op = gradient.take().op
        else:
            initial_grad_op = Tensor[Self.dtype].ones_like(self).op

        var target_ops = List[OpRef]()
        for t in targets:
            target_ops.append(t.op)

        var grads = Grad[Self.dtype].compute(self.op, initial_grad_op, target_ops)

        var result = List[Tensor[Self.dtype]]()
        for i in range(len(grads)):
            if grads[i]:
                result.append(Tensor[Self.dtype](self.device, grads[i].value()))
            else:
                if not self.device:
                    raise Error("gradient requires a device context")
                result.append(Tensor[Self.dtype].empty(self.device.value(), targets[i].op.layout()))
        return result^

    # ===-------------------------------------------------------------------===#
    # Display
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        self.op.write_to(writer)
