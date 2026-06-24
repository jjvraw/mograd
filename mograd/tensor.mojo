from std.memory import ArcPointer
from std.gpu.host import DeviceContext

from layout import IntTuple

from mograd import Device
from mograd.op import AttrVal, Op, OpRef, OpType
from mograd.layout import Layout
from mograd.buffer import AnyBuffer
from mograd.runtime import NativeRuntime
from mograd.scheduler import SchedulerRules
from mograd.grad import Grad

# ===-------------------------------------------------------------------===#
# Tensor
# ===-------------------------------------------------------------------===#


struct Tensor(Copyable, ImplicitlyCopyable, Movable, Writable):
    var op: OpRef
    var dtype: DType
    var requires_grad: Bool
    # TODO: Use ArcPointer when Optional[ArcPointer] is resolved:
    # https://github.com/modular/modular/issues/3293
    var _grad: ArcPointer[Optional[Tensor]]
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
        self._grad = ArcPointer(Optional[Tensor](None))
        self.dtype = self.op.dtype()

    def __init__[
        D: DType = DType.float32, /
    ](out self, device: Device, data: List[Scalar[D]], shape: Layout, requires_grad: Bool = False,) raises:
        var b = Buffer.from_data(device, data)
        self.device = device
        self.op = OpRef(Op(OpType.BUFFER, shape, D, [], b^))
        self.requires_grad = requires_grad
        self.dtype = D
        self._grad = ArcPointer(Optional[Tensor](None))

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def empty(
        device: Device,
        shape: Layout,
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) raises -> Self:
        var b = AnyBuffer.create(dtype, device, shape.numel())
        return Tensor(device, OpRef(Op(OpType.BUFFER, shape, dtype, [], b^)), requires_grad)

    @staticmethod
    def ones(device: Device, shape: Layout, dtype: DType = DType.float32, requires_grad: Bool = False) -> Self:
        return Self.full(device, shape, 1.0, dtype, requires_grad)

    @staticmethod
    def uniform(
        device: Device,
        shape: Layout,
        dtype: DType = DType.float32,
        low: Float32 = 0.0,
        high: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Self:
        attrs: Dict[String, AttrVal] = {"low": low, "high": high, "seed": Float32(seed)}
        return Tensor(device, OpRef(Op(OpType.UNIFORM, shape, dtype, [], attrs=attrs^)), requires_grad)

    @staticmethod
    def randn(
        device: Device,
        shape: Layout,
        dtype: DType = DType.float32,
        mean: Float32 = 0.0,
        std: Float32 = 1.0,
        seed: UInt32 = 42,
        requires_grad: Bool = False,
    ) -> Self:
        attrs: Dict[String, AttrVal] = {"mean": mean, "std": std, "seed": Float32(seed)}
        return Tensor(device, OpRef(Op(OpType.RANDN, shape, dtype, [], attrs^)), requires_grad)

    @staticmethod
    def disk(device: Device, path: String, shape: Layout, dtype: DType = DType.float32) -> Self:
        return Tensor(device, OpRef(Op(OpType.DISK, shape, dtype, [], {"path": path})))

    @staticmethod
    def full(
        device: Device,
        shape: Layout,
        fill_value: Float32,
        dtype: DType = DType.float32,
        requires_grad: Bool = False,
    ) -> Self:
        return Tensor(device, OpRef(Op(OpType.FULL, shape, dtype, [], {"value": fill_value})), requires_grad)

    @staticmethod
    def full_like(other: Tensor, fill_value: Float32, requires_grad: Bool = False) raises -> Tensor:
        return Tensor.full(other.device.value(), other.op.layout(), fill_value, other.dtype, requires_grad)

    @staticmethod
    def ones_like(other: Tensor, requires_grad: Bool = False) raises -> Tensor:
        return Tensor.ones(other.device.value(), other.op.layout(), other.dtype, requires_grad)

    @staticmethod
    def from_buffer(device: Device, layout: Layout, var buf: AnyBuffer) raises -> Self:
        var dtype = buf.dtype()
        return Tensor(device, OpRef(Op(OpType.BUFFER, layout, dtype, [], buf^)))

    # ===-------------------------------------------------------------------===#
    # Materialisation / Device I/O
    # ===-------------------------------------------------------------------===#

    def value(self, rules: Optional[SchedulerRules] = None) raises -> AnyBuffer:
        return NativeRuntime.run(self.op, self.device, rules.copy())

    def item[T: DType = DType.float32](self, rules: Optional[SchedulerRules] = None) raises -> Scalar[T]:
        if T != self.dtype:
            raise Error("Tensor.item: requested dtype does not match tensor dtype")
        return self.value(rules).item[T]()

    def to_list[T: DType = DType.float32](self, rules: Optional[SchedulerRules] = None) raises -> List[Scalar[T]]:
        if T != self.dtype:
            raise Error("Tensor.to_list: requested dtype does not match tensor dtype")
        return self.value(rules).to_list[T](self.op.layout())

    # ===-------------------------------------------------------------------===#
    # Layout
    # ===-------------------------------------------------------------------===#

    def is_contiguous(self) -> Bool:
        return self.op.layout().is_contiguous()

    def shape(self) -> IntTuple:
        return self.op.shape()

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

    def expand(self, *shape: Int) raises -> Self:
        """Broadcasts size-1 axes out to `shape`.

        `shape` must have the same rank as this tensor.

        Args:
            shape: Target shape, same rank as this tensor.

        Returns:
            Tensor viewing the same data with size-1 axes stretched to `shape`.

        Raises:
            If rank doesn't match, or a non-size-1 axis disagrees with `shape`.
        """
        return Self(self.device, self.op.expand(*shape), self.requires_grad)

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

    def flatten(self, start_dim: Int = 0, end_dim: Int = -1) raises -> Self:
        """Returns a new tensor with dimensions flattened from start_dim to end_dim.

        Args:
            start_dim: The starting dimension to flatten (default: 0).
            end_dim: The ending dimension to flatten (default: -1, the last dimension).

        Returns:
            A new tensor with the specified dimensions flattened.

        Raises:
            Error: If start_dim or end_dim are out of bounds.
        """
        return Self(self.device, self.op.flatten(start_dim, end_dim), self.requires_grad)

    def transpose(self, dim0: Int = -2, dim1: Int = -1) raises -> Self:
        """Returns a transposed view of the tensor by swapping `dim0` and `dim1`.
        All other dimensions are left in their original positions.
        Defaults to swapping the last two dimensions.

        Args:
            dim0: First dimension to swap.
            dim1: Second dimension to swap.

        Returns:
            Transposed tensor view.

        Raises:
            If dim0 equals dim1 or either dim is out of bounds.
        """
        return Self(self.device, self.op.transpose(dim0, dim1), self.requires_grad)

    def squeeze(self, dim: Optional[Int] = None) raises -> Self:
        """Remove a dimension of size 1.

        Args:
            dim: Dimension to squeeze. If None, removes all dimensions of size 1.

        Returns:
            Tensor with specified dimension(s) removed.

        Raises:
            If the dimension is not of size 1.
        """
        return Self(self.device, self.op.squeeze(dim), self.requires_grad)

    def unsqueeze(self, dim: Int) raises -> Self:
        """Insert a dimension of size 1 at the specified position.

        Args:
            dim: Position to insert the new dimension.

        Returns:
            Tensor with new dimension inserted.

        Raises:
            If dim is out of bounds.
        """
        return Self(self.device, self.op.unsqueeze(dim), self.requires_grad)

    def triu(self, diagonal: Int = 0) raises -> Self:
        """Returns the upper triangular part of the tensor, zeroing the rest.

        Args:
            diagonal: Diagonal offset. 0 keeps main diagonal and above,
                     1 keeps above main diagonal, -1 keeps main diagonal and below.

        Returns:
            Upper triangular tensor.

        Raises:
            If tensor rank is less than 2.
        """
        return Self(self.device, self.op.triu(diagonal), self.requires_grad)

    def __getitem__(self, s: Slice) raises -> Self:
        var dim = self.op.layout().shape(0)
        var start, stop, step = s.indices(dim)
        return Self(self.device, self.op.slice(start, stop, step), self.requires_grad)

    def cast(self, dtype: DType) -> Tensor:
        return Tensor(self.device, self.op.cast(dtype), self.requires_grad)

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

    def __mul__(self, scalar: Float32) -> Self:
        return self.scale(scalar)

    def __rmul__(self, scalar: Float32) -> Self:
        return self.scale(scalar)

    def scale(self, scalar: Float32) -> Self:
        return Self(self.device, self.op.scale(scalar), self.requires_grad)

    def __truediv__(self, other: Self) -> Self:
        return Self(self.device, self.op / other.op, self.requires_grad or other.requires_grad)

    def exp(self) -> Self:
        return Self(self.device, self.op.exp(), self.requires_grad)

    def sqrt(self) -> Self:
        return Self(self.device, self.op.sqrt(), self.requires_grad)

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

    def mean(self, axis: Optional[Int] = None, keepdim: Bool = False) raises -> Self:
        if not axis:
            return self.sum() * (1.0 / Float32(self.op.layout().numel()))
        var ax = self.op.layout().normalise_dim(axis.value())
        var denom = Float32(self.op.layout().shape(ax))
        return self.sum(ax, keepdim) * (1.0 / denom)

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

    def one_hot(self, num_classes: Int, out_dtype: DType = DType.int64) -> Tensor:
        return Tensor(self.device, self.op.one_hot(num_classes, out_dtype), False)

    def gather(self, indices: Tensor) raises -> Tensor:
        return Tensor(self.device, self.op.gather(indices.op), self.requires_grad)

    def scatter_add(self, indices: Tensor, num_rows: Int) raises -> Tensor:
        return Tensor(self.device, self.op.scatter_add(indices.op, num_rows), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Contraction operations
    # ===-------------------------------------------------------------------===#

    def __matmul__(self, other: Self) raises -> Self:
        return self.matmul(other)

    def matmul(self, other: Self) raises -> Self:
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

    def cross_entropy(self, labels: Self) -> Self:
        return Self(self.device, self.op.cross_entropy(labels.op), self.requires_grad)

    # ===-------------------------------------------------------------------===#
    # Autograd
    # ===-------------------------------------------------------------------===#

    def gradient(
        mut self,
        targets: List[Tensor],
        var gradient: Optional[Tensor] = None,
    ) raises -> List[Tensor]:
        var initial_grad_op: OpRef
        if gradient:
            initial_grad_op = gradient.take().op
        else:
            initial_grad_op = Tensor.ones_like(self).op

        var target_ops = List[OpRef]()
        for t in targets:
            target_ops.append(t.op)

        var grads = Grad.compute(self.op, initial_grad_op, target_ops)

        var result = List[Tensor]()
        for i in range(len(grads)):
            if grads[i]:
                result.append(Tensor(self.device, grads[i].value()))
            else:
                if not self.device:
                    raise Error("gradient requires a device context")
                result.append(Tensor.empty(self.device.value(), targets[i].op.layout()))
        return result^

    # ===-------------------------------------------------------------------===#
    # Display
    # ===-------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        self.op.write_to(writer)
