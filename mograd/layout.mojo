from std.format.tstring import TString
from std.builtin.builtin_slice import StridedSlice
from std.gpu.host import DeviceBuffer, DeviceContext

from layout.int_tuple import IntTuple, reverse, prefix_product, product, sorted, compact_order

# ===-------------------------------------------------------------------===#
# Layout
# ===-------------------------------------------------------------------===#


@fieldwise_init
struct Layout(Copyable, ImplicitlyCopyable, Movable, Writable):
    """Fully dynamic Layout containing rank, shape, strides and base offset.

    Defaults to row-major contiguous layout.
    """

    var _rank: Int
    var _shape: IntTuple
    var _strides: IntTuple
    var base_offset: Int

    # ===-------------------------------------------------------------------===#
    # Lifecycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self, *shape: Int):
        """Initializes a row-major contigious layout of shape provided."""
        self._rank = len(shape)
        self._shape = IntTuple(*shape)
        self._strides = Self.row_major_strides(self._shape)
        self.base_offset = 0

    # ===-------------------------------------------------------------------===#
    # Implicit overloads for nicer syntax
    # ===-------------------------------------------------------------------===#
    # TODO: Is there a better way to do this?

    @implicit
    def __init__(out self, t: Tuple[Int]):
        self = Self(t[0])

    @implicit
    def __init__(out self, t: Tuple[Int, Int]):
        self = Self(t[0], t[1])

    @implicit
    def __init__(out self, t: Tuple[Int, Int, Int]):
        self = Self(t[0], t[1], t[2])

    @implicit
    def __init__(out self, t: Tuple[Int, Int, Int, Int]):
        self = Self(t[0], t[1], t[2], t[3])

    @implicit
    def __init__(out self, t: Tuple[Int, Int, Int, Int, Int]):
        self = Self(t[0], t[1], t[2], t[3], t[4])

    # ===-------------------------------------------------------------------===#
    # Helpers
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def row_major_strides(ref shape: IntTuple) -> IntTuple:
        """Returns strides provided a shape in row major contiguous layout."""
        return reverse(prefix_product(reverse(shape)))

    def _handle_bounds(self, idx: Int, bound: Int, error_msg: TString) raises -> Int:
        """Normalises a possibly-negative index against `bound`."""
        var i = idx

        if i < 0:
            i += bound

        if i < 0 or i >= bound:
            raise Error(error_msg)

        return i

    def normalise_idx(self, idx: Int) raises -> Int:
        """Normalises a logical element index against `numel()`."""
        return self._handle_bounds(
            idx,
            self.numel(),
            t"Index {String(idx)} is out of bounds for layout with {String(self.numel())} elements",
        )

    def normalise_dim(self, dim: Int) raises -> Int:
        """Normalises a dimension index against `rank`."""
        return self._handle_bounds(
            dim,
            self.rank(),
            t"Dimension {String(dim)} is out of bounds for rank {String(self.rank())}",
        )

    def normalise_axes(self, axes: IntTuple) raises -> IntTuple:
        """Normalises a tuple of dimension indices `rank`."""
        if len(axes) != self.rank():
            raise Error(
                t"Invalid permutation: expected {String(self.rank())} axes, got {String(len(axes))}: {String(axes)}."
            )

        var out = IntTuple()

        for i in range(self.rank()):
            out.append(IntTuple(self.normalise_dim(axes.value(i))))

        return out

    @staticmethod
    def arange(rank: Int, out tupl: IntTuple):
        """Returns the identity axis order `(0, 1, ..., rank - 1)`."""

        tupl = IntTuple()
        for i in range(rank):
            tupl.append(IntTuple(i))

    # ===-------------------------------------------------------------------===#
    # Getters
    # ===-------------------------------------------------------------------===#

    @always_inline
    def rank(self) -> Int:
        """Returns the rank of the layout."""
        return self._rank

    @always_inline
    def numel(self) -> Int:
        """Returns the total number of elements in layout."""
        return product(self.shape())

    @always_inline
    def stride(self) -> IntTuple:
        """Returns the strides of the layout."""
        return self._strides

    @always_inline
    def stride(self, idx: Int) raises -> Int:
        """Returns the specified stride of the layout."""
        return self._strides.value(self.normalise_dim(idx))

    @always_inline
    def shape(self, idx: Int) -> Int:
        return self._shape.value(idx)

    @always_inline
    def shape(self) -> IntTuple:
        return self._shape

    @always_inline
    def is_contiguous(self) -> Bool:
        """Returns true if layout is contiguous in memory."""
        return self.stride() == Self.row_major_strides(self.shape())

    def batch_dims_collapsible(self) raises -> Bool:
        """True if every batch axis (positions 0..rank-3) nests row-major
        into the batch axis right after it, so all of them fold into one
        flat dim of stride `self.stride(rank - 3)`.

        A single batch axis (rank == 3) is always representable as-is,
        however it's strided, i.e. there's nothing to collapse. This only
        matters once there are >= 2 batch axes (rank >= 4): e.g.
        `transpose(1, 2)` on a (B, T, H, Dh) tensor scrambles B's and H's
        relative memory order, so they can no longer be addressed via a single
        combined stride.

        A broadcast (stride-0) batch axis sitting next to a non-broadcast one
        will also report False, since 0 won't equal the neighbor's
        `shape * stride`. If all batch axes are broadcast (or there's only
        one), this trivially returns True, with stride 0 as the folded dim's
        stride.
        """
        var rank = self.rank()
        for i in range(rank - 4, -1, -1):
            if self.stride(i) != self.shape(i + 1) * self.stride(i + 1):
                return False
        return True

    def as_contiguous(self) -> Layout:
        """Returns a new row-major layout with the same shape, zeroing base_offset."""
        return Layout(self.rank(), self.shape(), Self.row_major_strides(self.shape()), 0)

    def inner_sizes(self) -> IntTuple:
        """Suffix products of shape, used to decompose a flat index into coordinates.

        For shape (3, 4, 5) returns (20, 5, 1).
        """
        var result = IntTuple()
        var acc = 1
        for i in range(self.rank() - 1, -1, -1):
            result.append(IntTuple(acc))
            acc *= self.shape(i)
        return reverse(result)

    def shape_ptr(self) -> UnsafePointer[Int, MutAnyOrigin]:
        """Returns a heap-allocated array of `rank` ints containing the shape dimensions."""
        var p = alloc[Int](self.rank())
        for i in range(self.rank()):
            p[i] = self.shape(i)
        return p

    def stride_ptr(self) -> UnsafePointer[Int, MutAnyOrigin]:
        """Returns a heap-allocated array of `rank` ints containing the strides."""
        var p = alloc[Int](self.rank())
        for i in range(self.rank()):
            p[i] = self._strides.value(i)
        return p

    def strides_buffer(self, ctx: DeviceContext) raises -> DeviceBuffer[DType.int64]:
        """Uploads strides to a device buffer."""
        var host = List[Int64](capacity=self.rank())
        for i in range(self.rank()):
            host.append(Int64(self.stride(i)))
        var buf = ctx.enqueue_create_buffer[DType.int64](self.rank())
        buf.enqueue_copy_from(Span(host))
        return buf^

    def inner_sizes_buffer(self, ctx: DeviceContext) raises -> DeviceBuffer[DType.int64]:
        """Uploads inner_sizes to a device buffer."""
        var inner = self.inner_sizes()
        var host = List[Int64](capacity=self.rank())
        for i in range(self.rank()):
            host.append(Int64(inner.value(i)))
        var buf = ctx.enqueue_create_buffer[DType.int64](self.rank())
        buf.enqueue_copy_from(Span(host))
        return buf^

    def reduce_dims(self, axis: Int) raises -> Tuple[Int, Int, Int]:
        """Returns (outer, reduce_size, inner) for a reduction along axis.

        outer      = product of dims before axis
        reduce_size = shape[axis]
        inner      = product of dims after axis
        """
        var ax = self.normalise_dim(axis)
        var reduce_size = self.shape(ax)
        var inner = self.inner_sizes().value(ax)
        var outer = self.numel() // (reduce_size * inner)
        return (outer, reduce_size, inner)

    def expand_axis(self, axis: Int, size: Int) raises -> Self:
        """Returns a layout with a new broadcast dimension inserted at axis.

        The inserted dimension has stride 0 so all elements along it map to the
        same underlying element. Existing strides are preserved unchanged.
        """
        var ax = self._handle_bounds(
            axis, self.rank() + 1, t"Axis {String(axis)} out of bounds for rank {String(self.rank())}"
        )
        var new_shape = IntTuple()
        var new_strides = IntTuple()
        for i in range(self.rank() + 1):
            if i == ax:
                new_shape.append(IntTuple(size))
                new_strides.append(IntTuple(0))
            else:
                var src = i if i < ax else i - 1
                new_shape.append(IntTuple(self.shape(src)))
                new_strides.append(IntTuple(self._strides.value(src)))
        return Self(self.rank() + 1, new_shape, new_strides, self.base_offset)

    def reduce_output_shape(self, axis: Int, keepdim: Bool) raises -> Self:
        """Returns the output layout after reducing along axis.

        With keepdim=False the axis dimension is removed; with keepdim=True it
        is replaced by 1. A rank-0 result (1-D tensor, keepdim=False) is
        represented as shape (1,) to match the scalar convention used elsewhere.
        """
        var ax = self.normalise_dim(axis)
        var new_shape = IntTuple()
        for i in range(self.rank()):
            if i == ax:
                if keepdim:
                    new_shape.append(IntTuple(1))
            else:
                new_shape.append(IntTuple(self.shape(i)))
        if len(new_shape) == 0:
            new_shape.append(IntTuple(1))
        var new_rank = len(new_shape)
        return Self(new_rank, new_shape, Self.row_major_strides(new_shape), 0)

    # ===-------------------------------------------------------------------===#
    # Permute
    # ===-------------------------------------------------------------------===#

    def permute(self, *axes: Int, out layout: Self) raises:
        """Returns a new layout view with its dimensions permuted."""
        layout = self.permute(IntTuple(*axes))

    def permute(self, axes: IntTuple, out layout: Self) raises:
        """Returns a new layout view with its dimensions permuted."""

        if len(axes) != self.rank():
            raise Error(
                t"Permutation rank must match layout rank: {String(len(axes))} axes != {String(self.rank())} rank."
            )

        var order = IntTuple()

        for i in range(self.rank()):
            order.append(IntTuple(self.normalise_dim(axes[i].value())))

        if sorted(order) != Self.arange(self.rank()):
            raise Error(t"Invalid permutation {String(order)} for rank {String(self.rank())}.")

        var new_shape = IntTuple()
        var new_strides = IntTuple()

        for i in range(self.rank()):
            var axis = order.value(i)

            new_shape.append(IntTuple(self.shape(axis)))
            new_strides.append(IntTuple(self.stride(axis)))

        layout = Self(self.rank(), new_shape, new_strides, self.base_offset)

    # ===-------------------------------------------------------------------===#
    # Transpose
    # ===-------------------------------------------------------------------===#

    def transpose(self, dim0: Int = -2, dim1: Int = -1, out layout: Self) raises:
        """Return a view with two dimensions swapped."""

        var d0 = self.normalise_dim(dim0)
        var d1 = self.normalise_dim(dim1)

        if d0 == d1:
            raise Error(t"Cannot transpose the same dimension {String(d0)}.")
        var axes = Self.arange(self.rank())

        axes = axes.replace_entry(d0, IntTuple(d1))
        axes = axes.replace_entry(d1, IntTuple(d0))

        layout = self.permute(axes)

    # ===-------------------------------------------------------------------===#
    # Permutation detection
    # ===-------------------------------------------------------------------===#

    def permutation_of_contiguous(self) raises -> Optional[IntTuple]:
        """Returns the axis order that recovers a contiguous layout, if one exists.

        The returned order lists axes from slowest- to fastest-varying (i.e. by
        stride descending). If this layout is exactly a contiguous buffer viewed
        through that axis order, `compact_order` reconstructs its strides
        exactly and the order is returned. Otherwise returns None — this
        rejects broadcasts (stride 0 on a dim with size > 1 can never match a
        compact stride) and slices (a narrowed stride can never match either).
        """

        def by_stride_desc(a: IntTuple, b: IntTuple) -> Bool:
            return a.value(0) > b.value(0)

        var pairs = IntTuple()
        for i in range(self.rank()):
            pairs.append(IntTuple(self.stride(i), i))

        var sorted_pairs = sorted[by_stride_desc](pairs)

        var axis_order = IntTuple()
        var rank_of = IntTuple(num_elems=self.rank())
        for pos in range(self.rank()):
            var axis = sorted_pairs[pos].value(1)
            axis_order.append(IntTuple(axis))
            rank_of.replace_entry(axis, int_value=self.rank() - 1 - pos)

        if compact_order(self.shape(), rank_of) != self.stride():
            return None
        return axis_order

    # ===-------------------------------------------------------------------===#
    # View
    # ===-------------------------------------------------------------------===#

    def view(self, *shape: Int, out layout: Self) raises:
        """Returns a layout view with a different shape."""
        layout = self.view(IntTuple(*shape))

    def view(self, shape: IntTuple, out layout: Self) raises:
        """Returns a layout view with a different shape.

        The new shape must have the same number of elements and must be compatible
        with the current shape and strides. No data is copied.
        """

        var inferred_shape = Self._infer_view_shape(shape, self.numel())
        if product(inferred_shape) != self.numel():
            raise Error(
                t"Cannot view layout with {String(self.numel())} elements "
                t"as shape {String(shape)} with {String(product(shape))} elements."
            )

        # Scalar / empty-shape edge case.
        if self.rank() == 0:
            var scalar_strides = IntTuple()
            for _ in range(len(inferred_shape)):
                scalar_strides.append(IntTuple(1))

            layout = Self(len(inferred_shape), inferred_shape, scalar_strides, self.base_offset)
            return

        var new_strides = IntTuple(num_elems=len(inferred_shape))
        var view_d = len(inferred_shape) - 1
        var block_base_stride = self.stride(self.rank() - 1)
        var tensor_numel = 1  # Number of logical elements accumulated in the current old contiguous block
        var view_numel = 1  # Number of logical elements consumed from the new shape for this contiguous.

        for tensor_d in range(self.rank() - 1, -1, -1):
            tensor_numel *= self.shape(tensor_d)

            # A stride-compatible boundary is at tensor_d == 0, or when the previous old
            # dimension cannot collapse into the current block:
            #     stride[tensor_d - 1] != tensor_numel * block_base_stride
            #
            # This is equivalent to:
            #     stride[i] == stride[i + 1] * shape[i + 1]
            var end_of_block = tensor_d == 0

            if not end_of_block:
                if self.shape(tensor_d - 1) != 1:
                    if self.stride(tensor_d - 1) != tensor_numel * block_base_stride:
                        end_of_block = True

            if end_of_block:
                # Consume new dimensions until this new-shape block has the same
                # number of elements as the current old contiguous block.
                #
                # The `shape.value(view_d) == 1` case lets size-1 dimensions be
                # inserted without changing the represented storage span.
                while view_d >= 0 and (view_numel < tensor_numel or inferred_shape.value(view_d) == 1):
                    new_strides.replace_entry(
                        view_d,
                        int_value=view_numel * block_base_stride,
                    )

                    view_numel *= inferred_shape.value(view_d)
                    view_d -= 1

                if view_numel != tensor_numel:
                    raise Error(
                        t"Cannot view shape {String(self.shape())} with strides "
                        t"{String(self.stride())} as shape {String(shape)}."
                    )

                if tensor_d > 0:
                    block_base_stride = self.stride(tensor_d - 1)
                    tensor_numel = 1
                    view_numel = 1

        if view_d != -1:
            raise Error(
                t"Cannot view shape {String(self.shape())} with strides {String(self.stride())} as shape"
                t" {String(shape)}."
            )

        layout = Self(len(inferred_shape), inferred_shape, new_strides, self.base_offset)

    @staticmethod
    def _infer_view_shape(shape: IntTuple, numel: Int) raises -> IntTuple:
        var inferred = IntTuple()
        var known = 1
        var infer_dim = -1

        for i in range(len(shape)):
            var dim = shape.value(i)

            if dim == -1:
                if infer_dim != -1:
                    raise Error(t"Only one dimension can be inferred in shape {String(shape)}.")
                infer_dim = i
                inferred.append(IntTuple(-1))

            elif dim < 0:
                raise Error(t"Invalid negative dimension {String(dim)} in shape {String(shape)}.")

            else:
                known *= dim
                inferred.append(IntTuple(dim))

        if infer_dim != -1:
            if known == 0:
                raise Error(t"Cannot infer reshape dimension for shape {String(shape)}.")

            if numel % known != 0:
                raise Error(t"Cannot infer reshape dimension for {String(numel)} elements and shape {String(shape)}.")

            inferred.replace_entry(
                infer_dim,
                int_value=numel // known,
            )

        return inferred

    # ===-------------------------------------------------------------------===#
    # Reshape
    # ===-------------------------------------------------------------------===#

    def reshape(self, *shape: Int, out layout: Self) raises:
        """Returns a contiguous layout view with a new shape.

        At the layout level, reshape is view-only. Tensor.reshape can later copy
        when a view is not possible.
        """
        layout = self.view(IntTuple(*shape))

    # ===-------------------------------------------------------------------===#
    # Slice
    # ===-------------------------------------------------------------------===#

    def __getitem__(self, *slices: StridedSlice, out layout: Self) raises:
        """Returns a strided slice view."""

        if len(slices) > self.rank():
            raise Error(t"Invalid slice: expected {String(self.rank())} slices, got {String(len(slices))}.")

        var new_shape = IntTuple()
        var new_strides = IntTuple()
        var new_base_offset = self.base_offset

        for dim in range(self.rank()):
            var dim_size = self.shape(dim)

            var start: Int
            var stop: Int
            var step: Int

            if dim < len(slices):
                start, stop, step = slices[dim].indices(dim_size)
            else:
                # Missing trailing dimensions become full slices
                start = 0
                stop = dim_size
                step = 1

            var size: Int

            if step > 0:
                if stop <= start:
                    size = 0
                else:
                    size = (stop - start + step - 1) // step
            else:
                var neg_step = -step

                if start <= stop:
                    size = 0
                else:
                    size = (start - stop + neg_step - 1) // neg_step

            new_base_offset += start * self.stride(dim)
            new_shape.append(IntTuple(size))
            new_strides.append(IntTuple(self.stride(dim) * step))

        layout = Self(self.rank(), new_shape, new_strides, new_base_offset)
