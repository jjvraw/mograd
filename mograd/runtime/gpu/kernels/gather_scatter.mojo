from max.algorithm.functional import elementwise
from std.atomic import Atomic
from std.gpu.host import DeviceContext

from layout import Coord

from mograd.runtime.gpu.kernels.utils import strided_offset

# ===-------------------------------------------------------------------===#
# Tensor.gather
# ===-------------------------------------------------------------------===#
#
# TODO: Indices are always int64 for now.


def gather[
    dtype: DType
](
    src: UnsafePointer[mut=False, Scalar[dtype], _],
    indices: UnsafePointer[mut=False, Scalar[DType.int64], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    n: Int,
    row_size: Int,
    src_row_stride: Int,
    src_col_stride: Int,
    idx_rank: Int,
    idx_inner: UnsafePointer[mut=False, Int64, _],
    idx_strides: UnsafePointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var row = flat // row_size
        var col = flat % row_size
        var idx_off = strided_offset(row, idx_rank, idx_inner, idx_strides)
        var src_row = Int(indices.load(idx_off))
        var src_off = src_row * src_row_stride + col * src_col_stride
        dst.store(flat, src.load(src_off))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Tensor.scatter_add
# ===-------------------------------------------------------------------===#


def scatter_add[
    dtype: DType
](
    indices: UnsafePointer[mut=False, Scalar[DType.int64], _],
    values: UnsafePointer[mut=False, Scalar[dtype], _],
    dst: UnsafePointer[mut=True, Scalar[dtype], _],
    n: Int,
    row_size: Int,
    idx_rank: Int,
    idx_inner: UnsafePointer[mut=False, Int64, _],
    idx_strides: UnsafePointer[mut=False, Int64, _],
    values_rank: Int,
    values_inner: UnsafePointer[mut=False, Int64, _],
    values_strides: UnsafePointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var row = flat // row_size
        var idx_off = strided_offset(row, idx_rank, idx_inner, idx_strides)
        var dst_row = Int(indices.load(idx_off))
        var values_off = strided_offset(flat, values_rank, values_inner, values_strides)
        var dst_ptr = dst + (dst_row * row_size + (flat % row_size))
        _ = Atomic.fetch_add(dst_ptr, values.load(values_off))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)
