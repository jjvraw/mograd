from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from layout import Coord
from mograd.runtime.gpu.kernels.strided import strided_offset
from std.atomic import Atomic

# ===-------------------------------------------------------------------===#
# Tensor.gather
# ===-------------------------------------------------------------------===#
#
# TODO: Indices are always int64 for now.


def gather[
    dtype: DType
](
    src: Pointer[mut=False, Scalar[dtype], _],
    indices: Pointer[mut=False, Scalar[DType.int64], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    n: Int,
    row_size: Int,
    src_row_stride: Int,
    src_col_stride: Int,
    idx_rank: Int,
    idx_inner: Pointer[mut=False, Int64, _],
    idx_strides: Pointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises:
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var row = flat // row_size
        var col = flat % row_size
        var idx_off = strided_offset(row, idx_rank, idx_inner, idx_strides)
        var src_row = Int(indices.unsafe_load(idx_off))
        var src_off = src_row * src_row_stride + col * src_col_stride
        dst.unsafe_store(flat, src.unsafe_load(src_off))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)


# ===-------------------------------------------------------------------===#
# Tensor.scatter_add
# ===-------------------------------------------------------------------===#


def scatter_add[
    dtype: DType
](
    indices: Pointer[mut=False, Scalar[DType.int64], _],
    values: Pointer[mut=False, Scalar[dtype], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    n: Int,
    row_size: Int,
    idx_rank: Int,
    idx_inner: Pointer[mut=False, Int64, _],
    idx_strides: Pointer[mut=False, Int64, _],
    values_rank: Int,
    values_inner: Pointer[mut=False, Int64, _],
    values_strides: Pointer[mut=False, Int64, _],
    ctx: DeviceContext,
) raises where dtype.is_floating_point():
    def apply[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var row = flat // row_size
        var idx_off = strided_offset(row, idx_rank, idx_inner, idx_strides)
        var dst_row = Int(indices.unsafe_load(idx_off))
        var values_off = strided_offset(flat, values_rank, values_inner, values_strides)
        var dst_ptr = dst.unsafe_offset((dst_row * row_size + (flat % row_size)))
        _ = Atomic.fetch_add(dst_ptr, values.unsafe_load(values_off))

    elementwise[simd_width=1, target="gpu"](apply, Coord(n), ctx)
