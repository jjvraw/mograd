from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext

from layout import Coord

from mograd.buffer import AnyBuffer, BufferArm
from mograd.layout import Layout

# ===-------------------------------------------------------------------===#
# Strided offset helpers
# ===-------------------------------------------------------------------===#


@always_inline
def strided_offset(
    flat: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    strides: Pointer[mut=False, Int64, _],
) -> Int:
    var rem = flat
    var off = 0
    for i in range(rank):
        var idx = rem // Int(inner[unsafe_offset=i])
        rem %= Int(inner[unsafe_offset=i])
        off += idx * Int(strides[unsafe_offset=i])
    return off


@always_inline
def strided_offsets(
    flat: Int,
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    sb: Pointer[mut=False, Int64, _],
) -> Tuple[Int, Int]:
    var rem = flat
    var a_off = 0
    var b_off = 0
    for i in range(rank):
        var idx = rem // Int(inner[unsafe_offset=i])
        rem %= Int(inner[unsafe_offset=i])
        a_off += idx * Int(sa[unsafe_offset=i])
        b_off += idx * Int(sb[unsafe_offset=i])
    return (a_off, b_off)


# ===-------------------------------------------------------------------===#
# Generic strided GPU kernels
# ===-------------------------------------------------------------------===#


def unary_strided_map[
    dtype: DType,
    op: def(Scalar[dtype]) thin -> Scalar[dtype],
](
    a: Pointer[mut=False, Scalar[dtype], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.unsafe_store(flat, op(a.unsafe_load(strided_offset(flat, rank, inner, sa))))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


def binary_strided_scalar_map[
    dtype: DType,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](
    a: Pointer[mut=False, Scalar[dtype], _],
    b: Pointer[mut=False, Scalar[dtype], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    n: Int,
    ctx: DeviceContext,
) raises:
    var scalar_val = b.unsafe_load(0)

    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.unsafe_store(flat, op(a.unsafe_load(strided_offset(flat, rank, inner, sa)), scalar_val))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


def binary_strided_map[
    dtype: DType,
    op: def(Scalar[dtype], Scalar[dtype]) thin -> Scalar[dtype],
](
    a: Pointer[mut=False, Scalar[dtype], _],
    b: Pointer[mut=False, Scalar[dtype], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    sb: Pointer[mut=False, Int64, _],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var offs = strided_offsets(flat, rank, inner, sa, sb)
        dst.unsafe_store(flat, op(a.unsafe_load(offs[0]), b.unsafe_load(offs[1])))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)


def strided_copy_map[
    dtype: DType,
](
    a: Pointer[mut=False, Scalar[dtype], _],
    dst: Pointer[mut=True, Scalar[dtype], _],
    rank: Int,
    inner: Pointer[mut=False, Int64, _],
    sa: Pointer[mut=False, Int64, _],
    sd: Pointer[mut=False, Int64, _],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var offs = strided_offsets(flat, rank, inner, sa, sd)
        dst.unsafe_store(offs[1], a.unsafe_load(offs[0]))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)
