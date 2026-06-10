from std.algorithm.functional import elementwise

# ===-------------------------------------------------------------------===#
# Add
# ===-------------------------------------------------------------------===#


def add[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    comptime width = simd_width_of[dtype, target=get_gpu_target()]()

    def apply_fast[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        dst.store[simd_width](flat, a.load[simd_width](flat) + b.load[simd_width](flat))

    elementwise[simd_width=width, target="gpu"](apply_fast, Coord(n), ctx)


def add_strided[
    dtype: DType
](
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    rank: Int,
    inner: UnsafePointer[Int64, MutAnyOrigin],
    sa: UnsafePointer[Int64, MutAnyOrigin],
    sb: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    def apply_strided[simd_width: Int, alignment: Int = 1](coord: Coord) {var}:
        var flat = Int(coord[0].value())
        var rem = flat
        var a_off = 0
        var b_off = 0
        for i in range(rank):
            var idx = rem // Int(inner[i])
            rem %= Int(inner[i])
            a_off += idx * Int(sa[i])
            b_off += idx * Int(sb[i])
        dst.store(flat, a.load(a_off) + b.load(b_off))

    elementwise[simd_width=1, target="gpu"](apply_strided, Coord(n), ctx)
