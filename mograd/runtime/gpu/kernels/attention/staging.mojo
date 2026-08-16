# Shared staging helpers for the attention kernels.
#
# These factor the body of a staging operation only. Loops, coordinates,
# commit/wait placement, barriers, and the choice between cp.async and
# register-path staging stay at each call site: those are exactly the
# scheduling decisions performance work keeps needing to vary per site.
#
# NOTE: nvidia_bwd.mojo deliberately keeps its staging loops raw: at its register
# ceiling, every reshaping through these helpers measured slower even though
# they inline (the forward absorbs them at zero cost, im not sure why).
# Re-verify with the bench before introducing them there.

from max.gpu.memory import CacheEviction, async_copy
from std.memory import AddressSpace
from std.sys import size_of

comptime COPY_VEC = 8  # 16 bytes of half elements per staged chunk


@always_inline
def chunk_valid(row_ok: Bool, col: Int, bound: Int) -> Int:
    """In-bounds element count of the 8-wide chunk at col, 0 for OOB rows."""
    var v = bound - col
    v = COPY_VEC if v > COPY_VEC else (0 if v < 0 else v)
    return v if row_ok else 0


@always_inline
def stage_chunk_async[
    dtype: DType, //, eviction: CacheEviction = CacheEviction.EVICT_NORMAL
](
    src: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    valid_elems: Int,
):
    """One 16B cp.async chunk, global to shared.

    Bypasses the register file: no scoreboard entry and no LSU issue slot on
    the critical path. valid_elems < COPY_VEC zero-fills the tail and 0
    reads nothing, which covers OOB rows and ragged row tails in a single
    instruction. The caller owns async_copy_commit_group / wait placement.
    """
    async_copy[16, fill=Scalar[dtype](0), eviction_policy=eviction](
        src.unsafe_address_space_cast[AddressSpace.GLOBAL](),
        dst,
        Int32(size_of[Scalar[dtype]]() * valid_elems),
    )


@always_inline
def stage_chunk_elems[
    dtype: DType, //
](
    src: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutUntrackedOrigin, address_space=AddressSpace.SHARED],
    row_ok: Bool,
    col: Int,
    bound: Int,
):
    """Register-path per-element fallback for rows that are not 16B aligned."""
    comptime for e in range(COPY_VEC):
        var ok = row_ok and col + e < bound
        dst[unsafe_offset=e] = src[unsafe_offset=e] if ok else Scalar[dtype](0)
