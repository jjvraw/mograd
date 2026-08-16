from std.gpu.primitives.warp import max as warp_max, sum as warp_sum
from std.gpu import thread_idx, block_idx
from max.gpu.sync import barrier
from std.memory import AddressSpace
from std.memory import stack_allocation

from mograd.runtime.gpu.kernels.strided import strided_offset

# ===-------------------------------------------------------------------===#
# Cross Entropy
# ===-------------------------------------------------------------------===#


def cross_entropy_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: Pointer[Scalar[dtype], ImmutAnyOrigin],
    labels: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    N_arg: Int64,
    C_arg: Int64,
) where dtype.is_floating_point():
    var N = Int(N_arg)
    var C = Int(C_arg)
    from std.math import exp, log
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tid = thread_idx.x

    var local_max = Scalar[dtype].MIN
    for i in range(tid, C, BLOCK_SIZE):
        var v = logits[unsafe_offset=row_offset + i]
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[unsafe_offset=w] > smem[unsafe_offset=0]:
                smem[unsafe_offset=0] = smem[unsafe_offset=w]
    barrier()
    var global_max = smem[unsafe_offset=0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += exp(logits[unsafe_offset=row_offset + i] - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[unsafe_offset=0] += smem[unsafe_offset=w]
        smem[unsafe_offset=0] = log(smem[unsafe_offset=0])
    barrier()
    var log_sum_exp = smem[unsafe_offset=0]

    var local_loss = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_loss += (logits[unsafe_offset=row_offset + i] - global_max - log_sum_exp) * labels[
            unsafe_offset=row_offset + i
        ]
    local_loss = warp_sum(local_loss)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_loss
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[unsafe_offset=0] += smem[unsafe_offset=w]
        dst[unsafe_offset=row] = -smem[unsafe_offset=0] / Scalar[dtype](N)


def cross_entropy_kernel_strided[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: Pointer[Scalar[dtype], ImmutAnyOrigin],
    labels: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    N_arg: Int64,
    C_arg: Int64,
    rank_arg: Int64,
    la_inner: Pointer[Int64, ImmutAnyOrigin],
    la_strides: Pointer[Int64, ImmutAnyOrigin],
    lb_inner: Pointer[Int64, ImmutAnyOrigin],
    lb_strides: Pointer[Int64, ImmutAnyOrigin],
) where dtype.is_floating_point():
    var N = Int(N_arg)
    var C = Int(C_arg)
    var rank = Int(rank_arg)
    # logits/labels always share shape but may carry independent strides
    # (e.g. labels coming through a one_hot/cast chain), so each gets its
    # own strided_offset rather than assuming a shared flat layout.
    from std.math import exp, log
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tid = thread_idx.x

    var local_max = Scalar[dtype].MIN
    for i in range(tid, C, BLOCK_SIZE):
        var v = logits.unsafe_load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[unsafe_offset=w] > smem[unsafe_offset=0]:
                smem[unsafe_offset=0] = smem[unsafe_offset=w]
    barrier()
    var global_max = smem[unsafe_offset=0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += exp(logits.unsafe_load(strided_offset(row_offset + i, rank, la_inner, la_strides)) - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[unsafe_offset=0] += smem[unsafe_offset=w]
        smem[unsafe_offset=0] = log(smem[unsafe_offset=0])
    barrier()
    var log_sum_exp = smem[unsafe_offset=0]

    var local_loss = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        var logit = logits.unsafe_load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        var label = labels.unsafe_load(strided_offset(row_offset + i, rank, lb_inner, lb_strides))
        local_loss += (logit - global_max - log_sum_exp) * label
    local_loss = warp_sum(local_loss)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_loss
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[unsafe_offset=0] += smem[unsafe_offset=w]
        dst[unsafe_offset=row] = -smem[unsafe_offset=0] / Scalar[dtype](N)


def sum_rows_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    src: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    N_arg: Int64,
) where dtype.is_floating_point():
    var N = Int(N_arg)
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var local = Scalar[dtype](0)
    for i in range(tid, N, BLOCK_SIZE):
        local += src[unsafe_offset=i]
    local = warp_sum(local)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[unsafe_offset=0] += smem[unsafe_offset=w]
        dst[unsafe_offset=0] = smem[unsafe_offset=0]


# ===-------------------------------------------------------------------===#
# Cross Entropy grad
# ===-------------------------------------------------------------------===#


def cross_entropy_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    logits: Pointer[Scalar[dtype], ImmutAnyOrigin],
    labels: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    N_arg: Int64,
    C_arg: Int64,
) where dtype.is_floating_point():
    var N = Int(N_arg)
    var C = Int(C_arg)
    from std.math import exp as math_exp
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tid = thread_idx.x

    var local_max = Scalar[dtype].MIN
    for i in range(tid, C, BLOCK_SIZE):
        var v = logits[unsafe_offset=row_offset + i]
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[unsafe_offset=w] > smem[unsafe_offset=0]:
                smem[unsafe_offset=0] = smem[unsafe_offset=w]
    barrier()
    var global_max = smem[unsafe_offset=0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += math_exp(logits[unsafe_offset=row_offset + i] - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[unsafe_offset=0] += smem[unsafe_offset=w]
    barrier()
    var sm_scale = Scalar[dtype](1) / smem[unsafe_offset=0]
    var d_by_nrows = grad[unsafe_offset=0] / Scalar[dtype](N)

    for i in range(tid, C, BLOCK_SIZE):
        dst[unsafe_offset=row_offset + i] = (
            math_exp(logits[unsafe_offset=row_offset + i] - global_max) * sm_scale
            - labels[unsafe_offset=row_offset + i]
        ) * d_by_nrows


def cross_entropy_grad_kernel_strided[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: Pointer[Scalar[dtype], ImmutAnyOrigin],
    logits: Pointer[Scalar[dtype], ImmutAnyOrigin],
    labels: Pointer[Scalar[dtype], ImmutAnyOrigin],
    dst: Pointer[Scalar[dtype], MutAnyOrigin],
    N_arg: Int64,
    C_arg: Int64,
    rank_arg: Int64,
    la_inner: Pointer[Int64, ImmutAnyOrigin],
    la_strides: Pointer[Int64, ImmutAnyOrigin],
    lb_inner: Pointer[Int64, ImmutAnyOrigin],
    lb_strides: Pointer[Int64, ImmutAnyOrigin],
) where dtype.is_floating_point():
    var N = Int(N_arg)
    var C = Int(C_arg)
    var rank = Int(rank_arg)
    from std.math import exp as math_exp
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C
    var tid = thread_idx.x

    var local_max = Scalar[dtype].MIN
    for i in range(tid, C, BLOCK_SIZE):
        var v = logits.unsafe_load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[unsafe_offset=w] > smem[unsafe_offset=0]:
                smem[unsafe_offset=0] = smem[unsafe_offset=w]
    barrier()
    var global_max = smem[unsafe_offset=0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += math_exp(
            logits.unsafe_load(strided_offset(row_offset + i, rank, la_inner, la_strides)) - global_max
        )
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[unsafe_offset=tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[unsafe_offset=0] += smem[unsafe_offset=w]
    barrier()
    var sm_scale = Scalar[dtype](1) / smem[unsafe_offset=0]
    var d_by_nrows = grad[unsafe_offset=0] / Scalar[dtype](N)

    for i in range(tid, C, BLOCK_SIZE):
        var logit = logits.unsafe_load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        var label = labels.unsafe_load(strided_offset(row_offset + i, rank, lb_inner, lb_strides))
        dst[unsafe_offset=row_offset + i] = (math_exp(logit - global_max) * sm_scale - label) * d_by_nrows
