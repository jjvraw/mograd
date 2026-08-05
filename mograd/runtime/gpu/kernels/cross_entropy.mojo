from std.gpu.primitives.warp import max as warp_max, sum as warp_sum
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation

from mograd.runtime.gpu.kernels.strided import strided_offset

# ===-------------------------------------------------------------------===#
# Cross Entropy
# ===-------------------------------------------------------------------===#


def cross_entropy_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
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
        var v = logits[row_offset + i]
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[w] > smem[0]:
                smem[0] = smem[w]
    barrier()
    var global_max = smem[0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += exp(logits[row_offset + i] - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        smem[0] = log(smem[0])
    barrier()
    var log_sum_exp = smem[0]

    var local_loss = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_loss += (logits[row_offset + i] - global_max - log_sum_exp) * labels[row_offset + i]
    local_loss = warp_sum(local_loss)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_loss
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        dst[row] = -smem[0] / Scalar[dtype](N)


def cross_entropy_kernel_strided[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
    rank: Int,
    la_inner: UnsafePointer[Int64, ImmutAnyOrigin],
    la_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    lb_inner: UnsafePointer[Int64, ImmutAnyOrigin],
    lb_strides: UnsafePointer[Int64, ImmutAnyOrigin],
) where dtype.is_floating_point():
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
        var v = logits.load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[w] > smem[0]:
                smem[0] = smem[w]
    barrier()
    var global_max = smem[0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += exp(logits.load(strided_offset(row_offset + i, rank, la_inner, la_strides)) - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        smem[0] = log(smem[0])
    barrier()
    var log_sum_exp = smem[0]

    var local_loss = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        var logit = logits.load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        var label = labels.load(strided_offset(row_offset + i, rank, lb_inner, lb_strides))
        local_loss += (logit - global_max - log_sum_exp) * label
    local_loss = warp_sum(local_loss)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_loss
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        dst[row] = -smem[0] / Scalar[dtype](N)


def sum_rows_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    src: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
) where dtype.is_floating_point():
    from std.gpu import lane_id, WARP_SIZE

    var smem = stack_allocation[BLOCK_SIZE // WARP_SIZE, Scalar[dtype], address_space=AddressSpace.SHARED]()
    var tid = thread_idx.x
    var local = Scalar[dtype](0)
    for i in range(tid, N, BLOCK_SIZE):
        local += src[i]
    local = warp_sum(local)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
        dst[0] = smem[0]


# ===-------------------------------------------------------------------===#
# Cross Entropy grad
# ===-------------------------------------------------------------------===#


def cross_entropy_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
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
        var v = logits[row_offset + i]
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[w] > smem[0]:
                smem[0] = smem[w]
    barrier()
    var global_max = smem[0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += math_exp(logits[row_offset + i] - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
    barrier()
    var sm_scale = Scalar[dtype](1) / smem[0]
    var d_by_nrows = grad[0] / Scalar[dtype](N)

    for i in range(tid, C, BLOCK_SIZE):
        dst[row_offset + i] = (
            math_exp(logits[row_offset + i] - global_max) * sm_scale - labels[row_offset + i]
        ) * d_by_nrows


def cross_entropy_grad_kernel_strided[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
    rank: Int,
    la_inner: UnsafePointer[Int64, ImmutAnyOrigin],
    la_strides: UnsafePointer[Int64, ImmutAnyOrigin],
    lb_inner: UnsafePointer[Int64, ImmutAnyOrigin],
    lb_strides: UnsafePointer[Int64, ImmutAnyOrigin],
) where dtype.is_floating_point():
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
        var v = logits.load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        if v > local_max:
            local_max = v
    local_max = warp_max(local_max)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_max
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            if smem[w] > smem[0]:
                smem[0] = smem[w]
    barrier()
    var global_max = smem[0]

    var local_sum = Scalar[dtype](0)
    for i in range(tid, C, BLOCK_SIZE):
        local_sum += math_exp(logits.load(strided_offset(row_offset + i, rank, la_inner, la_strides)) - global_max)
    local_sum = warp_sum(local_sum)
    if lane_id() == 0:
        smem[tid // WARP_SIZE] = local_sum
    barrier()
    if tid == 0:
        for w in range(1, BLOCK_SIZE // WARP_SIZE):
            smem[0] += smem[w]
    barrier()
    var sm_scale = Scalar[dtype](1) / smem[0]
    var d_by_nrows = grad[0] / Scalar[dtype](N)

    for i in range(tid, C, BLOCK_SIZE):
        var logit = logits.load(strided_offset(row_offset + i, rank, la_inner, la_strides))
        var label = labels.load(strided_offset(row_offset + i, rank, lb_inner, lb_strides))
        dst[row_offset + i] = (math_exp(logit - global_max) * sm_scale - label) * d_by_nrows
