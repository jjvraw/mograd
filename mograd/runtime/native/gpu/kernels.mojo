from std.gpu import global_idx, thread_idx, block_idx, block_dim, barrier
from std.gpu.primitives.warp import max as warp_max, sum as warp_sum
from std.gpu.memory import AddressSpace, external_memory
from std.memory import stack_allocation
from std.math import exp, log
from std.atomic import Atomic

# ===-------------------------------------------------------------------===#
# GPU Kernels
# ===-------------------------------------------------------------------===#

# ===-------------------------------------------------------------------===#
# Transpose
# ===-------------------------------------------------------------------===#


def transpose_kernel[
    dtype: DType, BLOCK_SIZE: Int
](src: UnsafePointer[Scalar[dtype], MutAnyOrigin], dst: UnsafePointer[Scalar[dtype], MutAnyOrigin], M: Int, N: Int):
    var shmem = stack_allocation[BLOCK_SIZE * (BLOCK_SIZE + 1), dtype, address_space=AddressSpace.SHARED]()

    x = block_idx.x * BLOCK_SIZE + thread_idx.x
    y = block_idx.y * BLOCK_SIZE + thread_idx.y

    if x < N and y < M:
        shmem[thread_idx.y * (BLOCK_SIZE + 1) + thread_idx.x] = src[y * N + x]

    barrier()

    x_out = block_idx.y * BLOCK_SIZE + thread_idx.x
    y_out = block_idx.x * BLOCK_SIZE + thread_idx.y
    if x_out < M and y_out < N:
        dst[y_out * M + x_out] = shmem[thread_idx.x * (BLOCK_SIZE + 1) + thread_idx.y]


# ===-------------------------------------------------------------------===#
# Cross Entropy
# ===-------------------------------------------------------------------===#
# https://huggingface.co/kaisser/LLM-Maroc/blob/main/llama.cpp/ggml/src/ggml-cuda/cross-entropy-loss.cu


def cross_entropy_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C

    var tmp = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=4,
    ]()

    # Load into shared memory + find max
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        tmp[i] = val
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)

    barrier()

    # Sum of exp
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        sum_exp += exp(tmp[i] - max_val)
    sum_exp = warp_sum(sum_exp)
    var log_sum = log(sum_exp)

    # Loss
    var loss = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        loss += (tmp[i] - max_val - log_sum) * labels[row_offset + i]
    loss = -warp_sum(loss) / Scalar[dtype](N)

    if thread_idx.x == 0:
        dst[row] = loss


def cross_entropy_kernel_no_smem[
    dtype: DType, BLOCK_SIZE: Int
](
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C

    # Find max via GMEM
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)

    # Sum of exp via GMEM
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        sum_exp += exp(logits[row_offset + i] - max_val)
    sum_exp = warp_sum(sum_exp)
    var log_sum = log(sum_exp)

    # Loss via GMEM
    var loss = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        loss += (logits[row_offset + i] - max_val - log_sum) * labels[row_offset + i]
    loss = -warp_sum(loss) / Scalar[dtype](N)

    if thread_idx.x == 0:
        dst[row] = loss


def cross_entropy_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C

    var tmp = external_memory[
        Scalar[dtype],
        address_space=AddressSpace.SHARED,
        alignment=4,
    ]()

    # Load into shared + find max
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        tmp[i] = val
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)

    # Compute exp(logit - max), store back to tmp
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = exp(tmp[i] - max_val)
        sum_exp += val
        tmp[i] = val
    sum_exp = warp_sum(sum_exp)
    var sm_scale = Scalar[dtype](1.0) / sum_exp

    # Gradient = (softmax - label) * grad / nrows
    var d_by_nrows = grad[0] / Scalar[dtype](N)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        dst[row_offset + i] = (tmp[i] * sm_scale - labels[row_offset + i]) * d_by_nrows


def cross_entropy_grad_kernel_no_smem[
    dtype: DType, BLOCK_SIZE: Int
](
    grad: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    logits: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    labels: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    C: Int,
) where dtype.is_floating_point():
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * C

    # Find max via GMEM
    var max_val = Scalar[dtype](-1e38)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = logits[row_offset + i]
        if val > max_val:
            max_val = val
    max_val = warp_max(max_val)

    # Compute exp, write to dst, accumulate sum
    var sum_exp = Scalar[dtype](0.0)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        val = exp(logits[row_offset + i] - max_val)
        sum_exp += val
        dst[row_offset + i] = val
    sum_exp = warp_sum(sum_exp)
    var sm_scale = Scalar[dtype](1.0) / sum_exp

    # Gradient = (softmax - label) * grad / nrows
    var d_by_nrows = grad[0] / Scalar[dtype](N)
    for i in range(thread_idx.x, C, BLOCK_SIZE):
        dst[row_offset + i] = (dst[row_offset + i] * sm_scale - labels[row_offset + i]) * d_by_nrows


# ===-------------------------------------------------------------------===#
# Softmax
# ===-------------------------------------------------------------------===#


def softmax_grad_kernel[
    dtype: DType, BLOCK_SIZE: Int
](
    y: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    upstream: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    N: Int,
    size: Int,
):
    var row = block_idx.x
    if row >= N:
        return
    var row_offset = row * size

    var dot = Scalar[dtype](0.0)
    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dot += y[row_offset + i] * upstream[row_offset + i]
    dot = warp_sum(dot)

    for i in range(thread_idx.x, size, BLOCK_SIZE):
        dst[row_offset + i] = y[row_offset + i] * (upstream[row_offset + i] - dot)
