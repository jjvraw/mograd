from std.gpu import global_idx, thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.math import exp, log
from std.atomic import Atomic

# ===-------------------------------------------------------------------===#
# GPU Kernels
# ===-------------------------------------------------------------------===#

comptime BLOCK_SIZE = 256

# ===-------------------------------------------------------------------===#
# Transpose
# ===-------------------------------------------------------------------===#


def transpose_kernel[
    BLOCK_SIZE: Int
](src: UnsafePointer[Float32, MutAnyOrigin], dst: UnsafePointer[Float32, MutAnyOrigin], M: Int, N: Int):
    var shmem = stack_allocation[BLOCK_SIZE * (BLOCK_SIZE + 1), DType.float32, address_space=AddressSpace.SHARED]()

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


def cross_entropy_kernel(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    labels: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    N: Int,
    C: Int,
):
    var i = global_idx.x
    if i < N:
        var row = i * C
        var max_val = logits[row]
        for j in range(1, C):
            if logits[row + j] > max_val:
                max_val = logits[row + j]
        var sum_exp = Float32(0.0)
        for j in range(C):
            sum_exp += exp(logits[row + j] - max_val)
        var label = Int(labels[i])
        var log_prob = logits[row + label] - max_val - log(sum_exp)
        _ = Atomic.fetch_add(dst, -log_prob / Float32(N))


def cross_entropy_grad_kernel(
    logits: UnsafePointer[Float32, MutAnyOrigin],
    labels: UnsafePointer[Float32, MutAnyOrigin],
    upstream: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    N: Int,
    C: Int,
):
    var tid = global_idx.x
    if tid < N * C:
        var i = tid // C
        var j = tid % C
        var row = i * C
        var max_val = logits[row]
        for k in range(1, C):
            if logits[row + k] > max_val:
                max_val = logits[row + k]
        var sum_exp = Float32(0.0)
        for k in range(C):
            sum_exp += exp(logits[row + k] - max_val)
        var sm = exp(logits[row + j] - max_val) / sum_exp
        var label = Int(labels[i])
        var one_hot = Float32(1.0) if j == label else Float32(0.0)
        dst[tid] = upstream[0] * (sm - one_hot) / Float32(N)


def softmax_grad_kernel(
    y: UnsafePointer[Float32, MutAnyOrigin],
    upstream: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = thread_idx.x
    var shared = stack_allocation[BLOCK_SIZE, Scalar[DType.float32], address_space=AddressSpace.SHARED]()

    shared[tid] = y[tid] * upstream[tid] if tid < size else Float32(0.0)
    barrier()

    var active = BLOCK_SIZE
    comptime for _ in range(8):
        active >>= 1
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()

    var dot = shared[0]

    if tid < size:
        dst[tid] = y[tid] * (upstream[tid] - dot)
