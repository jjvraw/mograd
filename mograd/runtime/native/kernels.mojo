from std.gpu import global_idx, thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.math import exp, log
from std.atomic import Atomic

# ===-------------------------------------------------------------------===#
# GPU Kernels
# ===-------------------------------------------------------------------===#

comptime BLOCK_SIZE = 256
comptime TILE_DIM = 16


def add_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        c[tid] = a[tid] + b[tid]


def mul_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        c[tid] = a[tid] * b[tid]


def relu_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = x[tid] if x[tid] > Float32(0.0) else Float32(0.0)


def relu_grad_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    upstream: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = upstream[tid] if x[tid] > Float32(0.0) else Float32(0.0)


def exp_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = exp(x[tid])


def log_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = log(x[tid])


def neg_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = -x[tid]


def div_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = a[tid] / b[tid]


def sum_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = thread_idx.x
    var gtid = block_idx.x * block_dim.x + tid
    var shared = stack_allocation[BLOCK_SIZE, Scalar[DType.float32], address_space=AddressSpace.SHARED]()
    shared[tid] = x[gtid] if gtid < size else Float32(0.0)
    barrier()

    var active = BLOCK_SIZE
    comptime for _ in range(8):
        active >>= 1
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()

    if tid == 0:
        _ = Atomic.fetch_add(dst, shared[0])


def sum_grad_kernel(
    upstream: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = upstream[0]


def matmul_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    M: Int,
    N: Int,
    K: Int,
):
    var row = global_idx.y
    var col = global_idx.x
    if row < M and col < N:
        var acc = Float32(0.0)
        for k in range(K):
            acc += a[row * K + k] * b[k * N + col]
        dst[row * N + col] = acc


def transpose_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    M: Int,
    N: Int,
):
    var row = global_idx.y
    var col = global_idx.x
    if row < M and col < N:
        dst[col * M + row] = a[row * N + col]


def softmax_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = thread_idx.x
    var shared = stack_allocation[BLOCK_SIZE, Scalar[DType.float32], address_space=AddressSpace.SHARED]()

    shared[tid] = x[tid] if tid < size else Float32(-1e38)
    barrier()

    var active = BLOCK_SIZE
    comptime for _ in range(8):
        active >>= 1
        if tid < active:
            if shared[tid + active] > shared[tid]:
                shared[tid] = shared[tid + active]
        barrier()

    var max_val = shared[0]
    barrier()

    var exp_val = Float32(0.0)
    if tid < size:
        exp_val = exp(x[tid] - max_val)
        dst[tid] = exp_val
    shared[tid] = exp_val
    barrier()

    active = BLOCK_SIZE
    comptime for _ in range(8):
        active >>= 1
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()

    var total = shared[0]

    if tid < size:
        dst[tid] = dst[tid] / total


def uniform_kernel(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
    low: Float32,
    high: Float32,
    seed: UInt32,
):
    var tid = global_idx.x
    if tid < size:
        var s = (seed ^ UInt32(tid + 1)) * UInt32(2654435761)
        s ^= s << 13
        s ^= s >> 17
        s ^= s << 5
        dst[tid] = low + (Float32(s) / Float32(4294967296.0)) * (high - low)


def eq_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = Float32(1.0) if a[tid] == b[tid] else Float32(0.0)


def argmax_rows_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    N: Int,
    C: Int,
):
    var i = global_idx.x
    if i < N:
        var row = i * C
        var max_val = src[row]
        var max_idx = 0
        for j in range(1, C):
            if src[row + j] > max_val:
                max_val = src[row + j]
                max_idx = j
        dst[i] = Float32(max_idx)


def scale_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    scalar: Float32,
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = src[tid] * scalar


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


def slice_rows_kernel(
    src: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    start_row: Int,
    cols: Int,
    size: Int,
):
    var tid = global_idx.x
    if tid < size:
        dst[tid] = src[start_row * cols + tid]


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
