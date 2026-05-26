from std.gpu import global_idx, thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.math import exp, log
from std.atomic import Atomic

# ===-------------------------------------------------------------------===#
# GPU Kernels
# ===-------------------------------------------------------------------===#

comptime BLOCK_SIZE = 256


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
    var shared = stack_allocation[
        BLOCK_SIZE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
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


def softmax_kernel(
    x: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = thread_idx.x
    var shared = stack_allocation[
        BLOCK_SIZE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

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


def softmax_grad_kernel(
    y: UnsafePointer[Float32, MutAnyOrigin],
    upstream: UnsafePointer[Float32, MutAnyOrigin],
    dst: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    var tid = thread_idx.x
    var shared = stack_allocation[
        BLOCK_SIZE, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()

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
