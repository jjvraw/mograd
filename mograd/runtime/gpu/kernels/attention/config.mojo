# Tile shapes follow FA2's design with under 144KB of shared mem. per block.
# Note, current values were tuned on CC 8.9.

from std.gpu import WARP_SIZE
from layout.tensor_core import TensorCore, shape_16x8x8

# Generic SIMT kernels
comptime Br = 16
comptime Bc = 16
comptime BLOCK_SIZE = Br * WARP_SIZE

comptime LOG2E = Float32(1.4426950408889634)
comptime LN2 = Float32(0.6931471805599453)

# NVIDIA MMA kernels
comptime MMA_M = 16  # NVIDIA m16n8k8 output M-dim
comptime MMA_N = 8  # NVIDIA m16n8k8 output N-dim
comptime MMA_K = 8  # NVIDIA m16n8k8 reduction K-dim
comptime NUM_WARPS_MMA = 4
comptime Br_MMA = NUM_WARPS_MMA * MMA_M  # = 64 query rows per block
comptime Bc_MMA = 64  # KV columns per tile
comptime Bc_HMMA = 32  # Half-precision forward KV tile
comptime KV_MMA_PAD = 4  # Padding for TF32/f32 K/V shared rows consumed by MMA load_b
comptime BLOCK_SIZE_MMA = NUM_WARPS_MMA * WARP_SIZE  # = 128

# TF32 tensor-core ops shared by the f32 forward and backward kernels.
# QKMma computes X @ Y^T products (Y stored row-major as (N, K)) and PVMma
# computes X @ Y products (Y stored row-major as (K, N)).
comptime QKMma = TensorCore[DType.float32, DType.float32, shape_16x8x8, True]
comptime PVMma = TensorCore[DType.float32, DType.float32, shape_16x8x8, False]

# 8 warps per block. Fewer resident warps starve delivered L2 throughput.
comptime BLOCK_SIZE_BWD_HALF = 8 * WARP_SIZE
comptime DQ_CONVERT_BLOCK = 256

# Register cap for the forward half kernel. 168 registers let three
# 128-thread blocks share a 64K-register SM, trading a few spills for
# occupancy on a latency-bound kernel.
comptime FWD_HALF_MAXNREG = 168
