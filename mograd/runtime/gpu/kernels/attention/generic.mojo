"""Portable SIMT flash-attention kernels (non-NVIDIA fallback)."""
from std.gpu import WARP_SIZE, MAX_THREADS_PER_BLOCK_METADATA, block_idx, thread_idx, barrier, syncwarp
from std.gpu.host import DeviceContext, FuncAttribute
from std.gpu.memory import external_memory, AddressSpace
from std.gpu.primitives.warp import shuffle_xor as warp_shuffle_xor
from std.gpu.primitives.warp import sum as warp_sum
from std.math import exp2, log, min
from std.sys import size_of
from std.utils import StaticTuple
from mograd.runtime.gpu.kernels.attention.config import Br, Bc, BLOCK_SIZE, LOG2E, LN2


# ===-------------------------------------------------------------------===#
# Layouts (shared by all three kernels):
#   Q, K, V     [b, s, h, d] → b*S*H*D + s*H*D + h*D + d  (BSHD, raw leaves
#               recovered by fuse_flash_attention, i.e. pre-transpose)
#   O (dst)     [b, h, s, d] → b*H*S*D + h*S*D + s*D + d  (BHSD, the fused
#               node keeps the softmax@V matmul output layout)
#   mask        [b, h, i, j] → b*H*S*S + h*S*S + i*S + j  (same dtype as Q/K/V)
#   lse         [b, h, s]    → b*H*S   + h*S   + s        (natural log, Float32)
#
# Forward + backward D<=64: lane-per-output decomposition (the MMA kernels'
# structure with the tensor core's cross-lane reduction replaced by a serial
# k-loop). Backward D>=128: row-per-warp dataflow (see the rowwarp banner
# for the measured crossover). Each kernel's tile variants (causal
# below-diagonal / diagonal / non-causal+bias) share one process_tile
# closure selected by comptime CAUSAL + runtime causal_diag.
# ===-------------------------------------------------------------------===#


# Forward geometry: GW_FWD warps per block, each owning GR query rows. A
# lane owns WHOLE score/output elements and loops the reduction dimension
# serially. Neither matmul phase uses warp_sum. The only shuffles are the
# per-tile softmax max/sum row ladders.
comptime GW_FWD = 4  # warps per block (GR rows/warp is a kernel parameter)
comptime FWD_PAD = 1  # +1 f32 per smem row breaks stride-bank conflicts


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GW_FWD * WARP_SIZE)))
@__name(t"flash_attn_fwd_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}_gr{GR}")
def flash_attn_fwd_kernel[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool, GR: Int = 8
](
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert WARP_SIZE == 32, "fwd lane layout assumes 32-lane warps"
    comptime BR = GW_FWD * GR  # query rows per block
    comptime LG = WARP_SIZE // GR  # lane groups per warp
    comptime JPL = Bc // LG  # j-columns per lane in the score pass
    comptime DPL = D_BUCKET // LG  # d-elements per lane in the output pass
    comptime assert D_BUCKET % LG == 0, "D_BUCKET must divide into lane groups"
    comptime STRIDE = D_BUCKET + FWD_PAD
    comptime PSTRIDE = Bc + 1
    comptime BLOCK = GW_FWD * WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var tid = Int(thread_idx.x)

    var j_tiles = (S + Bc - 1) // Bc
    var num_tiles = (S + BR - 1) // BR
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H

    # Row r picks the lane's row within the warp. Group g indexes j-columns
    # in the score pass and the d-chunk in the output pass.
    var r = lane % GR
    var g = lane // GR
    var row_local = warp_id * GR + r
    var i = tile_i * BR + row_local

    # Q/K/V are BSHD, so row s of head h lives at qkv_bh + s*HD.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D

    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_fwd_smem"]()
    )
    # NOTE: sub-pointers derived from external_memory crash Metal codegen,
    # so every view indexes the one smem base through comptime offsets.
    comptime Q_OFF = 0
    comptime K_OFF = BR * STRIDE
    comptime V_OFF = K_OFF + Bc * STRIDE
    comptime P_OFF = V_OFF + Bc * STRIDE  # [BR][PSTRIDE], warp-private rows

    # Stage the block's Q rows once (reused across every KV tile).
    comptime Q_ELEMS = BR * D_BUCKET
    comptime for idx in range((Q_ELEMS + BLOCK - 1) // BLOCK):
        var slot = tid + idx * BLOCK
        if slot < Q_ELEMS:
            var qr = slot // D_BUCKET
            var qc = slot % D_BUCKET
            var gq = tile_i * BR + qr
            smem[Q_OFF + qr * STRIDE + qc] = Float32(q[qkv_bh + gq * HD + qc]) if gq < S and qc < D else Float32(0)
    barrier()

    var o_acc = stack_allocation[DPL, Float32]()
    comptime for u in range(DPL):
        o_acc[u] = Float32(0)
    var m = Float32(-1e38)  # running max in log2 space
    var l = Float32(0)
    var scale_log2e = scale * LOG2E

    @always_inline
    def process_tile(
        j_tile: Int, causal_diag: Bool
    ) {
        read i,
        read S,
        read D,
        read H,
        read HD,
        read b,
        read h,
        read qkv_bh,
        read k,
        read v,
        read mask,
        read tid,
        read r,
        read g,
        read row_local,
        read scale_log2e,
        read smem,
        read o_acc,
        mut m,
        mut l,
    }:
        var j_base = j_tile * Bc
        comptime KV_ELEMS = Bc * D_BUCKET
        comptime for idx in range((KV_ELEMS + BLOCK - 1) // BLOCK):
            var slot = tid + idx * BLOCK
            if slot < KV_ELEMS:
                var kr = slot // D_BUCKET
                var kc = slot % D_BUCKET
                var gk = j_base + kr
                var kv_base = qkv_bh + gk * HD + kc
                smem[K_OFF + kr * STRIDE + kc] = Float32(k[kv_base]) if gk < S and kc < D else Float32(0)
                smem[V_OFF + kr * STRIDE + kc] = Float32(v[kv_base]) if gk < S and kc < D else Float32(0)
        barrier()

        # Score pass. Each lane computes JPL full dots serially.
        var s_reg = stack_allocation[JPL, Float32]()
        comptime for t in range(JPL):
            s_reg[t] = Float32(0)
        for dd in range(D_BUCKET):
            var qv = smem[Q_OFF + row_local * STRIDE + dd]
            comptime for t in range(JPL):
                s_reg[t] += qv * smem[K_OFF + (g * JPL + t) * STRIDE + dd]

        # Scale, mask, sentinel. Invalid positions take -1e38.
        var tile_full = j_base + Bc <= S
        comptime for t in range(JPL):
            var j = j_base + g * JPL + t
            var valid = i < S
            comptime if CAUSAL:
                if causal_diag:
                    valid = valid and (tile_full or j < S) and j <= i
            else:
                valid = valid and (tile_full or j < S)
            var sc = s_reg[t] * scale_log2e
            comptime if HAS_BIAS and not CAUSAL:
                # mask is additive bias in natural-log space → log2 space
                sc += Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E if valid else Float32(0)
            s_reg[t] = sc if valid else Float32(-1e38)

        # Per-tile online softmax. Row max and sum reduce across the LG
        # row-mates (lanes r, r+8, r+16, r+24) with two shuffles.
        var mx = s_reg[0]
        comptime for t in range(1, JPL):
            mx = mx if mx > s_reg[t] else s_reg[t]
        comptime for sh in range(5):
            comptime OFF = GR << sh
            comptime if OFF < WARP_SIZE:
                var peer = warp_shuffle_xor(mx, UInt32(OFF))
                mx = mx if mx > peer else peer
        var m_new = m if m > mx else mx
        var corr = exp2(m - m_new)
        m = m_new

        var lpart = Float32(0)
        comptime for t in range(JPL):
            var es = exp2(s_reg[t] - m_new)
            lpart += es
            smem[P_OFF + row_local * PSTRIDE + g * JPL + t] = es
        comptime for sh in range(5):
            comptime OFF = GR << sh
            comptime if OFF < WARP_SIZE:
                lpart += warp_shuffle_xor(lpart, UInt32(OFF))
        l = l * corr + lpart
        syncwarp()  # p rows are warp-private: warp-visible before the O pass

        # Output pass. The lane owns d-chunk g of its row and loops j
        # serially with p broadcast across the row's lane groups.
        comptime for u in range(DPL):
            o_acc[u] *= corr
        for jr in range(Bc):
            var pv = smem[P_OFF + row_local * PSTRIDE + jr]
            comptime for u in range(DPL):
                o_acc[u] += pv * smem[V_OFF + jr * STRIDE + g * DPL + u]
        barrier()  # K/V smem is reused by the next tile

    comptime if CAUSAL:
        # BR exceeds Bc so the diagonal straddles multiple tiles and every
        # tile from diag_start on needs the predicate. A single diag_tile
        # would let the block's earlier rows attend to their future.
        var diag_start = (tile_i * BR) // Bc
        var j_tile_end = min(j_tiles, (tile_i * BR + BR + Bc - 1) // Bc)
        for j_tile in range(j_tile_end):
            process_tile(j_tile, j_tile >= diag_start)
    else:
        for j_tile in range(j_tiles):
            process_tile(j_tile, False)

    if i < S:
        var o_base = b * H * S * D + h * S * D + i * D
        comptime for u in range(DPL):
            var d = g * DPL + u
            if d < D:
                dst[o_base + d] = Scalar[dtype](o_acc[u] / l)
        # m is in log2 space, convert to natural log for LSE
        if g == 0:
            lse[b * H * S + h * S + i] = m * LN2 + log(l)


# ===-------------------------------------------------------------------===#
# Backward: dQ  (also writes delta = dot(O, dO) per query row)
#
# Same lane-per-output decomposition as the forward: GW_FWD warps of GR
# query rows per block, serial k-loops.
# Two passes per KV tile: (A) each lane computes its (row, j-group) score
# and dO.V dots serially and stages dS through warp-private smem
# (B) each lane accumulates its (row, d-chunk) of dQ serially over j.
# LSE is read in natural-log space and converted to log2 for exp2-based
# p_ij = exp2(score_l2 - lse_i * LOG2E).
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GW_FWD * WARP_SIZE)))
@__name(t"flash_attn_dq_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}_gr{GR}")
def flash_attn_dq_kernel[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool, GR: Int = 8
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    delta: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert WARP_SIZE == 32, "bwd lane layout assumes 32-lane warps"
    comptime BR = GW_FWD * GR
    comptime LG = WARP_SIZE // GR
    comptime JPL = Bc // LG
    comptime DPL = D_BUCKET // LG
    comptime assert D_BUCKET % LG == 0 and Bc % LG == 0
    comptime STRIDE = D_BUCKET + FWD_PAD
    comptime PSTRIDE = Bc + 1
    comptime BLOCK = GW_FWD * WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var tid = Int(thread_idx.x)

    var j_tiles = (S + Bc - 1) // Bc
    var num_tiles = (S + BR - 1) // BR
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H

    var r = lane % GR
    var g = lane // GR
    var row_local = warp_id * GR + r
    var i = tile_i * BR + row_local

    # Q/K/V (and dQ) are BSHD. O and dO keep the fused node's BHSD layout.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var o_bh = b * H * S * D + h * S * D

    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_dq_smem"]()
    )
    comptime Q_OFF = 0
    comptime DO_OFF = BR * STRIDE
    comptime K_OFF = DO_OFF + BR * STRIDE
    comptime V_OFF = K_OFF + Bc * STRIDE
    comptime DS_OFF = V_OFF + Bc * STRIDE  # [BR][PSTRIDE], warp-private rows

    # Stage the block's Q and dO rows once.
    comptime QD_ELEMS = BR * D_BUCKET
    comptime for idx in range((QD_ELEMS + BLOCK - 1) // BLOCK):
        var slot = tid + idx * BLOCK
        if slot < QD_ELEMS:
            var qr = slot // D_BUCKET
            var qc = slot % D_BUCKET
            var gq = tile_i * BR + qr
            var ok = gq < S and qc < D
            smem[Q_OFF + qr * STRIDE + qc] = Float32(q[qkv_bh + gq * HD + qc]) if ok else Float32(0)
            smem[DO_OFF + qr * STRIDE + qc] = Float32(dy[o_bh + gq * D + qc]) if ok else Float32(0)
    barrier()

    # delta_i = dot(dO_i, O_i), computed serially over the lane's d-chunk
    # and reduced with the row-mate shuffle ladder once per kernel.
    var dpart = Float32(0)
    if i < S:
        for u in range(DPL):
            var d = g * DPL + u
            if d < D:
                dpart += Float32(o[o_bh + i * D + d]) * smem[DO_OFF + row_local * STRIDE + d]
    comptime for sh in range(5):
        comptime OFF = GR << sh
        comptime if OFF < WARP_SIZE:
            dpart += warp_shuffle_xor(dpart, UInt32(OFF))
    var delta_i = dpart
    if g == 0 and i < S:
        delta[b * H * S + h * S + i] = delta_i

    var lse_l2 = lse[b * H * S + h * S + i] * LOG2E if i < S else Float32(0)
    var scale_log2e = scale * LOG2E

    var dq_acc = stack_allocation[DPL, Float32]()
    comptime for u in range(DPL):
        dq_acc[u] = Float32(0)

    @always_inline
    def process_tile(
        j_tile: Int, causal_diag: Bool
    ) {
        read i,
        read S,
        read D,
        read H,
        read HD,
        read b,
        read h,
        read qkv_bh,
        read k,
        read v,
        read mask,
        read tid,
        read g,
        read row_local,
        read scale_log2e,
        read lse_l2,
        read delta_i,
        read smem,
        read dq_acc,
    }:
        var j_base = j_tile * Bc
        comptime KV_ELEMS = Bc * D_BUCKET
        comptime for idx in range((KV_ELEMS + BLOCK - 1) // BLOCK):
            var slot = tid + idx * BLOCK
            if slot < KV_ELEMS:
                var kr = slot // D_BUCKET
                var kc = slot % D_BUCKET
                var gk = j_base + kr
                var kv_base = qkv_bh + gk * HD + kc
                smem[K_OFF + kr * STRIDE + kc] = Float32(k[kv_base]) if gk < S and kc < D else Float32(0)
                smem[V_OFF + kr * STRIDE + kc] = Float32(v[kv_base]) if gk < S and kc < D else Float32(0)
        barrier()

        # Pass A: score and dO.V dots for the lane's (row, j-group), both serial
        var s_qk = stack_allocation[JPL, Float32]()
        var s_dp = stack_allocation[JPL, Float32]()
        comptime for t in range(JPL):
            s_qk[t] = Float32(0)
            s_dp[t] = Float32(0)
        for dd in range(D_BUCKET):
            var qv = smem[Q_OFF + row_local * STRIDE + dd]
            var dov = smem[DO_OFF + row_local * STRIDE + dd]
            comptime for t in range(JPL):
                s_qk[t] += qv * smem[K_OFF + (g * JPL + t) * STRIDE + dd]
                s_dp[t] += dov * smem[V_OFF + (g * JPL + t) * STRIDE + dd]
        var tile_full = j_base + Bc <= S
        comptime for t in range(JPL):
            var j = j_base + g * JPL + t
            var take = i < S
            comptime if CAUSAL:
                if causal_diag:
                    take = take and (tile_full or j < S) and j <= i
            else:
                take = take and (tile_full or j < S)
            var s_l2 = s_qk[t] * scale_log2e
            comptime if HAS_BIAS and not CAUSAL:
                s_l2 += Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E if take else Float32(0)
            var p_ij = exp2(s_l2 - lse_l2) if take else Float32(0)
            smem[DS_OFF + row_local * PSTRIDE + g * JPL + t] = p_ij * (s_dp[t] - delta_i)
        syncwarp()

        # Pass B: dQ accumulation, serial over j, no reductions. The scale factor is applied once at the epilogue.
        for jr in range(Bc):
            var dsv = smem[DS_OFF + row_local * PSTRIDE + jr]
            comptime for u in range(DPL):
                dq_acc[u] += dsv * smem[K_OFF + jr * STRIDE + g * DPL + u]
        barrier()

    comptime if CAUSAL:
        var diag_start = (tile_i * BR) // Bc
        var j_tile_end = min(j_tiles, (tile_i * BR + BR + Bc - 1) // Bc)
        for j_tile in range(j_tile_end):
            process_tile(j_tile, j_tile >= diag_start)
    else:
        for j_tile in range(j_tiles):
            process_tile(j_tile, False)

    if i < S:
        var out_base = qkv_bh + i * HD
        comptime for u in range(DPL):
            var d = g * DPL + u
            if d < D:
                dq[out_base + d] = Scalar[dtype](dq_acc[u] * scale)


# ===-------------------------------------------------------------------===#
# Backward: dK and dV
#
# Mirror of dQ with i/j roles swapped. The block owns BR KEY rows (K/V
# resident in smem), loops Q tiles, and stages BOTH p and dS per tile (dV
# needs p, dK needs dS). LSE and delta are read from the buffers the dQ
# kernel wrote (same stream, ordering guaranteed).
#
# Causal tiles fully below the block's first key row are all-masked and
# skipped. Straddling tiles get the j<=i predicate and tiles fully above
# need none.
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(GW_FWD * WARP_SIZE)))
@__name(t"flash_attn_dkdv_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}_gr{GR}")
def flash_attn_dkdv_kernel[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool, GR: Int = 8
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    delta: UnsafePointer[Float32, ImmutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert WARP_SIZE == 32, "bwd lane layout assumes 32-lane warps"
    comptime BR = GW_FWD * GR
    comptime LG = WARP_SIZE // GR
    comptime JPL = Bc // LG  # here: q-rows per lane in pass A
    comptime DPL = D_BUCKET // LG
    comptime assert D_BUCKET % LG == 0 and Bc % LG == 0
    comptime STRIDE = D_BUCKET + FWD_PAD
    comptime PSTRIDE = Bc + 1
    comptime BLOCK = GW_FWD * WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE
    var tid = Int(thread_idx.x)

    var i_tiles = (S + Bc - 1) // Bc
    var num_tiles = (S + BR - 1) // BR
    if bid >= B * H * num_tiles:
        return

    var tile_j = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H

    var r = lane % GR
    var g = lane // GR
    var row_local = warp_id * GR + r
    var j = tile_j * BR + row_local

    # Q/K/V (and dK/dV) are BSHD. dO keeps the fused node's BHSD layout.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var o_bh = b * H * S * D + h * S * D

    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_dkdv_smem"]()
    )
    comptime K_OFF = 0
    comptime V_OFF = BR * STRIDE
    comptime Q_OFF = V_OFF + BR * STRIDE
    comptime DO_OFF = Q_OFF + Bc * STRIDE
    comptime P_OFF = DO_OFF + Bc * STRIDE  # [BR][PSTRIDE], warp-private rows
    comptime DS_OFF = P_OFF + BR * PSTRIDE
    comptime LSE_OFF = DS_OFF + BR * PSTRIDE  # [Bc]
    comptime DELTA_OFF = LSE_OFF + Bc  # [Bc]

    # Stage the block's K and V rows once (resident across all Q tiles).
    comptime KV_ELEMS = BR * D_BUCKET
    comptime for idx in range((KV_ELEMS + BLOCK - 1) // BLOCK):
        var slot = tid + idx * BLOCK
        if slot < KV_ELEMS:
            var kr = slot // D_BUCKET
            var kc = slot % D_BUCKET
            var gk = tile_j * BR + kr
            var ok = gk < S and kc < D
            var kv_base = qkv_bh + gk * HD + kc
            smem[K_OFF + kr * STRIDE + kc] = Float32(k[kv_base]) if ok else Float32(0)
            smem[V_OFF + kr * STRIDE + kc] = Float32(v[kv_base]) if ok else Float32(0)
    barrier()

    var dk_acc = stack_allocation[DPL, Float32]()
    var dv_acc = stack_allocation[DPL, Float32]()
    comptime for u in range(DPL):
        dk_acc[u] = Float32(0)
        dv_acc[u] = Float32(0)
    var scale_log2e = scale * LOG2E

    @always_inline
    def process_tile(
        i_tile: Int, causal_diag: Bool
    ) {
        read j,
        read S,
        read D,
        read H,
        read HD,
        read b,
        read h,
        read qkv_bh,
        read o_bh,
        read q,
        read dy,
        read mask,
        read lse,
        read delta,
        read tid,
        read g,
        read row_local,
        read scale_log2e,
        read smem,
        read dk_acc,
        read dv_acc,
    }:
        var i_base = i_tile * Bc
        comptime QD_ELEMS = Bc * D_BUCKET
        comptime for idx in range((QD_ELEMS + BLOCK - 1) // BLOCK):
            var slot = tid + idx * BLOCK
            if slot < QD_ELEMS:
                var qr = slot // D_BUCKET
                var qc = slot % D_BUCKET
                var gq = i_base + qr
                var ok = gq < S and qc < D
                smem[Q_OFF + qr * STRIDE + qc] = Float32(q[qkv_bh + gq * HD + qc]) if ok else Float32(0)
                smem[DO_OFF + qr * STRIDE + qc] = Float32(dy[o_bh + gq * D + qc]) if ok else Float32(0)
        if tid < Bc:
            var gq = i_base + tid
            smem[LSE_OFF + tid] = lse[b * H * S + h * S + gq] if gq < S else Float32(0)
            smem[DELTA_OFF + tid] = delta[b * H * S + h * S + gq] if gq < S else Float32(0)
        barrier()

        # Pass A: q.k and dO.v dots for the lane's (key row, i-group)
        var s_qk = stack_allocation[JPL, Float32]()
        var s_dp = stack_allocation[JPL, Float32]()
        comptime for t in range(JPL):
            s_qk[t] = Float32(0)
            s_dp[t] = Float32(0)
        for dd in range(D_BUCKET):
            var kv_val = smem[K_OFF + row_local * STRIDE + dd]
            var vv = smem[V_OFF + row_local * STRIDE + dd]
            comptime for t in range(JPL):
                s_qk[t] += smem[Q_OFF + (g * JPL + t) * STRIDE + dd] * kv_val
                s_dp[t] += smem[DO_OFF + (g * JPL + t) * STRIDE + dd] * vv
        var tile_full = i_base + Bc <= S
        comptime for t in range(JPL):
            var i = i_base + g * JPL + t
            var take = j < S
            comptime if CAUSAL:
                if causal_diag:
                    take = take and (tile_full or i < S) and j <= i
            else:
                take = take and (tile_full or i < S)
            var s_l2 = s_qk[t] * scale_log2e
            comptime if HAS_BIAS and not CAUSAL:
                s_l2 += Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E if take else Float32(0)
            var p_ij = exp2(s_l2 - smem[LSE_OFF + g * JPL + t] * LOG2E) if take else Float32(0)
            smem[P_OFF + row_local * PSTRIDE + g * JPL + t] = p_ij
            smem[DS_OFF + row_local * PSTRIDE + g * JPL + t] = p_ij * (s_dp[t] - smem[DELTA_OFF + g * JPL + t])
        syncwarp()

        # Pass B: dV and dK accumulation, serial over the tile's q rows.
        for qi in range(Bc):
            var pv = smem[P_OFF + row_local * PSTRIDE + qi]
            var dsv = smem[DS_OFF + row_local * PSTRIDE + qi]
            comptime for u in range(DPL):
                dv_acc[u] += pv * smem[DO_OFF + qi * STRIDE + g * DPL + u]
                dk_acc[u] += dsv * smem[Q_OFF + qi * STRIDE + g * DPL + u]
        barrier()

    comptime if CAUSAL:
        # Tiles with every i below the block's first key row are all-masked
        # and skipped. The straddle region gets the predicate.
        var i_start = (tile_j * BR) // Bc
        for i_tile in range(i_start, i_tiles):
            process_tile(i_tile, i_tile * Bc < tile_j * BR + BR)
    else:
        for i_tile in range(i_tiles):
            process_tile(i_tile, False)

    if j < S:
        var out_base = qkv_bh + j * HD
        comptime for u in range(DPL):
            var d = g * DPL + u
            if d < D:
                dk[out_base + d] = Scalar[dtype](dk_acc[u] * scale)
                dv[out_base + d] = Scalar[dtype](dv_acc[u])


# ===-------------------------------------------------------------------===#
# Backward, row-per-warp variants. Row operands stay in registers and p/dS are
# consumed in-place with no staging round trip.

# Backward: dQ  (also writes delta = dot(O, dO) per query row)
#
# Grid=(B*H*ceil(S/Br),), Block=(WARP_SIZE*Br,) = Br warps.
# Warp warp_id owns query row i = tile_i*Br + warp_id.
# delta_i is computed inline and written to the delta buffer for dKdV
# (same stream).
#
# LSE is read in natural-log space (as written by fwd) and converted to log2
# for exp2-based p_ij computation: p_ij = exp2(score_l2 - lse_i * LOG2E).
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE)))
@__name(t"flash_attn_dq_rowwarp_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_dq_kernel_rowwarp[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    delta: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert D_BUCKET % WARP_SIZE == 0, "D_BUCKET must be divisible by WARP_SIZE"
    comptime D_PT = D_BUCKET // WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE

    var j_tiles = (S + Bc - 1) // Bc
    var num_tiles = (S + Br - 1) // Br
    if bid >= B * H * num_tiles:
        return

    var tile_i = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H
    var i = tile_i * Br + warp_id

    # Q/K/V (and dQ) are BSHD. O and dO keep the fused node's BHSD layout.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var o_bh = b * H * S * D + h * S * D

    var q_reg = stack_allocation[D_PT, Float32]()
    var do_reg = stack_allocation[D_PT, Float32]()
    var dq_reg = stack_allocation[D_PT, Float32]()

    var delta_partial = Float32(0)
    comptime for di in range(D_PT):
        dq_reg[di] = Float32(0)
        var d = lane + di * WARP_SIZE
        if i < S and d < D:
            var o_idx = o_bh + i * D + d
            q_reg[di] = Float32(q[qkv_bh + i * HD + d])
            var do_val = Float32(dy[o_idx])
            do_reg[di] = do_val
            delta_partial += Float32(o[o_idx]) * do_val
        else:
            q_reg[di] = Float32(0)
            do_reg[di] = Float32(0)
    var delta_i = warp_sum(delta_partial)
    if lane == 0 and i < S:
        delta[b * H * S + h * S + i] = delta_i

    var lse_i_l2 = lse[b * H * S + h * S + i] * LOG2E if i < S else Float32(0)
    var scale_log2e = scale * LOG2E

    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_dq_smem"]()
    )
    comptime K_OFF = 0
    comptime V_OFF = Bc * D_BUCKET

    @always_inline
    def process_tile(
        j_tile: Int, causal_diag: Bool
    ) {
        read i,
        read S,
        read D,
        read H,
        read HD,
        read b,
        read h,
        read qkv_bh,
        read k,
        read v,
        read mask,
        read lane,
        read warp_id,
        read scale,
        read scale_log2e,
        read lse_i_l2,
        read delta_i,
        read q_reg,
        read do_reg,
        read dq_reg,
        read smem,
    }:
        var j_base = j_tile * Bc
        var j_load = j_base + warp_id
        var kv_base = qkv_bh + j_load * HD
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            var kv_idx = warp_id * D_BUCKET + lane + di * WARP_SIZE
            smem[K_OFF + kv_idx] = Float32(k[kv_base + d]) if j_load < S and d < D else Float32(0)
            smem[V_OFF + kv_idx] = Float32(v[kv_base + d]) if j_load < S and d < D else Float32(0)
        barrier()
        if i < S:
            var tile_full = j_base + Bc <= S
            comptime for jr in range(Bc):
                var j = j_base + jr
                var take = True
                comptime if CAUSAL:
                    if causal_diag:
                        take = (tile_full or j < S) and j <= i
                else:
                    take = tile_full or j < S
                if take:
                    var partial_qk = Float32(0)
                    comptime for di in range(D_PT):
                        partial_qk += q_reg[di] * smem[K_OFF + jr * D_BUCKET + lane + di * WARP_SIZE]
                    var p_ij: Float32
                    comptime if CAUSAL:
                        p_ij = exp2(warp_sum(partial_qk) * scale_log2e - lse_i_l2)
                    else:
                        var mask_val_l2 = Float32(0)
                        comptime if HAS_BIAS:
                            mask_val_l2 = Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E
                        p_ij = exp2(warp_sum(partial_qk) * scale_log2e + mask_val_l2 - lse_i_l2)
                    var partial_dp = Float32(0)
                    comptime for di in range(D_PT):
                        partial_dp += do_reg[di] * smem[V_OFF + jr * D_BUCKET + lane + di * WARP_SIZE]
                    var ds_ij = p_ij * (warp_sum(partial_dp) - delta_i)
                    comptime for di in range(D_PT):
                        dq_reg[di] += ds_ij * scale * smem[K_OFF + jr * D_BUCKET + lane + di * WARP_SIZE]
        barrier()

    comptime if CAUSAL:
        var diag_tile = (tile_i * Br + Br - 1) // Bc
        for j_tile in range(diag_tile):
            process_tile(j_tile, False)
        if diag_tile < j_tiles:
            process_tile(diag_tile, True)
    else:
        for j_tile in range(j_tiles):
            process_tile(j_tile, False)

    if i < S:
        var out_base = qkv_bh + i * HD
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            if d < D:
                dq[out_base + d] = Scalar[dtype](dq_reg[di])


# ===-------------------------------------------------------------------===#
# Backward: dK and dV
#
# Grid=(B*H*ceil(S/Bc),), Block=(WARP_SIZE*Bc,) = Bc warps.
# Warp warp_id owns key row j = tile_j*Bc + warp_id.
# LSE and delta read from buffers written by dQ kernel (same stream).
#
# CAUSAL structure (i_tile axis, key row j fixed):
#   [tile_j]        diagonal tile, apply j<=i per qi
#   (tile_j, ...]   fully above diagonal, no predicate, j<=i guaranteed
#   [0, tile_j)     all i < j, fully masked, skipped entirely
# ===-------------------------------------------------------------------===#


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE)))
@__name(t"flash_attn_dkdv_rowwarp_{dtype}_d{D_BUCKET}_causal_{CAUSAL}_bias_{HAS_BIAS}")
def flash_attn_dkdv_kernel_rowwarp[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    delta: UnsafePointer[Float32, ImmutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
):
    comptime assert D_BUCKET % WARP_SIZE == 0, "D_BUCKET must be divisible by WARP_SIZE"
    comptime D_PT = D_BUCKET // WARP_SIZE

    var bid = Int(block_idx.x)
    var warp_id = Int(thread_idx.x) // WARP_SIZE
    var lane = Int(thread_idx.x) % WARP_SIZE

    var i_tiles = (S + Bc - 1) // Bc
    var num_tiles = (S + Bc - 1) // Bc
    if bid >= B * H * num_tiles:
        return

    var tile_j = bid % num_tiles
    var bh = bid // num_tiles
    var h = bh % H
    var b = bh // H
    var j = tile_j * Bc + warp_id

    # Q/K/V (and dK/dV) are BSHD. dO keeps the fused node's BHSD layout.
    var HD = H * D
    var qkv_bh = b * S * H * D + h * D
    var o_bh = b * H * S * D + h * S * D

    var k_reg = stack_allocation[D_PT, Float32]()
    var v_reg = stack_allocation[D_PT, Float32]()
    var dk_reg = stack_allocation[D_PT, Float32]()
    var dv_reg = stack_allocation[D_PT, Float32]()

    comptime for di in range(D_PT):
        dk_reg[di] = Float32(0)
        dv_reg[di] = Float32(0)
        var d = lane + di * WARP_SIZE
        if j < S and d < D:
            var kv_base = qkv_bh + j * HD
            k_reg[di] = Float32(k[kv_base + d])
            v_reg[di] = Float32(v[kv_base + d])
        else:
            k_reg[di] = Float32(0)
            v_reg[di] = Float32(0)

    var scale_log2e = scale * LOG2E

    var smem = rebind[UnsafePointer[Float32, MutUntrackedOrigin, address_space=AddressSpace.SHARED]](
        external_memory[Float32, address_space=AddressSpace.SHARED, alignment=16, name="attn_dkdv_smem"]()
    )
    comptime Q_OFF = 0
    comptime DO_OFF = Bc * D_BUCKET
    comptime LSE_OFF = 2 * Bc * D_BUCKET
    comptime DELTA_OFF = LSE_OFF + Bc

    @always_inline
    def process_tile(
        i_tile: Int, causal_diag: Bool
    ) {
        read j,
        read S,
        read D,
        read H,
        read HD,
        read b,
        read h,
        read qkv_bh,
        read o_bh,
        read q,
        read dy,
        read mask,
        read lse,
        read delta,
        read lane,
        read warp_id,
        read scale,
        read scale_log2e,
        read k_reg,
        read v_reg,
        read dk_reg,
        read dv_reg,
        read smem,
    }:
        var i_base = i_tile * Bc
        var i_load = i_base + warp_id
        var q_src = qkv_bh + i_load * HD
        var do_src = o_bh + i_load * D
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            var qi_idx = warp_id * D_BUCKET + lane + di * WARP_SIZE
            smem[Q_OFF + qi_idx] = Float32(q[q_src + d]) if i_load < S and d < D else Float32(0)
            smem[DO_OFF + qi_idx] = Float32(dy[do_src + d]) if i_load < S and d < D else Float32(0)
        if lane == 0:
            smem[LSE_OFF + warp_id] = lse[b * H * S + h * S + i_load] if i_load < S else Float32(0)
            smem[DELTA_OFF + warp_id] = delta[b * H * S + h * S + i_load] if i_load < S else Float32(0)
        barrier()
        if j < S:
            var tile_full = i_base + Bc <= S
            comptime for qi in range(Bc):
                var i = i_base + qi
                var take = tile_full or i < S
                comptime if CAUSAL:
                    # Only the diagonal tile needs the j<=i predicate.
                    # Tiles fully above the diagonal have i > j_max for every j.
                    if causal_diag:
                        take = take and j <= i
                if take:
                    var lse_i_l2 = smem[LSE_OFF + qi] * LOG2E
                    var delta_i = smem[DELTA_OFF + qi]
                    var partial_qk = Float32(0)
                    comptime for di in range(D_PT):
                        partial_qk += Float32(smem[Q_OFF + qi * D_BUCKET + lane + di * WARP_SIZE]) * k_reg[di]
                    var p_ij: Float32
                    comptime if CAUSAL:
                        p_ij = exp2(warp_sum(partial_qk) * scale_log2e - lse_i_l2)
                    else:
                        var mask_val_l2 = Float32(0)
                        comptime if HAS_BIAS:
                            mask_val_l2 = Float32(mask[b * H * S * S + h * S * S + i * S + j]) * LOG2E
                        p_ij = exp2(warp_sum(partial_qk) * scale_log2e + mask_val_l2 - lse_i_l2)
                    var partial_dp = Float32(0)
                    comptime for di in range(D_PT):
                        partial_dp += Float32(smem[DO_OFF + qi * D_BUCKET + lane + di * WARP_SIZE]) * v_reg[di]
                    var ds_ij = p_ij * (warp_sum(partial_dp) - delta_i)
                    comptime for di in range(D_PT):
                        dv_reg[di] += p_ij * Float32(smem[DO_OFF + qi * D_BUCKET + lane + di * WARP_SIZE])
                        dk_reg[di] += ds_ij * scale * Float32(smem[Q_OFF + qi * D_BUCKET + lane + di * WARP_SIZE])
        barrier()

    comptime if CAUSAL:
        # Diagonal tile first (j<=i predicate), then tiles fully above the
        # diagonal. Tiles below (all i < j) contribute nothing and are skipped.
        if tile_j < i_tiles:
            process_tile(tile_j, True)
        for i_tile in range(tile_j + 1, i_tiles):
            process_tile(i_tile, False)
    else:
        for i_tile in range(i_tiles):
            process_tile(i_tile, False)

    if j < S:
        var out_base = qkv_bh + j * HD
        comptime for di in range(D_PT):
            var d = lane + di * WARP_SIZE
            if d < D:
                dk[out_base + d] = Scalar[dtype](dk_reg[di])
                dv[out_base + d] = Scalar[dtype](dv_reg[di])


def _flash_attn_fwd_launch[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    dst: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    lse: UnsafePointer[Float32, MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    comptime STRIDE = D_BUCKET + FWD_PAD

    @parameter
    @always_inline
    def launch[GR: Int]() raises:
        comptime BR = GW_FWD * GR
        comptime smem_bytes = ((BR + 2 * Bc) * STRIDE + BR * (Bc + 1)) * size_of[Float32]()
        var num_tiles = (S + BR - 1) // BR
        comptime kernel_fn = flash_attn_fwd_kernel[dtype, D_BUCKET, CAUSAL, HAS_BIAS, GR]
        ctx.enqueue_function[kernel_fn](
            q,
            k,
            v,
            mask,
            dst,
            lse,
            B,
            S,
            H,
            D,
            scale,
            grid_dim=(B * H * num_tiles,),
            block_dim=(GW_FWD * WARP_SIZE,),
            shared_mem_bytes=smem_bytes,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_bytes)),
        )

    # Measured heuristic: 32-row blocks win via Q reuse, except on small
    # grids where they leave SMs idle (16-row blocks double the block
    # count). Compute density beats occupancy here, so only the grid size influences the choice.
    # D_BUCKET 512 is the exception: GR=8 would need a 128-register output
    # accumulator per lane, so shrink to GR=2.
    comptime if D_BUCKET >= 512:
        launch[2]()
    else:
        if B * H * ((S + 31) // 32) < 96:
            launch[4]()
        else:
            launch[8]()


def _flash_attn_bwd_launch[
    dtype: DType, D_BUCKET: Int, CAUSAL: Bool, HAS_BIAS: Bool
](
    dy: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    o: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    q: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    k: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    v: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    mask: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    lse: UnsafePointer[Float32, ImmutAnyOrigin],
    dq: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dk: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    dv: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    B: Int,
    S: Int,
    H: Int,
    D: Int,
    scale: Float32,
    ctx: DeviceContext,
) raises:
    # dQ writes delta and dKdV reads it.
    var delta_buf = ctx.enqueue_create_buffer[DType.float32](B * H * S)
    var delta_ptr = delta_buf.unsafe_ptr().as_unsafe_any_origin()

    comptime STRIDE = D_BUCKET + FWD_PAD
    comptime GR_BASE = 8

    @parameter
    @always_inline
    def launch[GR: Int]() raises:
        comptime BR = GW_FWD * GR
        comptime dq_smem = ((2 * BR + 2 * Bc) * STRIDE + BR * (Bc + 1)) * size_of[Float32]()
        comptime dkdv_smem = ((2 * BR + 2 * Bc) * STRIDE + 2 * BR * (Bc + 1) + 2 * Bc) * size_of[Float32]()
        var num_tiles = (S + BR - 1) // BR
        comptime dq_fn = flash_attn_dq_kernel[dtype, D_BUCKET, CAUSAL, HAS_BIAS, GR]
        ctx.enqueue_function[dq_fn](
            dy,
            o,
            q,
            k,
            v,
            mask,
            lse,
            dq,
            delta_ptr,
            B,
            S,
            H,
            D,
            scale,
            grid_dim=(B * H * num_tiles,),
            block_dim=(GW_FWD * WARP_SIZE,),
            shared_mem_bytes=dq_smem,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(dq_smem)),
        )
        comptime dkdv_fn = flash_attn_dkdv_kernel[dtype, D_BUCKET, CAUSAL, HAS_BIAS, GR]
        ctx.enqueue_function[dkdv_fn](
            dy,
            q,
            k,
            v,
            mask,
            lse,
            delta_ptr,
            dk,
            dv,
            B,
            S,
            H,
            D,
            scale,
            grid_dim=(B * H * num_tiles,),
            block_dim=(GW_FWD * WARP_SIZE,),
            shared_mem_bytes=dkdv_smem,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(dkdv_smem)),
        )

    @parameter
    @always_inline
    def launch_rowwarp() raises:
        comptime smem_kv = 2 * Bc * D_BUCKET * size_of[Float32]()
        comptime smem_qdelta = (2 * Bc * D_BUCKET + 2 * Bc) * size_of[Float32]()
        var num_tiles_q = (S + Br - 1) // Br
        var num_tiles_kv = (S + Bc - 1) // Bc
        comptime dq_fn = flash_attn_dq_kernel_rowwarp[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
        ctx.enqueue_function[dq_fn](
            dy,
            o,
            q,
            k,
            v,
            mask,
            lse,
            dq,
            delta_ptr,
            B,
            S,
            H,
            D,
            scale,
            grid_dim=(B * H * num_tiles_q,),
            block_dim=(Br * WARP_SIZE,),
            shared_mem_bytes=smem_kv,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_kv)),
        )
        comptime dkdv_fn = flash_attn_dkdv_kernel_rowwarp[dtype, D_BUCKET, CAUSAL, HAS_BIAS]
        ctx.enqueue_function[dkdv_fn](
            dy,
            q,
            k,
            v,
            mask,
            lse,
            delta_ptr,
            dk,
            dv,
            B,
            S,
            H,
            D,
            scale,
            grid_dim=(B * H * num_tiles_kv,),
            block_dim=(Bc * WARP_SIZE,),
            shared_mem_bytes=smem_qdelta,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(UInt32(smem_qdelta)),
        )

    # NOTE: lane-per-output wins at D <= 64 (the per-score reduction tax dominates short dots)
    # row-per-warp wins at D >= 128 (long dots amortize the tax while dual staged tiles cost
    # occupancy).
    comptime if D_BUCKET <= 64:
        if B * H * ((S + GW_FWD * GR_BASE - 1) // (GW_FWD * GR_BASE)) < 96:
            launch[GR_BASE // 2]()
        else:
            launch[GR_BASE]()
    else:
        launch_rowwarp()

    _ = delta_buf^
