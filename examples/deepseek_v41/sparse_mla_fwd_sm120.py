"""DeepSeek-V4.1-Flash compressed sparse MLA forward for SM120 / SM121.

Every V4.1 attention row reads two streams of 512-wide MQA latents. Each
latent serves as both K and V:

- the sliding-window (SWA) cache: 528-byte records of 512 E4M3 values and
  16 UE8M0 scales, one per 32 coordinates;
- the indexed cache, holding top-k selected compressed entries: 288-byte
  records of 256 bytes of packed E2M1 (low nibble first) and 32 E4M3 scales,
  one per 16 coordinates.

SWA entries are physical slots. Indexed entries are logical positions,
resolved through a per-row page table as in vLLM. All 512 coordinates are
quantized, RoPE included.

Each CTA resolves the slots it needs up front, copies the next chunk's raw
records into shared memory with ``cp.async`` while the current chunk
computes, dequantizes them exactly to BF16 in shared memory, and multiplies
on the warp-level MMA path. Masked entries are zero-filled.

The arithmetic is defined over chunks of ``block_I`` entries of the
concatenated [SWA | indexed] list, grouped into segments of
``segment_chunks`` chunks:

- within a segment, chunks update a state ``(O_g, m_g, l_g)`` in order, as
  in flash attention: ``m_g' = max(m_g, max(s))`` with scores
  ``s = q.k * sm_scale * log2(e)``, ``p = exp2(s - m_g')``,
  ``l_g = fma(l_g, a, sum(p))`` and ``O_g = O_g * a + P V`` with
  ``a = exp2(m_g - m_g')``;
- the segment states then fold in order into a state that starts from the
  head's attention sink, with ``M' = max(M, m_g)``,
  ``L = fma(L, exp2(M - M'), l_g * exp2(m_g - M'))``, the same update for
  ``O``, and finally ``out = O * (1 / L)``.

Two schedules implement this contract:

- ``extend``: one CTA per row and head block computes and folds every
  segment in registers;
- ``decode``: one CTA per segment, head block and row writes its segment
  state to a caller-owned workspace, and a merge kernel folds them. This
  gives small batches up to ``segments x head_blocks`` CTAs per row instead
  of one.

Both schedules use the same segment code, and explicit ``T.fma``/``T.fmul``
keep the fold's rounding fixed, so they produce identical bits, for any head
block size. A row's output depends neither on its neighbours nor on the
kernel the batch size selects. ``segment_chunks`` and ``block_I`` are part of
the arithmetic; ``schedule`` and ``block_H`` are not. ``SparseMlaV41Plan``
compiles the kernels for one layer shape, owns the workspace and output, and
picks a kernel from the row count.

SM120/SM121 have no WGMMA and allow at most 99 KiB of shared memory per
block, so the tiles are ``mma.sync`` tiles of 16..64 heads by 16..32
entries, instead of the 64x64 SM90 tiles in
``examples/deepseek_v4/sparse_attn_fwd_sm90.py``.
"""

import argparse

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

HEAD_DIM = 512
SWA_RECORD_BYTES = 528  # 512 x E4M3 + 16 x UE8M0
SWA_GROUP = 32
IDX_RECORD_BYTES = 288  # 256 x packed E2M1 + 32 x E4M3
IDX_GROUP = 16
LOG2E = 1.44269504
# Segments of four 32-entry chunks: the 128-entry sliding window is one segment.
SEGMENT_CHUNKS = 4
# The plan runs decode while it needs at most this many waves of CTAs.
DECODE_MAX_WAVES = 3


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def sparse_mla_fwd_v41(
    heads: int,
    swa_width: int,
    idx_width: int,
    idx_page_size: int,
    schedule: str = "extend",
    sm_scale: float = HEAD_DIM**-0.5,
    block_H: int = 32,
    block_I: int = 32,
    threads: int = 256,
    segment_chunks: int = SEGMENT_CHUNKS,
):
    """Build one schedule of the sparse MLA kernel.

    Both schedules take:

    - ``Q``: ``[rows, heads, 512]`` BF16 queries, with RoPE applied;
    - ``SwaRecords``: ``[slots, 528]`` uint8 SWA records;
    - ``SwaIndices``: ``[rows, swa_width]`` int32 SWA slots, ``-1`` for padding;
    - ``SwaLengths``: ``[rows]`` int32 count of leading SWA entries to use;
    - ``IdxRecords``: ``[slots, 288]`` uint8 indexed records;
    - ``IdxIndices``: ``[rows, idx_width]`` int32 logical indexed positions,
      ``-1`` for padding;
    - ``IdxLengths``: ``[rows]`` int32 count of leading indexed entries to use;
    - ``IdxPageTable``: ``[rows, pages]`` int32 physical page of each logical
      ``idx_page_size`` page, as in vLLM's indexed page table;
    - ``Sinks``: ``[heads]`` FP32 attention-sink logits.

    ``decode`` then takes the FP32 workspaces ``Partial``
    (``[rows, segments, heads, 512]``) and ``PartialStats``
    (``[rows, segments, heads, 2]``). Both write ``Output``
    (``[rows, heads, 512]`` BF16). ``rows``, the slot counts and the
    page-table width are dynamic.
    """
    rows, swa_slots, idx_slots, idx_pages = T.dynamic("rows, swa_slots, idx_slots, idx_pages")

    D = HEAD_DIM
    BI = block_I
    dtype = T.bfloat16
    accum_dtype = T.float32
    scale_log2 = sm_scale * LOG2E

    assert schedule in ("extend", "decode"), schedule
    assert block_H % 16 == 0, "block_H must be a multiple of the 16-row MMA atom"
    assert BI % 16 == 0, "block_I must be a multiple of 16"
    head_blocks = tilelang.cdiv(heads, block_H)
    pad_heads = heads % block_H != 0
    swa_blocks = tilelang.cdiv(swa_width, BI)
    idx_blocks = tilelang.cdiv(idx_width, BI)
    chunks = swa_blocks + idx_blocks
    G = segment_chunks
    segments = tilelang.cdiv(chunks, G)
    # extend resolves every slot of its row up front; decode only its segment's.
    slot_count = chunks * BI if schedule == "extend" else G * BI
    copies = tilelang.cdiv(BI * SWA_RECORD_BYTES // 16, threads)
    # Q, the dequantized KV block and P in BF16, the raw record staging rows,
    # the slots, and reduction scratch.
    smem_bytes = 2 * (block_H * D + BI * D + block_H * BI) + BI * SWA_RECORD_BYTES + 4 * slot_count + 2048
    assert smem_bytes <= 99 * 1024, (
        f"block_H={block_H}, block_I={BI} needs ~{smem_bytes} B of shared memory; SM120 allows 99 KiB (use block_I=16 with block_H=64)"
    )

    @T.macro
    def gather_swa(SwaIndices, SwaLengths, row, first, count, slots, base):
        for i in T.Parallel(count):
            pos = first + i
            slot = SwaIndices[row, T.min(pos, swa_width - 1)]
            slots[base + i] = T.if_then_else((pos < swa_width) & (pos < SwaLengths[row]) & (slot >= 0), slot, -1)

    @T.macro
    def gather_idx(IdxIndices, IdxLengths, IdxPageTable, row, first, count, slots, base):
        # Map logical positions to physical slots through the row's page table.
        for i in T.Parallel(count):
            pos = first + i
            logical = IdxIndices[row, T.min(pos, idx_width - 1)]
            page = IdxPageTable[row, T.min(T.max(logical, 0) // idx_page_size, idx_pages - 1)]
            slots[base + i] = T.if_then_else(
                (pos < idx_width) & (pos < IdxLengths[row]) & (logical >= 0),
                page * idx_page_size + logical % idx_page_size,
                -1,
            )

    @T.macro
    def stage_records(Records, record_bytes, slots, base, staged, copy_slot):
        # cp.async each record into its staging row, 16 bytes per copy; masked
        # entries are zero-filled, so they dequantize to zero. Reading the slots
        # first keeps shared-memory reads out of the issue loop.
        segments = record_bytes // 16
        tx = T.get_thread_binding()
        for k in T.unroll(tilelang.cdiv(BI * segments, threads)):
            copy_slot[k] = slots[base + T.min((k * threads + tx) // segments, BI - 1)]
        for k in T.unroll(tilelang.cdiv(BI * segments, threads)):
            e = k * threads + tx
            if e < BI * segments:
                T.ptx_cp_async(
                    T.access_ptr(staged[e // segments, e % segments * 16], "w", 16),
                    T.access_ptr(Records[T.max(copy_slot[k], 0), e % segments * 16], "r", 16),
                    16,
                    copy_slot[k] >= 0,
                )
        T.ptx_commit_group()

    @T.macro
    def stage_chunk(SwaRecords, IdxRecords, chunk, slots, base, staged, copy_slot):
        if chunk < swa_blocks:
            stage_records(SwaRecords, SWA_RECORD_BYTES, slots, base, staged, copy_slot)
        else:
            stage_records(IdxRecords, IDX_RECORD_BYTES, slots, base, staged, copy_slot)

    @T.macro
    def dequant_chunk(chunk, staged, staged_fp8, staged_fp4, KV_shared):
        if chunk < swa_blocks:
            for i, d in T.Parallel(BI, D):
                # UE8M0 scale 2^(e - 127), built from the exponent bits.
                scale = T.reinterpret(T.shift_left(T.cast(staged[i, D + d // SWA_GROUP], T.uint32), 23), accum_dtype)
                KV_shared[i, d] = T.cast(T.cast(staged_fp8[i, d], accum_dtype) * scale, dtype)
        else:
            for i, d in T.Parallel(BI, D):
                scale = T.cast(staged_fp8[i, D // 2 + d // IDX_GROUP], accum_dtype)
                KV_shared[i, d] = T.cast(T.cast(staged_fp4[i, d], accum_dtype) * scale, dtype)

    @T.macro
    def load_q(Q, row, h0, Q_shared):
        if pad_heads:
            for h, d in T.Parallel(block_H, D):
                Q_shared[h, d] = T.if_then_else(h0 + h < heads, Q[row, T.min(h0 + h, heads - 1), d], 0)
        else:
            T.copy(Q[row, h0 : h0 + block_H, :], Q_shared)

    @T.macro
    def chunk_update(Q_shared, KV_shared, slots, base, P_shared, acc_s, acc_g, m_g, l_g, m_prev, alpha, row_sum):
        """Add one chunk to the segment state (acc_g, m_g, l_g) for heads [h0, h0 + block_H)."""
        T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
        for h, i in T.Parallel(block_H, BI):
            acc_s[h, i] = T.if_then_else(slots[base + i] >= 0, T.fmul(acc_s[h, i], scale_log2), -T.infinity(accum_dtype))
        T.copy(m_g, m_prev)
        T.reduce_max(acc_s, m_g, dim=1, clear=True)
        for h in T.Parallel(block_H):
            m_g[h] = T.max(m_g[h], m_prev[h])
            # The state is empty until the first unmasked entry.
            alpha[h] = T.if_then_else(m_prev[h] == -T.infinity(accum_dtype), 0, T.exp2(m_prev[h] - m_g[h]))
        for h, i in T.Parallel(block_H, BI):
            acc_s[h, i] = T.if_then_else(slots[base + i] >= 0, T.exp2(acc_s[h, i] - m_g[h]), 0)
        T.reduce_sum(acc_s, row_sum, dim=1)
        for h in T.Parallel(block_H):
            l_g[h] = T.fma(l_g[h], alpha[h], row_sum[h])
        for h, d in T.Parallel(block_H, D):
            acc_g[h, d] = T.fmul(acc_g[h, d], alpha[h])
        T.copy(acc_s, P_shared)
        T.gemm(P_shared, KV_shared, acc_g, policy=T.GemmWarpPolicy.FullRow)

    @T.macro
    def clear_segment(acc_g, m_g, l_g):
        T.fill(acc_g, 0)
        T.fill(l_g, 0)
        T.fill(m_g, -T.infinity(accum_dtype))

    @T.macro
    def sink_log2(Sinks, h):
        return T.fmul(T.if_then_else(h < heads, Sinks[T.min(h, heads - 1)], 0), LOG2E)

    if schedule == "extend":

        @T.prim_func
        def main(
            Q: T.Tensor([rows, heads, D], dtype),
            SwaRecords: T.Tensor([swa_slots, SWA_RECORD_BYTES], T.uint8),
            SwaIndices: T.Tensor([rows, swa_width], T.int32),
            SwaLengths: T.Tensor([rows], T.int32),
            IdxRecords: T.Tensor([idx_slots, IDX_RECORD_BYTES], T.uint8),
            IdxIndices: T.Tensor([rows, idx_width], T.int32),
            IdxLengths: T.Tensor([rows], T.int32),
            IdxPageTable: T.Tensor([rows, idx_pages], T.int32),
            Sinks: T.Tensor([heads], accum_dtype),
            Output: T.Tensor([rows, heads, D], dtype),
        ):
            with T.Kernel(head_blocks, rows, threads=threads) as (hb, row):
                Q_shared = T.alloc_shared([block_H, D], dtype)
                KV_shared = T.alloc_shared([BI, D], dtype)
                P_shared = T.alloc_shared([block_H, BI], dtype)
                # Raw records, one 528-byte row per entry; indexed records use the first 288 bytes.
                staged = T.alloc_shared([BI, SWA_RECORD_BYTES], T.uint8)
                staged_fp8 = T.view(staged, dtype=T.float8_e4m3fn)
                staged_fp4 = T.view(staged, [BI, SWA_RECORD_BYTES * 2], T.float4_e2m1fn)
                slots = T.alloc_shared([slot_count], T.int32)
                copy_slot = T.alloc_local([copies], T.int32)
                acc_s = T.alloc_fragment([block_H, BI], accum_dtype)
                acc_g = T.alloc_fragment([block_H, D], accum_dtype)
                acc_o = T.alloc_fragment([block_H, D], accum_dtype)
                m_g = T.alloc_fragment([block_H], accum_dtype)
                l_g = T.alloc_fragment([block_H], accum_dtype)
                m_prev = T.alloc_fragment([block_H], accum_dtype)
                alpha = T.alloc_fragment([block_H], accum_dtype)
                row_sum = T.alloc_fragment([block_H], accum_dtype)
                M = T.alloc_fragment([block_H], accum_dtype)
                L = T.alloc_fragment([block_H], accum_dtype)
                a = T.alloc_fragment([block_H], accum_dtype)
                b = T.alloc_fragment([block_H], accum_dtype)

                # Resolve every slot of the row once, so record copies never wait on an index load.
                gather_swa(SwaIndices, SwaLengths, row, 0, swa_blocks * BI, slots, 0)
                gather_idx(IdxIndices, IdxLengths, IdxPageTable, row, 0, idx_blocks * BI, slots, swa_blocks * BI)
                T.sync_threads()
                stage_chunk(SwaRecords, IdxRecords, 0, slots, 0, staged, copy_slot)
                h0 = hb * block_H
                load_q(Q, row, h0, Q_shared)
                T.fill(acc_o, 0)
                T.fill(L, 1)
                for h in T.Parallel(block_H):
                    M[h] = sink_log2(Sinks, h0 + h)

                for seg in T.serial(segments):
                    clear_segment(acc_g, m_g, l_g)
                    for c in T.serial(G):
                        chunk = seg * G + c
                        if chunk < chunks:
                            T.ptx_wait_group(0)
                            T.sync_threads()
                            dequant_chunk(chunk, staged, staged_fp8, staged_fp4, KV_shared)
                            T.sync_threads()
                            # The next chunk's records load while this chunk computes.
                            if chunk + 1 < chunks:
                                stage_chunk(SwaRecords, IdxRecords, chunk + 1, slots, (chunk + 1) * BI, staged, copy_slot)
                            chunk_update(Q_shared, KV_shared, slots, chunk * BI, P_shared, acc_s, acc_g, m_g, l_g, m_prev, alpha, row_sum)
                    # Fold the segment into the running state, as the decode merge kernel does.
                    for h in T.Parallel(block_H):
                        m_new = T.max(M[h], m_g[h])
                        a[h] = T.exp2(M[h] - m_new)
                        b[h] = T.exp2(m_g[h] - m_new)
                        L[h] = T.fma(L[h], a[h], T.fmul(l_g[h], b[h]))
                        M[h] = m_new
                    for h, d in T.Parallel(block_H, D):
                        acc_o[h, d] = T.fma(acc_o[h, d], a[h], T.fmul(acc_g[h, d], b[h]))

                for h in T.Parallel(block_H):
                    L[h] = 1.0 / L[h]
                for h, d in T.Parallel(block_H, D):
                    if h0 + h < heads:
                        Output[row, h0 + h, d] = T.fmul(acc_o[h, d], L[h])

        return main

    @T.prim_func
    def main(
        Q: T.Tensor([rows, heads, D], dtype),
        SwaRecords: T.Tensor([swa_slots, SWA_RECORD_BYTES], T.uint8),
        SwaIndices: T.Tensor([rows, swa_width], T.int32),
        SwaLengths: T.Tensor([rows], T.int32),
        IdxRecords: T.Tensor([idx_slots, IDX_RECORD_BYTES], T.uint8),
        IdxIndices: T.Tensor([rows, idx_width], T.int32),
        IdxLengths: T.Tensor([rows], T.int32),
        IdxPageTable: T.Tensor([rows, idx_pages], T.int32),
        Sinks: T.Tensor([heads], accum_dtype),
        Partial: T.Tensor([rows, segments, heads, D], accum_dtype),
        PartialStats: T.Tensor([rows, segments, heads, 2], accum_dtype),
        Output: T.Tensor([rows, heads, D], dtype),
    ):
        # One CTA per segment: unnormalized partials and their (max, sum).
        with T.Kernel(segments, head_blocks, rows, threads=threads) as (seg, hb, row):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            P_shared = T.alloc_shared([block_H, BI], dtype)
            staged = T.alloc_shared([BI, SWA_RECORD_BYTES], T.uint8)
            staged_fp8 = T.view(staged, dtype=T.float8_e4m3fn)
            staged_fp4 = T.view(staged, [BI, SWA_RECORD_BYTES * 2], T.float4_e2m1fn)
            slots = T.alloc_shared([slot_count], T.int32)
            copy_slot = T.alloc_local([copies], T.int32)
            acc_s = T.alloc_fragment([block_H, BI], accum_dtype)
            acc_g = T.alloc_fragment([block_H, D], accum_dtype)
            m_g = T.alloc_fragment([block_H], accum_dtype)
            l_g = T.alloc_fragment([block_H], accum_dtype)
            m_prev = T.alloc_fragment([block_H], accum_dtype)
            alpha = T.alloc_fragment([block_H], accum_dtype)
            row_sum = T.alloc_fragment([block_H], accum_dtype)

            for c in T.serial(G):
                chunk = seg * G + c
                if chunk < swa_blocks:
                    gather_swa(SwaIndices, SwaLengths, row, chunk * BI, BI, slots, c * BI)
                elif chunk < chunks:
                    gather_idx(IdxIndices, IdxLengths, IdxPageTable, row, (chunk - swa_blocks) * BI, BI, slots, c * BI)
            T.sync_threads()
            stage_chunk(SwaRecords, IdxRecords, seg * G, slots, 0, staged, copy_slot)
            h0 = hb * block_H
            load_q(Q, row, h0, Q_shared)
            clear_segment(acc_g, m_g, l_g)
            for c in T.serial(G):
                chunk = seg * G + c
                if chunk < chunks:
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    dequant_chunk(chunk, staged, staged_fp8, staged_fp4, KV_shared)
                    T.sync_threads()
                    if (c + 1 < G) & (chunk + 1 < chunks):
                        stage_chunk(SwaRecords, IdxRecords, chunk + 1, slots, (c + 1) * BI, staged, copy_slot)
                    chunk_update(Q_shared, KV_shared, slots, c * BI, P_shared, acc_s, acc_g, m_g, l_g, m_prev, alpha, row_sum)
            for h, d in T.Parallel(block_H, D):
                if h0 + h < heads:
                    Partial[row, seg, h0 + h, d] = acc_g[h, d]
            for h in T.Parallel(block_H):
                if h0 + h < heads:
                    PartialStats[row, seg, h0 + h, 0] = m_g[h]
                    PartialStats[row, seg, h0 + h, 1] = l_g[h]

        # Fold the segment partials in order, starting from the sink.
        with T.Kernel(heads, rows, threads=128) as (hc, rc):
            o = T.alloc_fragment([D], accum_dtype)
            M = T.alloc_var(accum_dtype)
            L = T.alloc_var(accum_dtype)
            a = T.alloc_var(accum_dtype)
            b = T.alloc_var(accum_dtype)
            m_new = T.alloc_var(accum_dtype)

            M = sink_log2(Sinks, hc)
            L = 1.0
            T.clear(o)
            for c in T.unroll(segments):
                m_new = T.max(M, PartialStats[rc, c, hc, 0])
                a = T.exp2(M - m_new)
                b = T.exp2(PartialStats[rc, c, hc, 0] - m_new)
                L = T.fma(L, a, T.fmul(PartialStats[rc, c, hc, 1], b))
                M = m_new
                for d in T.Parallel(D):
                    o[d] = T.fma(o[d], a, T.fmul(Partial[rc, c, hc, d], b))
            L = 1.0 / L
            for d in T.Parallel(D):
                Output[rc, hc, d] = T.fmul(o[d], L)

    return main


# ---------------------------------------------------------------------------
# V4.1 cache records and a reference implementation
# ---------------------------------------------------------------------------

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_BOUNDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def pack_swa_records(kv: torch.Tensor) -> torch.Tensor:
    """Encode ``[n, 512]`` latents as SWA records: E4M3 values, UE8M0 scale per 32."""
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // SWA_GROUP, SWA_GROUP)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-126, 127)
    payload = (groups / torch.exp2(exponent).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    scales = (exponent + 127).to(torch.uint8)
    return torch.cat([payload.view(torch.uint8).reshape(n, HEAD_DIM), scales], dim=-1).contiguous()


def pack_idx_records(kv: torch.Tensor) -> torch.Tensor:
    """Encode ``[n, 512]`` latents as indexed records: E2M1 values, E4M3 scale per 16."""
    n = kv.shape[0]
    groups = kv.float().reshape(n, HEAD_DIM // IDX_GROUP, IDX_GROUP)
    scales = (groups.abs().amax(dim=-1).clamp_min(6 * 2**-9) / 6.0).clamp(max=448).to(torch.float8_e4m3fn)
    normalized = groups / scales.float().unsqueeze(-1)
    magnitude = normalized.abs()
    bounds = torch.tensor(_E2M1_BOUNDS, device=kv.device)
    codes = torch.bucketize(magnitude, bounds)
    # Round ties to the even code.
    tie = (codes < 7) & (magnitude == bounds[codes.clamp(max=6)]) & ((codes & 1) != 0)
    codes = (codes + tie.to(codes.dtype)).to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)
    codes = codes.reshape(n, HEAD_DIM)
    payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return torch.cat([payload, scales.view(torch.uint8)], dim=-1).contiguous()


def decode_swa_records(records: torch.Tensor) -> torch.Tensor:
    payload = records[:, :HEAD_DIM].contiguous().view(torch.float8_e4m3fn).float()
    scales = torch.exp2(records[:, HEAD_DIM:].float() - 127.0)
    return (payload.reshape(-1, HEAD_DIM // SWA_GROUP, SWA_GROUP) * scales.unsqueeze(-1)).reshape(-1, HEAD_DIM)


def decode_idx_records(records: torch.Tensor) -> torch.Tensor:
    packed = records[:, : HEAD_DIM // 2].long()
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(-1, HEAD_DIM)
    lut = torch.tensor(_E2M1_VALUES, device=records.device)
    values = torch.where((codes & 8) != 0, -lut[codes & 7], lut[codes & 7])
    scales = records[:, HEAD_DIM // 2 :].contiguous().view(torch.float8_e4m3fn).float()
    return (values.reshape(-1, HEAD_DIM // IDX_GROUP, IDX_GROUP) * scales.unsqueeze(-1)).reshape(-1, HEAD_DIM)


def physical_slots(logical: torch.Tensor, page_table: torch.Tensor, page_size: int) -> torch.Tensor:
    """Resolve logical indexed positions through per-row page tables; padding stays -1."""
    pages = page_table.long().gather(1, (logical.long().clamp_min(0) // page_size).clamp_max(page_table.shape[1] - 1))
    return torch.where(logical >= 0, pages * page_size + logical.long() % page_size, -1).to(torch.int32)


def ref_sparse_mla_fwd_v41(q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_slots, idx_lengths, sinks, sm_scale=HEAD_DIM**-0.5):
    """FP32 reference over physical slots: softmax across the valid SWA and indexed entries plus the sink."""
    rows, heads, _ = q.shape
    out = torch.empty((rows, heads, HEAD_DIM), dtype=torch.float32, device=q.device)
    for r in range(rows):
        swa = swa_indices[r, : int(swa_lengths[r])]
        idx = idx_slots[r, : int(idx_lengths[r])]
        swa = swa[swa >= 0].long()
        idx = idx[idx >= 0].long()
        kv = torch.cat([decode_swa_records(swa_cache[swa]), decode_idx_records(idx_cache[idx])], dim=0)
        scores = q[r].float() @ kv.T * sm_scale
        row_max = torch.maximum(scores.max(dim=-1).values if kv.shape[0] else sinks.float(), sinks.float())
        weights = torch.exp(scores - row_max[:, None])
        denom = weights.sum(dim=-1) + torch.exp(sinks.float() - row_max)
        out[r] = weights @ kv / denom[:, None]
    return out.to(torch.bfloat16)


class SparseMlaV41Plan:
    """The kernels for one layer shape, with their workspace and output.

    All kernels implement the same arithmetic, fixed by ``segment_chunks`` and
    ``block_I``, so choosing one by row count never changes results:

    - ``decode`` with ``split_block_H``-head blocks while it fits one wave of
      CTAs (``split_max_rows``);
    - ``decode`` with ``block_H``-head blocks while it needs at most
      ``DECODE_MAX_WAVES`` waves (``decode_max_rows``);
    - ``extend`` with ``block_H``-head blocks above that.

    Allocation happens here, once; ``run`` only launches kernels and can be
    captured in a CUDA graph.
    """

    def __init__(
        self,
        heads,
        swa_width,
        idx_width,
        idx_page_size=64,
        *,
        max_rows,
        segment_chunks=SEGMENT_CHUNKS,
        decode_max_rows=None,
        split_max_rows=None,
        sm_scale=HEAD_DIM**-0.5,
        block_H=32,
        split_block_H=16,
        block_I=32,
        threads=256,
        device="cuda",
    ):
        shape = (heads, swa_width, idx_width, idx_page_size)
        config = dict(sm_scale=sm_scale, block_I=block_I, segment_chunks=segment_chunks)
        chunks = tilelang.cdiv(swa_width, block_I) + tilelang.cdiv(idx_width, block_I)
        segments = tilelang.cdiv(chunks, segment_chunks)
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        if decode_max_rows is None:
            decode_max_rows = DECODE_MAX_WAVES * sms // (segments * tilelang.cdiv(heads, block_H))
        if split_max_rows is None:
            split_max_rows = sms // (segments * tilelang.cdiv(heads, split_block_H)) if split_block_H else 0
        self.decode_max_rows = min(decode_max_rows, max_rows)
        self.split_max_rows = min(split_max_rows, self.decode_max_rows)
        self.kernels = {"extend": sparse_mla_fwd_v41(*shape, "extend", block_H=block_H, threads=threads, **config)}
        if self.decode_max_rows:
            self.kernels["decode"] = sparse_mla_fwd_v41(*shape, "decode", block_H=block_H, threads=threads, **config)
        if self.split_max_rows:
            # One warp per 16x8 tile of the 16-head x block_I score block, at most four.
            split_threads = 32 * min(4, (split_block_H // 16) * (block_I // 8))
            self.kernels["split"] = sparse_mla_fwd_v41(*shape, "decode", block_H=split_block_H, threads=split_threads, **config)
        self.partial = torch.empty((self.decode_max_rows, segments, heads, HEAD_DIM), dtype=torch.float32, device=device)
        self.partial_stats = torch.empty((self.decode_max_rows, segments, heads, 2), dtype=torch.float32, device=device)
        self.out = torch.empty((max_rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)

    def kernel_for(self, rows):
        if rows <= self.split_max_rows:
            return "split"
        return "decode" if rows <= self.decode_max_rows else "extend"

    def run(self, q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_indices, idx_lengths, idx_page_table, sinks, kernel=None):
        """Attend over raw uint8 ``[slots, 528]`` / ``[slots, 288]`` records.

        ``kernel`` forces ``"split"``, ``"decode"`` or ``"extend"``. A vLLM page
        buffer of shape ``[pages, page_size * record_bytes]`` is the same memory
        as ``cache.view(-1, record_bytes)``.
        """
        rows = q.shape[0]
        kernel = kernel or self.kernel_for(rows)
        out = self.out[:rows]
        args = (q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_indices, idx_lengths, idx_page_table, sinks)
        if kernel == "extend":
            self.kernels["extend"](*args, out)
        else:
            assert rows <= self.decode_max_rows, f"{rows} rows exceed the decode workspace ({self.decode_max_rows})"
            self.kernels[kernel](*args, self.partial[:rows], self.partial_stats[:rows], out)
        return out


def sparse_mla_fwd_v41_interface(*inputs, idx_page_size=64, kernel=None, **config):
    """One-off convenience: build a plan sized to this batch and run it."""
    q, swa_indices, idx_indices = inputs[0], inputs[2], inputs[5]
    plan = SparseMlaV41Plan(q.shape[1], swa_indices.shape[1], idx_indices.shape[1], idx_page_size, max_rows=q.shape[0], **config)
    return plan.run(*inputs, kernel=kernel).clone()


def make_inputs(
    rows,
    heads,
    swa_width=128,
    idx_width=512,
    swa_slots=4096,
    idx_pages=256,
    idx_page_size=64,
    idx_context=8192,
    seed=0,
    device="cuda",
):
    """Random queries, both caches and padded index lists in the V4.1 serving layout.

    SWA indices are physical slots. Indexed entries are logical positions in
    a context of ``idx_context`` compressed tokens, mapped through a shuffled
    per-row page table.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn((rows, heads, HEAD_DIM), generator=gen, device=device).to(torch.bfloat16)
    swa_cache = pack_swa_records(torch.randn((swa_slots, HEAD_DIM), generator=gen, device=device))
    idx_cache = pack_idx_records(torch.randn((idx_pages * idx_page_size, HEAD_DIM), generator=gen, device=device))
    sinks = torch.randn((heads,), generator=gen, device=device)

    def index_lists(width, extent, min_len):
        lengths = torch.randint(min_len, width + 1, (rows,), generator=gen, device=device, dtype=torch.int32)
        indices = torch.stack([torch.randperm(extent, generator=gen, device=device)[:width] for _ in range(rows)]).to(torch.int32)
        pos = torch.arange(width, device=device)[None, :]
        return torch.where(pos < lengths[:, None], indices, -1).contiguous(), lengths

    swa_indices, swa_lengths = index_lists(swa_width, swa_slots, 1)
    idx_indices, idx_lengths = index_lists(idx_width, idx_context, 0)
    logical_pages = idx_context // idx_page_size
    idx_page_table = torch.stack([torch.randperm(idx_pages, generator=gen, device=device)[:logical_pages] for _ in range(rows)])
    return q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_indices, idx_lengths, idx_page_table.to(torch.int32), sinks


def test_sparse_mla_fwd_v41(rows=8, heads=24, swa_width=128, idx_width=512, check_invariance=True, **config):
    inputs = make_inputs(rows, heads, swa_width, idx_width)
    q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_indices, idx_lengths, idx_page_table, sinks = inputs
    config.setdefault("split_max_rows", rows)
    plan = SparseMlaV41Plan(heads, swa_width, idx_width, max_rows=rows, decode_max_rows=rows, **config)
    outs = {name: plan.run(*inputs, kernel=name).clone() for name in plan.kernels}
    extend = outs["extend"]
    idx_slots = physical_slots(idx_indices, idx_page_table, 64)
    ref = ref_sparse_mla_fwd_v41(q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_slots, idx_lengths, sinks)
    torch.testing.assert_close(extend.float(), ref.float(), rtol=2e-2, atol=2e-2)
    # Every kernel implements one contract: identical bits.
    for name, out in outs.items():
        assert torch.equal(out, extend), f"{name} and extend differ by {(out.float() - extend.float()).abs().max().item()}"

    if check_invariance:
        # The same row alone and in a batch must give identical bits.
        per_row = (0, 2, 3, 5, 6, 7)  # q, both index lists and lengths, the page table
        for r in (0, rows - 1):
            solo_inputs = [t[r : r + 1].contiguous() if i in per_row else t for i, t in enumerate(inputs)]
            for name in plan.kernels:
                solo = plan.run(*solo_inputs, kernel=name)
                assert torch.equal(solo[0], extend[r]), f"row {r} changed with batch composition ({name})"
    print(f"rows={rows} heads={heads} swa={swa_width} idx={idx_width} {config}: OK, {' == '.join(outs)}")


def run_regression_perf(rows=8, heads=24, swa_width=128, idx_width=512, kernel=None, **config):
    inputs = make_inputs(rows, heads, swa_width, idx_width)
    plan = SparseMlaV41Plan(heads, swa_width, idx_width, max_rows=rows, **config)
    return do_bench(lambda: plan.run(*inputs, kernel=kernel), backend="cupti")


def benchmark(rows, heads, swa_width, idx_width, kernel=None, backend="cudagraph", **config):
    inputs = make_inputs(rows, heads, swa_width, idx_width)
    plan = SparseMlaV41Plan(heads, swa_width, idx_width, max_rows=rows, **config)
    kernel = kernel or plan.kernel_for(rows)
    ms = do_bench(lambda: plan.run(*inputs, kernel=kernel), warmup=50, rep=200, backend=backend)
    record_bytes = rows * (swa_width * SWA_RECORD_BYTES + idx_width * IDX_RECORD_BYTES)
    print(f"rows={rows} heads={heads} {kernel} {config} [{backend}]: {ms * 1e3:.1f} us, {record_bytes / ms * 1e-6:.1f} GB/s of records")
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--heads", type=int, default=24, help="local heads per rank (24 at TP3)")
    parser.add_argument("--swa-width", type=int, default=128)
    parser.add_argument("--idx-width", type=int, default=512)
    parser.add_argument("--kernel", choices=["split", "decode", "extend"], default=None)
    parser.add_argument("--segment-chunks", type=int, default=SEGMENT_CHUNKS)
    parser.add_argument("--block-h", type=int, default=32)
    parser.add_argument("--block-i", type=int, default=32)
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()
    config = dict(segment_chunks=args.segment_chunks, block_H=args.block_h, block_I=args.block_i, threads=args.threads)
    test_sparse_mla_fwd_v41(args.rows, args.heads, args.swa_width, args.idx_width, **config)
    benchmark(args.rows, args.heads, args.swa_width, args.idx_width, args.kernel, **config)


if __name__ == "__main__":
    main()
