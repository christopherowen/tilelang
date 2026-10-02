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

Records are gathered by slot, dequantized exactly to BF16 in shared memory,
and multiplied on the warp-level MMA path. The arithmetic is defined per
chunk of ``block_I`` entries of the concatenated [SWA | indexed] list:

- chunk ``c`` yields scores ``s = q.k * sm_scale * log2(e)``, its maximum
  ``m_c``, ``p = exp2(s - m_c)``, ``l_c = sum(p)`` and ``O_c = P V``;
- the chunk partials then fold in chunk order into a state that starts from
  the head's attention sink, with ``M' = max(M, m_c)``,
  ``L = fma(L, exp2(M - M'), l_c * exp2(m_c - M'))``, the same update for
  ``O``, and finally ``out = O * (1 / L)``.

Two schedules implement this contract:

- ``extend``: one CTA per row and head block computes and folds every
  chunk in registers;
- ``decode``: one CTA per chunk, head block and row writes its partial to a
  caller-owned workspace, and a merge kernel folds the partials. This gives
  small batches up to ``chunks x head_blocks`` CTAs per row instead of one.

Both schedules use the same chunk code, and explicit ``T.fma``/``T.fmul``
keep the fold's rounding fixed, so they produce identical bits. A row's
output depends neither on its neighbours nor on the schedule the batch size
selects. ``SparseMlaV41Plan`` compiles both, owns the workspace and output,
and picks the schedule from the row count.

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
# Batches up to this many rows use the decode schedule by default.
DECODE_MAX_ROWS = 8


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
):
    """Build one schedule of the sparse MLA kernel.

    Both schedules take:

    - ``Q``: ``[rows, heads, 512]`` BF16 queries, with RoPE applied;
    - ``SwaFP8``, ``SwaBytes``: E4M3 and uint8 views of the ``[slots, 528]``
      SWA records;
    - ``SwaIndices``: ``[rows, swa_width]`` int32 SWA slots, ``-1`` for padding;
    - ``SwaLengths``: ``[rows]`` int32 count of leading SWA entries to use;
    - ``IdxFP4``: E2M1 view (``[slots, 576]`` logical) of the ``[slots, 288]``
      indexed records;
    - ``IdxFP8``: E4M3 view of the same records, scales from byte 256;
    - ``IdxIndices``: ``[rows, idx_width]`` int32 logical indexed positions,
      ``-1`` for padding;
    - ``IdxLengths``: ``[rows]`` int32 count of leading indexed entries to use;
    - ``IdxPageTable``: ``[rows, pages]`` int32 physical page of each logical
      ``idx_page_size`` page, as in vLLM's indexed page table;
    - ``Sinks``: ``[heads]`` FP32 attention-sink logits.

    ``decode`` then takes the FP32 workspaces ``Partial``
    (``[rows, chunks, heads, 512]``) and ``PartialStats``
    (``[rows, chunks, heads, 2]``). Both write ``Output``
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
    # Q, the dequantized KV block and P in BF16, plus reduction scratch.
    smem_bytes = 2 * (block_H * D + BI * D + block_H * BI) + 4096
    assert smem_bytes <= 99 * 1024, (
        f"block_H={block_H}, block_I={BI} needs ~{smem_bytes} B of shared memory; SM120 allows 99 KiB (use block_I=16 with block_H=64)"
    )
    head_blocks = tilelang.cdiv(heads, block_H)
    pad_heads = heads % block_H != 0
    swa_blocks = tilelang.cdiv(swa_width, BI)
    chunks = swa_blocks + tilelang.cdiv(idx_width, BI)

    @T.macro
    def gather(Indices, Lengths, width, row, blk, slot_shared):
        for i in T.Parallel(BI):
            pos = blk * BI + i
            slot = Indices[row, T.min(pos, width - 1)]
            slot_shared[i] = T.if_then_else((pos < width) & (pos < Lengths[row]) & (slot >= 0), slot, -1)

    @T.macro
    def gather_paged(Indices, Lengths, PageTable, row, blk, slot_shared):
        # Map logical positions to physical slots through the row's page table.
        for i in T.Parallel(BI):
            pos = blk * BI + i
            logical = Indices[row, T.min(pos, idx_width - 1)]
            page = PageTable[row, T.min(T.max(logical, 0) // idx_page_size, idx_pages - 1)]
            slot_shared[i] = T.if_then_else(
                (pos < idx_width) & (pos < Lengths[row]) & (logical >= 0),
                page * idx_page_size + logical % idx_page_size,
                -1,
            )

    @T.macro
    def dequant_swa(SwaFP8, SwaBytes, slot_shared, KV_shared):
        for i, d in T.Parallel(BI, D):
            slot = T.max(slot_shared[i], 0)
            # UE8M0 scale 2^(e - 127), built from the exponent bits.
            scale = T.reinterpret(T.shift_left(T.cast(SwaBytes[slot, D + d // SWA_GROUP], T.uint32), 23), accum_dtype)
            KV_shared[i, d] = T.cast(T.cast(SwaFP8[slot, d], accum_dtype) * scale, dtype)

    @T.macro
    def dequant_idx(IdxFP4, IdxFP8, slot_shared, KV_shared):
        for i, d in T.Parallel(BI, D):
            slot = T.max(slot_shared[i], 0)
            scale = T.cast(IdxFP8[slot, D // 2 + d // IDX_GROUP], accum_dtype)
            KV_shared[i, d] = T.cast(T.cast(IdxFP4[slot, d], accum_dtype) * scale, dtype)

    @T.macro
    def load_q(Q, row, h0, Q_shared):
        if pad_heads:
            for h, d in T.Parallel(block_H, D):
                Q_shared[h, d] = T.if_then_else(h0 + h < heads, Q[row, T.min(h0 + h, heads - 1), d], 0)
        else:
            T.copy(Q[row, h0 : h0 + block_H, :], Q_shared)

    @T.macro
    def chunk_partial(
        SwaFP8, SwaBytes, SwaIndices, SwaLengths, IdxFP4, IdxFP8, IdxIndices, IdxLengths, IdxPageTable,
        row, chunk, Q_shared, KV_shared, slot_shared, P_shared, acc_s, acc_c, m_c, l_c,
    ):  # fmt: skip
        """The contract's per-chunk partial: (acc_c, m_c, l_c) for heads [h0, h0 + block_H)."""
        if chunk < swa_blocks:
            gather(SwaIndices, SwaLengths, swa_width, row, chunk, slot_shared)
            dequant_swa(SwaFP8, SwaBytes, slot_shared, KV_shared)
        else:
            gather_paged(IdxIndices, IdxLengths, IdxPageTable, row, chunk - swa_blocks, slot_shared)
            dequant_idx(IdxFP4, IdxFP8, slot_shared, KV_shared)
        T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)
        for h, i in T.Parallel(block_H, BI):
            acc_s[h, i] = T.if_then_else(slot_shared[i] >= 0, T.fmul(acc_s[h, i], scale_log2), -T.infinity(accum_dtype))
        T.reduce_max(acc_s, m_c, dim=1, clear=True)
        for h, i in T.Parallel(block_H, BI):
            acc_s[h, i] = T.if_then_else(slot_shared[i] >= 0, T.exp2(acc_s[h, i] - m_c[h]), 0)
        T.reduce_sum(acc_s, l_c, dim=1)
        T.copy(acc_s, P_shared)
        T.gemm(P_shared, KV_shared, acc_c, policy=T.GemmWarpPolicy.FullRow, clear_accum=True)

    @T.macro
    def sink_log2(Sinks, h):
        return T.fmul(T.if_then_else(h < heads, Sinks[T.min(h, heads - 1)], 0), LOG2E)

    if schedule == "extend":

        @T.prim_func
        def main(
            Q: T.Tensor([rows, heads, D], dtype),
            SwaFP8: T.Tensor([swa_slots, SWA_RECORD_BYTES], T.float8_e4m3fn),
            SwaBytes: T.Tensor([swa_slots, SWA_RECORD_BYTES], T.uint8),
            SwaIndices: T.Tensor([rows, swa_width], T.int32),
            SwaLengths: T.Tensor([rows], T.int32),
            IdxFP4: T.Tensor([idx_slots, IDX_RECORD_BYTES * 2], T.float4_e2m1fn),
            IdxFP8: T.Tensor([idx_slots, IDX_RECORD_BYTES], T.float8_e4m3fn),
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
                slot_shared = T.alloc_shared([BI], T.int32)
                acc_s = T.alloc_fragment([block_H, BI], accum_dtype)
                acc_c = T.alloc_fragment([block_H, D], accum_dtype)
                acc_o = T.alloc_fragment([block_H, D], accum_dtype)
                m_c = T.alloc_fragment([block_H], accum_dtype)
                l_c = T.alloc_fragment([block_H], accum_dtype)
                M = T.alloc_fragment([block_H], accum_dtype)
                L = T.alloc_fragment([block_H], accum_dtype)
                a = T.alloc_fragment([block_H], accum_dtype)
                b = T.alloc_fragment([block_H], accum_dtype)

                h0 = hb * block_H
                load_q(Q, row, h0, Q_shared)
                T.fill(acc_o, 0)
                T.fill(L, 1)
                for h in T.Parallel(block_H):
                    M[h] = sink_log2(Sinks, h0 + h)

                for chunk in T.serial(chunks):
                    chunk_partial(
                        SwaFP8, SwaBytes, SwaIndices, SwaLengths, IdxFP4, IdxFP8, IdxIndices, IdxLengths, IdxPageTable,
                        row, chunk, Q_shared, KV_shared, slot_shared, P_shared, acc_s, acc_c, m_c, l_c,
                    )  # fmt: skip
                    # Fold the chunk into the running state, as the decode merge kernel does.
                    for h in T.Parallel(block_H):
                        m_new = T.max(M[h], m_c[h])
                        a[h] = T.exp2(M[h] - m_new)
                        b[h] = T.exp2(m_c[h] - m_new)
                        L[h] = T.fma(L[h], a[h], T.fmul(l_c[h], b[h]))
                        M[h] = m_new
                    for h, d in T.Parallel(block_H, D):
                        acc_o[h, d] = T.fma(acc_o[h, d], a[h], T.fmul(acc_c[h, d], b[h]))

                for h in T.Parallel(block_H):
                    L[h] = 1.0 / L[h]
                for h, d in T.Parallel(block_H, D):
                    if h0 + h < heads:
                        Output[row, h0 + h, d] = T.fmul(acc_o[h, d], L[h])

        return main

    @T.prim_func
    def main(
        Q: T.Tensor([rows, heads, D], dtype),
        SwaFP8: T.Tensor([swa_slots, SWA_RECORD_BYTES], T.float8_e4m3fn),
        SwaBytes: T.Tensor([swa_slots, SWA_RECORD_BYTES], T.uint8),
        SwaIndices: T.Tensor([rows, swa_width], T.int32),
        SwaLengths: T.Tensor([rows], T.int32),
        IdxFP4: T.Tensor([idx_slots, IDX_RECORD_BYTES * 2], T.float4_e2m1fn),
        IdxFP8: T.Tensor([idx_slots, IDX_RECORD_BYTES], T.float8_e4m3fn),
        IdxIndices: T.Tensor([rows, idx_width], T.int32),
        IdxLengths: T.Tensor([rows], T.int32),
        IdxPageTable: T.Tensor([rows, idx_pages], T.int32),
        Sinks: T.Tensor([heads], accum_dtype),
        Partial: T.Tensor([rows, chunks, heads, D], accum_dtype),
        PartialStats: T.Tensor([rows, chunks, heads, 2], accum_dtype),
        Output: T.Tensor([rows, heads, D], dtype),
    ):
        # One CTA per chunk: unnormalized partials and their (max, sum).
        with T.Kernel(chunks, head_blocks, rows, threads=threads) as (chunk, hb, row):
            Q_shared = T.alloc_shared([block_H, D], dtype)
            KV_shared = T.alloc_shared([BI, D], dtype)
            P_shared = T.alloc_shared([block_H, BI], dtype)
            slot_shared = T.alloc_shared([BI], T.int32)
            acc_s = T.alloc_fragment([block_H, BI], accum_dtype)
            acc_c = T.alloc_fragment([block_H, D], accum_dtype)
            m_c = T.alloc_fragment([block_H], accum_dtype)
            l_c = T.alloc_fragment([block_H], accum_dtype)

            h0 = hb * block_H
            load_q(Q, row, h0, Q_shared)
            chunk_partial(
                SwaFP8, SwaBytes, SwaIndices, SwaLengths, IdxFP4, IdxFP8, IdxIndices, IdxLengths, IdxPageTable,
                row, chunk, Q_shared, KV_shared, slot_shared, P_shared, acc_s, acc_c, m_c, l_c,
            )  # fmt: skip
            for h, d in T.Parallel(block_H, D):
                if h0 + h < heads:
                    Partial[row, chunk, h0 + h, d] = acc_c[h, d]
            for h in T.Parallel(block_H):
                if h0 + h < heads:
                    PartialStats[row, chunk, h0 + h, 0] = m_c[h]
                    PartialStats[row, chunk, h0 + h, 1] = l_c[h]

        # Fold the partials in chunk order, starting from the sink.
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
            for c in T.unroll(chunks):
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
    """Both schedules for one layer shape, with their workspace and output.

    The schedules produce identical bits, so ``decode_max_rows`` only moves
    the performance crossover. Allocation happens here, once; ``run`` only
    launches kernels and can be captured in a CUDA graph.
    """

    def __init__(
        self,
        heads,
        swa_width,
        idx_width,
        idx_page_size=64,
        *,
        max_rows,
        decode_max_rows=DECODE_MAX_ROWS,
        sm_scale=HEAD_DIM**-0.5,
        block_H=32,
        block_I=32,
        threads=256,
        device="cuda",
    ):
        shape = (heads, swa_width, idx_width, idx_page_size)
        config = dict(sm_scale=sm_scale, block_H=block_H, block_I=block_I, threads=threads)
        self.decode_max_rows = min(decode_max_rows, max_rows)
        self.extend = sparse_mla_fwd_v41(*shape, "extend", **config)
        self.decode = sparse_mla_fwd_v41(*shape, "decode", **config) if self.decode_max_rows else None
        chunks = tilelang.cdiv(swa_width, block_I) + tilelang.cdiv(idx_width, block_I)
        self.partial = torch.empty((self.decode_max_rows, chunks, heads, HEAD_DIM), dtype=torch.float32, device=device)
        self.partial_stats = torch.empty((self.decode_max_rows, chunks, heads, 2), dtype=torch.float32, device=device)
        self.out = torch.empty((max_rows, heads, HEAD_DIM), dtype=torch.bfloat16, device=device)

    def schedule_for(self, rows):
        return "decode" if rows <= self.decode_max_rows else "extend"

    def run(self, q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_indices, idx_lengths, idx_page_table, sinks, schedule=None):
        """Attend over raw uint8 ``[slots, 528]`` / ``[slots, 288]`` records.

        A vLLM page buffer of shape ``[pages, page_size * record_bytes]`` is the
        same memory as ``cache.view(-1, record_bytes)``.
        """
        rows = q.shape[0]
        schedule = schedule or self.schedule_for(rows)
        out = self.out[:rows]
        args = (
            q,
            swa_cache.view(torch.float8_e4m3fn),
            swa_cache,
            swa_indices,
            swa_lengths,
            idx_cache.view(torch.int8),
            idx_cache.view(torch.float8_e4m3fn),
            idx_indices,
            idx_lengths,
            idx_page_table,
            sinks,
        )
        if schedule == "decode":
            assert rows <= self.decode_max_rows, f"{rows} rows exceed the decode workspace ({self.decode_max_rows})"
            self.decode(*args, self.partial[:rows], self.partial_stats[:rows], out)
        else:
            self.extend(*args, out)
        return out


def sparse_mla_fwd_v41_interface(*inputs, idx_page_size=64, schedule=None, **config):
    """One-off convenience: build a plan sized to this batch and run it."""
    q, swa_indices, idx_indices = inputs[0], inputs[2], inputs[5]
    plan = SparseMlaV41Plan(q.shape[1], swa_indices.shape[1], idx_indices.shape[1], idx_page_size, max_rows=q.shape[0], **config)
    return plan.run(*inputs, schedule=schedule).clone()


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
    plan = SparseMlaV41Plan(heads, swa_width, idx_width, max_rows=rows, decode_max_rows=rows, **config)
    extend = plan.run(*inputs, schedule="extend").clone()
    decode = plan.run(*inputs, schedule="decode").clone()
    idx_slots = physical_slots(idx_indices, idx_page_table, 64)
    ref = ref_sparse_mla_fwd_v41(q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_slots, idx_lengths, sinks)
    torch.testing.assert_close(extend.float(), ref.float(), rtol=2e-2, atol=2e-2)
    # The two schedules implement one contract: identical bits.
    assert torch.equal(decode, extend), f"decode and extend differ by {(decode.float() - extend.float()).abs().max().item()}"

    if check_invariance:
        # The same row alone and in a batch must give identical bits.
        per_row = (0, 2, 3, 5, 6, 7)  # q, both index lists and lengths, the page table
        for r in (0, rows - 1):
            solo_inputs = [t[r : r + 1].contiguous() if i in per_row else t for i, t in enumerate(inputs)]
            for schedule in ("decode", "extend"):
                solo = plan.run(*solo_inputs, schedule=schedule)
                assert torch.equal(solo[0], extend[r]), f"row {r} changed with batch composition ({schedule})"
    print(f"rows={rows} heads={heads} swa={swa_width} idx={idx_width} {config}: OK, decode == extend")


def run_regression_perf(rows=8, heads=24, swa_width=128, idx_width=512, schedule=None, **config):
    inputs = make_inputs(rows, heads, swa_width, idx_width)
    plan = SparseMlaV41Plan(heads, swa_width, idx_width, max_rows=rows, **config)
    return do_bench(lambda: plan.run(*inputs, schedule=schedule), backend="cupti")


def benchmark(rows, heads, swa_width, idx_width, schedule=None, backend="cudagraph", **config):
    inputs = make_inputs(rows, heads, swa_width, idx_width)
    plan = SparseMlaV41Plan(heads, swa_width, idx_width, max_rows=rows, decode_max_rows=max(rows, DECODE_MAX_ROWS), **config)
    schedule = schedule or plan.schedule_for(rows)
    ms = do_bench(lambda: plan.run(*inputs, schedule=schedule), warmup=50, rep=200, backend=backend)
    record_bytes = rows * (swa_width * SWA_RECORD_BYTES + idx_width * IDX_RECORD_BYTES)
    print(f"rows={rows} heads={heads} {schedule} {config} [{backend}]: {ms * 1e3:.1f} us, {record_bytes / ms * 1e-6:.1f} GB/s of records")
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--heads", type=int, default=24, help="local heads per rank (24 at TP3)")
    parser.add_argument("--swa-width", type=int, default=128)
    parser.add_argument("--idx-width", type=int, default=512)
    parser.add_argument("--schedule", choices=["decode", "extend"], default=None)
    parser.add_argument("--block-h", type=int, default=32)
    parser.add_argument("--block-i", type=int, default=32)
    parser.add_argument("--threads", type=int, default=256)
    args = parser.parse_args()
    config = dict(block_H=args.block_h, block_I=args.block_i, threads=args.threads)
    test_sparse_mla_fwd_v41(args.rows, args.heads, args.swa_width, args.idx_width, **config)
    benchmark(args.rows, args.heads, args.swa_width, args.idx_width, args.schedule, **config)


if __name__ == "__main__":
    main()
