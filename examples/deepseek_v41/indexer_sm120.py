"""DeepSeek-V4.1-Flash sparse-attention indexer for SM120 / SM121: scores and top-k.

The V4.1 indexer has 32 heads of width 128. It keeps its keys in a paged
MXFP4 cache: each page holds ``page_size`` rows of 64 bytes of packed E2M1
(low nibble first), followed by ``page_size`` rows of 4 UE8M0 scales (one per
32 dims). Queries are quantized to MXFP4 the same way, and per-head weights
arrive as BF16, already multiplied by ``1/64`` and rounded.

The published score keeps distinct BF16 rounding stages. For each position
``j`` and head ``h``:

    dot   = bf16(q_h . k_j)                     # FP32 accumulation
    c_h   = bf16(max(dot, 0) * w_h)
    score = bf16(c_0 + c_1 + ... + c_31)         # FP32, increasing head order

``indexer_score_v41`` computes ``q . k`` straight from the packed operands with
SM120's MXFP4 block-scaled MMA
(``mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::2X.ue8m0``), then
applies the rounding stages and the ordered head sum in the epilogue. It
scores either every position of a row's context, or only the positions in a
candidate list (V4.1's reindex layers).

``IndexerTopK`` selects the top ``k`` entries of each row exactly. Each
4096-entry segment caches order-preserving 16-bit keys of its BF16 scores in
shared memory, runs a two-pass 8-bit radix select and an ordered compaction
(ties at the threshold keep the lowest positions), and keeps its own top
``k``. Long rows then merge the segment winners with the same selection, so
small batches still spread over many CTAs. Positions come back in ascending
order, padded with ``-1``.
``group=8`` ranks blocks of eight positions by their maximum, always keeping
the block that holds the last visible position, and expands them: V4.1's
candidate-domain selection at its candidate source layer.

Selection is deterministic: the same scores give the same positions
regardless of thread timing or batch composition.
"""

import argparse

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

INDEX_HEADS = 32
INDEX_DIM = 128
INDEX_TOPK = 512
CANDIDATE_BLOCKS = 2048
CANDIDATE_BLOCK = 8
SF_BLOCK = 32
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
# Order-preserving uint16 key of BF16 -inf: entries at or below it are never selected.
_KEY_NEG_INF = 0x007F


def page_bytes(page_size: int) -> int:
    """Index-cache page: ``[page_size, 64]`` packed data, then ``[page_size, 4]`` UE8M0."""
    return page_size * (INDEX_DIM // 2 + INDEX_DIM // SF_BLOCK)


@tilelang.jit
def indexer_score_v41(
    width: int,
    page_size: int,
    candidates: bool = False,
    heads: int = INDEX_HEADS,
    block_P: int = 64,
    threads: int = 128,
):
    """Build the score kernel. Inputs:

    - ``Q``: ``[rows, heads, 128]`` E2M1 queries (``[rows, heads, 64]`` int8 storage);
    - ``QScale``: ``[rows, heads]`` uint32, the four UE8M0 scales of each head;
    - ``Weights``: ``[rows, heads]`` BF16 head weights;
    - ``KPool``: E2M1 view (``[pages, page_bytes * 2]`` logical) of the index cache;
    - ``KScale``: uint32 view (``[pages, page_bytes / 4]``) of the same cache;
    - ``PageTable``: ``[rows, table_width]`` int32 physical page per logical page;
    - ``Lengths``: ``[rows]`` int32 positions per row; ``Active``: ``[1]`` int32 width cap;
    - candidate mode: ``Candidates`` ``[rows, width]`` int32 positions and
      ``CandidateLengths`` ``[rows]``.

    Writes ``Scores`` (``[rows, width]`` BF16), ``-inf`` past each row's extent.
    """
    assert page_size % block_P == 0 or candidates, "dense tiles must not straddle a page"
    rows, pool_pages, table_width = T.dynamic("rows, pool_pages, table_width")
    D = INDEX_DIM
    fp4 = T.float4_e2m1fn
    accum_dtype = T.float32
    page_words = page_bytes(page_size) // 4
    scale_word0 = page_size * D // 2 // 4  # first scale word of a page

    @T.macro
    def score_tile(
        Q, QScale, Weights, KPool, KScale, PageTable, row, p0, extent, position, slot_valid, Scores,
    ):  # fmt: skip
        K_shared = T.alloc_shared([block_P, D], fp4)
        Q_shared = T.alloc_shared([heads, D], fp4)
        SFK_shared = T.alloc_shared([block_P, 1], T.uint32)
        SFQ_shared = T.alloc_shared([heads, 1], T.uint32)
        W_shared = T.alloc_shared([heads], accum_dtype)
        contrib = T.alloc_shared([block_P, heads], accum_dtype)
        acc = T.alloc_fragment([block_P, heads], accum_dtype)
        total = T.alloc_fragment([block_P], accum_dtype)

        for h, d in T.Parallel(heads, D):
            Q_shared[h, d] = Q[row, h, d]
        for h in T.Parallel(heads):
            SFQ_shared[h, 0] = QScale[row, h]
            W_shared[h] = T.cast(Weights[row, h], accum_dtype)
        # Invalid slots load a clamped, in-pool key and are masked at the end.
        for p, d in T.Parallel(block_P, D):
            pos = position(p)
            page = T.min(T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0), pool_pages - 1)
            K_shared[p, d] = KPool[page, (pos % page_size) * D + d]
        for p in T.Parallel(block_P):
            pos = position(p)
            page = T.min(T.max(PageTable[row, T.min(pos // page_size, table_width - 1)], 0), pool_pages - 1)
            SFK_shared[p, 0] = KScale[page, scale_word0 + pos % page_size]
        T.mma_gemm_blockscaled(
            K_shared,
            Q_shared,
            acc,
            SFK_shared,
            SFQ_shared,
            transpose_B=True,
            clear_accum=True,
            k_start=0,
            sf_a_granularity_k=SF_BLOCK,
            sf_b_granularity_k=SF_BLOCK,
        )
        # The published rounding stages: BF16 dot, ReLU, BF16 weighted product.
        for p, h in T.Parallel(block_P, heads):
            dot = T.max(T.cast(T.cast(acc[p, h], T.bfloat16), accum_dtype), 0.0)
            contrib[p, h] = T.cast(T.cast(T.fmul(dot, W_shared[h]), T.bfloat16), accum_dtype)
        T.fill(total, 0)
        for h in T.serial(heads):  # increasing head order, FP32
            for p in T.Parallel(block_P):
                total[p] += contrib[p, h]
        for p in T.Parallel(block_P):
            if p0 + p < width:
                Scores[row, p0 + p] = T.if_then_else(slot_valid(p), T.cast(total[p], T.bfloat16), -T.infinity(T.bfloat16))

    @T.macro
    def fill_neg_inf(row, p0, Scores):
        for p in T.Parallel(block_P):
            if p0 + p < width:
                Scores[row, p0 + p] = -T.infinity(T.bfloat16)

    if not candidates:

        @T.prim_func
        def main(
            Q: T.Tensor([rows, heads, D], fp4),
            QScale: T.Tensor([rows, heads], T.uint32),
            Weights: T.Tensor([rows, heads], T.bfloat16),
            KPool: T.Tensor([pool_pages, page_words * 8], fp4),
            KScale: T.Tensor([pool_pages, page_words], T.uint32),
            PageTable: T.Tensor([rows, table_width], T.int32),
            Lengths: T.Tensor([rows], T.int32),
            Active: T.Tensor([1], T.int32),
            Scores: T.Tensor([rows, width], T.bfloat16),
        ):
            with T.Kernel(T.ceildiv(width, block_P), rows, threads=threads) as (tile, row):
                p0 = tile * block_P
                extent = T.min(T.min(T.max(Lengths[row], 0), T.max(Active[0], 0)), width)
                if p0 < extent:
                    score_tile(
                        Q, QScale, Weights, KPool, KScale, PageTable, row, p0, extent,
                        lambda p: p0 + p, lambda p: p0 + p < extent, Scores,
                    )  # fmt: skip
                else:
                    fill_neg_inf(row, p0, Scores)

        return main

    @T.prim_func
    def main(
        Q: T.Tensor([rows, heads, D], fp4),
        QScale: T.Tensor([rows, heads], T.uint32),
        Weights: T.Tensor([rows, heads], T.bfloat16),
        KPool: T.Tensor([pool_pages, page_words * 8], fp4),
        KScale: T.Tensor([pool_pages, page_words], T.uint32),
        PageTable: T.Tensor([rows, table_width], T.int32),
        Lengths: T.Tensor([rows], T.int32),
        Active: T.Tensor([1], T.int32),
        Candidates: T.Tensor([rows, width], T.int32),
        CandidateLengths: T.Tensor([rows], T.int32),
        Scores: T.Tensor([rows, width], T.bfloat16),
    ):
        with T.Kernel(T.ceildiv(width, block_P), rows, threads=threads) as (tile, row):
            p0 = tile * block_P
            extent = T.min(T.max(CandidateLengths[row], 0), width)
            if p0 < extent:
                score_tile(
                    Q, QScale, Weights, KPool, KScale, PageTable, row, p0, extent,
                    lambda p: T.max(Candidates[row, T.min(p0 + p, width - 1)], 0),
                    lambda p: (p0 + p < extent) & (Candidates[row, T.min(p0 + p, width - 1)] >= 0),
                    Scores,
                )  # fmt: skip
            else:
                fill_neg_inf(row, p0, Scores)

    return main


def _key16(x):
    """Order-preserving uint16 key of a BF16 value, as int32."""
    bits = T.cast(T.reinterpret(x, T.uint16), T.int32)
    return T.if_then_else(bits >= 0x8000, bits ^ 0xFFFF, bits | 0x8000)


def _value16(key):
    """The BF16 value of an order-preserving key."""
    return T.reinterpret(T.cast(T.if_then_else(key >= 0x8000, key ^ 0x8000, key ^ 0xFFFF), T.uint16), T.bfloat16)


_TOPK_THREADS = 512
_SEGMENT_ENTRIES = _TOPK_THREADS * 8


def _select_macros(threads, topk):
    """Radix threshold and ordered compaction over int32 keys held in shared memory."""
    radix = 256
    steps = threads.bit_length() - 1

    @T.macro
    def block_scan(values, tx, n_steps):
        # Inclusive Hillis-Steele scan of values[0:threads].
        carry = T.alloc_var(T.int32)
        for s in T.serial(n_steps):
            off = 1 << s
            carry = T.if_then_else(tx >= off, values[T.max(tx - off, 0)], 0)
            T.sync_threads()
            values[tx] += carry
            T.sync_threads()

    @T.macro
    def suffix_counts(hist, scan, tx):
        # hist[b] <- number of keys whose digit is >= b; hist[radix] = 0.
        if tx < radix:
            scan[tx] = hist[radix - 1 - tx]
        T.sync_threads()
        block_scan(scan, tx, 8)
        if tx < radix:
            hist[radix - 1 - tx] = scan[tx]
        if tx == 0:
            hist[radix] = 0
        T.sync_threads()

    @T.macro
    def threshold(keys, n, tx, hist, scan, params):
        """params[0] <- threshold key, params[1] <- ties to keep, for the top ``topk`` of keys[0:n]."""
        key = T.alloc_var(T.int32)
        if tx < radix + 1:
            hist[tx] = 0
        if tx == 0:
            params[2] = 0
            params[3] = 0
        T.sync_threads()
        for i in T.serial(T.ceildiv(n, threads)):
            if i * threads + tx < n:
                key = keys[i * threads + tx]
                if key > _KEY_NEG_INF:
                    T.atomic_add(hist[key >> 8], 1)
        T.sync_threads()
        suffix_counts(hist, scan, tx)
        if tx < radix:  # noqa: SIM102 - the guard keeps hist[tx + 1] in bounds; TIR & is not short-circuit
            if (hist[tx] >= topk) & (hist[tx + 1] < topk):
                params[2] = tx
        T.sync_threads()
        hi = T.alloc_var(T.int32)
        above = T.alloc_var(T.int32)
        selectable = T.alloc_var(T.int32)
        hi = params[2]
        above = hist[hi + 1]
        selectable = hist[0]
        T.sync_threads()
        if tx < radix + 1:
            hist[tx] = 0
        T.sync_threads()
        for i in T.serial(T.ceildiv(n, threads)):
            if i * threads + tx < n:
                key = keys[i * threads + tx]
                if (key > _KEY_NEG_INF) & ((key >> 8) == hi):
                    T.atomic_add(hist[key & 0xFF], 1)
        T.sync_threads()
        suffix_counts(hist, scan, tx)
        if tx < radix:  # noqa: SIM102 - the guard keeps hist[tx + 1] in bounds
            if (above + hist[tx] >= topk) & (above + hist[tx + 1] < topk):
                params[3] = tx
        T.sync_threads()
        if tx == 0:
            if selectable <= topk:
                # Everything selectable survives.
                params[0] = _KEY_NEG_INF + 1
                params[1] = selectable
            else:
                params[0] = (hi << 8) | params[3]
                params[1] = topk - above - hist[params[3] + 1]
        T.sync_threads()

    @T.macro
    def compact(keys, per, tx, scan, params, sel_idx, sel_key):
        """Write the selected indices and keys to sel_idx/sel_key in index order.

        Ties at the threshold keep the lowest indices. Each thread owns
        keys[tx * per : (tx + 1) * per]. Leaves the selected count in
        scan[threads - 1].
        """
        key = T.alloc_var(T.int32)
        n_gt = T.alloc_var(T.int32)
        n_eq = T.alloc_var(T.int32)
        sel_eq = T.alloc_var(T.int32)
        out_pos = T.alloc_var(T.int32)
        taken = T.alloc_var(T.int32)
        thr = T.alloc_var(T.int32)
        need = T.alloc_var(T.int32)
        thr = params[0]
        need = params[1]
        n_gt = 0
        n_eq = 0
        for v in T.serial(per):
            key = keys[tx * per + v]
            if key > thr:
                n_gt += 1
            elif key == thr:
                n_eq += 1
        scan[tx] = n_eq
        T.sync_threads()
        block_scan(scan, tx, steps)
        sel_eq = T.max(T.min(need - (scan[tx] - n_eq), n_eq), 0)
        T.sync_threads()
        scan[tx] = n_gt + sel_eq
        T.sync_threads()
        block_scan(scan, tx, steps)
        out_pos = scan[tx] - n_gt - sel_eq
        taken = 0
        for v in T.serial(per):
            key = keys[tx * per + v]
            if (key > thr) | ((key == thr) & (taken < sel_eq)):
                if key == thr:
                    taken += 1
                sel_idx[out_pos] = tx * per + v
                sel_key[out_pos] = key
                out_pos += 1
        T.sync_threads()

    return threshold, compact


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def indexer_topk_segment_v41(
    width: int,
    topk: int = INDEX_TOPK,
    group: int = 1,
    final: bool = True,
    gather: bool = False,
    threads: int = _TOPK_THREADS,
):
    """Stage 1: the exact top-``topk`` of each ``_SEGMENT_ENTRIES``-entry segment of a row.

    Inputs: ``Scores`` (``[rows, width]`` BF16) and ``Extent`` (``[rows]`` int32
    positions to consider). With ``group > 1`` an entry is a block of ``group``
    positions keyed by its maximum; the block holding position ``Extent - 1``
    always ranks first.

    ``final`` (one segment per row) writes ``Indices`` (``[rows, topk * group]``
    int32 ascending positions, mapped through ``Gather`` when given, ``-1``
    padded) and ``Values`` (``[rows, topk]`` BF16). Otherwise it writes each
    segment's winners, in entry order, to ``WorkValues`` / ``WorkEntries``
    (``[rows, segments * topk]``, ``-1`` padded).
    """
    rows = T.dynamic("rows")
    entries = tilelang.cdiv(width, group)
    segment = _SEGMENT_ENTRIES
    segments = tilelang.cdiv(entries, segment)
    assert not final or segments == 1, "a final segment must cover the row"
    per = segment // threads
    threshold, compact = _select_macros(threads, topk)

    @T.macro
    def select_segment(Scores, Extent, row, seg, keys, hist, scan, params, sel_idx, sel_key):
        tx = T.get_thread_binding()
        best = T.alloc_var(T.int32)
        extent = T.min(T.max(Extent[row], 0), width)
        n_entries = T.ceildiv(extent, group)
        for i in T.serial(per):
            j = i * threads + tx
            e = seg * segment + j
            if group == 1:
                keys[j] = T.if_then_else(e < n_entries, _key16(Scores[row, T.min(e, width - 1)]), 0)
            else:
                best = 0
                for o in T.serial(group):
                    if (e < n_entries) & (e * group + o < extent):
                        best = T.max(best, _key16(Scores[row, T.min(e * group + o, width - 1)]))
                # The block holding the last visible position always survives.
                keys[j] = T.if_then_else((e < n_entries) & (e == (extent - 1) // group), 0xFF80, best)
        T.sync_threads()
        threshold(keys, segment, tx, hist, scan, params)
        compact(keys, per, tx, scan, params, sel_idx, sel_key)

    @T.macro
    def write_final(Extent, Gather, Indices, Values, row, base, total, sel_idx, sel_key):
        tx = T.get_thread_binding()
        extent = T.min(T.max(Extent[row], 0), width)
        for i in T.serial(T.ceildiv(topk, threads)):
            slot = i * threads + tx
            if slot < topk:
                if slot < total:
                    e = base + sel_idx[slot]
                    Values[row, slot] = _value16(sel_key[slot])
                    if group == 1:
                        if gather:
                            Indices[row, slot] = Gather[row, e]
                        else:
                            Indices[row, slot] = e
                    else:
                        for o in T.serial(group):
                            Indices[row, slot * group + o] = T.if_then_else(e * group + o < extent, e * group + o, -1)
                else:
                    Values[row, slot] = -T.infinity(T.bfloat16)
                    for o in T.serial(group):
                        Indices[row, slot * group + o] = -1

    if final:

        @T.macro
        def run_final(Scores, Extent, Gather, Indices, Values):
            with T.Kernel(rows, threads=threads) as row:
                keys = T.alloc_shared([segment], T.int32)
                hist = T.alloc_shared([257], T.int32)
                scan = T.alloc_shared([threads], T.int32)
                params = T.alloc_shared([4], T.int32)
                sel_idx = T.alloc_shared([topk], T.int32)
                sel_key = T.alloc_shared([topk], T.int32)
                select_segment(Scores, Extent, row, 0, keys, hist, scan, params, sel_idx, sel_key)
                write_final(Extent, Gather, Indices, Values, row, 0, scan[threads - 1], sel_idx, sel_key)

        if gather:

            @T.prim_func
            def main(
                Scores: T.Tensor([rows, width], T.bfloat16),
                Extent: T.Tensor([rows], T.int32),
                Gather: T.Tensor([rows, width], T.int32),
                Indices: T.Tensor([rows, topk * group], T.int32),
                Values: T.Tensor([rows, topk], T.bfloat16),
            ):
                run_final(Scores, Extent, Gather, Indices, Values)

            return main

        @T.prim_func
        def main(
            Scores: T.Tensor([rows, width], T.bfloat16),
            Extent: T.Tensor([rows], T.int32),
            Indices: T.Tensor([rows, topk * group], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
        ):
            run_final(Scores, Extent, None, Indices, Values)

        return main

    @T.prim_func
    def main(
        Scores: T.Tensor([rows, width], T.bfloat16),
        Extent: T.Tensor([rows], T.int32),
        WorkValues: T.Tensor([rows, segments * topk], T.bfloat16),
        WorkEntries: T.Tensor([rows, segments * topk], T.int32),
    ):
        with T.Kernel(segments, rows, threads=threads) as (seg, row):
            tx = T.get_thread_binding()
            keys = T.alloc_shared([segment], T.int32)
            hist = T.alloc_shared([257], T.int32)
            scan = T.alloc_shared([threads], T.int32)
            params = T.alloc_shared([4], T.int32)
            sel_idx = T.alloc_shared([topk], T.int32)
            sel_key = T.alloc_shared([topk], T.int32)
            select_segment(Scores, Extent, row, seg, keys, hist, scan, params, sel_idx, sel_key)
            total = scan[threads - 1]
            for i in T.serial(T.ceildiv(topk, threads)):
                slot = i * threads + tx
                if slot < topk:
                    if slot < total:
                        WorkValues[row, seg * topk + slot] = _value16(sel_key[slot])
                        WorkEntries[row, seg * topk + slot] = seg * segment + sel_idx[slot]
                    else:
                        WorkValues[row, seg * topk + slot] = -T.infinity(T.bfloat16)
                        WorkEntries[row, seg * topk + slot] = -1

    return main


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def indexer_topk_merge_v41(
    width: int,
    topk: int = INDEX_TOPK,
    group: int = 1,
    gather: bool = False,
    threads: int = _TOPK_THREADS,
):
    """Stage 2: the exact top-``topk`` of every row's segment winners.

    The winners arrive in ascending entry order, so ties keep the lowest
    positions exactly as a one-pass selection would. Writes ``Indices`` and
    ``Values`` like the final segment kernel.
    """
    rows = T.dynamic("rows")
    segments = tilelang.cdiv(tilelang.cdiv(width, group), _SEGMENT_ENTRIES)
    n = segments * topk
    n_pad = tilelang.cdiv(n, threads) * threads
    per = n_pad // threads
    threshold, compact = _select_macros(threads, topk)

    @T.macro
    def merge(WorkValues, WorkEntries, Extent, Gather, Indices, Values):
        with T.Kernel(rows, threads=threads) as row:
            tx = T.get_thread_binding()
            keys = T.alloc_shared([n_pad], T.int32)
            hist = T.alloc_shared([257], T.int32)
            scan = T.alloc_shared([threads], T.int32)
            params = T.alloc_shared([4], T.int32)
            sel_idx = T.alloc_shared([topk], T.int32)
            sel_key = T.alloc_shared([topk], T.int32)
            extent = T.min(T.max(Extent[row], 0), width)
            for i in T.serial(per):
                j = i * threads + tx
                c = T.min(j, n - 1)
                keys[j] = T.if_then_else((j < n) & (WorkEntries[row, c] >= 0), _key16(WorkValues[row, c]), 0)
            T.sync_threads()
            threshold(keys, n_pad, tx, hist, scan, params)
            compact(keys, per, tx, scan, params, sel_idx, sel_key)
            total = scan[threads - 1]
            for i in T.serial(T.ceildiv(topk, threads)):
                slot = i * threads + tx
                if slot < topk:
                    if slot < total:
                        e = WorkEntries[row, sel_idx[slot]]
                        Values[row, slot] = _value16(sel_key[slot])
                        if group == 1:
                            if gather:
                                Indices[row, slot] = Gather[row, e]
                            else:
                                Indices[row, slot] = e
                        else:
                            for o in T.serial(group):
                                Indices[row, slot * group + o] = T.if_then_else(e * group + o < extent, e * group + o, -1)
                    else:
                        Values[row, slot] = -T.infinity(T.bfloat16)
                        for o in T.serial(group):
                            Indices[row, slot * group + o] = -1

    if gather:

        @T.prim_func
        def main(
            WorkValues: T.Tensor([rows, n], T.bfloat16),
            WorkEntries: T.Tensor([rows, n], T.int32),
            Extent: T.Tensor([rows], T.int32),
            Gather: T.Tensor([rows, width], T.int32),
            Indices: T.Tensor([rows, topk * group], T.int32),
            Values: T.Tensor([rows, topk], T.bfloat16),
        ):
            merge(WorkValues, WorkEntries, Extent, Gather, Indices, Values)

        return main

    @T.prim_func
    def main(
        WorkValues: T.Tensor([rows, n], T.bfloat16),
        WorkEntries: T.Tensor([rows, n], T.int32),
        Extent: T.Tensor([rows], T.int32),
        Indices: T.Tensor([rows, topk * group], T.int32),
        Values: T.Tensor([rows, topk], T.bfloat16),
    ):
        merge(WorkValues, WorkEntries, Extent, None, Indices, Values)

    return main


class IndexerTopK:
    """Exact top-k over BF16 row scores: one pass, or segments plus a merge.

    Both routes return the same positions: the merge sees each segment's exact
    winners in entry order and applies the same score-then-position order.
    """

    def __init__(self, width, topk=INDEX_TOPK, group=1, gather=False, *, max_rows, device="cuda"):
        self.topk, self.group, self.gather = topk, group, gather
        entries = tilelang.cdiv(width, group)
        self.segments = tilelang.cdiv(entries, _SEGMENT_ENTRIES)
        if self.segments == 1:
            self.single = indexer_topk_segment_v41(width, topk, group, True, gather)
        else:
            self.stage1 = indexer_topk_segment_v41(width, topk, group, False)
            self.stage2 = indexer_topk_merge_v41(width, topk, group, gather)
            self.work_values = torch.empty((max_rows, self.segments * topk), dtype=torch.bfloat16, device=device)
            self.work_entries = torch.empty((max_rows, self.segments * topk), dtype=torch.int32, device=device)

    def run(self, scores, extent, indices, values, gather=None):
        rows = scores.shape[0]
        mapped = (gather,) if self.gather else ()
        if self.segments == 1:
            self.single(scores, extent, *mapped, indices, values)
        else:
            work = (self.work_values[:rows], self.work_entries[:rows])
            self.stage1(scores, extent, *work)
            self.stage2(*work, extent, *mapped, indices, values)
        return indices


# ---------------------------------------------------------------------------
# MXFP4 encoding, cache writes and references
# ---------------------------------------------------------------------------


def quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., 128]`` -> packed E2M1 ``[..., 64]`` uint8 and UE8M0 ``[..., 4]`` uint8."""
    groups = x.float().reshape(*x.shape[:-1], INDEX_DIM // SF_BLOCK, SF_BLOCK)
    amax = groups.abs().amax(dim=-1).clamp_min(6.0 * 2.0**-126)
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    normalized = (groups / torch.exp2(exponent).unsqueeze(-1)).reshape(*x.shape)
    lut = torch.tensor(_E2M1_VALUES[:8], device=x.device)
    magnitude = normalized.abs().clamp(max=6.0)
    codes = (magnitude.unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    codes |= torch.signbit(normalized).to(torch.uint8) << 3
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    return packed.contiguous(), (exponent + 127).to(torch.uint8).contiguous()


def decode_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    u = packed.long()
    lut = torch.tensor(_E2M1_VALUES, device=packed.device)
    values = torch.stack([lut[u & 15], lut[u >> 4]], dim=-1).reshape(*packed.shape[:-1], INDEX_DIM)
    scale = torch.exp2(scales.float() - 127.0).repeat_interleave(SF_BLOCK, dim=-1)
    return values * scale


def write_index_cache(pool: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor, slots: torch.Tensor, page_size: int):
    """Write ``[n, 64]`` keys and ``[n, 4]`` scales into ``[pages, page_bytes]`` uint8 pages."""
    page, token = slots.long() // page_size, slots.long() % page_size
    data = pool[:, : page_size * 64].view(-1, page_size, 64)
    scale_rows = pool[:, page_size * 64 :].view(-1, page_size, 4)
    data[page, token] = packed
    scale_rows[page, token] = scales


def ref_index_scores(q_values, weights, k_values, page_table, lengths, active, page_size, width, candidates=None, candidate_lengths=None):
    """FP32 reference of the staged BF16 score over dequantized MXFP4 operands."""
    rows = q_values.shape[0]
    scores = torch.full((rows, width), float("-inf"), dtype=torch.bfloat16, device=q_values.device)
    for r in range(rows):
        if candidates is None:
            extent = min(max(int(lengths[r]), 0), max(int(active[0]), 0), width)
            positions = torch.arange(extent, device=q_values.device)
        else:
            extent = min(max(int(candidate_lengths[r]), 0), width)
            positions = candidates[r, :extent].long()
        valid = positions >= 0
        slots = page_table[r].long()[positions.clamp_min(0) // page_size] * page_size + positions.clamp_min(0) % page_size
        keys = k_values[slots]
        dots = (keys.double() @ q_values[r].double().T).float()  # exact for the test inputs
        dots = dots.bfloat16().float().clamp_min(0)
        contrib = (dots * weights[r].float()).bfloat16().float()
        total = torch.zeros(len(positions), device=q_values.device)
        for h in range(contrib.shape[1]):
            total = total + contrib[:, h]
        row_scores = torch.where(valid, total.bfloat16(), torch.tensor(float("-inf"), dtype=torch.bfloat16, device=q_values.device))
        scores[r, :extent] = row_scores
    return scores


def ref_topk(scores, extent, topk, group=1, gather=None):
    """Exact top-k: larger scores first, then lower positions; returned ascending, -1 padded."""
    rows, width = scores.shape
    out = torch.full((rows, topk * group), -1, dtype=torch.int32, device=scores.device)
    for r in range(rows):
        n = min(max(int(extent[r]), 0), width)
        values = scores[r, :n].float()
        if group > 1:
            blocks = (n + group - 1) // group
            padded = torch.full((blocks * group,), float("-inf"), device=scores.device)
            padded[:n] = values
            values = padded.view(blocks, group).amax(dim=1)
            if blocks:
                values[(n - 1) // group] = float("inf")
        entries = torch.arange(values.shape[0], device=scores.device)
        keep = values > float("-inf")
        values, entries = values[keep], entries[keep]
        order = sorted(range(len(entries)), key=lambda i: (-values[i].item(), entries[i].item()))[:topk]
        chosen = entries[order].sort().values
        if group > 1:
            pos = (chosen[:, None] * group + torch.arange(group, device=scores.device)).reshape(-1)
            chosen = torch.where(pos < n, pos, -1)
        elif gather is not None:
            chosen = gather[r, chosen.long()]
        out[r, : chosen.numel()] = chosen.to(torch.int32)
    return out


# ---------------------------------------------------------------------------
# Plan, test inputs, tests
# ---------------------------------------------------------------------------


class IndexerV41Plan:
    """Score and top-k kernels for one context width, with scratch buffers."""

    def __init__(self, width, page_size, *, max_rows, topk=INDEX_TOPK, candidates=False, device="cuda"):
        self.width, self.page_size, self.topk, self.candidates = width, page_size, topk, candidates
        self.score = indexer_score_v41(width, page_size, candidates)
        self.select = IndexerTopK(width, topk, 1, candidates, max_rows=max_rows, device=device)
        self.scores = torch.empty((max_rows, width), dtype=torch.bfloat16, device=device)
        self.extent = torch.empty((max_rows,), dtype=torch.int32, device=device)
        self.indices = torch.empty((max_rows, topk), dtype=torch.int32, device=device)
        self.values = torch.empty((max_rows, topk), dtype=torch.bfloat16, device=device)

    def run(self, q_packed, q_scales, weights, pool, page_table, lengths, active, candidates=None, candidate_lengths=None):
        rows = q_packed.shape[0]
        scores, extent = self.scores[:rows], self.extent[:rows]
        args = (
            q_packed.view(torch.int8),
            q_scales.view(torch.uint32).reshape(rows, -1),
            weights,
            pool.view(torch.int8),
            pool.view(torch.uint32),
            page_table,
            lengths,
            active,
        )
        if self.candidates:
            self.score(*args, candidates, candidate_lengths, scores)
            torch.clamp(candidate_lengths, 0, self.width, out=extent)
            self.select.run(scores, extent, self.indices[:rows], self.values[:rows], gather=candidates)
        else:
            self.score(*args, scores)
            torch.minimum(lengths.clamp_min(0), active.clamp_min(0), out=extent)
            extent.clamp_(max=self.width)
            self.select.run(scores, extent, self.indices[:rows], self.values[:rows])
        return self.indices[:rows], scores


def make_inputs(rows, context, page_size=64, pool_pages=None, seed=0, device="cuda", exact=True):
    """MXFP4 queries, a paged MXFP4 key cache and per-row page tables.

    With ``exact``, values and scales make every dot product exactly
    representable in FP32, so scores are comparable bit for bit.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    logical_pages = (context + page_size - 1) // page_size
    pool_pages = pool_pages or logical_pages * max(rows, 1) + 3
    if exact:
        lut = torch.tensor(_E2M1_VALUES, device=device)

        def operand(shape):
            values = lut[torch.randint(0, 16, (*shape, INDEX_DIM), generator=gen, device=device)]
            exps = torch.randint(-3, 4, (*shape, INDEX_DIM // SF_BLOCK), generator=gen, device=device)
            return values * torch.exp2(exps.float()).repeat_interleave(SF_BLOCK, dim=-1)

        q = operand((rows, INDEX_HEADS))
        k = operand((pool_pages * page_size,))
    else:
        q = torch.randn((rows, INDEX_HEADS, INDEX_DIM), generator=gen, device=device)
        k = torch.randn((pool_pages * page_size, INDEX_DIM), generator=gen, device=device)
    q_packed, q_scales = quantize_mxfp4(q)
    k_packed, k_scales = quantize_mxfp4(k)
    pool = torch.zeros((pool_pages, page_bytes(page_size)), dtype=torch.uint8, device=device)
    write_index_cache(pool, k_packed, k_scales, torch.arange(pool_pages * page_size, device=device), page_size)
    weights = (torch.randn((rows, INDEX_HEADS), generator=gen, device=device) / 64).bfloat16()
    page_table = torch.stack([torch.randperm(pool_pages, generator=gen, device=device)[:logical_pages] for _ in range(rows)])
    lengths = torch.randint(context // 2, context + 1, (rows,), generator=gen, device=device, dtype=torch.int32)
    active = torch.tensor([context], dtype=torch.int32, device=device)
    return (
        q_packed,
        q_scales,
        weights,
        pool,
        page_table.to(torch.int32),
        lengths,
        active,
        decode_mxfp4(q_packed, q_scales),
        decode_mxfp4(k_packed, k_scales),
    )


def test_indexer_v41(rows=4, context=4096, page_size=64, topk=INDEX_TOPK):
    q_packed, q_scales, weights, pool, page_table, lengths, active, q_values, k_values = make_inputs(rows, context, page_size)
    width = (context + page_size - 1) // page_size * page_size
    plan = IndexerV41Plan(width, page_size, max_rows=rows, topk=topk)
    indices, scores = plan.run(q_packed, q_scales, weights, pool, page_table, lengths, active)
    ref_scores = ref_index_scores(q_values, weights, k_values, page_table, lengths, active, page_size, width)
    assert torch.equal(scores, ref_scores), f"scores differ at {(scores != ref_scores).nonzero()[:4].tolist()}"
    assert torch.equal(indices, ref_topk(scores, plan.extent[:rows], topk)), "top-k selection differs"
    print(f"dense rows={rows} context={context} page={page_size}: scores and top-{topk} exact")


def test_indexer_candidates_v41(rows=3, context=4096, page_size=64, topk=INDEX_TOPK):
    q_packed, q_scales, weights, pool, page_table, lengths, active, q_values, k_values = make_inputs(rows, context, page_size)
    width = CANDIDATE_BLOCKS
    gen = torch.Generator(device="cuda").manual_seed(7)
    candidate_lengths = torch.randint(width // 2, width + 1, (rows,), generator=gen, device="cuda", dtype=torch.int32)
    candidates = torch.stack([torch.randperm(int(lengths.min()), generator=gen, device="cuda")[:width].sort().values for _ in range(rows)])
    candidates = torch.where(torch.arange(width, device="cuda") < candidate_lengths[:, None], candidates, -1).to(torch.int32)
    plan = IndexerV41Plan(width, page_size, max_rows=rows, topk=topk, candidates=True)
    indices, scores = plan.run(q_packed, q_scales, weights, pool, page_table, lengths, active, candidates, candidate_lengths)
    ref_scores = ref_index_scores(q_values, weights, k_values, page_table, lengths, active, page_size, width, candidates, candidate_lengths)
    assert torch.equal(scores, ref_scores), "candidate scores differ"
    assert torch.equal(indices, ref_topk(scores, plan.extent[:rows], topk, gather=candidates)), "candidate top-k differs"
    print(f"candidates rows={rows} width={width}: scores and top-{topk} exact")


def test_topk_ties_and_blocks(rows=3, width=20000, topk=INDEX_TOPK):
    gen = torch.Generator(device="cuda").manual_seed(3)
    # Few distinct values force large ties at the threshold.
    scores = (torch.randint(0, 6, (rows, width), generator=gen, device="cuda").float() * 0.25).bfloat16()
    scores[0, 100:] = float("-inf")  # fewer selectable entries than topk
    extent = torch.tensor([width, width - 17, 1000], dtype=torch.int32, device="cuda")[:rows]
    for w in (width, 4000):  # segmented and single-pass routes
        topk_op = IndexerTopK(w, topk, max_rows=rows)
        indices = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
        values = torch.empty((rows, topk), dtype=torch.bfloat16, device="cuda")
        e = extent.clamp(max=w)
        topk_op.run(scores[:, :w].contiguous(), e, indices, values)
        assert torch.equal(indices, ref_topk(scores[:, :w], e, topk)), f"tie handling differs (width {w}, {topk_op.segments} segments)"
    block_topk = 64
    for w in (width, 4000 * CANDIDATE_BLOCK + 8):
        block_op = IndexerTopK(w, block_topk, CANDIDATE_BLOCK, max_rows=rows)
        blocks = torch.empty((rows, block_topk * CANDIDATE_BLOCK), dtype=torch.int32, device="cuda")
        block_values = torch.empty((rows, block_topk), dtype=torch.bfloat16, device="cuda")
        wide = scores.repeat(1, (w + width - 1) // width)[:, :w].contiguous()
        e = torch.tensor([w, w - 17, 1000], dtype=torch.int32, device="cuda")[:rows]
        block_op.run(wide, e, blocks, block_values)
        assert torch.equal(blocks, ref_topk(wide, e, block_topk, group=CANDIDATE_BLOCK)), f"block selection differs (width {w})"
    print(f"top-k ties, short rows, segmented and {CANDIDATE_BLOCK}-block routes: exact")


def run_regression_perf(rows=8, context=32768, page_size=64):
    q_packed, q_scales, weights, pool, page_table, lengths, active, *_ = make_inputs(rows, context, page_size, exact=False)
    plan = IndexerV41Plan((context + page_size - 1) // page_size * page_size, page_size, max_rows=rows)
    return do_bench(lambda: plan.run(q_packed, q_scales, weights, pool, page_table, lengths, active), backend="cupti")


def benchmark(rows, context, page_size=64, backend="cudagraph"):
    q_packed, q_scales, weights, pool, page_table, lengths, active, *_ = make_inputs(rows, context, page_size, exact=False)
    width = (context + page_size - 1) // page_size * page_size
    plan = IndexerV41Plan(width, page_size, max_rows=rows)
    args = (q_packed, q_scales, weights, pool, page_table, lengths, active)
    ms_all = do_bench(lambda: plan.run(*args), warmup=25, rep=100, backend=backend)
    ms_score = do_bench(lambda: plan.score(*_score_args(plan, *args)), warmup=25, rep=100, backend=backend)
    print(f"rows={rows} context={context}: score {ms_score * 1e3:.1f} us, score + top-{plan.topk} {ms_all * 1e3:.1f} us [{backend}]")


def _score_args(plan, q_packed, q_scales, weights, pool, page_table, lengths, active):
    rows = q_packed.shape[0]
    return (
        q_packed.view(torch.int8),
        q_scales.view(torch.uint32).reshape(rows, -1),
        weights,
        pool.view(torch.int8),
        pool.view(torch.uint32),
        page_table,
        lengths,
        active,
        plan.scores[:rows],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--context", type=int, default=32768, help="compressed positions per row")
    parser.add_argument("--page-size", type=int, default=64)
    args = parser.parse_args()
    test_indexer_v41(4, 4096, args.page_size)
    test_indexer_candidates_v41()
    test_topk_ties_and_blocks()
    benchmark(args.rows, args.context, args.page_size)


if __name__ == "__main__":
    main()
