"""DeepSeek-V4.1-Flash MoE dispatch and combine for SM120 / SM121.

V4.1 Flash routes every token to 6 of 384 experts (the drafter: 3 of 128).
A router, such as TileKernels' ``moe_topk_gate_forward`` or vLLM's
``dsv4_topk``, writes the experts as ``topk_idx[rows, topk]`` (``-1`` for
padding tokens) and their weights as ``topk_weights[rows, topk]``. Each
(row, slot) pair is a route, numbered ``row * topk + slot``.

``moe_dispatch_v41`` groups the routes by expert in the contiguous layout of
grouped GEMMs (as in DeepGEMM's ``m_grouped_*_contiguous``): each expert's
routes are consecutive, in route order, and padded to a multiple of
``block_M``. It writes:

- ``order[tiles * block_M]``: the route at each row of the layout, ``-1`` for
  padding;
- ``route_pos[rows, topk]``: the layout row of each route, ``-1`` for
  padding tokens;
- ``layout_weights[tiles * block_M]``: the route weight at each row, ``0``
  for padding;
- ``tile_expert[tiles]``: the expert of each ``block_M`` tile, ``-1`` past
  the last used tile, so a grouped GEMM with a fixed grid can stop early;
- ``expert_count[E]``, ``expert_offset[E]``, ``num_tiles[1]``, and
  ``psum[E] = expert_offset + expert_count``, the end of each expert's routes
  as TileKernels' ``psum_num_tokens_per_expert`` expects (with
  ``tile_kernels.config.set_token_alignment(block_M)``).

A route's rank within its expert is its position among that expert's routes
in route order, computed with warp match masks and per-warp counts. The
layout is therefore a pure function of ``topk_idx``.

DeepSeek applies each route's weight before the down projection: TileKernels'
``swiglu_forward_and_per_token_cast`` takes ``layout_weights`` and fuses the
weight, the SwiGLU clamp and the MXFP8 cast of the down projection's input.
``moe_combine_v41`` then sums each token's expert outputs, read from the
layout through ``route_pos``, in slot order in FP32, and optionally adds the
shared expert's output. With ``weighted=True`` it applies the weights itself
with explicit ``T.fma``. Neither kernel uses atomics on
results, so a token's output does not depend on the rest of the batch.
"""

import argparse

import torch

import tilelang
import tilelang.language as T

# Routes handled by one CTA of the dispatch kernels.
_CHUNK = 512


def _dispatch_macros(num_experts, threads):
    warps = threads // 32
    steps = threads.bit_length() - 1
    assert threads >= num_experts and threads & (threads - 1) == 0

    @T.macro
    def rank_routes(TopkIdx, routes, base, topk, tx, warp_count):
        """For the route base + tx: its expert and its rank among this chunk's routes of that expert.

        Leaves the chunk's per-expert counts in warp_count[warps, :].
        """
        expert = T.alloc_var(T.int32)
        rank = T.alloc_var(T.int32)
        peers = T.alloc_var(T.uint32)
        run = T.alloc_var(T.int32)
        held = T.alloc_var(T.int32)
        lane = tx % 32
        warp = tx // 32
        for i in T.serial(T.ceildiv((warps + 1) * num_experts, threads)):
            if i * threads + tx < (warps + 1) * num_experts:
                warp_count[(i * threads + tx) // num_experts, (i * threads + tx) % num_experts] = 0
        T.sync_threads()
        r = T.min(base + tx, routes - 1)
        expert = T.if_then_else(base + tx < routes, T.cast(TopkIdx[r // topk, r % topk], T.int32), -1)
        peers = T.match_any_sync(expert)
        rank = T.cast(T.popcount(peers & (T.shift_left(T.uint32(1), T.cast(lane, T.uint32)) - T.uint32(1))), T.int32)
        if (expert >= 0) & (rank == 0):
            warp_count[warp, expert] = T.cast(T.popcount(peers), T.int32)
        T.sync_threads()
        # Exclusive per-warp bases of each expert, in warp order.
        if tx < num_experts:
            run = 0
            for w in T.serial(warps):
                held = warp_count[w, tx]
                warp_count[w, tx] = run
                run += held
            warp_count[warps, tx] = run
        T.sync_threads()
        if expert >= 0:
            rank += warp_count[warp, expert]
        return expert, rank

    @T.macro
    def layout(count, block_M, tx, scan, ExpertCount, ExpertOffset, Psum, Order, LayoutWeights, TileExpert, NumTiles, tiles):
        """From per-expert route counts: the padded offsets, tile map and padding rows."""
        padded = T.alloc_var(T.int32)
        carry = T.alloc_var(T.int32)
        offset = T.alloc_var(T.int32)
        used_tiles = T.alloc_var(T.int32)
        padded = T.if_then_else(tx < num_experts, T.ceildiv(count, block_M) * block_M, 0)
        # scan may share storage with buffers the caller read last.
        T.sync_threads()
        scan[tx] = padded
        T.sync_threads()
        for s in T.serial(steps):
            carry = T.if_then_else(tx >= (1 << s), scan[T.max(tx - (1 << s), 0)], 0)
            T.sync_threads()
            scan[tx] += carry
            T.sync_threads()
        offset = scan[tx] - padded
        used_tiles = scan[threads - 1] // block_M
        if tx < num_experts:
            ExpertCount[tx] = count
            ExpertOffset[tx] = offset
            Psum[tx] = offset + count
            for i in T.serial(padded - count):
                Order[offset + count + i] = -1
                LayoutWeights[offset + count + i] = 0
            for t in T.serial(padded // block_M):
                TileExpert[offset // block_M + t] = tx
        if tx == 0:
            NumTiles[0] = used_tiles
        for i in T.serial(T.ceildiv(tiles, threads)):
            if (i * threads + tx < tiles) & (i * threads + tx >= used_tiles):
                TileExpert[i * threads + tx] = -1
        return offset

    return rank_routes, layout


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def moe_dispatch_v41(num_experts: int, topk: int, block_M: int, part: str = "fused", idx_dtype=T.int64, threads: int = _CHUNK):
    """Build one dispatch kernel.

    ``part="fused"`` handles up to ``threads`` routes in one CTA. Larger batches
    run ``"count"`` (per-chunk ranks and counts), ``"layout"`` (offsets and the
    tile map) and ``"scatter"`` (positions).
    """
    rows, tiles, chunks = T.dynamic("rows, tiles, chunks")
    warps = threads // 32
    rank_routes, layout = _dispatch_macros(num_experts, threads)

    if part == "fused":

        @T.prim_func
        def main(
            TopkIdx: T.Tensor([rows, topk], idx_dtype),
            TopkWeights: T.Tensor([rows, topk], T.float32),
            Order: T.Tensor([tiles * block_M], T.int32),
            LayoutWeights: T.Tensor([tiles * block_M], T.float32),
            RoutePos: T.Tensor([rows, topk], T.int32),
            TileExpert: T.Tensor([tiles], T.int32),
            ExpertCount: T.Tensor([num_experts], T.int32),
            ExpertOffset: T.Tensor([num_experts], T.int32),
            Psum: T.Tensor([num_experts], T.int32),
            NumTiles: T.Tensor([1], T.int32),
        ):
            with T.Kernel(1, threads=threads):
                warp_count = T.alloc_shared([warps + 1, num_experts], T.int32)
                scan = T.alloc_shared([threads], T.int32)
                count = T.alloc_var(T.int32)
                tx = T.get_thread_binding()
                routes = rows * topk
                expert, rank = rank_routes(TopkIdx, routes, 0, topk, tx, warp_count)
                count = T.if_then_else(tx < num_experts, warp_count[warps, T.min(tx, num_experts - 1)], 0)
                offset = layout(
                    count, block_M, tx, scan, ExpertCount, ExpertOffset, Psum, Order, LayoutWeights, TileExpert, NumTiles, tiles
                )
                T.sync_threads()
                if tx < num_experts:
                    scan[tx] = offset
                T.sync_threads()
                if tx < routes:
                    if expert >= 0:
                        Order[scan[expert] + rank] = tx
                        LayoutWeights[scan[expert] + rank] = TopkWeights[tx // topk, tx % topk]
                        RoutePos[tx // topk, tx % topk] = scan[expert] + rank
                    else:
                        RoutePos[tx // topk, tx % topk] = -1

        return main

    if part == "count":

        @T.prim_func
        def main(
            TopkIdx: T.Tensor([rows, topk], idx_dtype),
            ChunkCount: T.Tensor([chunks, num_experts], T.int32),
            Rank: T.Tensor([rows * topk], T.int32),
        ):
            with T.Kernel(chunks, threads=threads) as c:
                warp_count = T.alloc_shared([warps + 1, num_experts], T.int32)
                tx = T.get_thread_binding()
                expert, rank = rank_routes(TopkIdx, rows * topk, c * threads, topk, tx, warp_count)
                if tx < num_experts:
                    ChunkCount[c, tx] = warp_count[warps, tx]
                if c * threads + tx < rows * topk:
                    Rank[c * threads + tx] = rank

        return main

    if part == "layout":

        @T.prim_func
        def main(
            ChunkCount: T.Tensor([chunks, num_experts], T.int32),
            Order: T.Tensor([tiles * block_M], T.int32),
            LayoutWeights: T.Tensor([tiles * block_M], T.float32),
            TileExpert: T.Tensor([tiles], T.int32),
            ExpertCount: T.Tensor([num_experts], T.int32),
            ExpertOffset: T.Tensor([num_experts], T.int32),
            Psum: T.Tensor([num_experts], T.int32),
            NumTiles: T.Tensor([1], T.int32),
        ):
            with T.Kernel(1, threads=threads):
                scan = T.alloc_shared([threads], T.int32)
                count = T.alloc_var(T.int32)
                tx = T.get_thread_binding()
                # Exclusive per-expert bases of each chunk, in chunk order.
                held = T.alloc_var(T.int32)
                count = 0
                if tx < num_experts:
                    for c in T.serial(chunks):
                        held = ChunkCount[c, tx]
                        ChunkCount[c, tx] = count
                        count += held
                layout(count, block_M, tx, scan, ExpertCount, ExpertOffset, Psum, Order, LayoutWeights, TileExpert, NumTiles, tiles)

        return main

    assert part == "scatter", part

    @T.prim_func
    def main(
        TopkIdx: T.Tensor([rows, topk], idx_dtype),
        TopkWeights: T.Tensor([rows, topk], T.float32),
        ChunkCount: T.Tensor([chunks, num_experts], T.int32),
        Rank: T.Tensor([rows * topk], T.int32),
        ExpertOffset: T.Tensor([num_experts], T.int32),
        Order: T.Tensor([tiles * block_M], T.int32),
        LayoutWeights: T.Tensor([tiles * block_M], T.float32),
        RoutePos: T.Tensor([rows, topk], T.int32),
    ):
        with T.Kernel(chunks, threads=threads) as c:
            tx = T.get_thread_binding()
            expert = T.alloc_var(T.int32)
            r = c * threads + tx
            if r < rows * topk:
                expert = T.cast(TopkIdx[r // topk, r % topk], T.int32)
                if expert >= 0:
                    pos = ExpertOffset[expert] + ChunkCount[c, expert] + Rank[r]
                    Order[pos] = r
                    LayoutWeights[pos] = TopkWeights[r // topk, r % topk]
                    RoutePos[r // topk, r % topk] = pos
                else:
                    RoutePos[r // topk, r % topk] = -1

    return main


@tilelang.jit
def moe_combine_v41(hidden: int, topk: int, shared: bool = False, weighted: bool = False, block_D: int = 1024, threads: int = 128):
    """``Out[t] = sum_k [Weights[t, k] *] Y[RoutePos[t, k]] (+ Shared[t])``, FP32 in slot order, BF16 out."""
    rows, layout_rows = T.dynamic("rows, layout_rows")
    while hidden % block_D:
        block_D //= 2
    assert block_D >= 8, f"hidden={hidden} must be a multiple of 8"

    @T.prim_func
    def main(
        Y: T.Tensor([layout_rows, hidden], T.bfloat16),
        RoutePos: T.Tensor([rows, topk], T.int32),
        Weights: T.Tensor([rows, topk], T.float32),
        Shared: T.Tensor([rows, hidden], T.bfloat16),
        Out: T.Tensor([rows, hidden], T.bfloat16),
    ):
        with T.Kernel(hidden // block_D, rows, threads=threads) as (bd, row):
            acc = T.alloc_fragment([block_D], T.float32)
            T.clear(acc)
            for k in T.serial(topk):
                pos = RoutePos[row, k]
                if pos >= 0:
                    for d in T.Parallel(block_D):
                        if weighted:
                            acc[d] = T.fma(Weights[row, k], T.cast(Y[pos, bd * block_D + d], T.float32), acc[d])
                        else:
                            acc[d] = acc[d] + T.cast(Y[pos, bd * block_D + d], T.float32)
            for d in T.Parallel(block_D):
                if shared:
                    Out[row, bd * block_D + d] = T.cast(acc[d] + T.cast(Shared[row, bd * block_D + d], T.float32), T.bfloat16)
                else:
                    Out[row, bd * block_D + d] = T.cast(acc[d], T.bfloat16)

    return main


class MoeDispatchV41Plan:
    """Dispatch and combine buffers and kernels for one routing shape.

    ``tiles`` is fixed by ``max_rows``: each used tile holds at least one
    route, and the padding adds at most one tile per expert. ``dispatch`` and
    ``combine`` only launch kernels, so they can be captured in a CUDA graph.
    """

    def __init__(self, num_experts, topk, block_M, hidden, *, max_rows, idx_dtype=torch.int64, device="cuda"):
        self.num_experts, self.topk, self.block_M, self.hidden = num_experts, topk, block_M, hidden
        max_routes = max_rows * topk
        self.tiles = min(max_routes, tilelang.cdiv(max_routes, block_M) + num_experts)
        self.max_chunks = tilelang.cdiv(max_routes, _CHUNK)
        tl_idx = {torch.int64: T.int64, torch.int32: T.int32}[idx_dtype]
        self.fused = moe_dispatch_v41(num_experts, topk, block_M, "fused", tl_idx)
        if self.max_chunks > 1:
            self.count = moe_dispatch_v41(num_experts, topk, block_M, "count", tl_idx)
            self.layout = moe_dispatch_v41(num_experts, topk, block_M, "layout", tl_idx)
            self.scatter = moe_dispatch_v41(num_experts, topk, block_M, "scatter", tl_idx)
            self.chunk_count = torch.empty((self.max_chunks, num_experts), dtype=torch.int32, device=device)
            self.rank = torch.empty((max_routes,), dtype=torch.int32, device=device)
        self.order = torch.empty((self.tiles * block_M,), dtype=torch.int32, device=device)
        self.layout_weights = torch.empty((self.tiles * block_M,), dtype=torch.float32, device=device)
        self.route_pos = torch.empty((max_rows, topk), dtype=torch.int32, device=device)
        self.tile_expert = torch.empty((self.tiles,), dtype=torch.int32, device=device)
        self.expert_count = torch.empty((num_experts,), dtype=torch.int32, device=device)
        self.expert_offset = torch.empty((num_experts,), dtype=torch.int32, device=device)
        self.psum = torch.empty((num_experts,), dtype=torch.int32, device=device)
        self.num_tiles = torch.empty((1,), dtype=torch.int32, device=device)
        self.combine_kernels = {(s, w): moe_combine_v41(hidden, topk, s, w) for s in (False, True) for w in (False, True)}
        self.out = torch.empty((max_rows, hidden), dtype=torch.bfloat16, device=device)

    def dispatch(self, topk_idx, topk_weights):
        rows = self.rows = topk_idx.shape[0]
        route_pos = self.route_pos[:rows]
        if rows * self.topk <= _CHUNK:
            self.fused(topk_idx, topk_weights, self.order, self.layout_weights, route_pos, self.tile_expert,
                       self.expert_count, self.expert_offset, self.psum, self.num_tiles)  # fmt: skip
        else:
            chunk_count = self.chunk_count[: tilelang.cdiv(rows * self.topk, _CHUNK)]
            rank = self.rank[: rows * self.topk]
            self.count(topk_idx, chunk_count, rank)
            self.layout(chunk_count, self.order, self.layout_weights, self.tile_expert, self.expert_count, self.expert_offset,
                        self.psum, self.num_tiles)  # fmt: skip
            self.scatter(topk_idx, topk_weights, chunk_count, rank, self.expert_offset, self.order, self.layout_weights, route_pos)
        return dict(order=self.order, layout_weights=self.layout_weights, route_pos=route_pos, tile_expert=self.tile_expert,
                    expert_count=self.expert_count, expert_offset=self.expert_offset, psum=self.psum, num_tiles=self.num_tiles)  # fmt: skip

    def combine(self, y, topk_weights=None, shared=None):
        """Sum the last dispatched tokens' expert outputs; pass ``topk_weights`` only if the experts did not apply them."""
        rows = self.rows
        out = self.out[:rows]
        weights = self.layout_weights[: rows * self.topk].view(rows, self.topk) if topk_weights is None else topk_weights
        kernel = self.combine_kernels[(shared is not None, topk_weights is not None)]
        kernel(y, self.route_pos[:rows], weights, out if shared is None else shared, out)
        return out


# ---------------------------------------------------------------------------
# References and tests
# ---------------------------------------------------------------------------


def ref_dispatch(topk_idx, num_experts, block_M):
    """Stable grouping by expert, padded per expert to block_M."""
    rows, topk = topk_idx.shape
    flat = topk_idx.reshape(-1).long()
    valid = flat >= 0
    counts = torch.bincount(flat[valid], minlength=num_experts)
    padded = (counts + block_M - 1) // block_M * block_M
    offsets = torch.cumsum(padded, 0) - padded
    routes = torch.nonzero(valid).flatten()
    routes = routes[torch.sort(flat[routes], stable=True).indices]
    experts = flat[routes]
    ranks = torch.arange(routes.numel(), device=flat.device) - (torch.cumsum(counts, 0) - counts)[experts]
    pos = offsets[experts] + ranks
    order = torch.full((int(padded.sum()),), -1, dtype=torch.int32, device=flat.device)
    order[pos] = routes.to(torch.int32)
    route_pos = torch.full((rows * topk,), -1, dtype=torch.int32, device=flat.device)
    route_pos[routes] = pos.to(torch.int32)
    tile_expert = torch.repeat_interleave(torch.arange(num_experts, device=flat.device), padded // block_M).to(torch.int32)
    return order, route_pos.reshape(rows, topk), tile_expert, counts.to(torch.int32), offsets.to(torch.int32)


def make_routes(rows, num_experts=384, topk=6, pad_rows=0, skew=0.0, seed=0, idx_dtype=torch.int64, device="cuda"):
    """Distinct experts per row; ``skew`` concentrates routes on a few experts; the last ``pad_rows`` are padding."""
    gen = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn((rows, num_experts), generator=gen, device=device)
    logits[:, : max(1, num_experts // 16)] += skew
    idx = logits.topk(topk, dim=-1).indices
    weights = torch.rand((rows, topk), generator=gen, device=device)
    if pad_rows:
        idx[rows - pad_rows :] = -1
        weights[rows - pad_rows :] = 0
    return idx.to(idx_dtype).contiguous(), weights.contiguous()


def test_moe_dispatch_v41(rows=8, num_experts=384, topk=6, block_M=16, hidden=1024, **kw):
    idx, weights = make_routes(rows, num_experts, topk, **kw)
    plan = MoeDispatchV41Plan(num_experts, topk, block_M, hidden, max_rows=rows, idx_dtype=idx.dtype)
    got = plan.dispatch(idx, weights)
    order, route_pos, tile_expert, counts, offsets = ref_dispatch(idx, num_experts, block_M)
    used = order.numel()
    assert int(got["num_tiles"][0]) * block_M == used
    assert torch.equal(got["order"][:used], order)
    assert torch.equal(got["route_pos"], route_pos)
    assert torch.equal(got["tile_expert"][: used // block_M], tile_expert)
    assert bool((got["tile_expert"][used // block_M :] == -1).all())
    assert torch.equal(got["expert_count"], counts) and torch.equal(got["expert_offset"], offsets)
    assert torch.equal(got["psum"], offsets + counts)
    ref_weights = torch.where(order >= 0, weights.reshape(-1)[order.clamp_min(0).long()], 0)
    assert torch.equal(got["layout_weights"][:used], ref_weights)

    # Combine: expert outputs in layout order, summed in slot order (weighted or not).
    y = torch.randn((plan.tiles * block_M, hidden), device=idx.device).to(torch.bfloat16)
    shared = torch.randn((rows, hidden), device=idx.device).to(torch.bfloat16)
    gathered = y[route_pos.clamp_min(0).long()].float() * (route_pos >= 0)[..., None]
    out = plan.combine(y, weights, shared).clone()
    torch.testing.assert_close(out.float(), (weights[..., None] * gathered).sum(1) + shared.float(), rtol=1e-2, atol=1e-2)
    unweighted = plan.combine(y).clone()
    torch.testing.assert_close(unweighted.float(), gathered.sum(1), rtol=1e-2, atol=1e-2)
    # A token's output does not depend on the rest of the batch.
    solo = MoeDispatchV41Plan(num_experts, topk, block_M, hidden, max_rows=1, idx_dtype=idx.dtype)
    solo.dispatch(idx[:1].contiguous(), weights[:1].contiguous())
    y_solo = torch.zeros((solo.tiles * block_M, hidden), device=idx.device, dtype=torch.bfloat16)
    y_solo[solo.route_pos[0].clamp_min(0).long()] = y[route_pos[0].clamp_min(0).long()]
    assert torch.equal(solo.combine(y_solo, weights[:1], shared[:1]), out[:1])
    assert torch.equal(solo.combine(y_solo), unweighted[:1])
    print(f"rows={rows} experts={num_experts} topk={topk} block_M={block_M} {kw}: OK, {int(got['num_tiles'][0])} tiles")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument("--experts", type=int, default=384)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--block-m", type=int, default=16)
    args = parser.parse_args()
    test_moe_dispatch_v41(args.rows, args.experts, args.topk, args.block_m)


if __name__ == "__main__":
    main()
