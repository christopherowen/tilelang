"""DeepSeek-V4.1-Flash grouped routed-expert GEMMs for SM120 / SM121.

The routed experts of V4.1 Flash are MXFP8 x MXFP4 linears (see
``fp8_fp4_gemm_sm120.py``). ``grouped_fp8_fp4_gemm`` runs one projection for
every expert at once, over the contiguous layout that ``moe_dispatch_v41``
builds: each ``block_M`` tile of the layout belongs to one expert,
``tile_expert[t]`` names it, and the grid covers the plan's fixed tile count
so the kernel can be captured in a CUDA graph. Tiles past the last used
tile exit immediately.

- ``gather=True`` (the gate/up projection) reads each layout row's token
  through ``order``: row ``i`` of tile ``t`` is route ``order[t * block_M + i]``,
  token ``route // topk``. The rows and their scale words are gathered with
  ``cp.async``; padding rows read token 0 and are never consumed.
- ``gather=False`` (the down projection) reads rows already in layout order,
  such as the output of TileKernels' ``swiglu_forward_and_per_token_cast``.

Each output row depends only on its own token and expert, so the result does
not depend on how many other tokens share the tile.

``MoeExpertsV41Plan`` runs a routed-expert layer: TileKernels' router
(``moe_topk_gate_forward``), ``moe_dispatch_v41``, the MXFP8 cast of the
tokens, the gate/up projection, TileKernels' fused SwiGLU (with the
checkpoint's clamp and the route weights) and MXFP8 cast, the down
projection, and ``moe_combine_v41``. TileKernels' packed row-major UE8M0
scales are the scale words these kernels read. Without TileKernels, PyTorch
stands in for the router and the casts.
"""

import argparse

import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

from fp8_fp4_gemm_sm120 import dequantize_mxfp4, quantize_mxfp4
from moe_dispatch_sm120 import MoeDispatchV41Plan, make_routes
from mxfp8_gemm_sm120 import SF_BLOCK, SF_WORD_K, dequantize, pack_scale_words, quantize_mxfp8

# Batches above this many tokens use the shallower prefill pipeline.
PREFILL_ROWS = 256


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
def grouped_fp8_fp4_gemm(
    num_experts: int,
    N: int,
    K: int,
    topk: int = 6,
    gather: bool = True,
    block_M: int = 32,
    block_N: int = 64,
    block_K: int = 128,
    num_stages: int = 4,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
):
    """``C[r] = (A[row(r)] * SFA) @ (B[e] * SFB[e])^T`` for each layout row ``r`` of expert ``e``.

    ``A``: ``[rows, K]`` E4M3 with packed UE8M0 words ``SFA[rows, K / 128]``;
    ``B``: ``[experts, N, K]`` E2M1 (packed bytes) with ``SFB[experts, N, K / 128]``.

    A ``num_stages``-deep pipeline loads the next K blocks while the current
    one multiplies: the expert's weights by TMA, the gathered activation rows
    and all scale words by ``cp.async``.
    """
    assert K % block_K == 0 and block_K % SF_WORD_K == 0, "block_K must divide K and be a multiple of 128"
    tokens, tiles = T.dynamic("tokens, tiles")
    a_rows = tokens if gather else tiles * block_M
    accum_dtype = T.float32
    sf_words = K // SF_WORD_K
    block_words = block_K // SF_WORD_K
    k_blocks = K // block_K
    a_segments = block_M * block_K // 16  # 16-byte cp.async copies of one A block
    a_copies = tilelang.cdiv(a_segments, threads)
    sfa_copies = tilelang.cdiv(block_M * block_words, threads)
    stages = num_stages
    assert stages >= 2

    @T.macro
    def load(A, SFA, B, SFB, A_shared, B_shared, SFA_shared, SFB_shared, a_row, sfa_row, loaded, expert, bx, t, tx, ko, s):
        T.tma_copy(B[expert, bx * block_N, ko * block_K], B_shared[s, :, :], barrier=loaded[s])
        if gather:
            for c in T.unroll(a_copies):
                e = c * threads + tx
                if e < a_segments:
                    T.ptx_cp_async(
                        T.access_ptr(A_shared[s, e // (block_K // 16), e % (block_K // 16) * 16], "w", 16),
                        T.access_ptr(A[a_row[c], ko * block_K + e % (block_K // 16) * 16], "r", 16),
                        16,
                    )
        else:
            T.tma_copy(A[t * block_M : (t + 1) * block_M, ko * block_K : (ko + 1) * block_K], A_shared[s, :, :], barrier=loaded[s])
        for c in T.unroll(sfa_copies):
            e = c * threads + tx
            if e < block_M * block_words:
                T.ptx_cp_async(
                    T.access_ptr(SFA_shared[s, e // block_words, e % block_words], "w", 1),
                    T.access_ptr(SFA[sfa_row[c], ko * block_words + e % block_words], "r", 1),
                    1,
                )
        for c in T.unroll(tilelang.cdiv(block_N * block_words, threads)):
            e = c * threads + tx
            if e < block_N * block_words:
                T.ptx_cp_async(
                    T.access_ptr(SFB_shared[s, e // block_words, e % block_words], "w", 1),
                    T.access_ptr(SFB[expert, bx * block_N + e // block_words, ko * block_words + e % block_words], "r", 1),
                    1,
                )
        T.ptx_commit_group()
        T.barrier_arrive(loaded[s])

    @T.prim_func
    def main(
        A: T.Tensor((a_rows, K), T.float8_e4m3fn),
        SFA: T.Tensor((a_rows, sf_words), T.uint32),
        B: T.Tensor((num_experts, N, K), T.float4_e2m1fn),
        SFB: T.Tensor((num_experts, N, sf_words), T.uint32),
        Order: T.Tensor((tiles * block_M,), T.int32),
        TileExpert: T.Tensor((tiles,), T.int32),
        C: T.Tensor((tiles * block_M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), tiles, threads=threads) as (bx, t):
            A_shared = T.alloc_shared((stages, block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared((stages, block_N, block_K), T.float4_e2m1_unpacked)
            SFA_shared = T.alloc_shared((stages, block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((stages, block_N, block_words), T.uint32)
            rows_shared = T.alloc_shared((block_M,), T.int32)
            a_row = T.alloc_local((a_copies,), T.int32)
            sfa_row = T.alloc_local((sfa_copies,), T.int32)
            loaded = T.alloc_barrier([threads] * stages)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            tx = T.get_thread_binding()

            expert = TileExpert[t]
            if expert < 0:
                T.thread_return()
            for i in T.Parallel(block_M):
                rows_shared[i] = T.max(Order[t * block_M + i], 0) // topk if gather else t * block_M + i
            T.sync_threads()
            # Each thread copies the same rows at every K block.
            for c in T.unroll(a_copies):
                a_row[c] = rows_shared[T.min((c * threads + tx) // (block_K // 16), block_M - 1)]
            for c in T.unroll(sfa_copies):
                sfa_row[c] = rows_shared[T.min((c * threads + tx) // block_words, block_M - 1)]

            T.clear(C_local)
            for p in T.unroll(stages - 1):
                if p < k_blocks:
                    load(A, SFA, B, SFB, A_shared, B_shared, SFA_shared, SFB_shared, a_row, sfa_row, loaded, expert, bx, t, tx, p, p)
                else:
                    T.ptx_commit_group()
            for ko in T.serial(k_blocks):
                # The stage loaded next was last read by iteration ko - 1.
                T.sync_threads()
                nxt = ko + stages - 1
                if nxt < k_blocks:
                    load(
                        A,
                        SFA,
                        B,
                        SFB,
                        A_shared,
                        B_shared,
                        SFA_shared,
                        SFB_shared,
                        a_row,
                        sfa_row,
                        loaded,
                        expert,
                        bx,
                        t,
                        tx,
                        nxt,
                        nxt % stages,
                    )
                else:
                    T.ptx_commit_group()
                T.ptx_wait_group(stages - 1)
                T.barrier_wait(loaded[ko % stages], (ko // stages) % 2)
                T.sync_threads()
                # The staged scale words are block-local, so this block's K starts at 0.
                T.mma_gemm_blockscaled(
                    A_shared[ko % stages, :, :],
                    B_shared[ko % stages, :, :],
                    C_local,
                    SFA_shared[ko % stages, :, :],
                    SFB_shared[ko % stages, :, :],
                    transpose_B=True,
                    k_start=0,
                    sf_a_granularity_k=SF_BLOCK,
                    sf_b_granularity_k=SF_BLOCK,
                )
            T.copy(C_local, C[t * block_M, bx * block_N])

    return main


def _tile_kernels():
    try:
        import tile_kernels

        return tile_kernels
    except ImportError:
        return None


def ref_route(logits, bias, topk, routed_scaling_factor):
    """DeepSeek's routing: sqrt-softplus scores, top-k of score + bias, normalized and scaled."""
    scores = F.softplus(logits.float()).sqrt()
    idx = (scores + bias).topk(topk, dim=-1).indices
    weights = scores.gather(1, idx)
    return idx, weights / (weights.sum(-1, keepdim=True) + 1e-20) * routed_scaling_factor


def ref_swiglu_cast(h, layout_weights, valid, limit):
    """Clamped SwiGLU of ``[gate | up]``, times the route weight, cast to MXFP8 with packed scale words."""
    gate, up = h.float().chunk(2, dim=-1)
    gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
    act = F.silu(gate) * up * (layout_weights * valid)[:, None]
    values, exponents = quantize_mxfp8(act)
    return values, pack_scale_words(exponents)


class MoeExpertsV41Plan:
    """A routed-expert layer over MXFP4 experts: route, dispatch, two grouped GEMMs, combine.

    ``w13`` is ``[experts, 2 * intermediate, hidden]`` E2M1 bytes, gate rows
    first, with packed scale words ``w13_sf``; ``w2`` is
    ``[experts, hidden, intermediate]`` with ``w2_sf``.
    """

    def __init__(self, num_experts, topk, hidden, intermediate, *, max_rows, block_M=32, block_N=64, swiglu_limit=10.0,
                 routed_scaling_factor=1.5, use_tile_kernels=True, device="cuda"):  # fmt: skip
        self.topk, self.swiglu_limit, self.routed_scaling_factor = topk, swiglu_limit, routed_scaling_factor
        self.dispatch = MoeDispatchV41Plan(num_experts, topk, block_M, hidden, max_rows=max_rows, device=device)
        rows = self.dispatch.tiles * block_M
        # Decode streams expert weights and wants a deep pipeline; prefill has many more tiles in flight.
        self.depths = sorted({4, 2} if max_rows > PREFILL_ROWS else {4})
        self.gate_up = {
            d: grouped_fp8_fp4_gemm(num_experts, 2 * intermediate, hidden, topk, True, block_M, block_N, num_stages=d) for d in self.depths
        }
        self.down = {
            d: grouped_fp8_fp4_gemm(num_experts, hidden, intermediate, topk, False, block_M, block_N, num_stages=d) for d in self.depths
        }
        self.h = torch.empty((rows, 2 * intermediate), dtype=torch.bfloat16, device=device)
        self.y = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
        self.tk = _tile_kernels() if use_tile_kernels else None
        if self.tk is not None:
            self.tk.config.set_token_alignment(block_M)

    def route(self, logits, bias):
        if self.tk is not None:
            return self.tk.moe.moe_topk_gate_forward(logits, self.topk, False, 0, self.routed_scaling_factor, 0, bias=bias)
        idx, weights = ref_route(logits, bias, self.topk, self.routed_scaling_factor)
        return idx.contiguous(), weights.contiguous()

    def run(self, x, logits, bias, w13, w13_sf, w2, w2_sf, shared=None):
        idx, weights = self.route(logits, bias)
        layout = self.dispatch.dispatch(idx, weights)
        depth = 4 if x.shape[0] <= PREFILL_ROWS else 2
        if self.tk is not None:
            xq, x_sf = self.tk.quant.per_token_cast(x, "e4m3", SF_BLOCK, round_sf=True, use_packed_ue8m0=True)
        else:
            xq, x_exp = quantize_mxfp8(x)
            x_sf = pack_scale_words(x_exp)
        self.gate_up[depth](xq, x_sf.view(torch.uint32), w13, w13_sf, layout["order"], layout["tile_expert"], self.h)
        if self.tk is not None:
            hq, h_sf = self.tk.quant.swiglu_forward_and_per_token_cast(
                self.h, "e4m3", SF_BLOCK, layout["psum"], layout["layout_weights"],
                clamp_value=self.swiglu_limit, round_sf=True, use_packed_ue8m0=True,
            )  # fmt: skip
        else:
            hq, h_sf = ref_swiglu_cast(self.h, layout["layout_weights"], layout["order"] >= 0, self.swiglu_limit)
        self.down[depth](hq, h_sf.view(torch.uint32), w2, w2_sf, layout["order"], layout["tile_expert"], self.y)
        return self.dispatch.combine(self.y, shared=shared)


# ---------------------------------------------------------------------------
# Inputs, references and tests
# ---------------------------------------------------------------------------


def make_expert_weights(num_experts, N, K, seed=1, device="cuda"):
    """MXFP4 weights ``[E, N, K / 2]`` (packed) and packed UE8M0 words ``[E, N, K / 128]``."""
    gen = torch.Generator(device=device).manual_seed(seed)
    packed, words, exps = [], [], []
    for _ in range(num_experts):
        b, sfb = quantize_mxfp4(torch.randn((N, K), generator=gen, device=device) * 0.02)
        packed.append(b)
        exps.append(sfb)
        words.append(pack_scale_words(sfb))
    return torch.stack(packed), torch.stack(words), torch.stack(exps)


def ref_grouped(a, sfa, b_exps, b_packed, layout_rows, layout_expert):
    """FP32 reference for each valid layout row: dequant(A[row]) @ dequant(B[expert])^T."""
    deq_a = dequantize(a, sfa)
    out = {}
    for e in layout_expert.unique().tolist():
        rows = (layout_expert == e).nonzero().flatten()
        w = dequantize_mxfp4(b_packed[e], b_exps[e])
        out[e] = (rows, deq_a[layout_rows[rows]] @ w.T)
    return out


def test_grouped_fp8_fp4_gemm(rows=16, num_experts=16, topk=6, N=256, K=512, block_M=32, **config):
    device = "cuda"
    idx, weights = make_routes(rows, num_experts, topk)
    plan = MoeDispatchV41Plan(num_experts, topk, block_M, 1024, max_rows=rows)
    layout = plan.dispatch(idx, weights)
    gen = torch.Generator(device=device).manual_seed(0)
    a, sfa = quantize_mxfp8(torch.randn((rows, K), generator=gen, device=device))
    b_packed, b_words, b_exps = make_expert_weights(num_experts, N, K)
    kernel = grouped_fp8_fp4_gemm(num_experts, N, K, topk, True, block_M, **config)
    c = torch.empty((plan.tiles * block_M, N), dtype=torch.bfloat16, device=device)
    kernel(a, pack_scale_words(sfa), b_packed.view(torch.int8), b_words, layout["order"], layout["tile_expert"], c)

    order = layout["order"][: int(layout["num_tiles"][0]) * block_M]
    valid = order >= 0
    layout_rows = (order.clamp_min(0) // topk).long()
    layout_expert = layout["tile_expert"][: order.numel() // block_M].repeat_interleave(block_M)
    layout_expert = torch.where(valid, layout_expert, -1)
    for e, (r, ref) in ref_grouped(a, sfa, b_exps, b_packed, layout_rows, layout_expert).items():
        if e < 0:
            continue
        got = c[r].float()
        torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())

    # gather=False reads the same rows already in layout order.
    a_layout = a[layout_rows]
    sfa_layout = pack_scale_words(sfa[layout_rows])
    pad = plan.tiles * block_M - a_layout.shape[0]
    a_layout = torch.cat([a_layout, a_layout.new_zeros((pad, K))])
    sfa_layout = torch.cat([sfa_layout, sfa_layout.new_zeros((pad, sfa_layout.shape[1]))])
    kernel2 = grouped_fp8_fp4_gemm(num_experts, N, K, topk, False, block_M, **config)
    c2 = torch.empty_like(c)
    kernel2(a_layout, sfa_layout, b_packed.view(torch.int8), b_words, layout["order"], layout["tile_expert"], c2)
    assert torch.equal(c2[: order.numel()][valid], c[: order.numel()][valid]), "gather and layout-order inputs differ"
    print(f"rows={rows} experts={num_experts} topk={topk} N={N} K={K} block_M={block_M} {config}: OK")


def test_moe_experts_v41(rows=12, num_experts=32, topk=6, hidden=512, intermediate=256, block_M=32, use_tile_kernels=True):
    """End to end against an FP32 reference that quantizes at the same points."""
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(0)
    x = torch.randn((rows, hidden), generator=gen, device=device).to(torch.bfloat16)
    logits = torch.randn((rows, num_experts), generator=gen, device=device)
    bias = torch.randn((num_experts,), generator=gen, device=device) * 0.1
    w13, w13_sf, w13_exp = make_expert_weights(num_experts, 2 * intermediate, hidden, seed=2)
    w2, w2_sf, w2_exp = make_expert_weights(num_experts, hidden, intermediate, seed=3)
    plan = MoeExpertsV41Plan(num_experts, topk, hidden, intermediate, max_rows=rows, block_M=block_M, use_tile_kernels=use_tile_kernels)
    out = plan.run(x, logits, bias, w13.view(torch.int8), w13_sf, w2.view(torch.int8), w2_sf).clone()

    idx, weights = plan.route(logits, bias)
    xq, x_exp = quantize_mxfp8(x)
    deq_x = dequantize(xq, x_exp)
    ref = torch.zeros((rows, hidden), device=device)
    for t in range(rows):
        for k in range(topk):
            e = int(idx[t, k])
            h = (deq_x[t : t + 1] @ dequantize_mxfp4(w13[e], w13_exp[e]).T).to(torch.bfloat16)
            hq, h_sf = ref_swiglu_cast(h, weights[t, k : k + 1], torch.ones(1, device=device), plan.swiglu_limit)
            h_exp = h_sf.view(torch.uint8).reshape(1, -1)
            y = (dequantize(hq, h_exp) @ dequantize_mxfp4(w2[e], w2_exp[e]).T).to(torch.bfloat16)
            ref[t] += y[0].float()
    torch.testing.assert_close(out.float(), ref, rtol=3e-2, atol=3e-2 * ref.abs().max().item())
    print(f"rows={rows} experts={num_experts} topk={topk} hidden={hidden} intermediate={intermediate} "
          f"tile_kernels={plan.tk is not None}: OK")  # fmt: skip


def benchmark(rows, num_experts=384, topk=6, N=1536, K=5120, block_M=32, backend="cudagraph", **config):
    device = "cuda"
    idx, weights = make_routes(rows, num_experts, topk)
    plan = MoeDispatchV41Plan(num_experts, topk, block_M, 1024, max_rows=rows)
    layout = plan.dispatch(idx, weights)
    a, sfa = quantize_mxfp8(torch.randn((rows, K), device=device))
    b_packed = torch.randint(-128, 127, (num_experts, N, K // 2), dtype=torch.int8, device=device)
    b_words = torch.randint(0x78787878, 0x7A7A7A7A, (num_experts, N, K // SF_WORD_K), dtype=torch.int32, device=device).view(torch.uint32)
    kernel = grouped_fp8_fp4_gemm(num_experts, N, K, topk, True, block_M, **config)
    c = torch.empty((plan.tiles * block_M, N), dtype=torch.bfloat16, device=device)
    words_a = pack_scale_words(sfa)
    ms = do_bench(lambda: kernel(a, words_a, b_packed, b_words, layout["order"], layout["tile_expert"], c), backend=backend)
    used = int((layout["expert_count"] > 0).sum())
    weight_bytes = used * N * K // 2
    print(f"rows={rows} experts used={used}/{num_experts} N={N} K={K} block_M={block_M} [{backend}]: {ms * 1e3:.1f} us, "
          f"{weight_bytes / ms * 1e-6:.0f} GB/s of expert weights")  # fmt: skip
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--block-m", type=int, default=32)
    args = parser.parse_args()
    test_grouped_fp8_fp4_gemm(rows=args.rows, block_M=args.block_m)
    benchmark(args.rows, block_M=args.block_m)


if __name__ == "__main__":
    main()
