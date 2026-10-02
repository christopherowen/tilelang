"""DeepSeek-V4.1-Flash block-32 FP8 linear for SM120 / SM121.

V4.1 Flash stores its dense projections as E4M3 weights with one UE8M0 scale
per 32x32 block (``weight_block_size = [32, 32]``, ``scale_fmt = "ue8m0"``) and
quantizes activations dynamically to E4M3 with one UE8M0 scale per 32
elements. Both operands are therefore MXFP8, and the GEMM maps directly onto
SM120's ``mma.sync.m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X``: the
tensor core applies both scales per K32 atom and accumulates in FP32, so no
per-block promotion loop is needed.

Scales reach ``T.mma_gemm_blockscaled`` as uint32 words, each packing the
UE8M0 exponents of four consecutive K32 blocks of one row (byte ``j`` covers
K block ``4 * word + j``). The weight's 32x32 block scales are broadcast to
per-row words once on the host.
"""

import argparse

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

SF_BLOCK = 32  # K elements per UE8M0 scale
SF_WORD_K = 4 * SF_BLOCK  # K elements per packed scale word

# Local (TP3) block-FP8 projections of V4.1 Flash, as (name, N, K).
DS41_TP3_PROJECTIONS = (
    ("fused_wqa_wkv", 1792, 5120),
    ("wq_b", 12288, 1280),
    ("index_wq_b", 4096, 1280),
    ("shared_gate_up", 1536, 5120),
    ("shared_down", 5120, 768),
    ("wo_b", 5120, 3072),
    ("engram_wkv", 8544, 6144),
)


@tilelang.jit(out_idx=[-1])
def mxfp8_gemm(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
):
    """``C[M, N] = (A * SFA) @ (B * SFB)^T`` for MXFP8 ``A[M, K]`` and ``B[N, K]``; M is dynamic."""
    assert K % SF_WORD_K == 0, f"K must be a multiple of {SF_WORD_K}"
    assert block_K % SF_WORD_K == 0 and K % block_K == 0, "block_K must divide K and be a multiple of 128"
    M = T.dynamic("M")
    in_dtype = T.float8_e4m3fn
    accum_dtype = T.float32
    sf_words = K // SF_WORD_K
    block_words = block_K // SF_WORD_K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        SFA: T.Tensor((M, sf_words), T.uint32),
        SFB: T.Tensor((N, sf_words), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_N, block_K), in_dtype)
            SFA_shared = T.alloc_shared((block_M, block_words), T.uint32)
            SFB_shared = T.alloc_shared((block_N, block_words), T.uint32)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(K // block_K, num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                for i, w in T.Parallel(block_M, block_words):
                    SFA_shared[i, w] = SFA[by * block_M + i, ko * block_words + w]
                for j, w in T.Parallel(block_N, block_words):
                    SFB_shared[j, w] = SFB[bx * block_N + j, ko * block_words + w]
                # The staged scale words are tile-local, so this tile's K starts at 0.
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    k_start=0,
                    sf_a_granularity_k=SF_BLOCK,
                    sf_b_granularity_k=SF_BLOCK,
                )
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ---------------------------------------------------------------------------
# MXFP8 quantization, scale packing and reference
# ---------------------------------------------------------------------------


def quantize_mxfp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row, per-32 MXFP8: E4M3 values and UE8M0 exponents ``[rows, K / 32]``."""
    rows, k = x.shape
    groups = x.float().reshape(rows, k // SF_BLOCK, SF_BLOCK)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    values = (groups / torch.exp2(exponent).unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.reshape(rows, k), (exponent + 127).to(torch.uint8)


def quantize_block32_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """V4.1 checkpoint format: E4M3 weights and one UE8M0 exponent per 32x32 block."""
    n, k = w.shape
    blocks = w.float().reshape(n // SF_BLOCK, SF_BLOCK, k // SF_BLOCK, SF_BLOCK)
    amax = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.exp2(exponent)[:, None, :, None]
    values = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return values.reshape(n, k), (exponent + 127).to(torch.uint8)


def pack_scale_words(exponents: torch.Tensor, rows: int | None = None) -> torch.Tensor:
    """Pack ``[rows, K / 32]`` UE8M0 bytes into ``[rows, K / 128]`` uint32 words.

    A ``[rows / 32, K / 32]`` block-scale grid is first broadcast to every row.
    """
    if rows is not None and exponents.shape[0] != rows:
        exponents = exponents.repeat_interleave(rows // exponents.shape[0], dim=0)
    return exponents.contiguous().view(torch.uint32).contiguous()


def dequantize(values: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    rows, k = values.shape
    if exponents.shape[0] != rows:
        exponents = exponents.repeat_interleave(rows // exponents.shape[0], dim=0)
    scale = torch.exp2(exponents.float() - 127.0).repeat_interleave(SF_BLOCK, dim=1)
    return values.float() * scale


def make_inputs(M, N, K, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, K), generator=gen, device=device)
    w = torch.randn((N, K), generator=gen, device=device) * 0.05
    # Make block scales vary so that a misapplied scale is visible.
    x = x * torch.exp2(torch.randint(-3, 4, (M, K // SF_BLOCK), generator=gen, device=device)).repeat_interleave(SF_BLOCK, 1)
    a, sfa = quantize_mxfp8(x)
    b, sfb = quantize_block32_weight(w)
    return a, sfa, b, sfb


def default_config(M: int) -> dict:
    """Tile choice by row count: decode steps need more, smaller N tiles to fill the GPU.

    Tiles stay at least 64 rows tall. On SM121, warp-specialized 32x64 tiles
    with cp.async scale staging intermittently read stale scales.
    """
    if M <= 64:
        return dict(block_M=64, block_N=64, num_stages=3)
    return dict(block_M=128, block_N=128, num_stages=2)


def mxfp8_linear(a, sfa, b, sfb, **config):
    """V4.1 block-FP8 linear: activation scales per (row, K32), weight scales per 32x32."""
    M, K = a.shape
    N = b.shape[0]
    kernel = mxfp8_gemm(N, K, **(config or default_config(M)))
    return kernel(a, b, pack_scale_words(sfa), pack_scale_words(sfb, rows=N))


def test_mxfp8_gemm(M=96, N=1792, K=1280, **config):
    a, sfa, b, sfb = make_inputs(M, N, K)
    out = mxfp8_linear(a, sfa, b, sfb, **config)
    ref = dequantize(a, sfa) @ dequantize(b, sfb).T
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())
    print(f"M={M} N={N} K={K} {config}: OK")


def run_regression_perf(M=128, N=1792, K=5120, **config):
    a, sfa, b, sfb = make_inputs(M, N, K)
    kernel = mxfp8_gemm(N, K, **(config or default_config(M)))
    words_a, words_b = pack_scale_words(sfa), pack_scale_words(sfb, rows=N)
    return do_bench(lambda: kernel(a, b, words_a, words_b), backend="cupti")


def benchmark(M, N, K, **config):
    a, sfa, b, sfb = make_inputs(M, N, K)
    kernel = mxfp8_gemm(N, K, **(config or default_config(M)))
    words_a, words_b = pack_scale_words(sfa), pack_scale_words(sfb, rows=N)
    ms = do_bench(lambda: kernel(a, b, words_a, words_b), warmup=50, rep=200)
    print(f"M={M} N={N} K={K}: {ms * 1e3:.1f} us, {2 * M * N * K / ms * 1e-9:.1f} TFLOP/s")
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--projection", choices=[p[0] for p in DS41_TP3_PROJECTIONS], default=None)
    args = parser.parse_args()
    shapes = [p for p in DS41_TP3_PROJECTIONS if args.projection in (None, p[0])]
    for name, N, K in shapes:
        print(name)
        test_mxfp8_gemm(min(args.m, 256), N, K)
        benchmark(args.m, N, K)


if __name__ == "__main__":
    main()
