"""DeepSeek-V4.1-Flash routed-expert FP8 x FP4 GEMM for SM120 / SM121.

V4.1 Flash ships its routed experts as native FP4: E2M1 values with one
UE8M0 scale per 32 K elements (MXFP4). Activations are quantized to MXFP8. One
expert projection is therefore an MXFP8 x MXFP4 GEMM, which SM120 runs as
``mma.sync.m16n8k32.kind::mxf8f6f4.block_scale.scale_vec::1X.e4m3.e2m1``.

The weights stay packed in global memory (two E2M1 values per byte). A TMA
copy into ``T.float4_e2m1_unpacked`` shared memory writes the padded layout
(16 values in 8 bytes, then 8 bytes of padding) that ``ldmatrix
.b8x16.b4x16_p64`` expands into the 8-bit MMA containers.

This is the SM120 counterpart of ``examples/deepseek_v4/fp8_fp4_gemm_1d1d_sm100.py``.
"""

import argparse

import torch

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench

from mxfp8_gemm_sm120 import SF_BLOCK, SF_WORD_K, default_config, dequantize, pack_scale_words, quantize_mxfp8

# Local (TP3) routed-expert projections of V4.1 Flash, as (name, N, K).
DS41_TP3_EXPERT_PROJECTIONS = (
    ("w13_gate_up", 2 * 768, 5120),
    ("w2_down", 5120, 768),
)

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


@tilelang.jit(out_idx=[-1])
def fp8_fp4_gemm(
    N: int,
    K: int,
    block_M: int = 128,
    block_N: int = 128,
    block_K: int = 128,
    num_stages: int = 2,
    threads: int = 128,
    out_dtype: T.dtype = T.bfloat16,
):
    """``C[M, N] = (A * SFA) @ (B * SFB)^T`` for MXFP8 ``A[M, K]`` and MXFP4 ``B[N, K]``; M is dynamic."""
    assert K % block_K == 0 and block_K % SF_WORD_K == 0, "block_K must divide K and be a multiple of 128"
    M = T.dynamic("M")
    accum_dtype = T.float32
    sf_words = K // SF_WORD_K
    block_words = block_K // SF_WORD_K

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float8_e4m3fn),
        B: T.Tensor((N, K), T.float4_e2m1fn),
        SFA: T.Tensor((M, sf_words), T.uint32),
        SFB: T.Tensor((N, sf_words), T.uint32),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1_unpacked)
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


def quantize_mxfp4(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 weights: packed E2M1 bytes ``[N, K / 2]`` and UE8M0 exponents ``[N, K / 32]``."""
    n, k = w.shape
    groups = w.float().reshape(n, k // SF_BLOCK, SF_BLOCK)
    amax = groups.abs().amax(dim=-1).clamp_min(1e-4)
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    normalized = (groups / torch.exp2(exponent).unsqueeze(-1)).reshape(n, k)
    lut = torch.tensor(_E2M1_VALUES, device=w.device)
    codes = (normalized.unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.view(torch.int8).contiguous(), (exponent + 127).to(torch.uint8)


def dequantize_mxfp4(packed: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    u = packed.view(torch.uint8).long()
    lut = torch.tensor(_E2M1_VALUES, device=packed.device)
    values = torch.stack([lut[u & 15], lut[u >> 4]], dim=-1).reshape(u.shape[0], -1)
    return values * torch.exp2(exponents.float() - 127.0).repeat_interleave(SF_BLOCK, dim=1)


def make_inputs(M, N, K, seed=0, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((M, K), generator=gen, device=device)
    w = torch.randn((N, K), generator=gen, device=device) * 0.02
    a, sfa = quantize_mxfp8(x)
    b, sfb = quantize_mxfp4(w)
    return a, sfa, b, sfb


def expert_linear(a, sfa, b, sfb, **config):
    M, K = a.shape
    N = b.shape[0]
    kernel = fp8_fp4_gemm(N, K, **(config or default_config(M)))
    return kernel(a, b, pack_scale_words(sfa), pack_scale_words(sfb))


def test_fp8_fp4_gemm(M=96, N=1536, K=1024, **config):
    a, sfa, b, sfb = make_inputs(M, N, K)
    out = expert_linear(a, sfa, b, sfb, **config)
    ref = dequantize(a, sfa) @ dequantize_mxfp4(b, sfb).T
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2 * ref.abs().max().item())
    print(f"M={M} N={N} K={K} {config}: OK")


def run_regression_perf(M=128, N=1536, K=5120, **config):
    a, sfa, b, sfb = make_inputs(M, N, K)
    kernel = fp8_fp4_gemm(N, K, **(config or default_config(M)))
    words_a, words_b = pack_scale_words(sfa), pack_scale_words(sfb)
    return do_bench(lambda: kernel(a, b, words_a, words_b), backend="cupti")


def benchmark(M, N, K, **config):
    a, sfa, b, sfb = make_inputs(M, N, K)
    kernel = fp8_fp4_gemm(N, K, **(config or default_config(M)))
    words_a, words_b = pack_scale_words(sfa), pack_scale_words(sfb)
    ms = do_bench(lambda: kernel(a, b, words_a, words_b), warmup=50, rep=200)
    print(f"M={M} N={N} K={K}: {ms * 1e3:.1f} us, {2 * M * N * K / ms * 1e-9:.1f} TFLOP/s")
    return ms


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--m", type=int, default=1024, help="routed rows for one expert")
    args = parser.parse_args()
    for name, N, K in DS41_TP3_EXPERT_PROJECTIONS:
        print(name)
        test_fp8_fp4_gemm(min(args.m, 256), N, K)
        benchmark(args.m, N, K)


if __name__ == "__main__":
    main()
