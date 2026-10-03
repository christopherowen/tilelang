import tilelang.testing

import fp8_fp4_gemm_sm120
import indexer_sm120
import moe_dispatch_sm120
import mxfp8_gemm_sm120
import sparse_mla_fwd_sm120


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_sparse_mla_fwd_tp3():
    # 24 local heads at TP3, padded into one 32-head block; every plan kernel gives the same bits.
    sparse_mla_fwd_sm120.test_sparse_mla_fwd_v41(rows=8, heads=24)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_sparse_mla_fwd_segments():
    # One chunk per segment, and segments that do not divide the 20 chunks.
    sparse_mla_fwd_sm120.test_sparse_mla_fwd_v41(rows=5, heads=24, segment_chunks=1)
    sparse_mla_fwd_sm120.test_sparse_mla_fwd_v41(rows=5, heads=24, segment_chunks=3)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_sparse_mla_fwd_small_heads_block():
    sparse_mla_fwd_sm120.test_sparse_mla_fwd_v41(rows=3, heads=24, block_H=16, threads=128)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_sparse_mla_fwd_tp1():
    sparse_mla_fwd_sm120.test_sparse_mla_fwd_v41(rows=4, heads=64, block_H=64, block_I=16)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_mxfp8_gemm():
    mxfp8_gemm_sm120.test_mxfp8_gemm(M=96, N=1792, K=1280)
    mxfp8_gemm_sm120.test_mxfp8_gemm(M=6, N=8544, K=6144)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_fp8_fp4_gemm():
    fp8_fp4_gemm_sm120.test_fp8_fp4_gemm(M=96, N=1536, K=1024)
    fp8_fp4_gemm_sm120.test_fp8_fp4_gemm(M=6, N=5120, K=768)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_indexer():
    # Bit-exact scores and selections; 40000 positions take the segmented top-k.
    indexer_sm120.test_indexer_v41(rows=4, context=4096, page_size=64)
    indexer_sm120.test_indexer_v41(rows=2, context=40000, page_size=128)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_indexer_candidates():
    indexer_sm120.test_indexer_candidates_v41()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_indexer_topk_ties_and_blocks():
    indexer_sm120.test_topk_ties_and_blocks()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
def test_example_moe_dispatch():
    # One CTA (48 rows), chunked (300 rows), padding tokens, a skewed load, and the drafter's 3 of 128.
    moe_dispatch_sm120.test_moe_dispatch_v41(rows=48)
    moe_dispatch_sm120.test_moe_dispatch_v41(rows=300, pad_rows=17, skew=4.0)
    moe_dispatch_sm120.test_moe_dispatch_v41(rows=40, num_experts=128, topk=3)


if __name__ == "__main__":
    tilelang.testing.main()
