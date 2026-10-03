import tilelang.testing

import fp8_fp4_gemm_sm120
import indexer_sm120
import mxfp8_gemm_sm120
import sparse_mla_fwd_sm120


def regression_sparse_mla_fwd_decode():
    tilelang.testing.process_func(sparse_mla_fwd_sm120.run_regression_perf, rows=8, heads=24, kernel="decode")


def regression_sparse_mla_fwd_extend():
    tilelang.testing.process_func(sparse_mla_fwd_sm120.run_regression_perf, rows=512, heads=24, kernel="extend")


def regression_mxfp8_gemm():
    tilelang.testing.process_func(mxfp8_gemm_sm120.run_regression_perf, M=4096, N=1792, K=5120)


def regression_fp8_fp4_gemm():
    tilelang.testing.process_func(fp8_fp4_gemm_sm120.run_regression_perf, M=1024, N=1536, K=5120)


def regression_indexer_decode():
    tilelang.testing.process_func(indexer_sm120.run_regression_perf, rows=8, context=32768)


if __name__ == "__main__":
    tilelang.testing.regression()
