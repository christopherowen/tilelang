# DeepSeek-V4.1-Flash on SM120 / SM121

TileLang kernels for the DeepSeek-V4.1-Flash operators that need SM120-family
features: sparse MLA over the model's quantized KV caches, block-32 FP8 linear
layers, and FP4 routed experts. They target the SM120 family: consumer and
workstation Blackwell (SM120) and GB10 / DGX Spark (SM121). This family has
warp-level `mma.sync`, block-scaled MMA and TMA, no WGMMA or tensor memory, and
at most 99 KiB of shared memory per block.

```text
deepseek_v41/
├── README.md
├── sparse_mla_fwd_sm120.py      # sparse MLA over the SWA (MXFP8) and indexed (NVFP4) caches
├── indexer_sm120.py             # indexer scores over the MXFP4 index cache, exact top-k
├── mxfp8_gemm_sm120.py          # block-32 FP8 linear: MXFP8 x MXFP8
├── fp8_fp4_gemm_sm120.py        # routed-expert linear: MXFP8 x MXFP4
├── test_tilelang_example_deepseek_v41.py
└── regression_tilelang_example_deepseek_v41.py
```

## Operator map

| V4.1 Flash operator | Kernel | SM120 instruction path |
|---|---|---|
| Sparse MLA, SWA and indexed caches, attention sink | `sparse_mla_fwd_sm120.py` | in-kernel dequant to BF16, `mma.sync.m16n8k16` |
| Indexer scores, MXFP4 queries and paged MXFP4 keys | `indexer_sm120.py` | `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::2X` |
| Top-512 selection, candidate-domain blocks | `indexer_sm120.py` | radix select in shared memory |
| Block-32 FP8 projections (query, KV, output, shared expert, Engram) | `mxfp8_gemm_sm120.py` | `mma.sync.m16n8k32.kind::mxf8f6f4.block_scale` E4M3 x E4M3 |
| Routed experts, one expert's projection | `fp8_fp4_gemm_sm120.py` | `mma.sync.m16n8k32.kind::mxf8f6f4.block_scale` E4M3 x E2M1 |
| Manifold hyper-connections (4 streams) | `../deepseek_mhc` | architecture-neutral; use `hc_mult=4`, hidden 5120, 20 Sinkhorn iterations |
| Activation quantization | `../deepseek_v4/act_quant.py` | architecture-neutral |

Routing and the grouped expert launch are not included.

## Sparse MLA

Each attention row reads two streams of 512-wide MQA latents (K and V are the
same latent):

| Cache | Record | Values | Scales |
|---|---|---|---|
| Sliding window (SWA) | 528 bytes | 512 x E4M3 | 16 x UE8M0, one per 32 coordinates |
| Indexed (top-k selected compressed entries) | 288 bytes | 256 bytes of packed E2M1, low nibble first | 32 x E4M3, one per 16 coordinates |

All 512 coordinates are quantized, RoPE included. The kernel gathers records
by slot index, dequantizes them into shared memory as BF16 (exactly, for both
formats), and multiplies on `mma.sync`. An index of `-1`, or a position past
the row's length, is masked.

### One contract, two schedules

The arithmetic is defined per chunk of `block_I` entries of the concatenated
[SWA | indexed] list. Each chunk yields its maximum score `m_c`, the sum
`l_c` of `exp2(s - m_c)`, and the unnormalized `O_c = P V`. The chunks then
fold in order into a state that starts from the head's sink, using explicit
`T.fma`/`T.fmul`, and the output is `O * (1 / L)`.

| Schedule | Grid | Suits |
|---|---|---|
| `extend` | one CTA per row and head block; chunks fold in registers | many rows |
| `decode` | one CTA per chunk, head block and row writes `(O_c, m_c, l_c)`; a merge kernel folds them | few rows: up to `chunks x head_blocks` CTAs per row |

Both schedules run the same chunk code and the same fold, so they produce
identical bits. The tests check that decode equals extend, and that a row
gives the same bits alone and in a batch. Choosing a schedule by batch size
therefore never changes results.

`SparseMlaV41Plan` compiles both schedules once, owns the decode workspace and
the output, and selects `decode` for batches up to `decode_max_rows`.
`run(..., schedule=...)` forces either schedule. `run` only launches kernels,
so it can be captured in a CUDA graph.

```python
plan = SparseMlaV41Plan(heads=24, swa_width=128, idx_width=512, idx_page_size=64, max_rows=48)
out = plan.run(q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_indices, idx_lengths, idx_page_table, sinks)
```

The default tile is 32 heads by 32 entries with 256 threads, about 70 KiB of
shared memory. At TP3 a rank owns 24 heads, which fit one 32-head block.
`block_H=16, threads=128` doubles the decode CTAs per row. `block_H=64,
block_I=16` serves the 64 heads of TP1 in one block (a 64x32 tile exceeds
99 KiB); two 32-head blocks also work.

## Indexer and top-k

The indexer has 32 heads of width 128. Its keys live in a paged MXFP4
cache: each page holds `page_size` rows of 64 bytes of packed E2M1, then
`page_size` rows of 4 UE8M0 scales. Queries are MXFP4 in the same format.
The head weights are BF16, already multiplied by 1/64. `indexer_score_v41`
feeds both packed operands directly to the MXFP4 block-scaled MMA, then
reproduces the published rounding stages:

```text
dot   = bf16(q_h . k_j)
c_h   = bf16(max(dot, 0) * w_h)
score = bf16(c_0 + c_1 + ... + c_31)     # FP32 sum, increasing head order
```

It scores every position of a row's context, or (`candidates=True`) only
the positions in a candidate list, as V4.1's reindex layers do.

`IndexerTopK` selects exactly. Each 4096-entry segment caches 16-bit
order-preserving keys in shared memory, runs a two-pass radix select, and
compacts its winners in position order, so ties keep the lowest positions.
Long rows merge the segment winners with the same selection. Positions come
back ascending and padded with `-1`. With `group=8`, it ranks blocks of eight
positions by their maximum (the block holding the last visible position
always survives) and expands them. This is the candidate-domain selection at
V4.1's candidate source layer.

## Block-scaled GEMMs

V4.1 Flash stores dense weights as E4M3 with one UE8M0 scale per 32x32 block,
quantizes activations to MXFP8 (one UE8M0 scale per 32 elements), and stores
routed experts as MXFP4. `T.mma_gemm_blockscaled` lowers both GEMMs to SM120's
MX block-scaled MMA, which applies both scales per K32 atom inside the tensor
core:

- scales are uint32 words packing four consecutive K32 scales of one row;
  `pack_scale_words` builds them, broadcasting the weight block scales to
  rows;
- E2M1 weights stay packed in global memory. A TMA copy into
  `T.float4_e2m1_unpacked` shared memory writes the padded layout that
  `ldmatrix .b8x16.b4x16_p64` expands into 8-bit MMA containers.

Use K tiles that are multiples of 128, so that the scale byte selectors
become compile-time constants.

## Measured on DGX Spark (GB10, SM121)

CUDA 13.0.2, `sm_121a`. Times are per call, from CUDA-graph replay of ten
back-to-back calls, which is how decode runs in serving. Eager calls add about
20 us of host launch overhead each.

| Sparse MLA, 24 heads, SWA 128 + indexed 512 | `decode` | `extend` |
|---|---:|---:|
| 1 row | 7.6 us | 66 us |
| 2 rows | 8.4 us | 73 us |
| 8 rows | 30 us | 72 us |
| 16 rows | 99 us | 76 us |
| 48 rows | 478 us | 75 us |
| 64 rows | 631 us | 138 us |

Both schedules give identical bits. The plan's default crossover is 8 rows.
The max error against the FP32 reference is 0.002-0.008.

| Indexer, score + top-512, page 64 | Time |
|---|---:|
| 1 row, 4096 positions | 10.7 us |
| 1 row, 32768 positions | 21.1 us |
| 8 rows, 32768 positions | 48.9 us |
| 48 rows, 32768 positions | 365 us |
| 1 row, 131072 positions | 37.3 us |

| GEMM, M = 4096 | TFLOP/s |
|---|---:|
| MXFP8, query/KV/shared/output projections | 101-140 |
| MXFP8, Engram 8544x6144 | 53 |
| MXFP8 x MXFP4, routed-expert gate/up and down | 104-144 |

For decode-sized M (6 to 48 rows), each GEMM projection takes 31-315 us.

## Running

```bash
python sparse_mla_fwd_sm120.py --rows 8 --heads 24
python indexer_sm120.py --rows 8 --context 32768
python mxfp8_gemm_sm120.py --projection wq_b
python fp8_fp4_gemm_sm120.py
python -m pytest test_tilelang_example_deepseek_v41.py
```

SM121 (`sm_121a`) needs CUDA 12.9 or newer.
