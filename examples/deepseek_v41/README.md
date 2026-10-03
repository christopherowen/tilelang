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
├── moe_dispatch_sm120.py        # routed-expert dispatch to the grouped layout, and combine
├── moe_expert_gemm_sm120.py     # grouped routed-expert GEMMs and a whole routed-expert layer
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
| Routed experts, dispatch and combine | `moe_dispatch_sm120.py` | warp match masks, ordered weighted sums |
| Routed experts, all experts at once | `moe_expert_gemm_sm120.py` | TMA weights, `cp.async` row gather, `kind::mxf8f6f4` E4M3 x E2M1 |
| Manifold hyper-connections (4 streams) | `../deepseek_mhc` | architecture-neutral; use `hc_mult=4`, hidden 5120, 20 Sinkhorn iterations |
| Activation quantization | `../deepseek_v4/act_quant.py` | architecture-neutral |

Expert selection (sqrt-softplus scores, biased top-k, normalized weights)
and the fused SwiGLU-and-cast between the two expert projections come from
DeepSeek's TileKernels, which runs on SM121 with these changes.

## Sparse MLA

Each attention row reads two streams of 512-wide MQA latents (K and V are the
same latent):

| Cache | Record | Values | Scales |
|---|---|---|---|
| Sliding window (SWA) | 528 bytes | 512 x E4M3 | 16 x UE8M0, one per 32 coordinates |
| Indexed (top-k selected compressed entries) | 288 bytes | 256 bytes of packed E2M1, low nibble first | 32 x E4M3, one per 16 coordinates |

All 512 coordinates are quantized, RoPE included. Each CTA resolves the
slots it needs up front, then copies the next chunk's raw records into
shared memory with `cp.async` while the current chunk computes. It
dequantizes them to BF16 in shared memory (exactly, for both formats) and
multiplies on `mma.sync`. An index of `-1`, or a position past the row's
length, is masked and its record zero-filled.

### One contract, several kernels

The arithmetic is defined over chunks of `block_I` entries of the
concatenated [SWA | indexed] list, grouped into segments of
`segment_chunks` chunks (default 4: the 128-entry window is one segment).
Within a segment, chunks update a flash-attention state `(O_g, m_g, l_g)` in
order. The segment states then fold in order into a state that starts from
the head's sink, using explicit `T.fma`/`T.fmul`, and the output is
`O * (1 / L)`.

| Schedule | Grid | Suits |
|---|---|---|
| `extend` | one CTA per row and head block; segments fold in registers | many rows |
| `decode` | one CTA per segment, head block and row writes `(O_g, m_g, l_g)`; a merge kernel folds them | few rows: `segments x head_blocks` CTAs per row |

Both schedules run the same segment code and the same fold, so they produce
identical bits, and so do different head block sizes. `segment_chunks` and
`block_I` are part of the arithmetic; the schedule and `block_H` are not.
The tests check that every kernel gives the same bits, and that a row gives
the same bits alone and in a batch. Choosing a kernel by batch size
therefore never changes results.

`SparseMlaV41Plan` compiles three kernels for one layer shape and picks one
by row count:

- `decode` with 16-head blocks while it fits one wave of CTAs;
- `decode` with `block_H`-head blocks while it needs at most three waves;
- `extend` above that.

It owns the decode workspace and the output. `run(..., kernel=...)` forces
a kernel. `run` only launches kernels, so it can be captured in a CUDA
graph.

```python
plan = SparseMlaV41Plan(heads=24, swa_width=128, idx_width=512, idx_page_size=64, max_rows=48)
out = plan.run(q, swa_cache, swa_indices, swa_lengths, idx_cache, idx_indices, idx_lengths, idx_page_table, sinks)
```

Fewer chunks per segment give small batches more CTAs. More chunks per
segment write fewer partial states: the decode workspace holds one
512-wide FP32 state per segment, which costs more memory traffic than the
records themselves when segments are a single chunk.

The default tile is 32 heads by 32 entries with 256 threads, about 90 KiB of
shared memory. At TP3 a rank owns 24 heads, which fit one 32-head block.
`block_H=64, block_I=16` serves the 64 heads of TP1 in one block (a 64x32
tile exceeds 99 KiB); two 32-head blocks also work.

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

## Expert dispatch and combine

V4.1 Flash routes each token to 6 of 384 experts (the drafter: 3 of 128).
`moe_dispatch_v41` takes the router's `topk_idx` (int64 from TileKernels or
int32 from vLLM, `-1` for padding tokens) and groups the routes by expert in
the contiguous layout of grouped GEMMs, as in DeepGEMM's
`m_grouped_*_contiguous`: each expert's routes are consecutive, in route
order, and padded to a multiple of `block_M`. Besides the route at each
layout row and each route's layout row, it writes the expert of every
`block_M` tile, with `-1` past the last used tile, so a grouped GEMM with a
fixed grid can stop early.

A route's rank within its expert comes from warp match masks and per-warp
counts, so the layout is a pure function of `topk_idx`. Up to 512 routes
take one CTA; larger batches count per 512-route chunk, scan, and scatter.
`moe_combine_v41` sums each token's expert outputs, weighted, in slot order,
and can add the shared expert's output. `MoeDispatchV41Plan` sizes the
buffers for `max_rows`, so both steps can be captured in a CUDA graph.

## Grouped expert GEMMs

`grouped_fp8_fp4_gemm` runs one routed-expert projection for every expert
in one launch, over the dispatch layout. Each `block_M` tile multiplies by
its expert's MXFP4 weights; the grid covers the plan's fixed tile count and
unused tiles exit at once, so the launch can be captured in a CUDA graph. A
pipeline loads the next K blocks while the current one multiplies: the
expert's weights by TMA into unpacked FP4 shared memory, and the activation
rows (gathered through the layout for the gate/up projection) by `cp.async`.

Two things keep decode close to the memory bandwidth. Weights are stored
tile-major (`tile_expert_weights`, done once at load time), so every TMA box
is one contiguous slab instead of `block_N` scattered 64-byte pieces. And
padding rows, which make up nearly every row of a tile when an expert
receives one or two tokens, are zero-filled without a memory read and never
stored.

`MoeExpertsV41Plan` chains a routed-expert layer: TileKernels' router,
dispatch, TileKernels' MXFP8 cast of the tokens, the gate/up projection,
TileKernels' clamped SwiGLU with the route weights and MXFP8 cast, the down
projection, and the combine. As in DeepSeek's kernels, the route weight is
applied before the down projection, so the combine is a plain sum.
TileKernels' packed row-major UE8M0 scales are exactly the scale words the
GEMMs read. Without TileKernels, PyTorch stands in for the router and the
casts.

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

| Sparse MLA, 24 heads, SWA 128 + indexed 512 | Plan's kernel | Time |
|---|---|---:|
| 1 row | `decode`, 16-head blocks | 11.5 us |
| 4 rows | `decode`, 16-head blocks | 12.3 us |
| 8 rows | `decode` | 14.8 us |
| 16 rows | `decode` | 27.8 us |
| 24 rows | `decode` | 43.2 us |
| 32 rows | `extend` | 48.3 us |
| 48 rows | `extend` | 50.5 us |
| 64 rows | `extend` | 96.7 us |

Every kernel gives the same bits, and the outputs match an FP32 reference
within the tests' tolerance (2e-2). With one chunk per segment the decode
workspace dominates instead: `decode` takes 30 us at 8 rows and 97 us at 16.

| Indexer, score + top-512, page 64 | Time |
|---|---:|
| 1 row, 4096 positions | 10.7 us |
| 1 row, 32768 positions | 21.1 us |
| 8 rows, 32768 positions | 48.9 us |
| 48 rows, 32768 positions | 365 us |
| 1 row, 131072 positions | 37.3 us |

| Grouped expert GEMMs, TP3, 384 experts, top-6 | Experts used | Gate/up 1536x5120 | Down 5120x768 |
|---|---:|---:|---:|
| 8 tokens | 46 | 812 us (223 GB/s) | 393 us (230 GB/s) |
| 16 tokens | 85 | 1469 us (228 GB/s) | 712 us (235 GB/s) |
| 48 tokens | 204 | 3525 us (228 GB/s) | 1719 us (233 GB/s) |
| 4096 tokens | 384 | 7989 us | 5577 us |

Decode is bound by reading the used experts' FP4 weights; the bandwidths
count those bytes only (GB10 peaks at 273 GB/s). All rows use 32x128 tiles,
tile-major weights and a three-deep pipeline.

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
python moe_dispatch_sm120.py --rows 48
python moe_expert_gemm_sm120.py --rows 16
python mxfp8_gemm_sm120.py --projection wq_b
python fp8_fp4_gemm_sm120.py
python -m pytest test_tilelang_example_deepseek_v41.py
```

SM121 (`sm_121a`) needs CUDA 12.9 or newer.
