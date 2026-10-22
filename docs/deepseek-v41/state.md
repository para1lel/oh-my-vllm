# DeepSeek V4.1 state contract

## Authority and scope

This document extends the [source contract](model.md), version 1.
Its `LOCAL-V41` identities have the hashes in that document.
`LOCAL-V41:inference/model.py` defines attention and DSpark computation.
`LOCAL-V41:inference/engram.py` defines token mapping and hash state.
This contract separates local mathematics from proposed service state handling.

## Sparse modes and source sharing

Layer numbers are zero-based.
Layers 0-1 and draft layers 40-42 have sliding-window attention only.
Target layers 2-19 use compression ratio 2.
Target layers 20-39 use ratio 1.
The sliding window contains at most 128 positions, with the current token included.

| Mode | Layers | Global state |
|---|---|---|
| Full | 2, 8, 14, 20 | Produce compressed KV, index keys, and query-dependent top-k indices |
| Reindex | 24, 28, 32, 36 | Read layer 20 KV and index keys, then compute new top-k indices |
| Reuse | Other compressed layers | Read the latest source's KV, keys, and top-k indices |

Full names a sparse source mode, not dense full-context attention.
Packed main KV uses `512/2+512/16=288` bytes per compressed row.
Packed index keys use `128/2+128/32=68` bytes per compressed row.
The four source stores thus use `(288+68)*(3/2+1)=890` bytes per token before replication and allocation rounding.
This excludes windows, partial groups, speculative state, and workspaces.

KV and index-key source ranges are 2-7, 8-13, 14-19, and 20-39.
Top-k source ranges split the last range at 24, 28, 32, and 36.
The source runs before its consumers for the same query rows.
The reference passes shared tensors through `SharedAttentionRuntime`.
Those process-global references are not persistent request identities.
The service must bind source identities, query-row layout, positions, and request ownership explicitly.

The local prefill path starts at position 0.
Its nonzero-position path accepts one token per sequence.
It does not supply arbitrary chunk continuation or multirow target verification.
New chunked operators must carry groups that are not full and causal visibility across chunk boundaries.
Reusing the initial-prefill path at nonzero positions would change the mathematics.

Each layer still produces its own sliding-window KV.
Source sharing applies to global compressed KV and index state, not local sliding-window KV.
Ratio-2 compression projects values and per-channel gate scores in FP32.

For each full pair, softmax over its two token scores supplies per-channel pooling weights.
The pooled value returns to BF16 before RMS normalization.
Ratio 1 uses a BF16 projection and RMS normalization without pooling.
Incomplete ratio-2 groups publish no global row.

Compressed row `j` uses RoPE position `j*ratio`, the first token of its group.
Index keys use the normalized latent before main-KV RoPE and FP4 quantization.
The index projection, normalization, RoPE, and FP4 quantization run before that latent is overwritten.
Main KV then receives its own RoPE and cache quantization.
At query position `p`, only `floor((p+1)/ratio)` full groups are visible.

The indexer has 32 heads of width 128.
It quantizes rotated queries and keys to FP4.
Its score sums `ReLU(q_h dot k)` times signed projected head weights.

The head-weight factor is `1/sqrt(128*32)`.
It selects at most 512 positions, then sorts their indices by position.
Invalid entries use `-1`.
Top-k score ties have no explicit local tie-breaking rule.

Layer 20 constructs a candidate pool for later reindex layers.
Each block contains eight compressed positions.
Its score is the maximum position score.
The newest reachable block receives an infinite score to force inclusion.

At most 2048 reachable blocks enter the pool.
Later reindex layers apply their own scores only to this pool.
The local reference computes dense index scores before masking.
It does not prove a bounded-memory sparse implementation for million-token prefill.

Attention concatenates local-window entries with selected global entries.
These are distinct cache entries even when their source token intervals overlap.
For each head, the mathematical output is `sum(exp(q dot k/sqrt(512))*k)/(exp(sink)+sum(exp(q dot k/sqrt(512))))`.
The sink contributes denominator mass but no value.
The local kernel uses FP32 softmax statistics and BF16 exponential operands for value accumulation.
All-invalid rows produce zero output.

## Engram mapping and positions

Engram runs before target layers 1 and 14.
It hashes 2-, 3-, and 4-grams with eight heads per size.
Each lookup has 256 channels, so each token fetches 24 rows per Engram layer.
The compressed vocabulary size must equal 99092.

`build_compressed_token_map` visits raw token IDs in ascending order.
It decodes each token without skipping special tokens.
Normalization applies NFKC, NFD, accent removal, lowercase, whitespace collapse, and strip in source order.
A private-use sentinel protects a single space from strip.

Empty normalized strings fall back to decoded text.
Tokens with the Unicode replacement character use their raw token spelling.
First occurrence assigns each compressed ID.
Tokenizer libraries and normalization versions are therefore identity inputs.

Hash multipliers use NumPy `default_rng(10007*layer_id)`.
The source draws four signed-int64 values, then applies `2*v+1`.
The exclusive upper bound is `(INT64_MAX//99092)//2`.

Successive lookback products combine with XOR, not addition.
Each n-gram/head uses a distinct prime modulus and cumulative table offset.
Prime search starts above 15999999 for each n-gram range.
The used-prime set spans all ranges and Engram layers.

Positions are absolute language-model token offsets.
Lookback stops at sequence start or a dead image-span token.
Unavailable tokens use the compressed mapping of PAD token 2.
An image token blocks itself and all earlier lookback positions.

It receives no Engram contribution.
Later text can use its own text suffix but cannot hash across that image boundary.
The reference caches compressed IDs by absolute position.
Continuation uses only the last three IDs with their dead markers and accepted length.
This reduction follows the maximum 4-gram lookback, not an attention replay assumption.

Lookup rows dequantize to BF16 before the Engram projection.
The projection supplies four keys and one shared value at width 5120.
The gate uses per-copy RMS-normalized key/residual dot products and the product of query/key norm weights.
It applies signed square root with absolute floor `1e-6`, then sigmoid.
The gated value adds to each residual copy before the target block.
Bounded device staging must not change row IDs, scales, or addition positions.

## DSpark hidden states and proposals

The local configuration has three draft layers and block size 5.
Each draft layer has 128 routed experts and three selected experts.
Target features concatenate the mean across four residual copies at layer inputs 37, 38, and 39.
These are pre-attention block inputs, not layer outputs or final normalized hidden states.
`main_proj` maps 15360 channels to 5120, followed by RMS normalization.

Draft prefill only seeds each draft layer's main-token window KV.
It does not execute draft attention/FFN for prompt proposals.
Decode consumes one target feature row at absolute position `p`.
The five draft input rows contain the target-sampled token first, followed by noise token 128799.

Their attention positions are `p+1` through `p+5`.
Each draft row sees the accepted main-token window and all five draft KV rows.
The draft block is bidirectional, not five causal target steps.
Draft embeddings and vocabulary head use the target's weights.

After collapse and draft normalization, the vocabulary head supplies five base-logit rows.
A rank-256 Markov head adds a vocabulary bias from the preceding sampled token.
Thus proposal distributions depend sequentially on earlier sampled proposals despite parallel draft attention.

`forward_head` returns six IDs: the supplied target token and five proposals.
Its returned logits include those Markov biases.
Confidence projects the unnormalized collapsed hidden state concatenated with the preceding token's Markov embedding.
The local result is a raw scalar, not a sigmoid probability.

DeepInfer's [confidence kernel](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/dflash.py) divides by temperature, clamps to `[-60,60]`, then applies sigmoid.
Its [draft path](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/mtp/dflash.py) stores proposal probabilities for verification.
Those operations do not occur in the local forward.
Confidence calibration and verification-budget selection must have a separate approved contract.
Confidence changes verification length, not the acceptance ratio.

The local sampler uses temperature softmax and exponential-race sampling, or argmax at temperature 0.
It does not implement service penalties, grammar masks, top-k/top-p, rejection correction, or RNG allocation.
For stochastic service verification, `q_i` must be the full constrained distribution that generated proposal `y_i`.
The target distribution `p_i` uses the same accepted prefix and constraints.
At later verification rows, that prefix includes the proposed tokens before the current row.
Grammar and penalty updates for those rows use private speculative state.

Exact rejection sampling accepts with `min(1,p_i(y_i)/q_i(y_i))`.
At first rejection, the correction distribution is proportional to `max(p_i-q_i,0)`.
An all-accepted prefix permits one next target sample.
Five drafts can thus use six target verification rows.
These are service requirements, not a claim that the local loop implements them.

DeepInfer's [sampler](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/layers/tilelang_ops/cuda/fused_sample.py) subtracts `exp(draft_logprob-DRAFT_PROB_PENALTY)` during correction.
Do not use that adjustment as the exact formula above.
Freeze probability storage, RNG streams, and absolute sample positions before implementation.
Rejected rows cannot advance accepted grammar, penalties, token history, or logical output positions.
Acceptance and correction draws have separate RNG allocations from output-token draws.
The local source does not define these service RNG streams.

## Accepted state and restoration

Cache length counts processed input tokens, not all tokens already emitted by sampling.
A sampled next token can be accepted but not yet materialized in target KV.
Keep that pending token separate from the processed-prefix length and target-feature position.
After rejection, a correction token must pass through target computation before its KV or target features can be used.

| State group | Accepted contents |
|---|---|
| Target windows | Per-layer 128-position KV, absolute length, and ring-slot mapping |
| Global sources | Four compressed-KV stores, four index-key stores, and valid lengths |
| Partial compressors | FP32 values and gate scores for unfinished ratio-2 groups |
| Sparse selections | Source identity, query positions, candidates, and top-k row layout |
| Engram | Last three compressed IDs, dead markers, and accepted length |
| DSpark | Three main-token windows, target features, proposal distributions, and verification positions |
| Sampling | Accepted tokens, grammar/penalty history, stopping state, and RNG allocation |

Candidate masks and top-k indices can be transient when recomputed before use.
They cannot come from rejected rows or a different request.
Keep speculative writes separate from accepted state.

Use shadow slots or save overwritten ring entries before verification.
Restore valid lengths and incomplete compressor contents after rejection at any position.
Truncating token arrays alone cannot undo ring overwrites or a completed speculative compression group.
Shared-source rollback must select one consistent state across all consumers and ranks.

Exact continuation is possible with full accepted state and the same arithmetic contract.
This source reading does not prove byte-identical output across different reduction layouts.
Prefix identity must bind weights, tokenizer mapping, media, absolute positions, formats, and parallel layout.
Host offload must store sufficient state for that continuation.

DeepInfer's [replay guidance](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md) calls truncated 128-token SWA replay approximate.
Local layer-dependent SWA inputs propagate dependencies through deeper layers.
A single 128-token tail is therefore not an exact-restoration proof.
Keep full accepted windows unless a sufficient replay construction has approval.
No fixed replay length is approved here.
