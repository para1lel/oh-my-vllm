# Model operator contracts

The [stage inventory](../../python/oh_my_vllm/kernels/specification.py) gives the inputs, outputs, computation, and state writes of the internal semantic stages.
The dimensions come from the checkpoint configuration and tensor headers.

Use the configuration SHA256 in that inventory to identify the model.
Do not change the checkpoint values or FP8 scales.
The loader can convert BF16 norm weights, scales, A_log, and dt_bias to FP32 without a change of value.

## Language model

| Property | Value |
|---|---|
| Hidden / FFN / vocabulary | 5120 / 17408 / 248320 |
| Layers | 48 GDN + 16 FA |
| GQA | 24 query heads, 4 KV heads, width 256 |
| GDN | 16 key heads, 48 value heads, width 128 |
| Convolution | 10240 channels, width 4 |
| Cache | Block 784, BF16 recurrent state, align mode |
| Context / step | 262144 / 32768 tokens |
| FP8 | E4M3, 128-by-128 weight scales, dynamic 128-element activation scales |
| MTP | One FA layer, four proposals |

The applicable norm weights use the checkpoint offset.
The GDN gate norm uses the checkpoint weight without an offset.

The packed GQA projection has 14336 columns.
Each query head has 256 query columns followed by 256 gate columns.

The last 2048 columns contain K and V.
The GQA output gate uses sigmoid. The GDN output gate uses SiLU.

MRoPE uses 64 of the 256 head elements, theta 10000000, and interleaved sections `(11,11,10)`.
Image and video tokens keep their temporal and spatial positions.

Use these GDN equations with the value-head to key-head mapping `h // 3`:

```text
q = q * rsqrt(sum(q*q) + epsilon)
k = k * rsqrt(sum(k*k) + epsilon)
g = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)
S_decay = exp(g) * S
delta = beta * (v - S_decay @ k)
S_next = S_decay + outer(delta, k)
o = S_next @ q / sqrt(128)
```

Keep the BF16 rounding points of the inference path.
Use FP32 live carry for prefill and recurrent verification rows.
Store persistent checkpoints in BF16.

Save checkpoints at each 784-token boundary and at the final row.
Verification writes private states. Only accepted rows can replace the accepted state.
Sequence boundaries and read, write, and checkpoint slots identify the convolution state for each request.

To accept tokens, select the saved state slots.
Do not copy the full recurrent or convolution state to accept tokens.

MTP context row `p` pairs target hidden row `p-1` with token `p`.
Position zero has no MTP row. Prefix restoration supplies the boundary hidden row.
Image and video rows use their vision embeddings in the MTP input norm.

Context fill projects K and V for all context rows.
Q, the gate, attention output, and FFN output are necessary only at the selected endpoint.

The GQA input variants use 14336 full columns, 2048 K/V columns, or 12288 Q/gate columns.
MTP target verification produces at most five vocabulary rows per request.
Draft entries that are not active form a suffix with token ID `-1`.
For each request, the target row after the active drafts supplies the bonus distribution without proposal subtraction.
The first draft uses the endpoint with the Q/gate-only variant.
The other 3 drafts use the full MTP input and layer path.

## Vision model

| Property | Value |
|---|---|
| Layers / hidden / FFN | 27 / 1152 / 4304 |
| Attention | 16 heads, width 72, noncausal |
| Patch | 2 frames by 16 by 16 pixels, RGB |
| Spatial merge | 2 by 2 patches |
| Position table | 2304 by 1152 |
| Output | 5120 columns |
| Weights | BF16, with bias |
| Activation | Block MLP: GELU tanh approximation. Merger: exact GELU |

If temporal patching makes another frame necessary, use the final frame again.
Use the checkpoint image normalization and resize limits.

Interpolate the absolute position table for the spatial grid.
Apply spatial RoPE to Q and K.

Attention connects patches in the same temporal frame group.
It does not connect different images or frame groups.

The merger normalizes each 1152-element patch before the 2-by-2 gather.
Its two projections have shapes `4608x4608` and `5120x4608`.

Put the output embeddings in the related language-token rows.
Keep the input order for text, images, and video.

## Intermediate precision

Follow [vLLM commit e8e1e961](https://github.com/vllm-project/vllm/commit/e8e1e96196725f097eae2394cbc458b1f5a679b8) for numerical behavior.
Use its BF16 model path with the project cache settings. This reference adds no vLLM dependency.
Use the source types and rounding points unless the selected Tensor Core path uses the permitted lower precision.

Use FP8 or BF16 Tensor Core operands when these types pass the numerical checks.
The implementation can round intermediate operands and prepared scales to a Tensor Core format.
Keep the original checkpoint tensors unchanged. Do not increase precision to pass a numerical check.
Use the full-model logits limits from the [evaluation procedure](evaluation.md).

FFN inputs with more than 128 rows use MXFP8 Tensor Core projections.
GDN inputs with more than 128 rows use MXFP8 input projection.
Their output projection also uses MXFP8 for prefill and verification with `max_queries > 1`.
Single-query verification keeps the previous output projection.
Weight preparation converts derived E4M3 codes and supplies UE8M0 scales with power-of-two values.
The [normalization kernel](../../python/oh_my_vllm/kernels/cuda_backend/mx_fp8_norm.cuh#L7)<!-- source: norm_mx_kernel(const --> supplies MXFP8 operands directly from its BF16 result.

The original checkpoint tensors and scales stay unchanged.

Full FA boundaries with more than 128 rows use MXFP8 input projection.
They use MXFP8 output projection when attention has one split.

Use the operand, accumulator, and storage types of the selected implementation for the work calculation.
Do not increase work for a slower instruction choice that supplies the same numerical computation.

| Region | Computation | Stored result |
|---|---|---|
| Other FP8 projections | E4M3 operands, FP32 scales and accumulation | BF16 |
| MXFP8 FFN and GDN projections, more than 128 rows | E4M3 operands, UE8M0 scales, FP32 accumulation | BF16 |
| Selected MXFP8 FA projections, more than 128 rows | E4M3 operands, UE8M0 scales, FP32 accumulation | BF16 |
| BF16 projections | BF16 operands, FP32 accumulation | BF16 |
| Residual add and RMS | FP32 sum and statistics. Normalize the unrounded sum | BF16 residual and normalized result |
| Q/K RoPE | BF16 normalized Q/K and cos/sin. FP32 products and sum | BF16 |
| GDN Q/K norm | FP32 | BF16 for prefill. FP32 live values for recurrent rows |
| GDN scan | FP32 live carry, products, and sums | BF16 checkpoints |
| GDN chunked prefill | BF16 matrix operands. FP32 accumulation, state decay, and live carry | BF16 checkpoints |
| GDN head norm and gate | BF16 attention result, then FP32 RMS, weight, SiLU, and product | One BF16 result |
| FA attention and gate | BF16 Q/K/V/P, FP32 softmax and accumulation | BF16 attention, sigmoid, and product before FP8 |
| Vision attention | BF16 Q/K/V/P, FP32 softmax and accumulation | BF16 |
| MTP dual RMS | FP32 statistics and gain computation | BF16 packed input |
| MTP FC, all row counts | BF16 operands, FP32 accumulation | BF16 |
| Sampling acceptance and residual CDF | FP32 probabilities, ratio, masses, and sums. FP64 random thresholds | Integer selections |

Chunked GDN prefill follows the pinned [solve and projection types](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/chunk.py#L38).
The triangular-system result and gate-scaled K/V operands use BF16 stores before the W/U matrix products.
W/U outputs also use BF16 stores.
The triangular inverse can use BF16 intermediate storage and BF16 block matrix operands with FP32 accumulation.
Diagonal solves stay FP32. State decay, live carry, full token history, and private checkpoint requirements stay unchanged.

The [carry path](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py#L154) rounds the state operand and scaled new values to BF16 before matrix products.
The state update scales the FP32 new values before BF16 conversion.
Its separate output export uses BF16.
State decay and carry updates stay FP32. BF16 checkpoint export must not replace the live carry.

The [output path](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/chunk_o.py#L94) uses BF16 state snapshots and new values, with FP32 accumulation.
It rounds the causal, gated QK matrix to BF16 before the new-value product.
The [projection source](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/third_party/flash_linear_attention/ops/wy_fast.py#L75) defines the gate-scaling rounding points.

The pinned [RMS reference](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/vllm/ir/ops/layernorm.py#L44) separates FP32 normalization input from BF16 residual storage.
The [GDN decode path](https://github.com/vllm-project/vllm/blob/e8e1e96196725f097eae2394cbc458b1f5a679b8/csrc/gdn/fused_gdn_decode_kernel.cu#L226) keeps recurrent Q/K in FP32.
The [vision attention dependency](https://github.com/vllm-project/flash-attention/blob/b273a02621d3c94bc747df67204f591b67f38fae/flash_attn/cute/flash_fwd_sm100.py#L2522) rounds online probabilities to BF16 before PV.
The full-model reference uses FP64 throughout. It does not reproduce intermediate low-precision rounding.

The [HTTP reference calculation](../../development/serving/accounting.py) uses fixed precision assumptions for performance accounting.
Temporary storage and duplicated work cannot increase the theoretical lower bound.

## Numerical checks

Use the [full-model evaluation procedure](evaluation.md).
The reference uses direct PyTorch FP64 computation with checkpoint weights and their original scales.
Compare the complete vocabulary after a short input and one decode iteration.
Do not use single-operator tolerance tables or per-stage error gates.

Keep the state ownership, checkpoint, and precision contracts above.
Use [the fusion plan](fusion.md) for the serving boundaries.
