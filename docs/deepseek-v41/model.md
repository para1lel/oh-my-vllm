# DeepSeek V4.1 source contract

## Authority and identity

This is the Phase 0 mathematical source contract, version 1.
It does not declare runtime support or numerical acceptance.
The [state contract](state.md) specifies sparse selection, Engram, DSpark, and restoration.
The [validation plan](validation.md) owns numerical references and acceptance limits.

`LOCAL-V41` identifies the supplied `DeepSeek-V4.1-Flash` checkpoint, not a host directory or an upstream revision.
Names after `LOCAL-V41:` are relative source identities.
The following SHA256 values bind the mathematical sources.
They do not verify the weight payload.

| Source identity | SHA256 |
|---|---|
| `LOCAL-V41:config.json` | `8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879` |
| `LOCAL-V41:model.safetensors.index.json` | `74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8` |
| `LOCAL-V41:inference/model.py` | `4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65` |
| `LOCAL-V41:inference/kernel.py` | `1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455` |
| `LOCAL-V41:inference/convert.py` | `035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76` |
| `LOCAL-V41:inference/engram.py` | `11f35ecbead8150c35aa002b3d180ef290b05a25afe883a11884f94d476d3897` |
| `LOCAL-V41:inference/vision.py` | `5d49edc196a4ef22384abe76d35a40098cbe1e74b586c8f66a2edff4f076b26c` |
| `LOCAL-V41:inference/image_processor.py` | `482759e3bcc4e9bb5ee582b244cc563f5d0e163d8b48dda91ebb7106e62f9272` |
| `LOCAL-V41:encoding/encoding.py` | `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1` |
| `LOCAL-V41:tokenizer.json` | `c90dfa01249db1be4245780a052ede752e1361c612ac6d08e2bdada7d599476b` |
| `LOCAL-V41:tokenizer_config.json` | `6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547` |
| `LOCAL-V41:chat_template.jinja` | `d959d804d5101b79a49b6ff1bf3c54cd5affa9a5a78c503d925f3090040b31c3` |
| `LOCAL-V41:README.md` | `347c9db4e5506acb531cbc3b724407ab88e9af8781679152f0823d7bac16d251` |
| `LOCAL-V41:LICENSE` | `f2c6c602815669d292889e5be8c802f2ed950653b77999b1584e8e6aed25d040` |

The local license is MIT.
[DeepInfer](https://github.com/deepseek-ai/DeepInfer/tree/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9) is a read-only algorithm reference at revision `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9`.
Its [license](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/LICENSE) is Apache-2.0.
Its [default checkpoint](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/examples/_config.py) is `deepseek-ai/DeepSeek-V4.1-Flash-1017`.
That name does not prove weight identity with the local checkpoint.

## Target computation

The configuration selects `DeepseekV41ForCausalLM`, width 5120, 40 target layers, and vocabulary 129280.
The total context limit is 1048576 tokens.
Attention has 64 query heads, one shared 512-channel KV vector, and 64 rotary channels.
Query rank is 1280.
Output projection has eight groups with rank 1024 per group.
The embedding and vocabulary head have different weights.

`Transformer.forward`, `Block`, and `Attention` in `LOCAL-V41:inference/model.py` define the local graph.
Four hyper-connection copies carry each residual.
The initial collapse selects copy 0.
Attention uses the previous FFN's pre-mix.
FFN uses the current attention's pre-mix.

Each sublayer produces post-mix and residual-mix coefficients for its own output.
Pre-mix is sigmoid plus `1e-6`.
Post-mix is twice sigmoid.
Residual mixing uses 20 Sinkhorn iterations with `1e-6` stabilization.
This finite computation is not an exact doubly stochastic projection.
Text RMS normalization uses FP32 statistics and epsilon `1e-20`.

Each target layer has 384 routed experts, six selected experts, and one shared expert.
Expert intermediate width is 2304.
Routing scores are `sqrt(softplus(xW))` at gate temperature 1.

Text or image correction bias selects experts but does not scale their outputs.
Selected unbiased scores are normalized with denominator epsilon `1e-20`, then multiplied by 1.5.
SwiGLU caps the gate projection at 10 and clamps the up projection to `[-10,10]`.
The routing weight multiplies the activated intermediate before the down projection.

## CED boundary

`LOCAL-V41:README.md` calls layers 0-19 the causal encoder and layers 20-39 the decoder.
The ratio-1 global KV source is layer 20's normalized attention input.
Later decoder layers read that global KV instead of generating separate global KV.
Decode executes all 40 target layers.

The local `Transformer.forward` also executes all 40 layers for the full prefill.
It does not implement encoder-only prefill or bounded decoder replay.
Its decoder sliding-window KV comes from each layer's own attention input.
Its target hidden features also depend on those decoder computations.

DeepInfer's [CED helper](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/model_executor/models/v4_modules/yoco.py) defaults to shared decoder SWA inputs.
Its [deployment guidance](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/doc/inference_deployment_qa.md) permits truncated, approximate replay.
Neither behavior is local mathematical equivalence.
Do not omit decoder prefill rows without an approved equivalence contract.

## Weight and scale formats

`Linear`, `ParallelEngramEmbedding`, and `LOCAL-V41:inference/convert.py` define storage interpretation.
Tensor names already use `layers`, `attn`, `ffn`, `mtp`, and `scale` in the local index.
Alias conversion must not change tensor identity.

FP8 matrix weights use E4M3 with one UE8M0 multiplier per 32-by-32 output/input block.
The reconstruction is `W[o,k]=E4M3[o,k]*S[o//32,k//32]`.
UE8M0 byte `b` represents `2^(b-127)` for `0<=b<=254`.
Byte 255 is NaN, not a finite scale or a zero point.
Scale names do not imply reciprocal multiplication.

Routed expert weights use signed-byte storage for two E2M1 values.
Logical shape `[O,K]` has stored shape `[O,K/2]`.
Read the byte as unsigned before nibble extraction.
The low nibble supplies channel `2j`.

The high nibble supplies channel `2j+1`.
Codes 0-7 represent `0,0.5,1,1.5,2,3,4,6`.
Codes 8-15 supply their negative counterparts, with signed zero at 8.
Each output row has one UE8M0 multiplier per 32 input channels.
This is not a 32-by-32 expert scale layout.

Engram FP8 tables have one UE8M0 scale per row and 32 channels.
Their stored shapes are `[384006168,256]` and `[384016682,256]`.
Each scale table has eight columns.
Authoritative tables and scales stay in pinned host storage.

The local converter expands grouped `wo_a` FP8 weights to BF16.
The model loads ratio-2 compressor BF16 projections into FP32 parameters.
The vocabulary head and confidence projection also use FP32 parameters from BF16 weights.
These are source computation choices, not original storage formats.
Do not silently substitute expert FP8 conversion or a different projection rounding boundary.

## Intermediate precision

`LOCAL-V41:inference/kernel.py` defines quantization and accumulation.
Dense and expert projections quantize activations to E4M3 in groups of 32.
Their scale is `2^ceil(log2(max(amax,1e-4)/448))`.
Quantized values clamp to `[-448,448]` before conversion.
GEMM uses FP32 accumulators and BF16 outputs in the local default configuration.

Sliding-window KV uses this FP8 quantize/dequantize operation after RoPE, across all 512 channels.
Indexer queries and keys use FP4 groups of 32 with UE8M0 scales.
Their scale uses `max(amax,6*2^-126)/6` rounded upward to a power of two.
Compressed main KV uses FP4 groups of 16 with E4M3 scales.

Its scale is `E4M3(max(amax,6*2^-9)/6)`, not UE8M0.
FP4 conversion clamps to `[-6,6]`.
The reference stores dequantized cache values in BF16 tensors.
A packed service cache must reproduce those quantization boundaries.

RoPE rotates adjacent pairs in the final 64 channels.
The attention output receives inverse query rotation before grouped output projection.
Window-only layers use theta 10000 without YaRN.
Compressed layers use theta 160000 and YaRN factor 16 from original length 65536.
YaRN uses beta values 32 and 1.
The local attention score scale is `1/sqrt(512)` without an additional YaRN amplitude multiplier.

## Prompt encoding

`LOCAL-V41:encoding/encoding.py` owns prompt text, tool order, reasoning markers, and ordered image records.
It supplies BOS in the rendered prompt.
It uses V4.1 DSML tags with a leading space before `calls`, `invoke`, and `parameter`.
Tool results follow their call order.
Tool conversations keep earlier reasoning instead of applying the usual reasoning removal.

Numeric reasoning effort is 1-100.
Aliases `low`, `high`, and `max` map to 50, 75, and 100.
The default is `high` at 75.
The effort prefix appears only at conversation index 0 in thinking mode.
These aliases do not define mappings for other public API effort names.
Freeze encoded token IDs and image order, not rendered-text similarity alone.

## Image and video boundary

`LOCAL-V41:inference/image_processor.py` owns resize, padding, normalization, patch order, and image-span expansion.
RGB pixels use `(pixel/255-0.5)/0.5`, then BF16.
Minimum pixel area is 295936.

The final image span has at most 1024 tokens.
The planner counts `H*(W+1)+2`, with one newline per aligned row.
Patch size is 14.
The reference uses PIL padding and resize behavior, not a generic center crop.

`LOCAL-V41:inference/vision.py` defines 32 bidirectional layers at width 1024, with 16 heads and intermediate width 2816.
Vision uses 2D RoPE and RMS epsilon `1e-6`.
The aligner zero-pads the patch grid to multiples of 3.
It gathers nonoverlapping 3-by-3 cells, then applies a two-layer GELU projector to width 5120.

Image features replace text embeddings before hyper-connection expansion.
Start, end, and newline positions use learned embeddings.
All image-span positions carry token ID 129264, with separate token types.
Language-model positions are sequential token positions, not Qwen three-axis positions.
The local forward accepts full image spans only in the first prefill chunk.

The local encoding source supports ordered images, not a video frame sampler or timestamp protocol.
Video must first produce ordered frame images and timestamp text through a separately specified codec.
No temporal vision encoder is present.
Frame selection, timestamp spelling, token positions, and numerical reference outputs are unresolved video prerequisites.

## Compatibility gate

DeepInfer's [tokenizer policy](https://github.com/deepseek-ai/DeepInfer/blob/9bd9cf243fc53ef9afbf8f04d391d989117d6ab9/deepinfer/hugging_face/tokenizer.py) adds multimodal IDs 129280-129300 and changes BOS/PAD behavior.
The local tokenizer disables automatic BOS/EOS insertion and aliases tokenizer PAD to EOS.
The model configuration separately selects PAD 2.
Do not apply those upstream changes before Engram map generation.

The local README denies a Jinja template, but `LOCAL-V41:chat_template.jinja` exists.
Use the identified Python encoding source as authority pending token-ID equivalence review.
Weight payload identities, independent endpoint vectors, and approved limits are separate Phase 0 gates.
This source reading supplies none of those numerical artifacts.
