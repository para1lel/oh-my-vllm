# DeepSeek fixed reference fixtures

## Scope and identity

The [fixture reader](../../development/model/deepseek_v41/fixtures.py) checks SHA256 identities, metadata, and complete arrays.
The linked indexes contain the fixed numerical limits and source identities.
Ordinary checks load these artifacts without PyTorch, GPU execution, or checkpoint access.
Do not regenerate expected values during acceptance or replace them with candidate outputs.

Operator fixtures cover isolated boundaries. They do not prove model, state, capacity, or performance acceptance.
Their engineering limits are not worst-case error proofs.
Use the [validation plan](validation.md) for those separate gates.

| Fixture | Content |
|---|---|
| [Short text](../../development/model/deepseek_v41/reference.npz) | Two complete FP64 vocabulary vectors |
| [Long text](../../development/model/deepseek_v41/long-context.npz) | Four complete FP64 vectors at 16385 and 65537 tokens |
| [Prompt codec](../../development/model/deepseek_v41/codec.json) | Twenty cases with complete token IDs and media order |
| [DSpark continuation](../../development/model/deepseek_v41/dspark.npz) | Eighteen vocabulary rows and two confidence vectors |
| [Media endpoints](../../development/model/deepseek_v41/media.npz) | FP64 prefill and continuation after still and frame images |
| [Exact operators](../../development/model/deepseek_v41/exact-operators.npz) | Quantization bytes, Engram hashes, and continuation history |
| [Engram lookup](../../development/model/deepseek_v41/engram-lookup.npz) | Synthetic encoded table and exact BF16 outputs |
| [Engram checkpoint](../../development/model/deepseek_v41/engram-checkpoint.npz) | Original samples and exact outputs from the two tables |
| [Engram blend](../../development/model/deepseek_v41/engram-blend/index.json) | Four cases with complete FP64 gates and residuals |
| [Hyper-connections](../../development/model/deepseek_v41/hyperconnection/index.json) | Five cases with complete outputs and field limits |
| [Routing](../../development/model/deepseek_v41/routing/index.json) | Five cases with exact expert IDs and FP64 coefficients |
| [Encoded projections](../../development/model/deepseek_v41/projection/index.json) | Five A8W8 and five A8W4 matrix products |
| [Rank-owned projection](../../development/model/deepseek_v41/rank-encoded-projection/index.json) | Exact quantization and complete FP64 products for 1, 128, and 384 rows |
| [Unquantized projections](../../development/model/deepseek_v41/unquantized-projection/index.json) | Seven BF16/FP32 cases with complete FP64 products |
| [Startup conversion](../../development/model/deepseek_v41/startup-promotion.npz) | Exact FP8-to-BF16 and BF16-to-F32 conversion |
| [Normalization and rotation](../../development/model/deepseek_v41/norm-rope/index.json) | Four RMS and three RoPE cases |
| [Activation](../../development/model/deepseek_v41/swiglu/index.json) | Four SwiGLU cases |
| [Ratio2 pooling](../../development/model/deepseek_v41/pooling/index.json) | Four complete-pair cases |
| [Sparse attention](../../development/model/deepseek_v41/sparse-attention/index.json) | Five packed-cache cases with complete FP64 outputs |

The reader binds the local checkpoint manifest. Full-model artifacts also contain fixed inputs, generator identities, shapes, and array digests.
Float comparisons use separate request limits. Integer decisions and encoded fields use exact limits where specified.

## Exact operators

`load_exact_operators()` covers FP8/UE8M0 group32, FP4/UE8M0 group32, FP4/E4M3 group16, and Engram chunk continuation with dead tokens.
Expected values come from hand-checked literals.
Compare encoded values, scales, BF16 roundtrip bits, hashes, and history.

`load_engram_lookup()` covers repeated indices, signed zeros, subnormal ties, and distinct scale groups in a synthetic 16-row table.
`load_engram_checkpoint()` supplies 32 original rows from each table, with exact BF16 outputs.
The [lookup checker](../../development/operators/check_engram_lookup.py) uses these samples against the full shared store.

Quantization callers must reject nonzero group status. Lookup callers must reject nonzero row status.
Engram hashing returns private continuation history without changing its input history.

Set `EXACT_REPORT` to a new external JSON path. Build and check the modules:

```bash
scripts/with-env.sh python -m development.native_build \
  --module device --module quantization --module engram
OH_MY_VLLM_GPU_COUNT=1 scripts/with-env.sh scripts/with-gpu.sh \
  python -m development.operators.check_deepseek_exact --output "$EXACT_REPORT"
```

## Engram blend

`load_engram_blend()` covers 1, 128, and 384 rows, plus synthetic decision cases.
Inputs contain BF16 residuals, four projected keys, a shared value, two normalization weights, and token masks.
The [boundary](../../python/oh_my_vllm/operators/engram/blend.py) normalizes each residual copy separately and applies the signed-square-root gate.
The [checker](../../development/operators/check_engram_blend.py) compares all gates and residuals, for tiny signed dots and masked rows.
Lookup and projection are separate boundaries.

## Hyper-connections

`load_hyperconnection()` covers decode, prefill, verification, zero, and cancellation cases with original F32 weights.
The [workspace](../../python/oh_my_vllm/operators/hyperconnection/workspace.py) separates preparation from expansion around the sublayer.
Collapse uses incoming coefficients. `next_pre` belongs to the next sublayer.
The [checker](../../development/operators/check_deepseek_hc.py) also permutes channels to detect addressing errors.
`Workspace.collapse()` supplies the final head without calculating new mixing coefficients.

## Routing

`load_routing()` starts with stored FP32 logits and original text/image biases.
Cases cover 384 experts with top-6 and 128 experts with top-3.
The [model contract](model.md) specifies unbiased coefficients and stable global expert IDs.
Expert IDs must match exactly. Non-tied selections must pass the index's score-interval checks.
Synthetic ties use identical logits and biases.

## Encoded projections

`load_projection()` supplies encoded activations, UE8M0 scales, and complete FP64 products.
Original weights stay external. The synthetic FP4 case includes its weights.
The [checker](../../development/operators/check_deepseek_projection.py) covers packing, K=576, and widths up to 15360.
Activation quantization and collective reduction are separate boundaries.

`load_rank_encoded_projection()` checks [quantization and projection](../../python/oh_my_vllm/models/deepseek_v41/linear.py) with original rank-owned `wq_a` weights.
Its FP64 products use quantized activations.
Reject nonzero quantization status before state commit.

`load_startup_promotion()` contains all finite FP8/UE8M0 pairs and all finite BF16 words.
It also contains overflow inputs, invalid BF16 words, and original and synthetic dense32 scales.
Conversion uses ties-to-even rounding, keeps signed zero, and rejects nonfinite outputs.
External generation metadata has SHA256 `b341dd5dcf26691c104757b29b7cd6d1ea1eecf0f5429f3f7638e5f29d7e0dcb`.

## Unquantized projections

`load_unquantized_projection()` supplies seven BF16/FP32 cases with complete FP64 matrix products.
The [workspace](../../python/oh_my_vllm/operators/linear/workspace.py) supports BF16-to-BF16, BF16-to-FP32, and FP32-to-FP32 products.
Grouped `wo_a` uses the full input and output row strides.
Router logits stay FP32. Accumulation uses FP32 without TF32 or intermediate BF16 conversion.
The [checker](../../development/operators/check_deepseek_linear.py) covers grouping, full vocabulary outputs, cancellation, and exact rounding decisions.

## Normalization, rotation, and activation

`load_norm_rope()` uses original BF16 RMS weights without an offset.
Text epsilon is `1e-20`. Vision epsilon is `1e-6`.
RoPE uses supplied FP32 cosine/sine coefficients for adjacent pairs in the last 64 channels. Prefix bits do not change.
Phase generation and two-dimensional vision positions are separate boundaries.

`load_swiglu()` supplies BF16 gate/up inputs and complete FP64 outputs.
The [API](../../python/oh_my_vllm/operators/activation/api.py) specifies text clamps and optional FP32 routing weights.
The [plan](../../python/oh_my_vllm/operators/activation/plan.py) specifies vision width.
The [checker](../../development/operators/check_deepseek_activation.py) applies fixed limits separately to each request.

## Ratio2 pooling

`load_pooling()` supplies complete FP32 KV and gate pairs with width 512.
For each channel, softmax acts across the pair. The weighted sum converts once to BF16 before RMS.
The fixed limits include rounding and cancellation allowances.
The [checker](../../development/operators/check_deepseek_pooling.py) excludes partial-group state, projection, RMS, and RoPE.

## Packed sparse attention

`load_sparse_attention()` supplies BF16 queries, original F32 sinks, encoded cache rows, and explicit slot IDs.
The [API](../../python/oh_my_vllm/operators/sparse_attention/api.py) defines formats, masks, repeated IDs, and error status.
Only selected cache tiles are decoded. The full logical cache is not expanded.
Reject nonzero status before state commit.
The [checker](../../development/operators/check_deepseek_sparse_attention.py) compares all outputs, exact zero guards, and invalid-ID behavior.

## Short text

The [reference guide](reference.md) specifies the equations and fixed 16-token prompt.
The artifact contains all 129280 prefill logits and all 129280 continuation logits.
The prefill argmax selects token 16 for continuation from the same state.
Weight decoding and model arithmetic use FP64 without activation quantization.

## DSpark continuation

`load_dspark()` supplies eighteen complete FP64 vocabulary rows and two five-value confidence vectors.
The 129-token prefix repeats the short prompt eight times, then adds its first token.
Arrays cover prefill, five proposals, six verification inputs, restored continuation, and five subsequent proposals.
The case selects two verification inputs, equivalent to one accepted draft, and independently recomputes that prefix without rejected state.

Subsequent proposals use only accepted features. This forced selection is not a sampled rejection event.
See the [construction procedure](reference.md#dspark-reference).

## Media endpoints

`load_media()` supplies complete FP64 prefill and continuation logits for one synthetic still image and two decoded video frames.
The 37-token placeholder prompt expands to 624 tokens. Token 9544 continues from accepted state.
Metadata binds the prompt, ordered images, patches, vision embeddings, and token map.
See the [reference procedure](reference.md#media-reference).

## Prompt codec

The [codec generator](../../development/model/deepseek_v41/codec_reference.py) executes verified local encoder and tokenizer bytes.
It disables automatic special-token insertion, padding, and truncation. The Python encoder supplies BOS where applicable.
Cases keep rendered text, complete IDs, and ordered media records. The generator does not fetch or preprocess images.

Cases cover chat, reasoning effort, reasoning removal and preservation, tools, system messages, assistant continuation, and images.
With explicit context, the encoder returns only new text and images. The caller keeps preceding context and media separately.
The identified Python encoder is authoritative. These fixtures do not prove Jinja equivalence.

### Tool-image order correction

`tool-result-images` records a source defect: images are collected before tool results are sorted into call order.
Reversed results misalign placeholders and images. This is diagnostic behavior.

`tool-result-images-canonical` sorts complete result records before encoding.
The service adapter must apply this order before media extraction and reject missing, duplicate, or unknown call identities.
Moving image records alone is not sufficient.

## Upstream checkpoint boundary

Use the local checkpoint for weights and references.
The [compared upstream checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) has matching layouts but different whole-file hashes for all 48 shards.
Do not substitute its weights or numerical results.

## Commands

Run the fixed artifact and generator checks:

```bash
scripts/with-env.sh python -m unittest \
  development.model.deepseek_v41.test_fixtures \
  development.model.deepseek_v41.test_reference_cases \
  development.model.deepseek_v41.test_codec_reference
```

Set `MODEL_ROOT`, `CHECKPOINT_MANIFEST`, and `CODEC_OUTPUT` to the local checkpoint, verified manifest, and new external JSON path.
Generate a codec artifact:

```bash
scripts/with-env.sh python -m development.model.deepseek_v41.codec_reference \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$CODEC_OUTPUT"
```

Review source changes and generated identities before a deliberate fixture replacement.
