# DeepSeek fixed reference fixtures

## Scope and identity

The [fixture reader](../../development/model/deepseek_v41/fixtures.py) binds fixed artifacts by SHA256.
Ordinary checks load them without PyTorch, GPU execution, or checkpoint access.
They are reference inputs and expected outputs, not service performance results.
Do not replace them with runtime outputs.
Model and operator numerical limits are separate gates in the [validation plan](validation.md).

| Fixture | Content | SHA256 |
|---|---|---|
| [Short text](../../development/model/deepseek_v41/reference.npz) | Two full FP64 vocabulary vectors | `3bda8a9f03c797e65222a6eb85c82b758636514591f70f855e8f691a1f1fead0` |
| [Prompt codec](../../development/model/deepseek_v41/codec.json) | Twenty encoding cases with complete token IDs | `30a3079291e9326961f3d79937fe25bac3235f49ad79c250edc6763619fe7a75` |
| [DSpark continuation](../../development/model/deepseek_v41/dspark.npz) | Eighteen full FP64 vocabulary rows and two confidence vectors | `c78d89d66d2ff38999af2aa44b845c6d2fea1702280c217a92e7a0d693ee6fa8` |
| [Media endpoints](../../development/model/deepseek_v41/media.npz) | Two full FP64 vocabulary vectors after still and video-frame images | `a847254bb8a1b858d67eeb7405edb4a5435c24f2280585ed18f6b4b0eee86223` |
| [Exact operators](../../development/model/deepseek_v41/exact-operators.npz) | Four synthetic quantization and Engram cases | `a422c1f4cadeeb4ef918a18253834eae848c32b5173b0b88398e9bf0ede23e79` |
| [Engram lookup](../../development/model/deepseek_v41/engram-lookup.npz) | FP8 table, scales, indices, and 49152 BF16 output bit patterns | `930cee947ae1ee8bcdd3d15571732713305d849c18df8c4d98418dc309e40c2c` |
| [Engram checkpoint](../../development/model/deepseek_v41/engram-checkpoint.npz) | Original bytes and exact BF16 outputs for 32 selected rows from each table | `967109d7d713191212dc37fc18335c926fa3e2420a79e9fe03f5f2bbe365efce` |
| [Hyper-connection index](../../development/model/deepseek_v41/hyperconnection/index.json) | Five cases with complete FP64 outputs and fixed field limits | `933323bf315a75806e9dc8f66b0da47777622cd36c8b98f262f29b3978afdb47` |
| [Routing index](../../development/model/deepseek_v41/routing/index.json) | Five post-logit cases with exact expert IDs and FP64 coefficients | `bbb8538a912bede95a6d4364f52a63a026b04385de2e34cad5030109d305bf28` |
| [Projection index](../../development/model/deepseek_v41/projection/index.json) | Five A8W8 and five A8W4 cases with complete FP64 outputs | `db9c81bb0dd4a2f10aa3fdd289349dfb9162ede223a7ceb5cbe0e80444646fc6` |

The artifacts bind checkpoint manifest SHA256 `0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67`.
Full-model artifacts contain the manifest, generator identities, fixed inputs, array shapes, and digests.
The exact-operator bundle binds the manifest digest.
Codec cases contain rendered text, complete token IDs, and media records.
Readers check the applicable metadata and complete outputs.
Normal tests never regenerate expected values.

## Exact operators

`load_exact_operators()` supplies four existing operator `Fixture` objects.
The cases cover FP8/UE8M0 group32, FP4/UE8M0 group32, FP4/E4M3 group16, and Engram chunk continuation with dead tokens.
Expected values come from hand-checked literals, not candidate outputs.
Compare encoded bytes, scale bytes, BF16 roundtrip bits, hashes, and history with each case's exact limits.
These synthetic cases do not set compound numerical or performance limits.

`load_engram_lookup()` supplies fixed inputs and independently computed BF16 output bits.
Its 16-row table covers repeated indices, signed zeros, subnormal ties, and distinct scale groups.

`load_engram_checkpoint()` supplies original table samples, row IDs, scales, and independently decoded outputs.
All selected values have exact BF16 representations.
The [checkpoint check](../../development/operators/check_engram_lookup.py) uses these fixed outputs against the full shared store.

The optional native modules supply quantization, integer Engram hashes, and lookup through a shared pinned-host mapping.
They do not supply service integration.
The quantization caller must reject any nonzero group status.
The lookup caller must reject any nonzero row status.
Engram returns private continuation history without changing its input history.

Set `EXACT_REPORT` to a new external JSON path.
Build the modules.
Run the shared Eager and CUDA Graph check:

```bash
scripts/with-env.sh python -m development.native_build \
  --module device --module quantization --module engram
OH_MY_VLLM_GPU_COUNT=1 scripts/with-env.sh scripts/with-gpu.sh \
  python -m development.operators.check_deepseek_exact --output "$EXACT_REPORT"
```

## Hyper-connections

`load_hyperconnection()` supplies decode, prefill, verification, zero, and cancellation cases.
They use original F32 weights, four residual copies, width 5120, and 1, 128, or 384 rows.
The index binds all arrays and the `deepseek-hc-precision-v1` policy.

| Output | Normalized L2 limit | Maximum absolute error |
|---|---:|---|
| `collapsed`, `expanded` | `2^-8` | `max(1e-7, 2^-8 * P)` |
| `next_pre` | `2^-16` | `max(1e-7, 2^-16 * P)` |

`P` is the FP64 peak for each field and request.
The RMS floor is `1e-6`. The near-zero absolute limit is `1e-6`.
These engineering limits do not prove full-model accuracy.

The [native workspace](../../python/oh_my_vllm/operators/hyperconnection/workspace.py) separates preparation from expansion around the sublayer.
Collapse uses incoming coefficients. `next_pre` belongs to the next sublayer.
The [checker](../../development/operators/check_deepseek_hc.py) also permutes channels to detect addressing errors.
It checks Eager and Graph execution, invalid arguments, empty inputs, and input immutability.
Performance limits and model integration are pending.

## Routing

`load_routing()` supplies target decode, prefill, verification, draft, and synthetic decision cases.
The boundary starts with stored FP32 logits. Original checkpoint biases control text and image selection.
The cases cover 384 experts with top-6 and 128 experts with top-3.
The [model contract](model.md) specifies unbiased coefficients and stable global expert IDs.

The coefficient normalized L2 limit is `2^-19`.
Its absolute-error limit is `max(1e-7, 2^-19 * P)`, with `P` equal to each request's FP64 peak.
The RMS floor and near-zero absolute limit are the same as for hyper-connections.
Expert IDs must match exactly.

The index binds the independent score-error policy.
Non-tied selection and order must pass its interval checks before implementation.
Synthetic ties use identical stored logits and biases.
This gate excludes projection rounding, expert dispatch, latency, and full-model accuracy.

## Encoded projections

`load_projection()` supplies encoded FP8 activations, UE8M0 scales, and complete FP64 matrix products.
Original checkpoint weights stay external. The synthetic FP4 case stores its weights.
The [checker](../../development/operators/check_deepseek_projection.py) covers weight packing, K=576, and widths up to 15360.

Each request uses normalized L2 `2^-7` and maximum absolute error `max(1e-7, 2^-6 * P)`.
`P` is that request's FP64 peak. The RMS floor and near-zero absolute limit are `1e-6`.
Constructed zero endpoints also have a `1e-6` absolute limit.
These engineering limits exclude activation quantization error, promoted weight formats, collectives, latency, and full-model accuracy.

## Short text

The [construction guide](reference.md) specifies the FP64 equations and fixed prompt.
The artifact contains all 129280 prefill logits and all 129280 continuation logits.
The prefill argmax selects token 16 as the fixed continuation input.
The continuation consumes the same prefill state, not a fresh model state.

Metadata records all 16 prompt token IDs, array hashes, checkpoint identity, source hashes, and the CPU execution settings.
Weight decoding and all subsequent model arithmetic use FP64 without activation quantization.
This artifact does not cover DSpark endpoints, media, long contexts, or native rounding.
Its existence does not prove runtime numerical acceptance.

## DSpark continuation

`load_dspark()` supplies eighteen full FP64 vocabulary rows and two five-value confidence vectors.
The 129-token prefix repeats the short prompt eight times, then adds its first token.
The arrays cover prefill, five proposals, six verification inputs, restored continuation, and five subsequent proposals.

The case selects two verification inputs, equivalent to one accepted draft.
It independently recomputes that prefix, without rejected state.
The next draft block uses only accepted features.
This forced selection is not a sampled rejection event.
The [construction guide](reference.md#dspark-reference) gives the generator and source checks.

## Media endpoints

`load_media()` supplies full FP64 prefill and continuation logits for one synthetic still image and two decoded video frames.
The 37-token placeholder prompt expands to 624 tokens.
The prefill argmax selects token 9544 for continuation from accepted state.
Metadata binds the complete prompt, ordered images, patches, vision embeddings, and token map.

The [media guide](media-reference.md) specifies generation.
This fixture does not prove decoder, service, performance, or capacity acceptance.

## Prompt codec

The [codec generator](../../development/model/deepseek_v41/codec_reference.py) executes verified local encoder and tokenizer bytes.
It checks input and source identities before exclusive publication, without hashing weight payloads again.
Automatic special-token insertion, padding, and truncation are disabled.
The Python encoder supplies BOS where applicable.
Cases store inputs, rendered text, complete IDs, and ordered media records.
The generator does not fetch or preprocess images.

The cases cover chat, effort aliases and numbers, reasoning removal, reasoning preservation, tools, system messages, assistant continuation, and images.
Tool conversations keep earlier reasoning as specified by the local rule.
With explicit context, the encoder returns only the continuation text and its new images.
The caller keeps the preceding context and media separately.
Jinja equivalence is not proved by these fixtures.
The identified Python encoder stays authoritative.

### Tool-image order correction

`tool-result-images` records a source defect: images are collected before tool results are sorted into call order.
Reversed results then misalign placeholders and images.
This is diagnostic behavior, not approved service behavior.

`tool-result-images-canonical` orders complete result records before encoding and keeps the correct image order.
The service adapter must apply this ordering before media extraction.
It must reject missing, duplicate, or unknown call identities.
Moving image records alone is not sufficient.

## Upstream checkpoint boundary

Use the local checkpoint for weights and references.
The compared [upstream checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170) has matching layouts but different whole-file hashes for all 48 shards.
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
Do not regenerate fixtures as part of numerical acceptance.
