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

DeepInfer revision `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` defaults to the `-1017` checkpoint.
The comparison uses [checkpoint revision 1bae0e1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
Its configuration, index bytes, and shard sizes match locally, but all 48 whole-file hashes differ.

Equal layouts do not prove equal tensor values.
Different file hashes do not identify which values differ.
Use the local checkpoint for weights and references.
Upstream substitution and numerical-result reuse are not approved.
Keep comparison records external to the repository.

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
