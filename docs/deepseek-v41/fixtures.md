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

The artifacts bind checkpoint manifest SHA256 `0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67`.
That manifest records full identities for the 48 local weight shards and identified control files.
The artifacts also record their generator sources and dependency versions.
The fixed artifact hash binds those records to their outputs.
Subsequent source changes do not regenerate expected values during tests.

## Exact operators

`load_exact_operators()` supplies four existing operator `Fixture` objects.
The cases cover FP8/UE8M0 group32, FP4/UE8M0 group32, FP4/E4M3 group16, and Engram chunk continuation with dead tokens.
Expected values come from hand-checked literals, not candidate outputs.
Compare encoded bytes, scale bytes, BF16 roundtrip bits, hashes, and history with each case's exact limits.
These synthetic cases do not set compound numerical or performance limits.

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

`load_dspark()` loads the fixed 17870563-byte `dspark.npz` through NumPy only.
The file contains the validated external reference bytes.
It uses schema 1, case `dspark-restored`, and reference kind `independent-pytorch-fp64-no-activation-quantization`.
The loader compares the full file hash with its fixed identity before it reads array data.
It also validates the archived full checkpoint manifest, fixed inputs, state selection, shapes, finite FP64 values, and array metadata hashes.
Its shape contract is fixed separately from subsequent generator changes.

The input prefix has 129 tokens.
It repeats the 16-token short prompt eight times, then adds its first token.
The target prefill selects pending token 4787.
The first five proposals are `[13769, 46012, 54994, 1060, 270]`.
The six verification inputs are `[4787, 13769, 46012, 54994, 1060, 270]`.

| Array | Shape |
|---|---|
| `prefill` | `(129280,)` |
| `draft_logits` | `(5, 129280)` |
| `draft_confidence` | `(5,)` |
| `verify_logits` | `(6, 129280)` |
| `restored_logits` | `(129280,)` |
| `next_draft_logits` | `(5, 129280)` |
| `next_draft_confidence` | `(5,)` |

All eighteen logit rows contain the complete vocabulary.
The two confidence vectors each contain five raw FP64 values.
They are separate from the vocabulary logits.

The case forces a two-input prefix from verification, equivalent to one accepted draft.
It independently recomputes that prefix instead of using rejected verification state.
The target argmax supplies correction token 46012.
The next pending position is 131, with positions counted from zero.
The next five proposals are `[54994, 1060, 270, 41638, 6397]`.
The next draft block uses only accepted features, and the target continuation uses the selected state.

This case is not a sampled rejection event.
It supplies independent CPU FP64 reference values, not service DSpark execution.
Long-context endpoints and model-specific numerical thresholds are later gates.
Phase 0 is not complete.

The [construction guide](reference.md) gives the generator and its source checks.

## Media endpoints

`load_media()` loads the fixed 2012626-byte `media.npz` through NumPy only.
It uses schema 1, case `still-and-video-frames`, and reference kind `independent-pytorch-fp64-no-activation-quantization`.
The loader checks the full file hash before it reads arrays.
It also checks the embedded full checkpoint manifest, full input identity, historical source records, and compressed-token-map identity.

The input identity includes messages, encoder options, rendered text, PNG bytes, token IDs, token types, image grids, and patch identities.
It also includes the decode token, frame selection, and vision-embedding identities.
Its SHA256 is `968e863392fe7ec3531e4df6a7d43851a295f070c486fec80565abc63a7449bc`.
The 21 historical source records have SHA256 `c099eb57db022052006567e726dbd3cd4f56c5bebd35eaf7bce45a3ed2383dd7`.

These identities use UTF-8 JSON with sorted keys, separators `(",", ":")`, and `allow_nan=False`, without a final newline.
JSON uses `ensure_ascii=True`.

The 37-token placeholder prompt expands to 624 tokens.
Three vision embeddings have shapes `(200, 5120)`, `(206, 5120)`, and `(184, 5120)` in image order.
The prefill argmax selects decode token 9544.
Decode continues from the accepted prefill state.

| Array | Shape | SHA256 of little-endian FP64 bytes |
|---|---|---|
| `prefill` | `(129280,)` | `7edd90ed9122b9dab48dffbcc231fd06a284591bc4f04cfbf65ecf2384edfc38` |
| `decode` | `(129280,)` | `b97f1fd3e53a8e90145a5db261ffa44b78228beb427f337145836336422b6c31` |

The loader checks that all arrays are present, use FP64, and have finite values.
Their shape, dtype, and hash metadata must agree with the arrays.
The [media construction guide](media-reference.md) specifies the synthetic decoded still image and video frames.
This fixture does not prove decoder, service, performance, or capacity acceptance.
Long-context references and numerical limits are still pending.

## Prompt codec

The [codec generator](../../development/model/deepseek_v41/codec_reference.py) executes the verified local Python encoding source.
It verifies consumed encoder, tokenizer, and tokenizer-configuration bytes against the full checkpoint manifest first.
It does not calculate the weight payload hashes again.
It checks input and generator file states before output publication.
Publication cannot replace an existing output file.

Tokenization uses `Tokenizer.from_buffer` with the verified bytes, not a new pathname read.
Automatic special-token insertion, padding, and truncation are disabled.
The Python encoder supplies BOS where applicable.
Each case stores inputs, rendered text, complete IDs, and the returned ordered media records.
The generator does not fetch image data or execute image preprocessing.

The cases cover chat, effort aliases and numbers, reasoning removal, reasoning preservation, tools, system messages, assistant continuation, and images.
Tool conversations keep earlier reasoning as specified by the local rule.
With explicit context, the encoder returns only the continuation text and its new images.
The caller keeps the preceding context and media separately.
Jinja equivalence is not proved by these fixtures.
The identified Python encoder stays authoritative.

### Tool-image order correction

The raw `tool-result-images` case shows a local source defect.
The encoder collects images before it sorts tool results into call order.
For results supplied in reverse order, rendered placeholders and returned image records refer to different results.
This case records source behavior for diagnosis, not approved service behavior.

The `tool-result-images-canonical` case supplies complete tool-result records in call order before encoding.
It has the same rendered text and token IDs, with the correct image order.
The service adapter must normalize complete tool-result records before media extraction.
It must reject missing, duplicate, or unknown call identities instead of silently moving image records alone.
This correction changes source ordering, not language-model or image-model arithmetic.

## Upstream checkpoint boundary

DeepInfer revision `9bd9cf243fc53ef9afbf8f04d391d989117d6ab9` defaults to the `-1017` checkpoint.
The comparison uses [checkpoint revision 1bae0e1](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/tree/1bae0e1088993b9b82de08f59de5be5a34e42170).
Its configuration and tensor-index bytes match the local files.
The 48 shard sizes also match.
Each upstream Git LFS whole-file SHA256 differs from the corresponding local shard hash.

Equal layouts do not prove equal tensor values.
Different whole-file hashes also do not identify which tensor values differ.
No upstream checkpoint substitution or numerical-result reuse is approved.
Use the local checkpoint identities for all runtime weights and references.
Keep the upstream metadata and comparison records external to the repository.

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
