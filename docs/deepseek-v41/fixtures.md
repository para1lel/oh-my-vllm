# DeepSeek fixed reference fixtures

## Scope and identity

The [fixture reader](../../development/model/deepseek_v41/fixtures.py) binds two fixed artifacts by SHA256.
Ordinary checks load them without PyTorch, GPU execution, or checkpoint access.
They are reference inputs and expected outputs, not service performance results.
Do not replace them with runtime outputs.
Model and operator numerical limits are separate gates in the [validation plan](validation.md).

| Fixture | Content | SHA256 |
|---|---|---|
| [Short text](../../development/model/deepseek_v41/reference.npz) | Two full FP64 vocabulary vectors | `3bda8a9f03c797e65222a6eb85c82b758636514591f70f855e8f691a1f1fead0` |
| [Prompt codec](../../development/model/deepseek_v41/codec.json) | Twenty encoding cases with complete token IDs | `30a3079291e9326961f3d79937fe25bac3235f49ad79c250edc6763619fe7a75` |

The two artifacts bind checkpoint manifest SHA256 `0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67`.
That manifest records full identities for the 48 local weight shards and identified control files.
The artifacts also record their generator sources and dependency versions.
The fixed artifact hash binds those records to their outputs.
Subsequent source changes do not regenerate expected values during tests.

## Short text

The [construction guide](reference.md) specifies the FP64 equations and fixed prompt.
The artifact contains all 129280 prefill logits and all 129280 continuation logits.
The prefill argmax selects token 16 as the fixed continuation input.
The continuation consumes the same prefill state, not a fresh model state.

Metadata records all 16 prompt token IDs, array hashes, checkpoint identity, source hashes, and the CPU execution settings.
Weight decoding and all subsequent model arithmetic use FP64 without activation quantization.
This artifact does not cover DSpark endpoints, media, long contexts, or native rounding.
Its existence does not prove runtime numerical acceptance.

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
