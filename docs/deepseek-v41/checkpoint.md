# DeepSeek checkpoint verification

## Identity command

Use the development environment from the [development guide](../development.md).
Set `MODEL_ROOT` to the local checkpoint directory.
Set `CHECKPOINT_MANIFEST` to a new JSON file external to the repository and checkpoint directory.

```bash
scripts/with-env.sh python -m development.model.deepseek_v41.identity \
  --model "$MODEL_ROOT" --output "$CHECKPOINT_MANIFEST" --workers 4
```

The command checks the released Flash configuration and its expected tensor layout.
The layout comes from model dimensions and mathematical sources, not from the tensor index.
Unknown configuration fields and unsupported formats cause an error.

The command checks tensor names, dtypes, shapes, offsets, file lengths, and index coverage.
It rejects duplicate JSON keys, links, path escapes, overlapping storage, and unused payload bytes.
It detects file replacement or modification during reading.
The command does not load weights into GPU memory.

Full verification calculates SHA256 for all indexed weight bytes and control files.
It also binds the supplied prompt encoder and reference source files when present.
It reports separate target, vision, DSpark, and Engram payload totals.
These totals are checkpoint storage, not a runtime memory budget.

Use `--metadata-only` for a header-only weight check.
This mode still hashes control files.
Its manifest says `metadata-only` and has no weight digests.
Do not use this mode as full weight identity evidence.

The command does not overwrite an existing manifest.
Keep the full manifest and its digest with external numerical-reference artifacts.
Do not change checkpoint files during verification or reference generation.
The [phase plan](plan.md) controls subsequent reference and runtime integration.

## Reference weight reader

The development-only `Weights` reader must use a full manifest and its matching tensor layout.
It checks control files at construction and hashes each weight shard before its first use.
Later reads reject file replacement or modification.
These checks do not make concurrent checkpoint writes safe.

The reader loads one tensor or selected rows into CPU memory.
Engram tables must use explicit row indices to prevent an accidental full FP64 copy.
Repeated row indices keep their requested order.
Dense FP8 row selection uses the source rows' block scales.

The independent NumPy decoder supports BF16, F32, E4M3FN, UE8M0, and packed E2M1 weights.
It decodes the low FP4 nibble first and multiplies the corresponding UE8M0 scales.
It keeps signed zero and rejects nonfinite encodings.
Its output is FP64 mathematical weight data, without runtime activation rounding.

This reader does not execute the model or supply full-model numerical evidence.
Reference generation must use independent model mathematics and bound temporary weight storage.
Inference code must not import these development tools.

## Engram reference

The independent CPU Engram reference supplies token normalization, prime buckets, hash multipliers, and row indices.
It uses the local tokenizer without a Transformers tokenizer adapter.
Text tokens with the same normalized text share a compressed ID.
Partial UTF-8 tokens keep their raw token identity.
Empty text and one space do not merge.

The configured compressed vocabulary size and table row counts must match the derived values.
Prime buckets are disjoint across layers.
The multiplier generator uses the source's layer-specific PCG64 seeds.
Hash products must be in the signed 64-bit integer range.

`hash_rows` accepts compressed token IDs and returns row indices with continuation history.
The dead-token marker `-1` stops lookback through media spans.
Missing or blocked history uses the compressed padding ID.
The history keeps at most `max_ngram_size-1` IDs.
Dead-token markers are part of that history.
This reference does not allocate pinned storage or execute Engram projections.
