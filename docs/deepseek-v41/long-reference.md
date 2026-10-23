# DeepSeek long-context reference

## Scope

`load_long()` in the [fixture reader](../../development/model/deepseek_v41/fixtures.py) loads four frozen CPU FP64 endpoint vectors.
The [fixture index](fixtures.md) gives their artifact identity.
Numerical limits and service capacity are separate prerequisites in the [phase plan](plan.md).

The prompt repeats the 16 token IDs from the [short reference](reference.md#short-text-generation).
It uses 65537 tokens without a position offset, truncated history, or approximate window replay.
The first endpoint follows 16385 prompt tokens.
This length is more than the 2048 blocks of eight positions in the layer-20 candidate pool.

The second endpoint follows 65537 prompt tokens.
This length is more than the original 65536-token rotary context.
These selected boundaries do not prove 1048576-token service capacity or 64-request admission.

## Computation and state

The [generator](../../development/model/deepseek_v41/reference_long_case.py) uses the target model and original quantized checkpoint values decoded to FP64.
It does not use native runtime operators or runtime activation quantization.
The [reference guide](reference.md) specifies the arithmetic and layer-streamed weight reads.

The frozen artifact uses contiguous prefill chunks of at most 4096 tokens.
The accepted state from each chunk supplies the next chunk's full continuation state.
The last chunk before an endpoint ends at that endpoint.
No token is skipped or repeated between chunks.
The tool keeps source KV, local windows, partial compressors, and Engram history through the target state.
Bounded chunks limit temporary row storage, not total prefix storage.

At each endpoint, the generator stores all 129280 prefill logits.
Their argmax selects one decode token.
A different decode branch starts from that endpoint's accepted state and supplies all 129280 continuation logits.
The branch does not become part of the prompt for the next endpoint.
The artifact contains four FP64 arrays: `prefill_16385`, `decode_16385`, `prefill_65537`, and `decode_65537`.

## Identity and publication

The metadata binds the full checkpoint manifest, 15 source identities, all prompt IDs, dependency versions, execution settings, and complete array hashes.
Checkpoint files, sources, and the reference environment must stay unchanged during generation.
The tool accepts only complete finite FP64 arrays and a new output path.

Keep generation logs and unreviewed artifacts external to the repository.
Normal acceptance must load fixed arrays instead of generating new expected outputs.

## Resume state

Each completed chunk and endpoint branch produces a checkpoint of the complete prompt continuation state.
The checkpoint also keeps completed endpoints and generation time.
Identity changes, invalid state shapes, and damaged data cause errors.

The tool completes file synchronization before atomic replacement.
`<stem>.previous.npz` keeps the previous valid generation.
The tool does not fall back automatically.
These checkpoint files are not endpoint fixtures.

## Commands

Use the pinned environment from `requirements/deepseek-reference.lock`.
Set `REFERENCE_PYTHON` to its Python executable.
Set `MODEL_ROOT` to the checkpoint directory.
Set `CHECKPOINT_MANIFEST` to its full identity manifest.
Set `LONG_OUTPUT` to a new external NPZ path.
Set `LONG_STATE` to a different external NPZ path.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_long_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$LONG_OUTPUT" --checkpoint-state "$LONG_STATE" \
  --threads 8 --projection-row-limit 2048 --chunk-size 4096
```

This command uses the frozen artifact's execution settings.
Different chunk sizes can change floating-point reduction order.

Repeat the same command after interruption.
An existing state file supplies the validated continuation.
To select the previous generation, set `LONG_STATE` to its path.
Without `--checkpoint-state`, the tool does not save continuation state.

The [small checks](../../development/model/deepseek_v41/test_reference_long_case.py) cover chunk and resume equivalence, identities, damaged checkpoints, and publication failures.
They do not regenerate full-checkpoint outputs.
