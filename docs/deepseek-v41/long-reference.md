# DeepSeek long-context reference generation

## Scope

The [generator](../../development/model/deepseek_v41/reference_long_case.py) supplies independent CPU FP64 reference construction at two prompt lengths.
It does not enable DeepSeek inference or prove service capacity.
The endpoint artifacts are not fixed at this time.
Numerical and performance limits are separate prerequisites in the [phase plan](plan.md).

The prompt repeats the 16 token IDs from the [short reference](reference.md#short-text-generation).
It uses 65537 tokens without a position offset, truncated history, or approximate window replay.
The first endpoint follows 16385 prompt tokens.
This length is more than the 2048 blocks of eight positions in the layer-20 candidate pool.

The second endpoint follows 65537 prompt tokens.
This length is more than the original 65536-token rotary context.
These selected boundaries do not prove 1048576-token service capacity or 64-request admission.

## Computation and state

The generator uses the existing target model and original quantized checkpoint values decoded to FP64.
It does not use native runtime operators or runtime activation quantization.
The [reference guide](reference.md) specifies the arithmetic and layer-streamed weight reads.

Prefill uses contiguous chunks of at most 2048 tokens by default.
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

The tool verifies the full checkpoint manifest before computation.
It reads configuration and tokenizer controls from verified byte buffers.
The short prompt must encode to its fixed 16 IDs.
The compressed-token vocabulary must agree with the checkpoint configuration.
The model context must have space for the prompt and one decode token.

The metadata includes all prompt IDs, endpoint lengths, decode tokens, full checkpoint identity, source identities, dependency versions, and token-map identity.
It also includes CPU settings, chunk size, generation time, and full array shapes, dtypes, and hashes.
Checkpoint file states and reference source identities must stay unchanged until publication.
The final check occurs after serialization and before an exclusive link publishes the artifact.
Existing output paths, missing arrays, wrong shapes, non-FP64 arrays, and nonfinite values cause errors.

Keep the source files and reference environment unchanged during generation.
Keep generation logs and unreviewed artifacts external to the repository.
Freeze endpoint fixtures only after independent identity and output review.
Normal acceptance must load fixed arrays instead of generating new expected outputs.

## Commands

Use the pinned environment from `requirements/deepseek-reference.lock`.
Set `REFERENCE_PYTHON` to its Python executable.
Set `MODEL_ROOT` to the checkpoint directory.
Set `CHECKPOINT_MANIFEST` to its full identity manifest.
Set `LONG_OUTPUT` to a new external NPZ path.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_long_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$LONG_OUTPUT"
```

Defaults are eight CPU threads, 2048 projection rows per weight read, and 2048 prompt rows per chunk.
The chunk-size option accepts integers from 1 through 4096.
Different chunk sizes can change floating-point reduction order.
Keep the recorded chunk size with the frozen outputs.

The small checks compare chunked mathematical endpoints with whole-prefix calls.
They also do checks of publication failure conditions.
They do not generate full-checkpoint outputs:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m unittest development.model.deepseek_v41.test_reference_long_case
```
