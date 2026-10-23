# DeepSeek reference construction

## Scope

The [phase plan](plan.md) controls reference construction before native operator tuning.
The tools supply checkpoint identities, decoded weights, Engram hashes, FP64 target and predictor mathematics, and a CPU text reference generator.
The [fixed fixtures](fixtures.md) supply short-text, DSpark restoration, media, and long-context endpoints, with prompt-codec cases.
No runtime numerical acceptance is declared.
The [checkpoint guide](checkpoint.md) specifies identity and row-loading checks.

Keep the reference separate from inference code and native operators.
Use the pinned environment from `requirements/deepseek-reference.lock`.
Its PyTorch and NumPy versions match `requirements/model-reference.lock`.
It also supplies the pinned tokenizer library for Engram normalization.
The inference runtime must not import PyTorch or these development modules.

## Arithmetic

Decode original BF16, FP8, and packed FP4 checkpoint values directly to FP64.
Multiply their original scales in FP64.
Use FP64 for projections, activations, normalization, residuals, attention, and persistent mathematical state.
Do not add runtime activation quantization or BF16/FP32 intermediate rounding to this reference.

The [mathematical functions](../../development/model/deepseek_v41/reference_math.py) implement hyper-connections, adjacent-pair rotary transforms, sparse attention sinks, and routing.
Hyper-connection residual coefficients have source-copy and destination-copy axes, in that order.
RMS weights multiply the normalized values without a Qwen-style weight offset.
Finite Sinkhorn iteration includes the source epsilon at each specified normalization.
The attention sink contributes to the softmax denominator but has zero value.

Exact expert-score ties select lower global expert IDs first.
The local `torch.topk` call does not specify a stable tie order.
This explicit project rule prevents rank placement from changing tied expert selection.
Correction bias selects experts.
Unbiased scores determine routing weights.

The [source contract](model.md) and [state contract](state.md) specify the other model equations.
Operator and full-model limits must be fixed before tuning.
These mathematical functions do not set or relax those limits.

## Source-precision quantization

The separate [quantization oracle](../../development/model/deepseek_v41/reference_quantization.py) checks activation and cache conversions with NumPy only.
It does not replace the FP64 mathematical reference or set compound acceptance limits.
`activation_fp8`, `indexer_fp4`, and `compressed_kv_fp4` accept FP32 arrays and first round inputs to BF16.
They return encoded bytes, scale bytes, and BF16 roundtrip values represented in FP32.

Groups contain 32, 32, and 16 channels, respectively.
UE8M0 scale calculation uses FP32 reciprocal multiplication.
E4M3 scale calculation uses FP32 division.

Conversion uses nearest rounding with ties to even.
Packed channel `2j` occupies the low nibble.
Nonfinite inputs, BF16 overflow, invalid shapes, and unsupported input dtypes cause errors.
The module records the released kernel source identity.

## Layer-streamed construction

Keep original packed weights in checkpoint files.
Load one projection or selected expert at a time through the checked weight reader.
Release temporary FP64 weight storage before the next projection.
Keep accepted activations and mathematical cache state separate from temporary weights.

Read embedding and Engram rows by their selected IDs.
Stream vocabulary-head rows in bounded pieces, then assemble all 129280 logits.
Do not substitute top-k logits for a full endpoint vector.
No full FP64 model copy is necessary.

Compute all 40 target layers for the short prefill reference.
Keep its FP64 state for the next decode token.
Use the highest prefill logit to select that fixed decode input.
Record the selected token with the two full endpoint vectors.
Draft, restoration, media, and long-context references are separate cases.

Bind each artifact to checkpoint, tokenizer, reference source, dependency versions, input tokens, and output array identities.
Reject existing output paths instead of overwriting reference artifacts.
Keep raw generation records external to the repository.
Add reviewed endpoint fixtures only after the generator and identities pass independent review.

## Short text generation

The [generator](../../development/model/deepseek_v41/reference.py) binds the fixed 16-token prompt to the local tokenizer.
It calculates the full checkpoint identity again before computation.
It checks checkpoint file states and reference source identities after computation.
Configuration parsing and tokenizer construction consume verified byte buffers, not new pathname reads.
Changed input files or source files cause an error instead of an output artifact.

The output contains two full FP64 vocabulary vectors and their metadata.
Metadata binds dependencies, reference sources, checkpoint identity, prompt tokens, the selected decode token, and array digests.
The command uses CPU execution only.
It does not generate DSpark, media, or long-context fixtures.

Set `REFERENCE_ENV` to a new environment directory external to the repository.
Set `MODEL_ROOT`, `CHECKPOINT_MANIFEST`, and `REFERENCE_OUTPUT` to the checkpoint, full manifest, and new external NPZ path.

```bash
scripts/with-env.sh uv venv "$REFERENCE_ENV"
scripts/with-env.sh uv pip sync --python "$REFERENCE_ENV/bin/python" \
  requirements/deepseek-reference.lock
scripts/with-env.sh "$REFERENCE_ENV/bin/python" \
  -m development.model.deepseek_v41.reference \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$REFERENCE_OUTPUT"
```

## Mathematical checks

Set `REFERENCE_PYTHON` to the Python executable in the pinned reference environment.
The following small checks use CPU tensors only:

```bash
scripts/with-env.sh "$REFERENCE_PYTHON" -m unittest \
  development.model.deepseek_v41.test_reference_math \
  development.model.deepseek_v41.test_reference_attention \
  development.model.deepseek_v41.test_reference_model \
  development.model.deepseek_v41.test_reference_media_model \
  development.model.deepseek_v41.test_reference_dspark \
  development.model.deepseek_v41.test_reference_vision
```

The checks use hand-calculated values for mixing direction, normalization, rotary pairs, attention sinks, and routing bias.
They are not full-model numerical evidence.
The normal CPU contract hook does not import PyTorch.
Use the [GPU execution rules](../execution.md) for later GPU reference generation.

## DSpark reference

The [predictor](../../development/model/deepseek_v41/reference_dspark.py) supplies three FP64 stages and five bidirectional draft rows.
`append_main` projects raw target features and updates only the three bounded main windows.
`forward` appends one explicit boundary row and returns full base logits, Markov-biased logits, selected IDs, and raw confidence.
Optional fixed proposal IDs permit conditional endpoint checks.
Default selection uses lowest-ID argmax, not stochastic service sampling.
Each call leaves its input state unchanged.

Predictor attention positions and output sample positions differ by one.
Use the [sampling contract](sampling.md) for pending tokens, verification, accepted feature updates, RNG, and exact correction.
The [case generator](../../development/model/deepseek_v41/reference_cases.py) constructs the [fixed restoration fixture](fixtures.md#dspark-continuation).
That fixture guide specifies its inputs, state selection, and complete outputs.
The generator checks consumed control bytes, file states, and source identities before exclusive publication.

Set `CASE_OUTPUT` to a new external NPZ path.
Use the pinned reference environment:

```bash
scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_cases \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$CASE_OUTPUT"
```

## Media reference

The [media generator](../../development/model/deepseek_v41/reference_media_case.py) supplies the [fixed media endpoints](fixtures.md#media-endpoints).
Its three synthetic RGB images measure 17 by 11, 11 by 17, and 13 by 13 pixels.
The first is a still image. The others represent frames at `<0.0 seconds>` and `<0.5 seconds>`.
This case does not test video decoding or frame extraction.

The [vision reference](../../development/model/deepseek_v41/reference_vision.py) includes learned start, newline, and end embeddings.
The target inserts complete image spans in prompt order before hyper-connection expansion.
Image positions receive no Engram contribution and use the visual routing bias.
Only the first prefill call inserts images.
Decode uses its accepted state without new images.
Metadata binds the PNG bytes, prompt, patches, vision embeddings, token map, and complete endpoint vectors.

Use the separate environment from `requirements/deepseek-media-reference.lock`.
Set `MEDIA_REFERENCE_PYTHON` to its Python executable.
Set `MEDIA_OUTPUT` to a new external NPZ path.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_media_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$MEDIA_OUTPUT"
```

The defaults use eight CPU threads and a projection-row limit of 2048.
The small `test_reference_media_case` checks do not regenerate full-model endpoints.

## Long-context reference

The [long-context generator](../../development/model/deepseek_v41/reference_long_case.py) repeats the short prompt to 65537 tokens.
It stores full prefill and decode vectors at 16385 and 65537 tokens.
These lengths cross the layer-20 candidate pool's 2048 blocks of eight positions and the original 65536-token rotary context.
They do not prove 1048576-token capacity or 64-request admission.

Contiguous prefill chunks contain at most 4096 tokens and end exactly at each endpoint.
All source KV, windows, partial compressors, and Engram history continue between chunks.
No position offset, history truncation, or approximate replay is used.
Each endpoint's argmax selects a decode token on a separate branch.
That branch does not become part of the next endpoint's prompt.
`load_long()` loads the four fixed FP64 arrays through the [fixture reader](../../development/model/deepseek_v41/fixtures.py).

Use the existing `REFERENCE_PYTHON`, model, and manifest variables.
Set `LONG_OUTPUT` and `LONG_STATE` to different external NPZ paths.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_long_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$LONG_OUTPUT" --checkpoint-state "$LONG_STATE" \
  --threads 8 --projection-row-limit 2048 --chunk-size 4096
```

These are the frozen artifact's settings. Different chunk sizes can change reduction order.
Each completed chunk and endpoint branch saves the complete continuation state and finished endpoints.
Identity changes, invalid shapes, and damaged data cause errors.
The tool synchronizes files before atomic replacement and keeps `<stem>.previous.npz`.
It does not fall back automatically.

Repeat the same command after interruption.
To select the previous state, set `LONG_STATE` to that path.
Without `--checkpoint-state`, no continuation state is saved.
State checkpoints are not numerical fixtures.
The small `test_reference_long_case` checks cover continuation, identities, corruption, and publication failures.

## Released-source baseline

The separate [capture tool](../../development/model/deepseek_v41/source_capture.py) records short-context FP32 logits from the verified released source.
It uses the fixed short prompt and continuation input.
The [loader](../../development/model/deepseek_v41/source_weights.py) reads bounded slices from the original checkpoint.
It does not create another converted checkpoint.
Model arguments come from verified root configuration bytes.
The capture records an in-memory correction that publishes an index owner's cache before each forward call.

This baseline uses released MP4 placement: owned experts, replicated shared experts and vision, and device-resident Engram shards.
It does not implement the framework's TP4, TEP4, DEP4, or pinned-host Engram contracts.
Its full-vector errors, cosine similarity, and top-1 agreement are diagnostics.
They do not set compound numerical limits or replace the FP64 reference.

Use a separate environment from `requirements/deepseek-baseline.lock`.
Set `BASELINE_ENV` to that external environment directory.
Set `BASELINE_OUTPUT` to a new external NPZ path.
Use the model and manifest variables from Short text generation.
Keep device selection settings in `LOCAL.md`.
Run the command through the four-GPU wrapper:

```bash
scripts/with-env.sh uv venv "$BASELINE_ENV"
scripts/with-env.sh uv pip sync --python "$BASELINE_ENV/bin/python" \
  requirements/deepseek-baseline.lock
PYTHONDONTWRITEBYTECODE=1 OH_MY_VLLM_GPU_COUNT=4 \
  scripts/with-env.sh scripts/with-gpu.sh \
  "$BASELINE_ENV/bin/python" -B -m torch.distributed.run \
  --standalone --nproc-per-node=4 \
  -m development.model.deepseek_v41.source_capture \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$BASELINE_OUTPUT" --run-gpu
```

Before publication, all four ranks must have identical full vectors.
Source, checkpoint, and environment identities must stay unchanged.
The JSON file contains provenance and diagnostics.
The NPZ file is the completion marker.
An existing output is never replaced.
