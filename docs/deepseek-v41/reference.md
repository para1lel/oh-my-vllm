# DeepSeek FP64 reference construction

## Scope

The [phase plan](plan.md) controls reference construction before native operator tuning.
The tools supply checkpoint identities, decoded weights, Engram hashes, FP64 target and predictor mathematics, and a CPU text reference generator.
The [fixed fixtures](fixtures.md) supply short-text endpoints, DSpark restoration endpoints, and prompt-codec cases.
Media and long-context endpoint fixtures are not fixed at this time.
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
Its NumPy oracle does not use model weights or native operators.
These small CPU checks are not local-checkpoint DSpark endpoint fixtures.

## DSpark restoration generation

The [case generator](../../development/model/deepseek_v41/reference_cases.py) repeats the fixed short-prompt token IDs.
The prefix has 129 tokens.
This prefix crosses the 128-token window boundary and leaves a ratio-2 compressor group that is not full.
The target computes the prefix, then supplies features for a five-row DSpark proposal.
Verification consumes the pending token and all five proposals.

The reference separately computes the first two verification inputs from the original prefix state.
It checks their features and logits against the corresponding full verification rows.
Their target argmax supplies a fixed correction input.
This is a forced state-selection case, not a stochastic rejection event.
The [sampling oracle](sampling.md) checks rejection probabilities separately.

Only accepted target features update the predictor main windows before the next proposal.
The next boundary feature is appended once.
The target then computes the correction input from the selected state.
The artifact contains 18 full vocabulary rows across prefill, two proposal blocks, verification, and restored decode.
It also contains five raw confidence values for each proposal block.

The generator records checkpoint identity, source identity, dependency versions, all input IDs, and all output-array hashes.
Before publication, it checks consumed control bytes and input states.
An exclusive link publishes the complete external NPZ file without replacement.
The [fixed fixture](fixtures.md) stores this restoration case.

Set `CASE_OUTPUT` to a new external NPZ path.
Use the pinned reference environment:

```bash
scripts/with-env.sh "$REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_cases \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$CASE_OUTPUT"
```

## Vision mathematics

The [vision reference](../../development/model/deepseek_v41/reference_vision.py) accepts normalized FP64 patches or normalized FP64 RGB pixels with an explicit patch grid.
It calculates vision-layer outputs with split-half 2D RoPE and full bidirectional attention.
The aligner uses zero-padded 3-by-3 groups, channel-first gather order, and exact-erf GELU.
Learned start, newline, and end embeddings form the complete language-model image span.

Query tiles limit score storage without dropping keys.
Projection-row limits bound weight reads.
The reference uses CPU FP64 arithmetic without source BF16/FP32 rounding.
It does not include image loading, resizing, normalization, video frame selection, or language-model image insertion.
Small mathematical checks do not supply a fixed local-checkpoint media endpoint.

## Image preparation

The [image reference](../../development/model/deepseek_v41/reference_image.py) accepts encoded image bytes, not paths or URLs.
Its grid calculation uses the source minimum pixel area, patch rounding, token bound, and optional wide-image resize rule.
Pillow supplies the source RGB conversion, resize, and gray padding.
No EXIF rotation occurs.
The output has FP64 normalized patches and ordered start, image, newline, and end token types.
FP64 normalization does not include the source FP32 arithmetic or the final BF16 conversion.

The tool rejects encoded inputs above 64 MiB, decoded images above 67108864 pixels, and prepared FP64 arrays above 512 MiB.
These are reference resource bounds, not a new service media contract.
It does not execute the vision or language model.

Use a separate environment from `requirements/deepseek-media-reference.lock` for image and vision checks together.
Its versions match the text reference lock, with Pillow 12.3.0 added.
Do not change packages used by an active reference-generation process.
Set `MEDIA_REFERENCE_ENV` to a new external environment directory:

```bash
scripts/with-env.sh uv venv "$MEDIA_REFERENCE_ENV"
scripts/with-env.sh uv pip sync --python "$MEDIA_REFERENCE_ENV/bin/python" \
  requirements/deepseek-media-reference.lock
scripts/with-env.sh "$MEDIA_REFERENCE_ENV/bin/python" -m unittest \
  development.model.deepseek_v41.test_reference_image \
  development.model.deepseek_v41.test_reference_vision
```

## Media prompt expansion

The [prompt expansion helper](../../development/model/deepseek_v41/reference_media.py) accepts tokenizer IDs and ordered encoded image bytes.
Each image placeholder becomes one full source image span.
Each span position keeps the configured image token ID, with start, image, newline, and end types.
Text positions have type -1.
The helper keeps image order and records each span's start position.
It rejects image-count mismatches, missing vision support, invalid token IDs, and expanded prompts above the context capacity.

Combined patch storage has a 512 MiB limit for this reference helper.
This limit does not include temporary storage for the preparation of one image.
Prompt IDs, types, and patch arrays are read-only after expansion.
The helper does not run vision, insert embeddings, or reserve output tokens.
The caller must make sure that combined input/output length is not more than its limit before service admission.

## Language-model image insertion

The [target reference](../../development/model/deepseek_v41/reference_model.py) accepts `token_types` and `image_embeddings` for the first prefill chunk.
The caller supplies one finite CPU FP64 embedding row for each image-span position, in prompt order.
These rows include learned start, newline, and end embeddings from the vision reference.
They replace token embeddings before hyper-connection expansion.
Each image position must carry the configured image token ID.

Image positions become dead markers in Engram history and receive no Engram contribution.
Their expert selection uses the visual routing bias.
Subsequent text calls use the accepted state without image inputs.
The reference rejects image insertion during continuation, missing image rows, invalid types, and incorrect embedding arrays.
It does not change the caller's input arrays or the previous state.

The small composition check includes insertion, masks, routing, and text continuation.
It does not supply a fixed local-checkpoint media endpoint.
