# DeepSeek media reference generation

## Scope

The [media generator](../../development/model/deepseek_v41/reference_media_case.py) calculates full CPU FP64 language-model endpoints after one still image and two video-frame images.
Its input frames are synthetic decoded images, not a compressed video file.
This case does not validate a video decoder, frame extraction, or service media handling.
The [video contract](video.md) specifies frame selection and timestamp blocks.

The generator supplies reference construction, not runtime acceptance.
No fixed media endpoint fixture is supplied yet.
The [reference guide](reference.md) specifies arithmetic, weight streaming, and the separate reference environment.

## Fixed input

Three nonuniform RGB images use sizes 17 by 11, 11 by 17, and 13 by 13 pixels.
Their PNG bytes are stored as base64 metadata with SHA256 identities.
The first image is a still image.
The other images are two synthetic video frames at two frames per second.
The timestamp blocks are `<0.0 seconds>` and `<0.5 seconds>`.

The verified local prompt encoder supplies the rendered text and ordered image records.
The generator rejects a different image order.
The tokenizer uses verified checkpoint bytes without automatic padding or truncation.
The image helper expands each placeholder into a full image span.
The prompt must leave one position for decode.

Metadata keeps the messages, encoder options, rendered text, placeholder IDs, expanded IDs, token types, image grids, and patch identities.
It also keeps the frame indices and the synthetic-frame limitation.

## Calculation and publication

The vision reference calculates all three image spans in prompt order.
It includes learned start, newline, and end embeddings.
The target reference inserts these embeddings in the first prefill call.
The highest prefill logit selects one decode input.
Decode continues from the accepted prefill state without new image inputs.
Each endpoint contains all 129280 FP64 logits.

The generator recalculates the full checkpoint identity before computation.
It checks each control file that it uses against that identity.
It records reference-source hashes, dependency versions, the tokenizer map identity, vision-embedding identities, and endpoint identities.
Checkpoint file states and reference-source identities must stay unchanged until publication.
An exclusive link publishes a full external NPZ file without replacement.
Invalid endpoint shapes, nonfinite values, and existing output paths cause errors.

Do not change the reference source files or environment during generation.
Keep raw artifacts external to the repository.
Add a fixed fixture only after independent identity and output review.

## Command

Use the separate environment from `requirements/deepseek-media-reference.lock`.
Set `MEDIA_REFERENCE_PYTHON` to its Python executable.
Set `MODEL_ROOT` to the local checkpoint directory.
Set `CHECKPOINT_MANIFEST` to its full identity manifest.
Set `MEDIA_OUTPUT` to a new external NPZ path.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.reference_media_case \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$MEDIA_OUTPUT"
```

The command uses CPU execution only.
Its default configuration uses eight CPU threads and a projection-row limit of 2048.
The small construction checks do not generate full-checkpoint endpoints:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m unittest development.model.deepseek_v41.test_reference_media_case
```
