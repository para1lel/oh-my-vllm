# DeepSeek V4.1 video codec reference

## Status and source

This Phase 0 reference supplies CPU frame selection and ordered content blocks.
It does not decode video, execute the model, or add service video support.
It does not supply numerical model outputs or performance results.

The [model contract](model.md#image-and-video-boundary) identifies `LOCAL-V41:encoding/encoding.py` and the independent image encoder.
That local source accepts ordered image blocks.
It supplies no video frame sampler or timestamp protocol.
The rules below are an explicit service design decision for the planned DeepSeek extension.
They do not claim a local model source video contract.

The selection rule follows `video_media` in [the Qwen media source](../../../python/oh_my_vllm/models/qwen38/media.py).
DeepSeek uses each selected frame as one image.
It does not pair frames, repeat an odd final frame, or use a temporal vision encoder.
It does not use Qwen vision markers or three-axis language-model positions.

## Metadata and selection

The [pure reference](../../../development/model/deepseek_v41/reference_video.py) has three public API items:

- `VideoMetadata(frame_count, fps)` stores immutable, validated metadata.
- `sample_plan(metadata)` returns an immutable tuple of selected source frame indices.
- `video_records(metadata, frames)` returns ordered timestamp-text and image blocks for one video.

`frame_count` must be a positive Python integer at most `2**53`.
The last source index is at most `2**53-1`.
Each integer endpoint in this range has an exact float64 value.
`fps` must be a positive, finite Python integer or float that can convert to a finite float.
The constructor rejects booleans, strings, zero, negative values, NaN, and infinity.
The selection helper accepts only validated `VideoMetadata`.

The requested rate is approximately two frames per second.
The sample count is:

```python
count = min(frame_count, max(4, min(768, int(frame_count / fps * 2))))
```

The implementation applies the upper bound before integer conversion to avoid overflow for small positive rates.
This order gives the same count for finite calculations.

The selected indices follow this NumPy rule:

```python
np.linspace(0, frame_count - 1, count).round()
```

Indices start at zero.
NumPy rounds halfway values to the nearest even integer.
The plan keeps the first and last source frames.
Videos with fewer than four frames use all available frames.
The plan contains at most 768 frames.

## Ordered content blocks

`frames` is an ordered sequence of `(source_index, image_block)` pairs.
The record count and the complete index sequence must match `sample_plan(metadata)`.
Each index must be a Python integer.
Missing, extra, repeated, reversed, boolean, and float indices cause an error.
The helper does not sort records or discard records.

Each image block uses the local encoder's internal format:

```python
{"type": "image", "data": b"caller-supplied-frame-image"}
```

The local source also accepts a nonempty `source` or `url` field in this format.
These fields are caller-supplied image references.
The helper does not open paths, fetch URLs, decode bytes, or import the inference runtime.
It does not calculate byte sizes from opaque records.
It copies the image dictionary and keeps its field values.
It does not change the caller's records.

For each selected frame, the helper returns this text block immediately before its image block:

```python
{"type": "text", "text": f"<{source_index / fps:.1f} seconds>"}
```

The timestamp is a source-rate estimate from the frame index.
It is not a presentation timestamp or a variable-frame-rate time.
The expansion helper rejects a timestamp that is not finite.

Distinct frames can have equal timestamp text after rounding.
Such frames keep their source order.
Timestamp strings are ordinary text for the local prompt encoder.
This helper does not tokenize them.

The caller inserts the returned tuple at the video's original content position:

```python
content = [*before_blocks, *video_records(metadata, frames), *after_blocks]
message = {"role": "user", "content": content}
```

This insertion keeps the surrounding text and still-image order.
The local encoder then collects image records in prompt order.
The image processor applies the local resize, padding, normalization, patch order, and image-span rules.
Language-model positions follow the sequential token order.

## Resource limits and later work

The decoder boundary must reject an encoded video above 64 MiB before this helper runs.
The total original decoded sampled RGB bytes must not be more than 512 MiB.
The decoded-byte check occurs before stacking, resize, padding, or image preparation.
These checks are prerequisites for the future decoder.
This pure reference cannot enforce them through opaque image records.
Decoder integration must also bound its temporary storage and decode work.

Each frame keeps the local image-span limit of 1024 tokens.
The maximum video image-span budget is `768*1024 = 786432` tokens before timestamp text and other prompt content.
This finite upper bound does not guarantee request admission.
The service must later check the complete encoded prompt plus requested output against the 1048576-token context limit.
It must reject excess input without silent truncation.
Separate images and multiple videos contribute to the same request budget.

The [fixed media fixture](fixtures.md#media-endpoints) supplies full model endpoints for one still image and two synthetic decoded frames.
Video decoding, runtime allocation bounds, and service integration are later work.
The [behavior tests](../../../development/model/deepseek_v41/test_reference_video.py) check only selection, content order, and invalid inputs.

## Fixed fixtures

`load_video_codec()` reads [three fixed prompt cases](../../../development/model/deepseek_v41/video_codec.json) without a tokenizer or inference runtime.
The file SHA256 is `21553a1d0c51f5361a202ed34048c8d4b39fab2cb8a64856699c653439975f7b`.
It binds the [checkpoint identity](fixtures.md) and verified local encoder bytes.
The loader checks complete token sequences, source identities, and ordered media.
Historical generator identities can differ from later implementation files.

| Case | Required behavior | Tokens |
|---|---|---:|
| `mixed-still-video` | Keep still images before and after two selected frames in prompt order | 40 |
| `rounded-timestamp-ties` | Keep four distinct frames even when all timestamps format as `<0.0 seconds>` | 46 |
| `context-video-continuation` | Return only the new turn's text and images, without a new BOS token | 45 |

Each case has four image records and four placeholders with token ID 129264.
The continuation tokens equal the tied case after its BOS token.
Opaque image URLs are not opened. These cases do not decode video or calculate model endpoints.

Use `MEDIA_REFERENCE_PYTHON` from the [reference guide](reference.md#media-reference).
Set `VIDEO_CODEC_OUTPUT` to a new external JSON path.
Keep the fixed artifact unchanged during candidate acceptance.

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.video_codec \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$VIDEO_CODEC_OUTPUT"
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m unittest development.model.deepseek_v41.test_video_codec
```
