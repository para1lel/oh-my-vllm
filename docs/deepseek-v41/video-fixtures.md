# DeepSeek fixed video prompt fixtures

## Scope

The [video codec module](../../development/model/deepseek_v41/video_codec.py) supplies three supplemental prompt cases.
The [original twenty codec cases](fixtures.md#prompt-codec) stay unchanged.
These cases check text, token IDs, and image-record order after the [video helper](video.md) expands selected frames.
Still images and frame images use opaque fixture URLs.
No image bytes are opened.

These fixtures do not decode video, process image pixels, expand image spans, or calculate model outputs.
They do not prove service, numerical, performance, or capacity acceptance.
Decoder integration, long-context references, and numerical limits are pending.

## Fixed identity

The [frozen artifact](../../development/model/deepseek_v41/video_codec.json) contains 24704 bytes.
Its SHA256 is `21553a1d0c51f5361a202ed34048c8d4b39fab2cb8a64856699c653439975f7b`.
It uses schema 1 and reference kind `local-python-video-prompt-codec-cpu`.
It binds full checkpoint manifest SHA256 `0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67`.

Generation verifies consumed encoder, tokenizer, and tokenizer-configuration bytes against that manifest.
It executes the verified encoder bytes and constructs the tokenizer with `Tokenizer.from_buffer(verified_bytes)`.
It disables automatic special-token insertion, padding, and truncation.
The source encoder supplies BOS for requests without context.
Generation does not scan weight payloads again.

The artifact stores the full manifest, consumed-file identities, eight historical source identities, and dependency versions.
It also stores the full inputs, video metadata, selected indices, expected image order, rendered text, token IDs, and returned media.
Three JSON hashes bind the case inputs, full case outputs, and source records.
These hashes use sorted keys, separators `(",", ":")`, `ensure_ascii=True`, and `allow_nan=False`, without a final newline.

`load_video_codec()` uses only the Python standard library.
It does a check of the fixed file hash before it reads JSON.
It does checks of the embedded manifest, consumed-file bindings, JSON hashes, tokenization options, case bounds, and ordered media.
It does not import the generator helpers, tokenizer, NumPy, reference computation, or inference runtime.
Historical source records identify construction and are not compared with future implementation files.
Changes to the loader's fixed file hash do not change the recorded historical generator identity.

## Cases

Each case contains four image records and four image-placeholder tokens.
The image-placeholder token ID is 129264.
The three full token sequences contain 131 tokens in total.

| Case | Content and media order | Tokens |
|---|---|---:|
| `mixed-still-video` | Text, still image, text, two timestamp/frame pairs, text, still image, text | 40 |
| `rounded-timestamp-ties` | Text, four timestamp/frame pairs, text | 46 |
| `context-video-continuation` | The tied-frame prompt after the mixed prompt and its assistant reply | 45 |

The mixed case selects source frames 0 and 1 at two frames per second.
Its timestamps are `<0.0 seconds>` and `<0.5 seconds>`.
The returned image order is `still-before`, `mixed-frame-0`, `mixed-frame-1`, and `still-after`, with the `fixture://` URL prefix.

The tied case selects all four source frames at 100 frames per second.
Indices 0, 1, 2, and 3 all format as `<0.0 seconds>`.
The four distinct records keep their source order.
Their URL names are `tied-frame-0` through `tied-frame-3`.
Equal timestamp text does not merge frame records.

The continuation case keeps the earlier mixed prompt and assistant reply in explicit context.
Rendered text and returned images contain only the new turn.
Its tokens equal the tied case's tokens after removal of the initial BOS token.
The caller keeps earlier text and media separately.

## Construction and checks

Use the environment from `requirements/deepseek-media-reference.lock`.
Set `MEDIA_REFERENCE_PYTHON` to its Python executable.
Set `MODEL_ROOT` to the local checkpoint directory.
Set `CHECKPOINT_MANIFEST` to the full manifest identified above.
Set `VIDEO_CODEC_OUTPUT` to a new external JSON path.
Generate the supplemental artifact:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m development.model.deepseek_v41.video_codec \
  --model "$MODEL_ROOT" --checkpoint-manifest "$CHECKPOINT_MANIFEST" \
  --output "$VIDEO_CODEC_OUTPUT"
```

The fixed case set accepts no external case inputs.
Each input is at most 32768 JSON bytes.
Each rendered text is at most 65536 UTF-8 bytes.
Each case has at most 4096 tokens and eight image records.
The total token bound is 8192, and the artifact bound is 1 MiB.
Consumed files use the shared codec reader's 64 MiB bound.

Generation does checks of consumed files and recorded sources for changes before publication.
It writes and flushes a temporary file in the output directory.
An exclusive link publishes the full file without replacement.
An existing or competing output stays unchanged.
The temporary file is removed after success or failure.
Copy reviewed external bytes into the repository only for a deliberate fixture update.

Run the focused checks:

```bash
CUDA_VISIBLE_DEVICES= scripts/with-env.sh "$MEDIA_REFERENCE_PYTHON" \
  -m unittest development.model.deepseek_v41.test_video_codec
```

The checks include fixed values, image order, timestamp ties, context, verified-byte generation, corruption, mutation, bounds, and output ownership.
They do not run a decoder or a model.
