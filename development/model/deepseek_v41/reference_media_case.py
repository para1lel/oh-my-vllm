"""Independent CPU FP64 endpoints for one still image and two decoded frames."""

import argparse
import base64
import copy
import hashlib
import importlib.metadata
import io
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image

from .codec_reference import load_encoder
from .identity import inspect_checkpoint, parse_json
from .layout import tensor_layout
from .reference import control_bytes, file_snapshot, output_path, source_identity
from .reference_media import expand
from .reference_video import VideoMetadata, video_records
from .weights import Weights


def case_inputs():
    images = []
    for index, (width, height) in enumerate(((17, 11), (11, 17), (13, 13))):
        y, x = np.indices((height, width))
        pixels = np.stack(
            (
                (x * 17 + y * 11 + index * 53) % 256,
                (x * 7 + y * 29 + index * 71) % 256,
                (x * 31 + y * 3 + index * 97) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        encoded = io.BytesIO()
        Image.fromarray(pixels).save(encoded, format="PNG")
        raw = encoded.getvalue()
        images.append(
            {
                "url": f"fixture://media-{index}",
                "role": "still" if index == 0 else "decoded-video-frame",
                "width": width,
                "height": height,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "encoded_base64": base64.b64encode(raw).decode("ascii"),
            }
        )
    records = [{"type": "image", "url": image["url"]} for image in images]
    video = VideoMetadata(frame_count=2, fps=2)
    frames = video_records(video, tuple(enumerate(records[1:])))
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Compare this still image and these video frames.",
                    },
                    records[0],
                    *frames,
                    {"type": "text", "text": "Describe the changes in order."},
                ],
            }
        ],
        "images": images,
        "video": {
            **asdict(video),
            "selected_indices": [0, 1],
            "scope": "synthetic-decoded-frames; no video decoder",
        },
    }


def render_prompt(encoder, tokenizer, config):
    inputs = case_inputs()
    options = {
        "messages": inputs["messages"],
        "thinking_mode": "chat",
        "context": None,
        "drop_thinking": True,
        "add_default_bos_token": True,
        "reasoning_effort": None,
        "return_multi_modal_data": True,
    }
    text, media = encoder.encode_messages(**copy.deepcopy(options))
    expected = [{"type": "image", "url": image["url"]} for image in inputs["images"]]
    if media != {"images": expected}:
        raise ValueError("source encoder changed the fixed media order")
    tokens = tokenizer.encode(text, add_special_tokens=False).ids
    raw = [
        base64.b64decode(image["encoded_base64"], validate=True)
        for image in inputs["images"]
    ]
    prompt = expand(tokens, raw, config)
    if len(prompt.tokens) + 1 > config["text_config"]["max_position_embeddings"]:
        raise ValueError("media reference must reserve one decode position")
    metadata = {
        **inputs,
        "encoder_options": options,
        "rendered_text": text,
        "placeholder_tokens": tokens,
        "expanded_tokens": prompt.tokens.tolist(),
        "token_types": prompt.token_types.tolist(),
        "spans": [
            {
                "start": span.start,
                "grid": asdict(span.image.plan),
                "input_sha256": span.image.input_sha256,
                "patches": array_identity(span.image.patches),
            }
            for span in prompt.spans
        ],
    }
    return prompt, metadata


def array_identity(value):
    return {
        "shape": list(value.shape),
        "dtype": "float64",
        "sha256": hashlib.sha256(np.asarray(value).astype("<f8").tobytes()).hexdigest(),
    }


def media_endpoints(target, vision, prompt, *, progress=None):
    import torch

    notify = progress or (lambda _: None)
    embeddings = []
    for index, span in enumerate(prompt.spans):
        notify(f"Computing CPU FP64 vision image {index + 1}/{len(prompt.spans)}")
        grid = span.image.plan
        value = vision.forward(
            torch.from_numpy(span.image.patches.copy()), grid.patch_h, grid.patch_w
        )
        embeddings.append(value)
    notify(f"Computing CPU FP64 media prefill: {len(prompt.tokens)} tokens")
    prefill = target.forward(
        prompt.tokens,
        token_types=prompt.token_types,
        image_embeddings=torch.cat(embeddings),
    )
    token = int(prefill.logits[0].argmax())
    notify(f"Computing CPU FP64 media continuation: token {token}")
    decoded = target.forward([token], previous=prefill.state)
    return {
        "decode_token": token,
        "vision_embeddings": [array_identity(value) for value in embeddings],
    }, {
        "prefill": np.asarray(prefill.logits[0]).copy(),
        "decode": np.asarray(decoded.logits[0]).copy(),
    }


def case_sources():
    sources = source_identity()
    repository = Path(__file__).resolve().parents[3]
    paths = [
        Path("development/model/deepseek_v41") / name
        for name in (
            "reference_media_case.py",
            "codec_reference.py",
            "reference_image.py",
            "reference_media.py",
            "reference_video.py",
            "reference_vision.py",
        )
    ]
    paths.extend(
        Path("requirements") / name
        for name in (
            "deepseek-media-reference.txt",
            "deepseek-media-reference.lock",
        )
    )
    for path in paths:
        sources[str(path)] = hashlib.sha256(
            (repository / path).read_bytes()
        ).hexdigest()
    return sources


def write_case(output, checkpoint_root, metadata, arrays, *, validate_inputs=None):
    output = output_path(output, checkpoint_root)
    if set(arrays) != {"prefill", "decode"}:
        raise ValueError("media reference endpoints are incomplete")
    for value in arrays.values():
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != (129280,)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "media endpoints must contain full finite FP64 vocabulary rows"
            )
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with NamedTemporaryFile(dir=output.parent) as temporary:
        np.savez_compressed(temporary, metadata=np.asarray(encoded), **arrays)
        temporary.flush()
        if validate_inputs is not None:
            validate_inputs()
        os.link(temporary.name, output)


def generate(
    model_root, manifest_path, output, *, threads=8, row_limit=2048, hash_workers=4
):
    root = Path(model_root).resolve(strict=True)
    output = output_path(output, root)
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError("reference thread count must be between one and 64")
    if type(row_limit) is not int or not 1 <= row_limit <= 4096:
        raise ValueError("reference projection row limit must be between one and 4096")
    sources = case_sources()
    import torch
    from tokenizers import Tokenizer

    from .engram import compressed_token_map
    from .reference_model import Model, TorchWeights
    from .reference_vision import Vision

    for package, expected in (
        ("numpy", "2.5.4"),
        ("torch", "2.12.0"),
        ("tokenizers", "0.23.3"),
        ("pillow", "12.3.0"),
    ):
        if importlib.metadata.version(package).split("+")[0] != expected:
            raise ValueError(
                f"reference dependency differs from its pinned version: {package}"
            )
    torch.set_num_threads(threads)
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = parse_json(manifest_bytes)
    if manifest.get("verification") != "full" or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("reference generation must use a full checkpoint manifest")
    snapshot = file_snapshot(root, manifest["files"])
    print("Checking full checkpoint identity before CPU media reference", flush=True)
    if inspect_checkpoint(root, workers=hash_workers) != manifest:
        raise ValueError("current checkpoint differs from the supplied full identity")
    if file_snapshot(root, manifest["files"]) != snapshot:
        raise ValueError("checkpoint changed during reference startup")
    config = parse_json(control_bytes(root, "config.json", manifest))
    token_config = parse_json(control_bytes(root, "tokenizer_config.json", manifest))
    if (
        token_config.get("add_bos_token") is not False
        or token_config.get("add_eos_token") is not False
    ):
        raise ValueError("source tokenizer must disable automatic BOS/EOS")
    encoder = load_encoder(
        control_bytes(root, "encoding/encoding.py", manifest),
        root / "encoding/encoding.py",
    )
    tokenizer = Tokenizer.from_buffer(control_bytes(root, "tokenizer.json", manifest))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    prompt, inputs = render_prompt(encoder, tokenizer, config)
    token_map, size = compressed_token_map(tokenizer)
    if size != config["text_config"]["engram_compressed_vocab_size"]:
        raise ValueError("compressed tokenizer vocabulary differs from checkpoint")
    reader = Weights(root, manifest, tensor_layout(config))
    weights = TorchWeights(reader, device="cpu", row_limit=row_limit)
    target = Model(weights, config, token_map)
    vision = Vision(weights, config)
    started = time.monotonic()
    continuation, arrays = media_endpoints(
        target, vision, prompt, progress=lambda message: print(message, flush=True)
    )

    def validate_inputs():
        if (
            file_snapshot(root, manifest["files"]) != snapshot
            or case_sources() != sources
        ):
            raise ValueError("checkpoint or reference source changed during generation")

    validate_inputs()
    metadata = {
        "schema_version": 1,
        "case": "still-and-video-frames",
        "reference_kind": "independent-pytorch-fp64-no-activation-quantization",
        "checkpoint": manifest,
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reference_sources": sources,
        "dependencies": {
            item.metadata["Name"]: item.version
            for item in importlib.metadata.distributions()
        },
        "inputs": {**inputs, **continuation},
        "token_map_sha256": hashlib.sha256(
            token_map.astype("<i8").tobytes()
        ).hexdigest(),
        "execution": {
            "device": "cpu",
            "threads": threads,
            "projection_row_limit": row_limit,
        },
        "generation_seconds": time.monotonic() - started,
        "arrays": {name: array_identity(value) for name, value in arrays.items()},
    }
    write_case(output, root, metadata, arrays, validate_inputs=validate_inputs)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate CPU FP64 still-image and video-frame endpoints"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--row-limit", type=int, default=2048)
    parser.add_argument("--hash-workers", type=int, default=4)
    args = parser.parse_args()
    generate(
        args.model,
        args.checkpoint_manifest,
        args.output,
        threads=args.threads,
        row_limit=args.row_limit,
        hash_workers=args.hash_workers,
    )


if __name__ == "__main__":
    main()
