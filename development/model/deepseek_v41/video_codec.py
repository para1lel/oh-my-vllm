"""Bounded video prompt-codec construction and standard-library fixture loading."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

VIDEO_CODEC_PATH = Path(__file__).with_name("video_codec.json")
VIDEO_CODEC_SHA256 = "21553a1d0c51f5361a202ed34048c8d4b39fab2cb8a64856699c653439975f7b"
CHECKPOINT_MANIFEST_SHA256 = (
    "0f42736d2943ec4725cb33d9e860a10ed994ba1d33e50d79c191847d3facea67"
)
CHECKPOINT_FILES = ("encoding/encoding.py", "tokenizer.json", "tokenizer_config.json")
CASE_NAMES = (
    "mixed-still-video",
    "rounded-timestamp-ties",
    "context-video-continuation",
)
IMAGE = "<\uff5cdeepseek_image\uff5c>"
MAX_ARTIFACT_BYTES = 1 << 20
MAX_CASE_INPUT_BYTES = 32768
MAX_CASE_TEXT_BYTES = 65536
MAX_CASE_TOKENS = 4096
MAX_TOTAL_TOKENS = 8192
TOKENIZATION = {
    "loader": "Tokenizer.from_buffer(verified_bytes)",
    "add_special_tokens": False,
    "padding": False,
    "truncation": False,
}


def identity(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def fixture_cases():
    from .reference_video import VideoMetadata, sample_plan, video_records

    def video(frame_count, fps, prefix):
        metadata = VideoMetadata(frame_count, fps)
        selected = sample_plan(metadata)
        frames = [
            (index, {"type": "image", "url": f"fixture://{prefix}-frame-{index}"})
            for index in selected
        ]
        return list(video_records(metadata, frames)), {
            "frame_count": metadata.frame_count,
            "fps": metadata.fps,
            "selected_indices": list(selected),
        }

    mixed_blocks, mixed_video = video(2, 2, "mixed")
    tied_blocks, tied_video = video(4, 100, "tied")
    mixed = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Before the still."},
                {"type": "image", "url": "fixture://still-before"},
                {"type": "text", "text": "Before the video."},
                *mixed_blocks,
                {"type": "text", "text": "After the video."},
                {"type": "image", "url": "fixture://still-after"},
                {"type": "text", "text": "After the still."},
            ],
        }
    ]
    tied = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Before the tied frames."},
                *tied_blocks,
                {"type": "text", "text": "After the tied frames."},
            ],
        }
    ]
    history = [*mixed, {"role": "assistant", "content": "I saw the earlier images."}]
    result = []
    for name, messages, metadata, context in (
        (CASE_NAMES[0], mixed, mixed_video, None),
        (CASE_NAMES[1], tied, tied_video, None),
        (CASE_NAMES[2], tied, tied_video, history),
    ):
        result.append(
            {
                "name": name,
                "inputs": {
                    "messages": messages,
                    "thinking_mode": "chat",
                    "context": context,
                    "drop_thinking": True,
                    "add_default_bos_token": True,
                    "reasoning_effort": None,
                    "return_multi_modal_data": True,
                },
                "video": metadata,
                "expected_images": [
                    block
                    for message in messages
                    for block in message["content"]
                    if block["type"] == "image"
                ],
            }
        )
    return copy.deepcopy(result)


def case_manifest(cases):
    return [
        {key: case[key] for key in ("name", "inputs", "video", "expected_images")}
        for case in cases
    ]


def validate_cases(cases):
    if not isinstance(cases, list) or [case["name"] for case in cases] != list(
        CASE_NAMES
    ):
        raise ValueError("video-codec case names or count differ")
    total = 0
    for case in cases:
        if (
            len(json.dumps(case["inputs"], allow_nan=False).encode())
            > MAX_CASE_INPUT_BYTES
        ):
            raise ValueError("video-codec inputs exceed their byte bound")
        text, tokens, images = (
            case["rendered_text"],
            case["token_ids"],
            case["expected_images"],
        )
        if not isinstance(text, str) or len(text.encode()) > MAX_CASE_TEXT_BYTES:
            raise ValueError("video-codec rendered text exceeds its byte bound")
        if (
            not isinstance(images, list)
            or not 1 <= len(images) <= 8
            or case["media"] != {"images": images}
            or text.count(IMAGE) != len(images)
        ):
            raise ValueError("video-codec ordered image records differ")
        if (
            not isinstance(tokens, list)
            or not 1 <= len(tokens) <= MAX_CASE_TOKENS
            or any(
                type(token) is not int or not 0 <= token < 129280 for token in tokens
            )
        ):
            raise ValueError("video-codec token IDs exceed their bound or are invalid")
        total += len(tokens)
    if total > MAX_TOTAL_TOKENS:
        raise ValueError("video-codec token IDs exceed their total bound")


def render_cases(encoder, tokenizer):
    cases = fixture_cases()
    for case in cases:
        text, media = encoder.encode_messages(**copy.deepcopy(case["inputs"]))
        if not isinstance(text, str) or len(text.encode()) > MAX_CASE_TEXT_BYTES:
            raise ValueError("video-codec rendered text exceeds its byte bound")
        case.update(
            rendered_text=text,
            token_ids=tokenizer.encode(text, add_special_tokens=False).ids,
            media=copy.deepcopy(media),
        )
    validate_cases(cases)
    return cases


def source_records(snapshots):
    from .codec_reference import read_tracked, reference_sources
    from .identity import local_file

    repository = Path(__file__).resolve().parents[3]
    sources = reference_sources(snapshots)
    for name in (
        "development/model/deepseek_v41/video_codec.py",
        "development/model/deepseek_v41/reference_video.py",
        "requirements/deepseek-media-reference.txt",
        "requirements/deepseek-media-reference.lock",
    ):
        sources[name] = read_tracked(local_file(repository, name), snapshots)[1]
    return sources


def generate(model_root, manifest_path, output):
    from .codec_reference import (
        TOKENIZERS_VERSION,
        check_unchanged,
        load_encoder,
        output_path,
        read_tracked,
    )
    from .identity import VERSION, local_file, parse_json

    root = Path(model_root).resolve(strict=True)
    output = output_path(output, root)
    snapshots = {}
    raw, manifest_identity = read_tracked(Path(manifest_path), snapshots)
    manifest = parse_json(raw)
    if (
        manifest.get("version") != VERSION
        or manifest.get("verification") != "full"
        or not isinstance(manifest.get("files"), dict)
    ):
        raise ValueError("video-codec generation requires a full checkpoint manifest")
    sources = source_records(snapshots)
    contents, verified = {}, {}
    for name in CHECKPOINT_FILES:
        data, actual = read_tracked(local_file(root, name), snapshots)
        if actual != manifest["files"].get(name):
            raise ValueError(f"video-codec checkpoint bytes differ: {name}")
        contents[name], verified[name] = data, actual
    config = parse_json(contents["tokenizer_config.json"])
    if (
        config.get("add_bos_token") is not False
        or config.get("add_eos_token") is not False
    ):
        raise ValueError("video-codec configuration must disable automatic BOS/EOS")
    if importlib.metadata.version("tokenizers") != TOKENIZERS_VERSION:
        raise ValueError("video-codec tokenizers dependency differs")

    import tokenizers

    if tokenizers.__version__ != TOKENIZERS_VERSION:
        raise ValueError("loaded video-codec tokenizers dependency differs")
    check_unchanged(snapshots)
    encoder = load_encoder(
        contents["encoding/encoding.py"], root / "encoding/encoding.py"
    )
    check_unchanged(snapshots)
    tokenizer = tokenizers.Tokenizer.from_buffer(contents["tokenizer.json"])
    tokenizer.no_padding()
    tokenizer.no_truncation()
    check_unchanged(snapshots)
    cases = render_cases(encoder, tokenizer)
    result = {
        "schema_version": 1,
        "reference_kind": "local-python-video-prompt-codec-cpu",
        "scope": "ordered-image-records-and-timestamp-text; no video decoding",
        "checkpoint_manifest": manifest,
        "checkpoint_manifest_sha256": manifest_identity["sha256"],
        "verification_scope": "prompt-codec-files-only; no weight payload rescan",
        "verified_checkpoint_files": verified,
        "reference_sources": sources,
        "reference_sources_sha256": identity(sources),
        "dependencies": {
            "python": sys.version,
            "python_implementation": platform.python_implementation(),
            "tokenizers": tokenizers.__version__,
            "numpy": importlib.metadata.version("numpy"),
        },
        "tokenization": dict(TOKENIZATION),
        "case_manifest_sha256": identity(case_manifest(cases)),
        "cases_sha256": identity(cases),
        "cases": cases,
    }
    encoded = (
        json.dumps(result, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise ValueError("video-codec artifact exceeds its byte bound")
    with NamedTemporaryFile(dir=output.parent) as target:
        target.write(encoded)
        target.flush()
        os.fsync(target.fileno())
        check_unchanged(snapshots)
        os.link(target.name, output)
    return result


def load_video_codec(path=VIDEO_CODEC_PATH):
    with Path(path).open("rb") as source:
        raw = source.read(MAX_ARTIFACT_BYTES + 1)
    if (
        len(raw) > MAX_ARTIFACT_BYTES
        or hashlib.sha256(raw).hexdigest() != VIDEO_CODEC_SHA256
    ):
        raise ValueError("frozen video-codec artifact identity differs")
    artifact = json.loads(raw)
    manifest = artifact["checkpoint_manifest"]
    encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    if (
        artifact["schema_version"] != 1
        or artifact["reference_kind"] != "local-python-video-prompt-codec-cpu"
        or artifact["checkpoint_manifest_sha256"] != CHECKPOINT_MANIFEST_SHA256
        or hashlib.sha256(encoded).hexdigest() != CHECKPOINT_MANIFEST_SHA256
        or manifest["verification"] != "full"
        or artifact["verified_checkpoint_files"]
        != {name: manifest["files"][name] for name in CHECKPOINT_FILES}
        or artifact["tokenization"] != TOKENIZATION
        or artifact["case_manifest_sha256"]
        != identity(case_manifest(artifact["cases"]))
        or artifact["cases_sha256"] != identity(artifact["cases"])
        or artifact["reference_sources_sha256"]
        != identity(artifact["reference_sources"])
    ):
        raise ValueError("video-codec inputs or historical provenance differ")
    validate_cases(artifact["cases"])
    return artifact


def main():
    parser = argparse.ArgumentParser(
        description="Generate bounded CPU video codec cases"
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", "--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = generate(args.model, args.manifest, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "case_count": len(result["cases"]),
                "token_count": sum(len(case["token_ids"]) for case in result["cases"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
