"""Independent CPU FP64 endpoints across candidate and rotary boundaries."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .identity import inspect_checkpoint, parse_json
from .layout import tensor_layout
from .reference import (
    PROMPT,
    PROMPT_IDS,
    control_bytes,
    file_snapshot,
    output_path,
    source_identity,
)
from .weights import Weights

CHECKPOINTS = (16385, 65537)
ARRAY_SHAPES = {
    f"{endpoint}_{length}": (129280,)
    for length in CHECKPOINTS
    for endpoint in ("prefill", "decode")
}


def long_endpoints(
    target, tokens, *, checkpoints=CHECKPOINTS, chunk_size=2048, progress=None
):
    if type(chunk_size) is not int or not 1 <= chunk_size <= 4096:
        raise ValueError("reference chunk size must be between one and 4096")
    if (
        not checkpoints
        or any(type(length) is not int or length < 1 for length in checkpoints)
        or list(checkpoints) != sorted(set(checkpoints))
        or checkpoints[-1] != len(tokens)
    ):
        raise ValueError(
            "reference checkpoints must increase to the full prompt length"
        )
    notify = progress or (lambda _: None)
    arrays, decode_tokens = {}, {}
    state, start = None, 0
    for length in checkpoints:
        while start < length:
            end = min(start + chunk_size, length)
            notify(f"Computing CPU FP64 prompt rows {start}:{end}/{len(tokens)}")
            output = target.forward(tokens[start:end], previous=state)
            state = output.state
            start = end
        token = int(output.logits[0].argmax())
        arrays[f"prefill_{length}"] = np.asarray(output.logits[0]).copy()
        notify(f"Computing CPU FP64 branch at {length}: token {token}")
        decoded = target.forward([token], previous=state)
        arrays[f"decode_{length}"] = np.asarray(decoded.logits[0]).copy()
        decode_tokens[str(length)] = token
        # Probe decode is a branch, not part of the next checkpoint's prompt.
        del decoded, output
    return {
        "tokens": list(tokens),
        "checkpoints": list(checkpoints),
        "decode_tokens": decode_tokens,
    }, arrays


def case_sources():
    sources = source_identity()
    path = Path(__file__).resolve()
    sources[str(path.relative_to(path.parents[3]))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    return sources


def write_case(output, checkpoint_root, metadata, arrays, *, validate_inputs):
    output = output_path(output, checkpoint_root)
    if set(arrays) != set(ARRAY_SHAPES):
        raise ValueError("long-context reference endpoints are incomplete")
    for name, shape in ARRAY_SHAPES.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != shape
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"long-context endpoint must contain full finite FP64 rows: {name}"
            )
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with NamedTemporaryFile(dir=output.parent) as temporary:
        np.savez_compressed(temporary, metadata=np.asarray(encoded), **arrays)
        temporary.flush()
        validate_inputs()
        os.link(temporary.name, output)


def generate(
    model_root,
    manifest_path,
    output,
    *,
    threads=8,
    row_limit=2048,
    chunk_size=2048,
    hash_workers=4,
):
    root = Path(model_root).resolve(strict=True)
    output = output_path(output, root)
    for name, value, maximum in (
        ("thread count", threads, 64),
        ("projection row limit", row_limit, 4096),
        ("chunk size", chunk_size, 4096),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"reference {name} must be between one and {maximum}")
    sources = case_sources()
    import torch
    from tokenizers import Tokenizer

    from .engram import compressed_token_map
    from .reference_model import Model, TorchWeights

    for package, expected in (
        ("numpy", "2.5.4"),
        ("torch", "2.12.0"),
        ("tokenizers", "0.23.3"),
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
    print(
        "Checking full checkpoint identity before CPU long-context reference",
        flush=True,
    )
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
    tokenizer = Tokenizer.from_buffer(control_bytes(root, "tokenizer.json", manifest))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tuple(tokenizer.encode(PROMPT, add_special_tokens=False).ids) != PROMPT_IDS:
        raise ValueError("reference prompt tokens differ from the fixed case")
    text = config["text_config"]
    if CHECKPOINTS[-1] + 1 > text["max_position_embeddings"]:
        raise ValueError("long-context reference must reserve one decode position")
    tokens = (
        PROMPT_IDS * ((CHECKPOINTS[-1] + len(PROMPT_IDS) - 1) // len(PROMPT_IDS))
    )[: CHECKPOINTS[-1]]
    token_map, size = compressed_token_map(tokenizer)
    if size != text["engram_compressed_vocab_size"]:
        raise ValueError("compressed tokenizer vocabulary differs from checkpoint")
    reader = Weights(root, manifest, tensor_layout(config))
    weights = TorchWeights(reader, device="cpu", row_limit=row_limit)
    target = Model(weights, config, token_map)
    started = time.monotonic()
    inputs, arrays = long_endpoints(
        target,
        tokens,
        chunk_size=chunk_size,
        progress=lambda message: print(message, flush=True),
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
        "case": "long-context-boundaries",
        "reference_kind": "independent-pytorch-fp64-no-activation-quantization",
        "checkpoint": manifest,
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reference_sources": sources,
        "dependencies": {
            item.metadata["Name"]: item.version
            for item in importlib.metadata.distributions()
        },
        "inputs": inputs,
        "token_map_sha256": hashlib.sha256(
            token_map.astype("<i8").tobytes()
        ).hexdigest(),
        "execution": {
            "device": "cpu",
            "threads": threads,
            "projection_row_limit": row_limit,
            "prefill_chunk_size": chunk_size,
        },
        "generation_seconds": time.monotonic() - started,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": "float64",
                "sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
            }
            for name, value in arrays.items()
        },
    }
    write_case(output, root, metadata, arrays, validate_inputs=validate_inputs)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate CPU FP64 long-context reference endpoints"
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--checkpoint-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--projection-row-limit", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--hash-workers", type=int, default=4)
    args = parser.parse_args()
    result = generate(
        args.model,
        args.checkpoint_manifest,
        args.output,
        threads=args.threads,
        row_limit=args.projection_row_limit,
        chunk_size=args.chunk_size,
        hash_workers=args.hash_workers,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "generation_seconds": result["generation_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
