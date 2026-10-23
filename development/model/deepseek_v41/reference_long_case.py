"""Independent CPU FP64 endpoints across candidate and rotary boundaries."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
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
    target,
    tokens,
    *,
    checkpoints=CHECKPOINTS,
    chunk_size=2048,
    progress=None,
    resume=None,
    save_checkpoint=None,
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
    point = resume or {
        "arrays": {},
        "decode_tokens": {},
        "state": None,
        "cursor": 0,
        "last_logits": None,
        "next_endpoint": 0,
        "phase": "prefill",
        "generation_seconds": 0.0,
    }
    arrays, decode_tokens = point["arrays"], point["decode_tokens"]
    state, start = point["state"], point["cursor"]
    elapsed = point["generation_seconds"]
    started = time.monotonic()
    if resume is not None:
        notify(f"Resuming CPU FP64 at prompt row {start}: {point['phase']}")

    def save(endpoint, phase):
        if save_checkpoint is not None:
            save_checkpoint(
                {
                    "arrays": arrays,
                    "decode_tokens": decode_tokens,
                    "state": state,
                    "cursor": start,
                    "last_logits": last_logits,
                    "next_endpoint": endpoint,
                    "phase": phase,
                    "generation_seconds": elapsed + time.monotonic() - started,
                }
            )

    last_logits = point["last_logits"]
    for index in range(point["next_endpoint"], len(checkpoints)):
        length = checkpoints[index]
        while start < length:
            end = min(start + chunk_size, length)
            notify(f"Computing CPU FP64 prompt rows {start}:{end}/{len(tokens)}")
            output = target.forward(tokens[start:end], previous=state)
            state = output.state
            start = end
            last_logits = np.asarray(output.logits[0]).copy()
            del output
            save(index, "probe" if start == length else "prefill")
        token = int(last_logits.argmax())
        arrays[f"prefill_{length}"] = last_logits.copy()
        notify(f"Computing CPU FP64 branch at {length}: token {token}")
        decoded = target.forward([token], previous=state)
        arrays[f"decode_{length}"] = np.asarray(decoded.logits[0]).copy()
        decode_tokens[str(length)] = token
        # Probe decode is a branch, not part of the next checkpoint's prompt.
        del decoded
        save(index + 1, "complete" if index + 1 == len(checkpoints) else "prefill")
    return {
        "tokens": list(tokens),
        "checkpoints": list(checkpoints),
        "decode_tokens": decode_tokens,
    }, arrays


def case_sources():
    sources = source_identity()
    path = Path(__file__).resolve()
    for source in (path, path.with_name("reference_long_checkpoint.py")):
        sources[str(source.relative_to(path.parents[3]))] = hashlib.sha256(
            source.read_bytes()
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
    checkpoint_state=None,
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

    def validate_inputs():
        if (
            file_snapshot(root, manifest["files"]) != snapshot
            or case_sources() != sources
        ):
            raise ValueError("checkpoint or reference source changed during generation")

    dependencies = {
        item.metadata["Name"]: item.version
        for item in importlib.metadata.distributions()
    }
    execution = {
        "device": "cpu",
        "dtype": "float64",
        "threads": threads,
        "projection_row_limit": row_limit,
        "prefill_chunk_size": chunk_size,
        "hash_workers": hash_workers,
    }
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    token_map_digest = hashlib.sha256(token_map.astype("<i8").tobytes()).hexdigest()
    store, resume = None, None
    if checkpoint_state is not None:
        from .reference_long_checkpoint import Checkpoint

        identity = {
            "config": config,
            "tokens": list(tokens),
            "checkpoints": list(CHECKPOINTS),
            "checkpoint_manifest_sha256": manifest_digest,
            "reference_sources": sources,
            "dependencies": dependencies,
            "python": sys.version,
            "token_map_sha256": token_map_digest,
            "execution": execution,
        }
        store = Checkpoint(
            checkpoint_state,
            identity,
            checkpoint_root=root,
            final_output=output,
            validate_inputs=validate_inputs,
        )
        resume = store.load()
    validate_inputs()
    started = time.monotonic()
    inputs, arrays = long_endpoints(
        target,
        tokens,
        chunk_size=chunk_size,
        progress=lambda message: print(message, flush=True),
        resume=resume,
        save_checkpoint=None if store is None else store.save,
    )
    validate_inputs()
    metadata = {
        "schema_version": 1,
        "case": "long-context-boundaries",
        "reference_kind": "independent-pytorch-fp64-no-activation-quantization",
        "checkpoint": manifest,
        "checkpoint_manifest_sha256": manifest_digest,
        "reference_sources": sources,
        "dependencies": dependencies,
        "inputs": inputs,
        "token_map_sha256": token_map_digest,
        "execution": execution,
        "generation_seconds": (
            time.monotonic() - started if store is None else store.generation_seconds
        ),
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
    parser.add_argument("--checkpoint-state", type=Path)
    args = parser.parse_args()
    result = generate(
        args.model,
        args.checkpoint_manifest,
        args.output,
        threads=args.threads,
        row_limit=args.projection_row_limit,
        chunk_size=args.chunk_size,
        hash_workers=args.hash_workers,
        checkpoint_state=args.checkpoint_state,
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
