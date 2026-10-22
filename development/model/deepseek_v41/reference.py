import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from .identity import file_state, inspect_checkpoint, local_file, parse_json
from .layout import tensor_layout
from .weights import Weights

PROMPT = "The quick brown fox jumps over the lazy dog. Explain why the sky is blue"
PROMPT_IDS = (
    671,
    4787,
    13769,
    46012,
    54994,
    1060,
    270,
    41638,
    6397,
    16,
    28231,
    3939,
    270,
    12709,
    344,
    8295,
)
REFERENCE_FILES = (
    "identity.py",
    "layout.py",
    "weights.py",
    "quantization.py",
    "engram.py",
    "reference.py",
    "reference_math.py",
    "reference_attention.py",
    "reference_model.py",
)


def output_path(output, checkpoint_root):
    output = Path(output).resolve()
    repository = Path(__file__).resolve().parents[3]
    if output.is_relative_to(repository) or output.is_relative_to(
        Path(checkpoint_root).resolve()
    ):
        raise ValueError(
            "reference output must be external to repository and checkpoint"
        )
    if output.suffix != ".npz":
        raise ValueError("reference output must use the .npz suffix")
    if output.exists():
        raise FileExistsError("reference output already exists")
    return output


def write_reference(output, checkpoint_root, metadata, vectors):
    output = output_path(output, checkpoint_root)
    if set(vectors) != {"prefill", "decode"}:
        raise ValueError("short reference must contain prefill and decode vectors")
    for value in vectors.values():
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != (129280,)
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                "reference vectors must contain the full finite FP64 vocabulary"
            )
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with output.open("xb") as target:
        np.savez_compressed(target, metadata=np.asarray(encoded), **vectors)


def source_identity():
    repository = Path(__file__).resolve().parents[3]
    paths = [Path("development/model/deepseek_v41") / name for name in REFERENCE_FILES]
    paths.extend(
        Path("requirements") / name
        for name in (
            "model-reference.txt",
            "model-reference.lock",
            "deepseek-reference.txt",
            "deepseek-reference.lock",
        )
    )
    return {
        str(path): hashlib.sha256((repository / path).read_bytes()).hexdigest()
        for path in paths
    }


def file_snapshot(root, files):
    return {
        name: file_state(local_file(root, name).stat(follow_symlinks=False))
        for name in files
    }


def generate(
    model_root, manifest_path, output, *, threads=8, row_limit=2048, hash_workers=4
):
    root = Path(model_root).resolve(strict=True)
    output = output_path(output, root)
    if type(threads) is not int or not 1 <= threads <= 64:
        raise ValueError("reference thread count must be between one and 64")
    if type(row_limit) is not int or not 1 <= row_limit <= 4096:
        raise ValueError("reference projection row limit must be between one and 4096")
    sources = source_identity()
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
        "Checking full checkpoint identity before CPU reference generation", flush=True
    )
    if inspect_checkpoint(root, workers=hash_workers) != manifest:
        raise ValueError("current checkpoint differs from the supplied full identity")
    if file_snapshot(root, manifest["files"]) != snapshot:
        raise ValueError("checkpoint changed during reference startup")
    config = parse_json((root / "config.json").read_bytes())
    reader = Weights(root, manifest, tensor_layout(config))
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokens = tokenizer.encode(PROMPT, add_special_tokens=False).ids
    if tuple(tokens) != PROMPT_IDS:
        raise ValueError("reference prompt tokens differ from the fixed case")
    token_map, vocabulary_size = compressed_token_map(tokenizer)
    if vocabulary_size != config["text_config"]["engram_compressed_vocab_size"]:
        raise ValueError("compressed tokenizer vocabulary differs from checkpoint")
    weights = TorchWeights(reader, device="cpu", row_limit=row_limit)
    model = Model(weights, config, token_map)
    started = time.monotonic()
    print(
        f"Computing FP64 prefill through all target layers: {len(tokens)} tokens",
        flush=True,
    )
    prefill = model.forward(tokens)
    decode_token = int(prefill.logits[0].argmax())
    print(
        f"Computing FP64 continuation from accepted state: token {decode_token}",
        flush=True,
    )
    decoded = model.forward([decode_token], previous=prefill.state)
    vectors = {
        "prefill": prefill.logits[0].numpy().copy(),
        "decode": decoded.logits[0].numpy().copy(),
    }
    if (
        file_snapshot(root, manifest["files"]) != snapshot
        or source_identity() != sources
    ):
        raise ValueError("checkpoint or reference source changed during generation")
    metadata = {
        "schema_version": 1,
        "case": "short",
        "reference_kind": "independent-pytorch-fp64-no-activation-quantization",
        "checkpoint": manifest,
        "checkpoint_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "reference_sources": sources,
        "dependencies": {
            item.metadata["Name"]: item.version
            for item in importlib.metadata.distributions()
        },
        "tokens": tokens,
        "decode_token": decode_token,
        "token_map_sha256": hashlib.sha256(
            token_map.astype("<i8").tobytes()
        ).hexdigest(),
        "execution": {
            "device": "cpu",
            "threads": threads,
            "projection_row_limit": row_limit,
        },
        "generation_seconds": time.monotonic() - started,
        "arrays": {
            name: {
                "shape": list(value.shape),
                "dtype": "float64",
                "sha256": hashlib.sha256(value.astype("<f8").tobytes()).hexdigest(),
            }
            for name, value in vectors.items()
        },
    }
    write_reference(output, root, metadata, vectors)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "decode_token": decode_token,
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate the independent DeepSeek short FP64 reference on CPU"
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
