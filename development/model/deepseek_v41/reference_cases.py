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
    PROMPT_IDS,
    control_bytes,
    file_snapshot,
    output_path,
    source_identity,
)
from .weights import Weights

ARRAY_SHAPES = {
    "prefill": (129280,),
    "draft_logits": (5, 129280),
    "draft_confidence": (5,),
    "verify_logits": (6, 129280),
    "restored_logits": (129280,),
    "next_draft_logits": (5, 129280),
    "next_draft_confidence": (5,),
}


def write_case(output, checkpoint_root, metadata, arrays):
    output = output_path(output, checkpoint_root)
    if set(arrays) != set(ARRAY_SHAPES):
        raise ValueError("DSpark reference endpoints are incomplete")
    for name, shape in ARRAY_SHAPES.items():
        value = arrays[name]
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != np.float64
            or value.shape != shape
            or not np.isfinite(value).all()
        ):
            raise ValueError(
                f"reference endpoint must contain finite full FP64 rows: {name}"
            )
    encoded = json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    with NamedTemporaryFile(dir=output.parent) as temporary:
        np.savez_compressed(temporary, metadata=np.asarray(encoded), **arrays)
        temporary.flush()
        os.link(temporary.name, output)


def dspark_restored(target, predictor, tokens, *, progress=None):
    if len(tokens) < 2:
        raise ValueError(
            "DSpark restoration reference needs at least two prompt tokens"
        )
    notify = progress or (lambda _: None)
    length = len(tokens)
    notify(f"Computing target prefill: {length} tokens")
    prefill = target.forward(tokens)
    pending = int(prefill.logits[0].argmax())
    main = predictor.append_main(prefill.features[:-1])
    notify("Computing first five-row DSpark block")
    draft = predictor.forward(
        predictor.project_features(prefill.features[-1:]),
        start_pos=length - 1,
        input_token=pending,
        previous=main,
    )
    verification_tokens = [pending, *draft.proposal_tokens]
    notify("Computing six target verification rows")
    verified = target.forward(
        verification_tokens, previous=prefill.state, endpoint_rows=list(range(6))
    )
    # Recompute the selected prefix independently; rejected ring writes are not reused.
    notify("Computing independent two-input accepted prefix")
    accepted = target.forward(
        verification_tokens[:2], previous=prefill.state, endpoint_rows=[0, 1]
    )
    for full, selected in (
        (verified.logits[:2], accepted.logits),
        (verified.features[:2], accepted.features),
    ):
        if not np.allclose(
            np.asarray(full), np.asarray(selected), rtol=1e-12, atol=1e-12
        ):
            raise ValueError("reference verification differs from its accepted prefix")
    correction = int(accepted.logits[-1].argmax())
    notify("Computing next DSpark block from accepted features only")
    main = predictor.append_main(
        accepted.features[:1], previous=draft.state, start_pos=length
    )
    next_draft = predictor.forward(
        predictor.project_features(accepted.features[-1:]),
        start_pos=length + 1,
        input_token=correction,
        previous=main,
    )
    notify("Computing target continuation from the selected state")
    restored = target.forward([correction], previous=accepted.state)
    arrays = {
        "prefill": np.asarray(prefill.logits[0]).copy(),
        "draft_logits": np.asarray(draft.logits).copy(),
        "draft_confidence": np.asarray(draft.raw_confidence).copy(),
        "verify_logits": np.asarray(verified.logits).copy(),
        "restored_logits": np.asarray(restored.logits[0]).copy(),
        "next_draft_logits": np.asarray(next_draft.logits).copy(),
        "next_draft_confidence": np.asarray(next_draft.raw_confidence).copy(),
    }
    inputs = {
        "tokens": list(tokens),
        "pending_token": pending,
        "proposal_tokens": list(draft.proposal_tokens),
        "verification_tokens": verification_tokens,
        "accepted_input_rows": 2,
        "accepted_drafts": 1,
        "correction_token": correction,
        "next_pending_position": length + 2,
        "next_proposal_tokens": list(next_draft.proposal_tokens),
    }
    return inputs, arrays


def case_sources():
    sources = source_identity()
    repository = Path(__file__).resolve().parents[3]
    for name in ("reference_cases.py", "reference_dspark.py"):
        path = Path("development/model/deepseek_v41") / name
        sources[str(path)] = hashlib.sha256(
            (repository / path).read_bytes()
        ).hexdigest()
    return sources


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
    from .reference_dspark import DSpark
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
    print("Checking full checkpoint identity before CPU DSpark reference", flush=True)
    if inspect_checkpoint(root, workers=hash_workers) != manifest:
        raise ValueError("current checkpoint differs from the supplied full identity")
    if file_snapshot(root, manifest["files"]) != snapshot:
        raise ValueError("checkpoint changed during reference startup")
    config = parse_json(control_bytes(root, "config.json", manifest))
    tokenizer = Tokenizer.from_buffer(control_bytes(root, "tokenizer.json", manifest))
    token_map, vocabulary_size = compressed_token_map(tokenizer)
    if vocabulary_size != config["text_config"]["engram_compressed_vocab_size"]:
        raise ValueError("compressed tokenizer vocabulary differs from checkpoint")
    reader = Weights(root, manifest, tensor_layout(config))
    weights = TorchWeights(reader, device="cpu", row_limit=row_limit)
    target, predictor = Model(weights, config, token_map), DSpark(weights, config)
    started = time.monotonic()
    inputs, arrays = dspark_restored(
        target,
        predictor,
        list((PROMPT_IDS * 9)[:129]),
        progress=lambda message: print(message, flush=True),
    )
    if file_snapshot(root, manifest["files"]) != snapshot or case_sources() != sources:
        raise ValueError("checkpoint or reference source changed during generation")
    metadata = {
        "schema_version": 1,
        "case": "dspark-restored",
        "reference_kind": "independent-pytorch-fp64-no-activation-quantization",
        "state_selection": (
            "forced-two-input-prefix; correction is target argmax, "
            "not sampled rejection"
        ),
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
    write_case(output, root, metadata, arrays)
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
        description="Generate the CPU FP64 DSpark restoration reference"
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
