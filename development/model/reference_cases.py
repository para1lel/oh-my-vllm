import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from development.operators.reference.shards import write_fixture

from . import reference_layers as layers
from .capture import PROMPT
from .cases import load_cases
from .reference import Weights


def positions(length, start=0):
    return torch.arange(start, start + length, dtype=torch.int64, device="cuda")[
        None
    ].expand(3, -1)


def make_inputs(case, config):
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(case.id.encode()).digest()[:8])
    )
    tokens = np.resize(np.asarray(PROMPT, np.int64), case.prefix_length)
    if case.id == "context-32769":
        tokens[::97] = (
            np.arange(len(tokens[::97]), dtype=np.int64) * 313 + 760
        ) % 120000
    coordinates = np.tile(np.arange(case.prefix_length, dtype=np.int64), (3, 1))
    arrays = dict(
        tokens=tokens, positions=coordinates, continuation=np.array([271], np.int64)
    )
    if case.verification_rows:
        arrays.update(
            verification=np.array(PROMPT[1:6], np.int64),
            draft_token=np.array([3069], np.int64),
        )
    cursor = base = 0
    for media in case.media:
        _, shape = case.input_shapes[media.name + "_pixels"]
        arrays[media.name + "_pixels"] = rng.integers(0, 256, shape, dtype=np.uint8)
        height, width = (side // 2 for side in media.grid)
        for first, count, _ in media.spans:
            coordinates[:, cursor:first] = np.arange(first - cursor)[None] + base
            base += first - cursor
            coordinates[:, first : first + count] = (
                np.indices((1, height, width)).reshape(3, -1) + base
            )
            base += max(height, width)
            cursor = first + count
            tokens[first - 1] = config["vision_start_token_id"]
            tokens[first : first + count] = config[media.name + "_token_id"]
            tokens[first + count] = config["vision_end_token_id"]
    if case.media:
        coordinates[:, cursor:] = np.arange(case.prefix_length - cursor)[None] + base
    return arrays


def reference(case, arrays, weights):
    tokens = arrays["tokens"]
    coordinates = torch.from_numpy(arrays["positions"]).to("cuda")
    embedded = layers.embeddings(weights, tokens)
    for media in case.media:
        pixels = torch.from_numpy(arrays[media.name + "_pixels"]).to("cuda")
        encoded = layers.vision(weights, pixels, media.grid)
        for first, count, offset in media.spans:
            embedded[first : first + count] = encoded[offset : offset + count]
    prefill, hidden, states = layers.forward(
        weights, tokens, coordinates, embedded=embedded
    )
    continuation = arrays["continuation"]
    if case.verification_rows:
        tail = arrays["verification"]
        verified, _, _ = layers.forward(
            weights,
            tail,
            positions(len(tail), len(tokens)),
            states=states,
            start=len(tokens),
            project_all=True,
        )
        accepted = tail[: case.accepted_rows]
        _, accepted_hidden, selected = layers.forward(
            weights,
            accepted,
            positions(len(accepted), len(tokens)),
            states=states,
            start=len(tokens),
        )
        endpoint, _, _ = layers.forward(
            weights,
            continuation,
            positions(1, len(tokens) + len(accepted)),
            states=selected,
            start=len(tokens) + len(accepted),
        )
        history_hidden = torch.cat((hidden, accepted_hidden))
        history_embedding = torch.cat(
            (
                embedded[1:],
                layers.embeddings(weights, np.concatenate((accepted, continuation))),
            )
        )
        context_positions = positions(len(history_hidden), 1)
        outputs = dict(verified=verified, restored=endpoint)
    else:
        next_position = int(coordinates.max()) + 1
        endpoint, _, _ = layers.forward(
            weights,
            continuation,
            positions(1, next_position),
            states=states,
            start=len(tokens),
        )
        outputs = dict(prefill=prefill, decode=endpoint)
        history_hidden = hidden
        history_embedding = torch.cat(
            (embedded[1:], layers.embeddings(weights, continuation))
        )
        context_positions = torch.cat(
            (coordinates[:, 1:], positions(1, next_position)), dim=1
        )
    if "mtp_first" in case.outputs:
        projected, cache = layers.mtp_context(
            weights, history_hidden, history_embedding, context_positions
        )
        first, draft_hidden, cache = layers.mtp_draft(
            weights, projected, cache, context_positions[:, -1:], first=True
        )
        outputs["mtp_first"] = first
        if "mtp_next" in case.outputs:
            projected = layers.mtp_input(
                weights, draft_hidden, layers.embeddings(weights, arrays["draft_token"])
            )
            following, _, _ = layers.mtp_draft(
                weights, projected, cache, context_positions[:, -1:] + 1, first=False
            )
            outputs["mtp_next"] = following
    return {name: value.cpu().numpy() for name, value in outputs.items()}


def source_identity():
    root = Path(__file__).resolve().parents[2]
    model = Path(__file__).parent
    files = [
        model / name
        for name in (
            "capture.py",
            "cases.py",
            "reference.py",
            "reference_cases.py",
            "reference_layers.py",
        )
    ]
    files += list((root / "development/operators/reference").glob("*.py"))
    files += [
        root / name
        for name in (
            "development/operators/cases.py",
            "development/operators/fixtures.py",
            "development/operators/storage.py",
            "python/oh_my_vllm/worker/checkpoint.py",
            "python/oh_my_vllm/models/qwen38/checkpoint.py",
            "python/oh_my_vllm/operators/specification.py",
        )
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    cases = load_cases()
    parser = argparse.ArgumentParser(
        description="Generate added full-model PyTorch FP64 fixtures"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=[case.id for case in cases], action="append")
    args = parser.parse_args()
    root = args.output.resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("use a new external staging directory")
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    config = json.loads((args.model / "config.json").read_text())
    provenance = dict(
        precision="torch.float64",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        numpy=np.__version__,
        source=source_identity(),
    )
    identities = {}
    with torch.inference_mode():
        for case in cases:
            if args.case is not None and case.id not in args.case:
                continue
            weights = Weights(args.model)
            inputs = make_inputs(case, config)
            expected = reference(case, inputs, weights)
            if set(expected) != set(case.outputs) or any(
                value.shape != case.output_shapes[name] or not np.isfinite(value).all()
                for name, value in expected.items()
            ):
                raise ValueError(
                    "reference did not return every finite full-vocabulary endpoint"
                )
            arrays = {"inputs." + name: value for name, value in inputs.items()}
            arrays.update(
                {"expected." + name: value for name, value in expected.items()}
            )
            identities[case.id] = write_fixture(
                root / case.id,
                case,
                arrays,
                {name: str(value.dtype) for name, value in arrays.items()},
                {},
                weights.checkpoint.verify_identity(),
                provenance,
            )
            print(f"generated {case.id}", flush=True)
    if source_identity() != provenance["source"]:
        raise ValueError("full-model reference source changed during generation")
    with (root / "index.json").open("x") as output:
        json.dump(
            dict(version="full-model-fixtures-v1", fixtures=identities),
            output,
            indent=2,
        )
        output.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
