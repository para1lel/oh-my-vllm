import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..cases import load_cases
from . import fa, ffn, gdn, mtp, output, vision
from .common import Weights, arrays, tensor
from .decisions import MINIMUM_MARGIN, DecisionMargins
from .inputs import make
from .shards import write_fixture


def source_identity():
    root = Path(__file__).resolve().parent.parent
    repository = root.parents[1]
    files = [
        *root.joinpath("reference").glob("*.py"),
        root / "bindings.py",
        root / "cases.py",
        root / "fixtures.py",
        root / "storage.py",
        root.parents[0] / "model/reference.py",
        repository / "python/oh_my_vllm/worker/checkpoint.py",
        repository / "python/oh_my_vllm/models/qwen38/checkpoint.py",
        repository / "python/oh_my_vllm/operators/specification.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def generate(case, weights, root, provenance):
    weights.select(case)
    data = make(case, weights)
    groups = {
        group: {
            name: tensor(value, data.dtypes[group + "." + name])
            for name, value in getattr(data, group).items()
        }
        for group in ("inputs", "state")
    }
    family = case.boundary.split(".")[0]
    reference = {
        "fa": fa,
        "gdn": gdn,
        "ffn": ffn,
        "output": output,
        "mtp": mtp,
        "vision": vision,
    }[family].reference
    probability_cap = {"mtp.draft": 0.125, "output.sample": 0.0078125}.get(
        case.boundary
    )
    conditioning = {}
    options = {"conditioning": conditioning} if family == "gdn" else {}
    with DecisionMargins(probability_cap) as decisions:
        expected = arrays(
            reference(case, groups["inputs"], groups["state"], weights, **options)
        )
    provenance = {
        **provenance,
        "decisions": {
            "minimum_margin": MINIMUM_MARGIN,
            "records": decisions.records,
        },
    }
    if conditioning:
        provenance["conditioning"] = conditioning
    if "guards" in case.outputs:
        expected["guards"] = np.ones(len(data.state), dtype=bool)
    if set(expected) != set(case.outputs):
        difference = set(expected) ^ set(case.outputs)
        raise ValueError(f"reference outputs differ from the case: {difference}")
    if any(
        value.dtype.kind == "f" and not np.isfinite(value).all()
        for value in expected.values()
    ):
        raise ValueError("reference returned nonfinite outputs")
    stored = {
        f"{group}.{name}": value
        for group in ("inputs", "state")
        for name, value in getattr(data, group).items()
    }
    stored.update({"expected." + name: value for name, value in expected.items()})
    dtypes = {
        **data.dtypes,
        **{"expected." + name: str(value.dtype) for name, value in expected.items()},
    }
    return write_fixture(
        root / case.id, case, stored, dtypes, data.writes, weights.identity, provenance
    )


def main():
    cases = [case for case in load_cases() if case.purpose == "numerics"]
    parser = argparse.ArgumentParser(
        description="Generate fixed direct PyTorch operator fixtures"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=[case.id for case in cases], action="append")
    args = parser.parse_args()
    root = args.output.resolve()
    repository = Path(__file__).resolve().parents[3]
    if root == repository or repository in root.parents:
        raise ValueError("generate fixtures in a new external staging directory")
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    provenance = {
        "precision": "torch.float64",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "numpy": np.__version__,
        "source": source_identity(),
    }
    identities = {}
    with torch.inference_mode():
        weights = Weights(args.model)
        for case in cases:
            if args.case is None or case.id in args.case:
                identities[case.id] = generate(case, weights, root, provenance)
                print(f"generated {case.id}", flush=True)
        weights.checkpoint.verify_identity()
    if source_identity() != provenance["source"]:
        raise ValueError("reference source changed during generation")
    with (root / "index.json").open("x") as destination:
        json.dump(identities, destination, indent=2)


if __name__ == "__main__":
    main()
