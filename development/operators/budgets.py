import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from .cases import load_cases
from .fixtures import file_digest, load_fixture
from .numerics import sequences

BF16_UNIT = 2**-8
FP8_UNIT = 2**-4
VERSION = "operator-precision-v1"


def coefficients(case, field):
    if field == "selection_cdf":
        return 1.0, None
    if field in {"probabilities", "selected_probability"}:
        return (0.5, None) if case.boundary.startswith("mtp.") else (0.03125, None)
    if field == "residual":
        return BF16_UNIT, BF16_UNIT
    if field == "fp8_kv":
        return 2 * FP8_UNIT, 3 * FP8_UNIT
    if field in {"kv", "kv_scales", "convolution", "recurrent"}:
        return FP8_UNIT, 2 * FP8_UNIT
    if case.boundary == "output.hidden":
        return 2 * BF16_UNIT, 4 * BF16_UNIT
    if case.boundary.startswith("output.") or case.boundary == "mtp.context":
        return 4 * BF16_UNIT, 8 * BF16_UNIT
    if case.boundary.startswith("vision."):
        return 4 * BF16_UNIT, 8 * BF16_UNIT
    if case.boundary == "mtp.chain":
        return 2 * FP8_UNIT, 4 * FP8_UNIT
    if case.boundary == "mtp.draft":
        return FP8_UNIT + 4 * BF16_UNIT, 2 * FP8_UNIT
    if case.boundary == "ffn":
        return FP8_UNIT, 2 * FP8_UNIT
    return 0.1, 2 * FP8_UNIT


def propose(case, fixture):
    fields = {}
    probability_cap = 2 * (FP8_UNIT if case.boundary.startswith("mtp.") else BF16_UNIT)
    for field, value in fixture.expected.items():
        if value.dtype.kind != "f":
            fields[field] = {"kind": "exact"}
            continue
        relative, absolute = coefficients(case, field)
        maximum = (
            [
                max(1e-7, float(np.abs(part).max()) * absolute) if part.size else 1e-7
                for part in sequences(value, case.definition(), field)
            ]
            if absolute is not None
            else [probability_cap for _ in sequences(value, case.definition(), field)]
        )
        if case.boundary == "gdn" and field == "value":
            scales = np.asarray(
                fixture.manifest["reference"]["conditioning"][field], dtype=np.float64
            )
            if (
                scales.shape != (len(maximum), value.shape[-1])
                or not np.isfinite(scales).all()
                or np.any(scales < 0)
            ):
                raise ValueError("invalid reference projection conditioning scales")
            maximum = np.maximum(
                np.asarray(maximum)[:, None], scales * absolute
            ).tolist()
        fields[field] = {
            "kind": "float",
            "normalized_l2": relative,
            "max_abs": maximum,
            "rms_floor": 1e-6,
            "near_zero_max_abs": 1e-6,
        }
    result = {
        "case_sha256": case.identity,
        "fixture_sha256": case.fixture_sha256,
        "fields": fields,
    }
    if "probabilities" in fixture.expected:
        result["probability"] = {
            "total_variation": probability_cap,
            "softmax_logit_envelope": case.options.get("sampling") == "smooth",
        }
    if "selection_cdf" in fixture.expected:
        edges = fixture.expected["selection_cdf"]
        result["cdf_budgets"] = np.minimum(
            probability_cap, np.diff(edges, axis=-1)[:, 0] / 3
        ).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Propose fixed numerical accuracy limits"
    )
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    if (
        args.output.resolve() == repository
        or repository in args.output.resolve().parents
    ):
        raise ValueError("propose limits in a new external file for review")
    numerical = {}
    for case in load_cases():
        if case.purpose != "numerics":
            continue
        root = args.fixtures / case.id
        bound = replace(case, fixture_sha256=file_digest(root / "manifest.json"))
        numerical[case.id] = propose(bound, load_fixture(bound, root))
    payload = {
        "version": "operator-limits-v1",
        "precision_policy": {
            "version": VERSION,
            "source_sha256": file_digest(__file__),
        },
        "numerics": numerical,
        "performance": {},
    }
    with args.output.open("x") as destination:
        json.dump(payload, destination, indent=2, allow_nan=False)
        destination.write("\n")


if __name__ == "__main__":
    main()
