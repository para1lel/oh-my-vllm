import hashlib
import json
from pathlib import Path

import numpy as np

BASELINE_SHA256 = "bb9558a2181a2f6095aef113f1defe4ab7ed7e36091883447230b27ab7e86bc3"
REFERENCE_PATH = Path(__file__).with_name("reference.npz")
LIMITS = dict(max_nrmse=0.05, min_cosine=0.999, max_abs=0.5)
CASE_LIMITS = {
    "short": LIMITS,
    "mtp-restored": dict(max_nrmse=0.09, min_cosine=0.996, max_abs=1.6),
    "image-video": dict(max_nrmse=0.06, min_cosine=0.998, max_abs=1.3),
    "context-32769": dict(max_nrmse=0.07, min_cosine=0.998, max_abs=1.1),
}


def difference(actual, reference, *, case_id="short"):
    limits = CASE_LIMITS[case_id]
    actual, reference = (
        np.asarray(value, dtype=np.float64) for value in (actual, reference)
    )
    if any(
        value.shape != (248320,) or not np.isfinite(value).all()
        for value in (actual, reference)
    ):
        raise ValueError("logits must contain the full finite vocabulary")
    delta = actual - reference
    norm = float(np.linalg.norm(reference))
    actual_norm = float(np.linalg.norm(actual))
    if norm == 0 or actual_norm == 0:
        raise ValueError("logits must have nonzero norms")
    metrics = dict(
        nrmse=float(np.linalg.norm(delta) / norm),
        max_abs=float(np.max(np.abs(delta))),
        cosine=float(np.clip(np.dot(actual, reference) / (norm * actual_norm), -1, 1)),
        actual_top1=int(np.argmax(actual)),
        reference_top1=int(np.argmax(reference)),
        top1_equal=bool(np.argmax(actual) == np.argmax(reference)),
    )
    metrics["passed"] = (
        metrics["nrmse"] <= limits["max_nrmse"]
        and metrics["cosine"] >= limits["min_cosine"]
        and metrics["max_abs"] <= limits["max_abs"]
    )
    return metrics


def validate_case(actual, reference, checkpoint, reference_checkpoint):
    if actual != reference or actual.get("iterations") != 1:
        raise ValueError("full-model input differs from the frozen case")
    if checkpoint != reference_checkpoint:
        raise ValueError("checkpoint differs from the frozen reference")


def compare_endpoints(case, actual, reference):
    for outputs in (actual, reference):
        if set(outputs) != set(case.outputs):
            raise ValueError("full-model endpoint fields differ from the fixed case")
        if any(
            value.shape != case.output_shapes[name] for name, value in outputs.items()
        ):
            raise ValueError("full-model endpoint rows differ from the fixed case")
    return {
        name: [
            difference(candidate, expected, case_id=case.id)
            for candidate, expected in zip(
                actual[name].reshape(-1, 248320),
                reference[name].reshape(-1, 248320),
                strict=True,
            )
        ]
        for name in case.outputs
    }


def load_reference(path=REFERENCE_PATH):
    with path.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != BASELINE_SHA256:
            raise ValueError("reference artifact differs from the frozen identity")
        source.seek(0)
        with np.load(source, allow_pickle=False) as data:
            return json.loads(str(data["metadata"])), {
                name: data[name].copy() for name in ("prefill", "decode")
            }
