"""Read the frozen DeepSeek reference and apply its fixed numerical limits."""

import hashlib
import json
from pathlib import Path

REFERENCE_VERSION = "deepseek-official-mp4-text-frozen-v1"
MANIFEST_SHA256 = "2c40ce592665c14a7334657be7fdcf88fd7ea29c9a276c6fa7b159f5cdd8bbe4"
ENDPOINT_SHA256 = "3089f8e3d0327a07f166c744b25d7751fc65175963fbbd588533d34643568093"
DRIVER_SHA256 = "c4ac8b03345dd465c1a20b34e7cfbfe2a4891b6bb99b8a920e4f05d89900c909"
LIMITS = dict(
    logits_relative_l2=0.01,
    hidden_relative_l2=0.01,
    greedy_ids="exact",
    confidence_atol=0.01,
)
CASE_NAMES = ("short", "window", "authentic", *(f"forced.{i}" for i in range(6)))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def endpoint_schema():
    schema = {}

    def add(label, shape, dtype="float32"):
        schema[label] = dict(shape=list(shape), dtype="torch." + dtype)

    def target(label):
        add(label + ".logits", (1, 129280))
        add(label + ".hidden", (1, 1, 15360), "bfloat16")
        add(label + ".argmax", (1,), "int64")

    def proposal(label):
        add(label + ".ids", (1, 6), "int64")
        add(label + ".logits", (1, 5, 129280))
        add(label + ".confidence", (1, 5))

    for endpoint in (1, 2, 3, 4):
        target(f"short.{endpoint}")
    for endpoint in (125, 126, 127, 128, 129, 130, 132):
        target(f"window.{endpoint}")
    add("authentic.base_logits", (1, 129280))
    add("authentic.base_hidden", (1, 1, 15360), "bfloat16")
    proposal("authentic")
    for name in CASE_NAMES[2:]:
        for stage in range(6):
            add(f"{name}.verify_hidden.{stage}", (1, 1, 15360), "bfloat16")
        add(name + ".verify_logits", (1, 6, 129280))
        add(name + ".accepted_logits", (1, 129280))
        add(name + ".accepted_hidden", (1, 1, 15360), "bfloat16")
        target(name + ".restored_probe")
        proposal(name + ".restored_spec")
    return schema


def validate_tensors(tensors, schema):
    import torch

    if tensors.keys() != schema.keys():
        raise ValueError(
            f"endpoint keys differ: missing={sorted(schema.keys() - tensors.keys())}, "
            f"extra={sorted(tensors.keys() - schema.keys())}"
        )
    for label, metadata in schema.items():
        value = tensors[label]
        if (
            list(value.shape) != metadata["shape"]
            or str(value.dtype) != metadata["dtype"]
        ):
            raise ValueError(f"endpoint metadata differs: {label}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"nonfinite reference endpoint: {label}")


def acceptance_record(chain, logits):
    predictions = logits.argmax(-1).reshape(-1).tolist()
    if len(chain) != 6 or len(predictions) != 6:
        raise ValueError("verification must contain one known token and five drafts")
    accepted = 0
    for draft, prediction in zip(chain[1:], predictions[:-1], strict=True):
        if draft != prediction:
            break
        accepted += 1
    return dict(
        accepted_drafts=accepted,
        accepted_endpoint=127 + accepted,
        correction=predictions[accepted],
        probe_endpoint=128 + accepted,
        emitted_ids=[*chain[1 : accepted + 1], predictions[accepted]],
    )


def validate_cases(manifest, tensors):
    cases = manifest["cases"]
    if tuple(case["name"] for case in cases) != CASE_NAMES:
        raise ValueError("reference case coverage differs from the original driver")
    for case in cases[2:]:
        actual = acceptance_record(
            case["chain"], tensors[case["name"] + ".verify_logits"]
        )
        if any(actual[key] != case[key] for key in actual if key != "emitted_ids"):
            raise ValueError(f"reference acceptance metadata differs: {case['name']}")
        if case["restore_oracle"] != "fresh-accepted-only-replay":
            raise ValueError(
                "reference restoration must use fresh accepted-only replay"
            )
    if [case["accepted_drafts"] for case in cases[3:]] != list(range(6)):
        raise ValueError("forced acceptance must cover zero through five")


def load_reference(root):
    from safetensors.torch import load_file

    root = Path(root)
    observed = {
        "manifest.json": digest(root / "manifest.json"),
        "endpoints.safetensors": digest(root / "endpoints.safetensors"),
    }
    if observed != {
        "manifest.json": MANIFEST_SHA256,
        "endpoints.safetensors": ENDPOINT_SHA256,
    }:
        raise ValueError(f"frozen reference SHA256 differs: {observed}")
    manifest = json.loads((root / "manifest.json").read_text())
    if (
        manifest["limits"] != LIMITS
        or manifest["driver_sha256"] != DRIVER_SHA256
        or manifest["fixture_sha256"] != ENDPOINT_SHA256
        or manifest["candidate_import"] is not None
        or manifest["complete"] is not True
        or manifest["tensors"] != endpoint_schema()
    ):
        raise ValueError("reference contract differs from the original driver")
    tensors = load_file(str(root / "endpoints.safetensors"), device="cpu")
    validate_tensors(tensors, manifest["tensors"])
    validate_cases(manifest, tensors)
    identity = dict(version=REFERENCE_VERSION, files=observed, limits=LIMITS)
    return manifest, tensors, identity


def compare_tensor(label, expected, actual):
    import torch

    result = dict(label=label, passed=False)
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return dict(result, reason="tensor metadata differs")
    if not expected.is_floating_point():
        return dict(result, metric="exact", passed=torch.equal(expected, actual))
    if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
        return dict(result, reason="nonfinite endpoint")
    delta = actual.double() - expected.double()
    if label.endswith(".confidence"):
        maximum = float(delta.abs().max())
        return dict(
            result,
            metric="maximum_absolute_error",
            value=maximum,
            limit=LIMITS["confidence_atol"],
            passed=maximum <= LIMITS["confidence_atol"],
        )
    width = expected.size(-1)
    energy = torch.linalg.vector_norm(expected.double().reshape(-1, width), dim=-1)
    error = torch.linalg.vector_norm(delta.reshape(-1, width), dim=-1)
    if bool(((energy == 0) & (error != 0)).any()):
        return dict(result, reason="nonzero error at a zero reference vector")
    relative = torch.where(energy == 0, 0, error / energy)
    maximum = float(relative.max())
    hidden = ".hidden" in label or "_hidden" in label
    limit = LIMITS["hidden_relative_l2" if hidden else "logits_relative_l2"]
    logits = label.endswith((".logits", "_logits"))
    greedy_equal = not logits or torch.equal(expected.argmax(-1), actual.argmax(-1))
    return dict(
        result,
        metric="maximum_endpoint_relative_l2",
        value=maximum,
        limit=limit,
        greedy_ids_equal=greedy_equal,
        passed=maximum <= limit and greedy_equal,
    )


def selected_schema(manifest, cases):
    return {
        label: metadata
        for label, metadata in manifest["tensors"].items()
        if any(label.startswith(name + ".") for name in cases)
    }
