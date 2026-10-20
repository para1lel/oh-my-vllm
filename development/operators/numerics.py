import itertools
from dataclasses import dataclass

import numpy as np

from .cases import digest
from .storage import equal_bits


@dataclass(frozen=True)
class NumericalResult:
    metrics: dict
    failures: dict

    @property
    def passed(self):
        return not self.failures


def sequences(value, case, field):
    if case["boundary"] == "mtp.chain" and field in {
        "hidden",
        "logits",
        "tokens",
        "probabilities",
    }:
        value = np.swapaxes(value, 0, 1)
        return [np.asarray(stage)[None] for request in value for stage in request]
    starts = case["options"].get("sequence_rows")
    if starts is not None and field in {
        "value",
        "residual",
        "hidden",
        "logits",
        "probabilities",
        "tokens",
        "selected_probability",
        "selection_cdf",
    }:
        if starts[0] != 0 or starts[-1] != len(value):
            raise ValueError("numerical sequence rows differ from the output")
        return [value[first:last] for first, last in itertools.pairwise(starts)]
    count = case["shape"].get("batch", len(starts) - 1 if starts is not None else 1)
    if len(value) % count:
        raise ValueError("output rows do not match fixed request boundaries")
    return np.split(value, count)


def float_metrics(actual, expected, budget):
    fields = {"kind", "normalized_l2", "max_abs", "rms_floor", "near_zero_max_abs"}
    if set(budget) != fields or any(
        not np.isscalar(budget[name])
        or not np.isfinite(budget[name])
        or budget[name] <= 0
        for name in fields - {"kind", "max_abs"}
    ):
        raise ValueError("invalid numerical limits")
    absolute = np.asarray(budget["max_abs"], dtype=np.float64)
    if (
        absolute.ndim > 1
        or (absolute.ndim == 1 and absolute.shape != expected.shape[-1:])
        or not np.isfinite(absolute).all()
        or np.any(absolute <= 0)
    ):
        raise ValueError("invalid absolute numerical limits")
    if expected.size == 0:
        return {"normalized_l2": 0.0, "max_abs": 0.0, "reference_rms": 0.0}, True
    reference_rms = float(np.sqrt(np.mean(np.square(expected))))
    error = actual.astype(np.float64) - expected
    normalized_l2 = float(np.sqrt(np.mean(np.square(error)))) / max(
        reference_rms, budget["rms_floor"]
    )
    maximum = float(np.max(np.abs(error)))
    passed = normalized_l2 <= budget["normalized_l2"] and bool(
        np.all(np.abs(error) <= absolute)
    )
    if reference_rms <= budget["rms_floor"]:
        passed = passed and maximum <= budget["near_zero_max_abs"]
    return {
        "normalized_l2": normalized_l2,
        "max_abs": maximum,
        "reference_rms": reference_rms,
        "max_abs_fraction": float(np.max(np.abs(error) / absolute)),
    }, passed


def compare(actual, fixture, limits):
    expected = fixture.expected
    if limits.get("case_sha256") != digest(fixture.manifest["case"]):
        raise ValueError("numerical limit identity differs from the fixture")
    if set(limits.get("fields", {})) != set(expected):
        raise ValueError("numerical limits do not cover every output")
    if set(actual) != set(expected):
        return NumericalResult({}, {"outputs": "missing or unexpected fields"})
    metrics, failures = {}, {}
    for name, reference in expected.items():
        value, budget = actual[name], limits["fields"][name]
        if value.shape != reference.shape or value.dtype.kind != reference.dtype.kind:
            failures[name] = "shape or datatype differs"
            continue
        if reference.dtype.kind != "f":
            if budget != {"kind": "exact"}:
                raise ValueError("integer and guard limits must be exact")
            if not equal_bits(value, reference):
                failures[name] = "exact output differs"
            continue
        if budget.get("kind") != "float":
            raise ValueError("missing floating-point limits")
        if not np.isfinite(reference).all() or not np.isfinite(value).all():
            failures[name] = "nonfinite output"
            continue
        observed = sequences(value, fixture.manifest["case"], name)
        references = sequences(reference, fixture.manifest["case"], name)
        metrics[name] = []
        for sequence, (output, target) in enumerate(
            zip(observed, references, strict=True)
        ):
            selected = dict(budget)
            if isinstance(selected.get("max_abs"), list):
                if len(selected["max_abs"]) != len(references):
                    raise ValueError("absolute limits do not cover every sequence")
                selected["max_abs"] = selected["max_abs"][sequence]
            measured, passed = float_metrics(output, target, selected)
            metrics[name].append(measured)
            if not passed:
                failures.setdefault(name, []).append(sequence)
    if "probabilities" in actual and "probabilities" not in failures:
        failures.update(probability_contract(actual, fixture.inputs))
        control = limits.get("probability")
        if control is None:
            raise ValueError("missing probability limits")
        variation = 0.5 * np.abs(
            actual["probabilities"] - expected["probabilities"]
        ).sum(axis=-1)
        metrics["total_variation"] = variation.tolist()
        if np.any(variation > control["total_variation"]):
            failures["total_variation"] = (
                "probability error exceeds its independent cap"
            )
        if control["softmax_logit_envelope"]:
            failures.update(softmax_envelope(actual, fixture, limits))
    if "selection_cdf" in actual and "selection_cdf" not in failures:
        failures.update(selection_contract(actual, fixture, limits))
    return NumericalResult(metrics, failures)


def probability_contract(actual, inputs):
    probability = actual["probabilities"]
    vocabulary = probability.shape[-1]
    permitted = np.unpackbits(inputs["mask"], axis=-1, bitorder="little")[
        ..., :vocabulary
    ]
    permitted = np.broadcast_to(permitted.astype(bool), probability.shape)
    if np.any(probability < 0) or np.any(probability[~permitted] != 0):
        return {"probability_contract": "negative or masked probability mass"}
    total = probability.sum(axis=-1, dtype=np.float64)
    if np.any(np.abs(total - 1) > 32 * np.finfo(np.float32).eps):
        return {"probability_contract": "probabilities are not normalized"}
    if "parameters" in inputs:
        greedy = np.broadcast_to(
            inputs["parameters"][:, 0] == 0, probability.shape[:-1]
        )
        if np.any((probability[greedy] != 0) & (probability[greedy] != 1)):
            return {"probability_contract": "greedy probabilities must be one-hot"}
    if "selected_probability" in actual:
        tokens = actual["tokens"]
        if np.any((tokens < 0) | (tokens >= vocabulary)):
            return {"probability_contract": "selected token is out of range"}
        selected = np.take_along_axis(probability, tokens[..., None], axis=-1)[..., 0]
        if not np.array_equal(selected, actual["selected_probability"]):
            return {
                "probability_contract": "selected probability differs from its token"
            }
    return {}


def softmax_envelope(actual, fixture, limits):
    references = sequences(
        fixture.expected["probabilities"], fixture.manifest["case"], "probabilities"
    )
    observed = sequences(
        actual["probabilities"], fixture.manifest["case"], "probabilities"
    )
    parameters = fixture.inputs["parameters"]
    offset = 0
    maxima = limits["fields"]["logits"]["max_abs"]
    if not isinstance(maxima, list):
        maxima = [maxima] * len(references)
    for reference, value, maximum in zip(references, observed, maxima, strict=True):
        rows = len(reference)
        settings = parameters[offset : offset + rows]
        offset += rows
        if np.any(settings[:, 0] <= 0):
            raise ValueError("softmax envelope requires positive temperature")
        sensitivity = np.maximum(settings[:, 6], 1 / settings[:, 6]) / settings[:, 0]
        ratio = np.exp(np.minimum(2 * maximum * sensitivity, 80))[:, None]
        lower = reference / (reference + (1 - reference) * ratio)
        upper = reference * ratio / (1 - reference + reference * ratio)
        allowance = 32 * np.finfo(np.float32).eps
        if np.any(value < lower - allowance) or np.any(value > upper + allowance):
            return {"softmax_envelope": "probabilities exceed the logit error envelope"}
    return {}


def selection_contract(actual, fixture, limits):
    budgets = np.asarray(limits.get("cdf_budgets", []), dtype=np.float64)
    edges = fixture.expected["selection_cdf"]
    if (
        budgets.shape != (len(edges),)
        or not np.isfinite(budgets).all()
        or np.any(budgets <= 0)
    ):
        raise ValueError("missing or invalid CDF limits")
    if np.any(np.abs(actual["selection_cdf"] - edges) > budgets[:, None]):
        return {
            "selection_contract": "selected-token CDF endpoints exceed the fixed budget"
        }
    records = iter(fixture.manifest["reference"]["decisions"]["records"])
    for index, temperature in enumerate(fixture.inputs["parameters"][:, 0]):
        if temperature == 0:
            continue
        record = next(records, None)
        if (
            record is None
            or record["kind"] != "cdf"
            or abs(record["budget"] - budgets[index]) > 1e-14
        ):
            raise ValueError("CDF limits differ from the reviewed fixture margins")
        if record["distance"] <= budgets[index] + 32 * np.finfo(np.float32).eps:
            raise ValueError("reference draw has insufficient CDF margin")
    if next(records, None) is not None:
        raise ValueError("unexpected reference selection decisions")
    return {}
