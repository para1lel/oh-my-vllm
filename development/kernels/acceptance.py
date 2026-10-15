import hashlib
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from oh_my_vllm.kernels.native import SOURCES, source_digest
from oh_my_vllm.kernels.specification import (
    MODEL,
    NUMERICAL_TOLERANCES,
    UPSTREAM_COMMIT,
)

from .provenance import ROOT, case_id, source_identity, suite_identity
from .providers import MODULES, RUNTIME_MODULES
from .roofline import ExpectedCase, Observation, expected_efficiencies
from .work import work_for
from .workload import SERVING_OPERATORS, modeled_cases

CHECKS = {
    "ffn": ("residual", "output", "all_rows_finite"),
    "fa_attention": (
        "residual",
        "output",
        "cache_writes",
        "unchanged_cache",
        "all_rows_finite",
    ),
    "gdn_attention": (
        "residual",
        "normalization",
        "activation_quantization",
        "input_projection",
        "ba_projection",
        "output",
        "state_writes",
        "convolution_writes",
        "unchanged_slots",
        "all_rows_finite",
    ),
    "sample": (
        "logits",
        "probabilities",
        "all_rows_finite",
        "normalized",
        "valid_selected_tokens",
        "rng_counter",
    ),
    "mtp_context": ("input", "zero_residual", "endpoints", "attention"),
    "mtp_draft": (
        "input",
        "attention",
        "ffn",
        "ffn_residual",
        "ffn_residual_store_exact",
        "sample",
    ),
    "vision_block": (
        "output",
        "patch_order",
        "patch_boundaries",
        "frame_isolation",
        "all_rows_finite",
    ),
}
FLAGS = {
    "ffn": ("all_rows_finite",),
    "fa_attention": ("all_rows_finite", "unchanged_cache"),
    "gdn_attention": ("all_rows_finite", "unchanged_slots"),
    "sample": ("all_rows_finite", "normalized", "valid_selected_tokens", "rng_counter"),
    "mtp_context": ("zero_residual",),
    "mtp_draft": ("ffn_residual_store_exact",),
    "vision_block": ("all_rows_finite",),
}
METRICS = {
    "ffn": ("residual", "output"),
    "fa_attention": ("residual", "output"),
    "gdn_attention": (
        "residual",
        "normalization",
        "input_projection",
        "ba_projection",
        "output",
    ),
    "sample": ("logits", "probabilities"),
    "mtp_context": ("input",),
    "mtp_draft": ("input", "ffn", "ffn_residual"),
    "vision_block": ("output",),
}


def verify_identity(report):
    if report.get("suite_id") != suite_identity():
        raise ValueError("report uses a different serving suite")
    operator = report["operator"]
    builds = {}
    for build in report["builds"]:
        names = [
            name
            for name, functions in MODULES.items()
            if list(functions) == build["functions"]
        ]
        if len(names) != 1 or names[0] in builds:
            raise ValueError("ambiguous native build manifest")
        name = names[0]
        binary = Path(build["library"])
        if json.loads((binary.parent / "manifest.json").read_text()) != build:
            raise ValueError("native build manifest has changed")
        source = (
            ROOT / "development/kernels" if name == "device" else SOURCES
        ) / f"{name}.cu"
        if source_digest(source) != build["source_fingerprint"]:
            raise ValueError("native source has changed")
        if report["binary_ids"].get(str(binary)) != build["binary_id"]:
            raise ValueError("binary and build identities differ")
        builds[name] = build
    names = RUNTIME_MODULES[operator]
    if set(builds) != {*names, "device"}:
        raise ValueError("missing or unexpected native modules")
    identities = "".join(builds[name]["source_id"] for name in names)
    if len(names) > 1:
        identities = hashlib.sha256(identities.encode()).hexdigest()
    context = report["context"]
    if context.get("upstream_commit") != UPSTREAM_COMMIT:
        raise ValueError("report uses a different upstream numerical contract")
    if (
        report["native_id"] != identities
        or context["device_source"] != builds["device"]["source_id"]
    ):
        raise ValueError("native provider identity differs from its modules")
    identity, binary_ids = source_identity(
        identities,
        tuple(Path(path) for path in report["binary_ids"]),
        context,
        operator,
    )
    if identity != report["source_id"] or binary_ids != report["binary_ids"]:
        raise ValueError("provider source or binary has changed")


def tolerance_for(operator, path, mtp=False):
    return {
        ("vision_block", ("output",)): "vision",
        ("mtp_draft", ("attention", "output")): "mtp_attention",
        ("mtp_draft", ("ffn",)): "mtp_hidden",
        ("mtp_draft", ("ffn_residual",)): "mtp_residual",
        ("mtp_draft", ("sample", "logits")): "mtp_logits",
        ("fa_attention", ("output",)): "mtp_attention" if mtp else "bf16",
        ("sample", ("logits",)): "mtp_logits" if mtp else "bf16",
    }.get((operator, path), "bf16")


def verify_checks(value, operator, *, state=False, path=(), mtp=False):
    numerical = 0
    if isinstance(value, dict):
        if "max_abs" in value:
            name = value.get("tolerance_name")
            expected = "state" if state else tolerance_for(operator, path, mtp)
            if name != expected:
                raise ValueError("unknown or incorrect numerical tolerance policy")
            limits = NUMERICAL_TOLERANCES[name]
            required_atol = value.get("required_atol")
            if not state and (
                not isinstance(required_atol, (int, float))
                or not math.isfinite(required_atol)
                or not 0 <= required_atol <= limits["atol"]
            ):
                raise ValueError("required absolute tolerance exceeds the contract")
            if not math.isfinite(value["max_abs"]) or value["max_abs"] < 0:
                raise ValueError("invalid maximum absolute error")
            ratio = value.get("max_tolerance_ratio")
            if (
                not isinstance(ratio, (int, float))
                or not math.isfinite(ratio)
                or not 0 <= ratio <= 1
            ):
                raise ValueError("recorded elementwise or state tolerance check failed")
            numerical += 1
        nrmse_key = "worst_chunk_nrmse" if "rows_checked" in value else "nrmse"
        if {"max_abs", nrmse_key, "reference_peak"} <= value.keys():
            maximum, nrmse, peak = (
                value[key] for key in ("max_abs", nrmse_key, "reference_peak")
            )
            if any(
                not math.isfinite(item) or item < 0 for item in (maximum, nrmse, peak)
            ):
                raise ValueError("invalid numerical error metrics")
            limits = NUMERICAL_TOLERANCES[value["tolerance_name"]]
            if maximum > (
                limits["peak"] * peak
                if state
                else limits["atol"] + limits["rtol"] * peak
            ) or (state and nrmse > limits["nrmse"]):
                raise ValueError("numerical error exceeds the contract")
        for key, item in value.items():
            if operator == "gdn_attention" and key == "input_projection_upstream_drift":
                continue
            numerical += verify_checks(
                item,
                operator,
                state=state
                or key in {"state_writes", "convolution_writes", "cache_writes"},
                path=path if key == "graph" else (*path, key),
                mtp=mtp,
            )
    elif isinstance(value, list):
        for item in value:
            numerical += verify_checks(item, operator, state=state, path=path, mtp=mtp)
    elif value is False:
        raise ValueError("numerical evidence contains a failed check")
    return numerical


def verify_metric(value, tolerance="bf16"):
    nrmse_key = (
        "worst_chunk_nrmse"
        if isinstance(value, dict) and "rows_checked" in value
        else "nrmse"
    )
    if (
        not isinstance(value, dict)
        or not {"max_abs", nrmse_key, "reference_peak", "max_tolerance_ratio"}
        <= value.keys()
        or value.get("passed") is not True
        or value.get("tolerance_name") != tolerance
    ):
        raise ValueError("required numerical error evidence is missing")


def verify_case(check, operator, *, variant="default", mtp=False):
    if not isinstance(check, dict) or not set(CHECKS[operator]) <= check.keys():
        raise ValueError("required numerical or guard evidence is missing")
    if any(check[key] is not True for key in FLAGS[operator]):
        raise ValueError("required boolean guard is missing or failed")
    for key in METRICS[operator]:
        if (operator, variant, key) not in {
            ("fa_attention", "kv_only", "output"),
            ("mtp_draft", "query_only", "input"),
        }:
            verify_metric(check[key], tolerance_for(operator, (key,), mtp))
    for key in CHECKS[operator]:
        if (
            key not in FLAGS[operator]
            and key
            not in {
                "cache_writes",
                "state_writes",
                "convolution_writes",
            }
            and (
                not isinstance(check[key], dict) or check[key].get("passed") is not True
            )
        ):
            raise ValueError("required component evidence is missing or failed")
    for key in {"cache_writes", "state_writes", "convolution_writes"} & check.keys():
        if not isinstance(check[key], list):
            raise ValueError("persistent-write numerical evidence is missing")
        if (
            operator == "fa_attention"
            and variant == "query_only"
            and key == "cache_writes"
        ):
            if check[key]:
                raise ValueError("query-only attention must not write KV")
            continue
        if not check[key]:
            raise ValueError("persistent-write numerical evidence is missing")
        for value in check[key]:
            verify_metric(value, "state")
    if operator == "gdn_attention" and any(
        check["activation_quantization"].get(key) is not True
        for key in ("codes_exact", "scales_exact")
    ):
        raise ValueError("FP8 quantization evidence is missing or failed")
    if operator in {"mtp_context", "mtp_draft"}:
        verify_case(
            check["attention"],
            "fa_attention",
            variant="kv_only" if operator == "mtp_context" else variant,
            mtp=operator == "mtp_draft",
        )
        if operator == "mtp_context":
            verify_metric(check["endpoints"].get("reference"))
        else:
            verify_case(check["sample"], "sample", mtp=True)
    if (
        operator == "sample"
        and variant == "verify"
        and check.get("acceptance_state_selection") is not True
    ):
        raise ValueError("MTP acceptance evidence is missing or failed")
    if check.get("passed") is not True or verify_checks(check, operator, mtp=mtp) == 0:
        raise ValueError("numerical check is missing or did not pass")


def assess(reports):
    measured, checkpoint, device = {}, {}, None
    for report in reports:
        operator, mode = report["operator"], report["mode"]
        if operator not in SERVING_OPERATORS or mode not in {"eager", "graph"}:
            raise ValueError("unknown operator or execution mode")
        if (operator, mode) in measured:
            raise ValueError("duplicate operator and execution mode")
        if any(
            report.get(key)
            for key in (
                "error",
                "synchronize_error",
                "profile_only",
                "missing_providers",
            )
        ):
            raise ValueError("report has a failed or incomplete evaluation")
        context = report["context"]
        if context.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError("report uses a different upstream numerical contract")
        if (
            context["mode"] != mode
            or not context["device"].startswith("GPU-")
            or context["device"] != report["device"].split(",", 1)[0].strip()
        ):
            raise ValueError(
                "execution context differs from the measured device or mode"
            )
        if device is not None and device != report["device"]:
            raise ValueError("reports use different devices or clock settings")
        device = report["device"]
        identities = context["checkpoint"]
        if (
            identities.get("config") != MODEL["config_sha256"]
            or "index" not in identities
        ):
            raise ValueError("checkpoint configuration or index identity is missing")
        for name, identity in identities.items():
            if len(identity) != 64 or any(
                char not in "0123456789abcdef" for char in identity
            ):
                raise ValueError("checkpoint identity is not a SHA-256 digest")
            if name in checkpoint and checkpoint[name] != identity:
                raise ValueError("reports use different checkpoint data")
            checkpoint[name] = identity
        policy = report["cache_policy"]
        if (
            policy["method"] != "ld.global.cg scan before start event"
            or not policy["bytes"] > 2 * policy["l2_bytes"] > 0
        ):
            raise ValueError("report does not use the cold-weight policy")
        cases = {
            case_id(case.shape): case
            for case in modeled_cases()
            if case.shape.operator == operator
        }
        required = {
            identity: ExpectedCase(operator, case.calls, work_for(case.shape))
            for identity, case in cases.items()
        }
        if any(
            set(report[key]) != set(required) for key in ("checks", "samples", "work")
        ):
            raise ValueError("missing or unexpected serving shapes")
        observations = [Observation(**item) for item in report["observations"]]
        efficiency = expected_efficiencies(observations, required, report["source_id"])[
            operator
        ]
        for item in observations:
            identity = item.case_id
            check = report["checks"][identity]
            variant = cases[identity].shape.variant
            verify_case(check, operator, variant=variant)
            if mode == "graph":
                verify_case(check["graph"], operator, variant=variant)
            samples = report["samples"][identity]
            if len(samples) != 5 or any(
                not math.isfinite(value) or value <= 0 for value in samples
            ):
                raise ValueError("report requires five valid measurements per shape")
            if not math.isclose(
                statistics.median(samples), item.measured_seconds, rel_tol=1e-12
            ):
                raise ValueError("observation differs from its timing samples")
            if report["work"][identity] != asdict(required[identity].work):
                raise ValueError(
                    "report uses a different precision or work calculation"
                )
        measured[operator, mode] = efficiency
    missing = [
        f"{operator}/{mode}"
        for operator in SERVING_OPERATORS
        for mode in ("eager", "graph")
        if (operator, mode) not in measured
    ]
    return {
        "accepted": not missing and all(value >= 0.60 for value in measured.values()),
        "missing": missing,
        "expected_efficiency": {
            f"{operator}/{mode}": value
            for (operator, mode), value in sorted(measured.items())
        },
    }


def audit(paths):
    suite_id = suite_identity()
    reports = [json.loads(path.read_text()) for path in paths]
    builds = {}
    for report in reports:
        verify_identity(report)
        for build in report["builds"]:
            key = tuple(build["functions"])
            if key in builds and builds[key] != build:
                raise ValueError(
                    "reports use different builds of a shared native module"
                )
            builds[key] = build
    result = assess(reports)
    if suite_identity() != suite_id:
        raise ValueError("serving source changed during the acceptance audit")
    return {
        **result,
        "suite_id": suite_id,
        "reports": [str(path.resolve()) for path in paths],
    }
