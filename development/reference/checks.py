import math

from oh_my_vllm.kernels.specification import NUMERICAL_TOLERANCES

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
        "greedy",
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


def tolerance_for(operator, path, mtp=False, *, rows, variant="default"):
    if operator == "ffn" and path == ("output",) and rows > 128 and not mtp:
        return "mxfp8_ffn"
    if (
        operator == "fa_attention"
        and path == ("output",)
        and rows > 128
        and not mtp
        and variant in {"default", "full"}
    ):
        return "mxfp8_fa_output"
    if operator == "gdn_attention" and rows > 128 and not mtp:
        if path == ("input_projection",):
            return "mxfp8_gdn_input"
        if path == ("output",):
            return "mxfp8_gdn_output"
    return {
        ("vision_block", ("output",)): "vision",
        ("mtp_draft", ("attention", "output")): "mtp_attention",
        ("mtp_draft", ("ffn",)): "mtp_hidden",
        ("mtp_draft", ("ffn_residual",)): "mtp_residual",
        ("mtp_draft", ("sample", "logits")): "mtp_logits",
        ("fa_attention", ("output",)): "mtp_attention" if mtp else "bf16",
        ("sample", ("logits",)): "mtp_logits" if mtp else "bf16",
    }.get((operator, path), "bf16")


def state_tolerance_for(operator, *, rows, mtp=False, variant="default"):
    if (
        operator == "fa_attention"
        and rows > 128
        and not mtp
        and variant in {"default", "full"}
    ):
        return "mxfp8_fa_state"
    if operator == "gdn_attention" and rows > 128 and not mtp:
        return "mxfp8_gdn_state"
    return "state"


def verify_checks(
    value, operator, *, rows, state=False, path=(), mtp=False, variant="default"
):
    numerical = 0
    if isinstance(value, dict):
        if "max_abs" in value:
            name = value.get("tolerance_name")
            expected = tolerance_for(operator, path, mtp, rows=rows, variant=variant)
            if state:
                expected = state_tolerance_for(
                    operator, rows=rows, mtp=mtp, variant=variant
                )
            if name != expected:
                raise ValueError("unknown or incorrect numerical tolerance policy")
            limits = NUMERICAL_TOLERANCES[name]
            if state and value.get("state_ftz_atol") != limits["ftz_atol"]:
                raise ValueError("state FTZ absolute floor differs from the contract")
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
            if any(not math.isfinite(item) or item < 0 for item in (maximum, peak)):
                raise ValueError("invalid numerical error metrics")
            limits = NUMERICAL_TOLERANCES[value["tolerance_name"]]
            absolute_state_pass = state and maximum <= limits["ftz_atol"]
            if isinstance(nrmse, str):
                if not (
                    nrmse == "infinity"
                    and absolute_state_pass
                    and peak == 0
                    and maximum > 0
                ):
                    raise ValueError("invalid numerical error metrics")
                nrmse = math.inf
            elif not math.isfinite(nrmse) or nrmse < 0:
                raise ValueError("invalid numerical error metrics")
            if not absolute_state_pass and (
                maximum
                > (
                    limits["peak"] * peak
                    if state
                    else limits["atol"] + limits["rtol"] * peak
                )
                or (state and nrmse > limits["nrmse"])
            ):
                raise ValueError("numerical error exceeds the contract")
        for key, item in value.items():
            if operator == "gdn_attention" and key == "input_projection_upstream_drift":
                continue
            numerical += verify_checks(
                item,
                operator,
                rows=rows,
                state=state
                or key in {"state_writes", "convolution_writes", "cache_writes"},
                path=path if key == "graph" else (*path, key),
                mtp=mtp,
                variant=variant,
            )
    elif isinstance(value, list):
        for item in value:
            numerical += verify_checks(
                item,
                operator,
                rows=rows,
                state=state,
                path=path,
                mtp=mtp,
                variant=variant,
            )
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


def verify_greedy(check):
    cases = check.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("greedy numerical cases are missing")
    for name in ("penalties_and_mask", "tied_logits", "penalized_tie", "empty_mask"):
        modes = cases.get(name)
        if not isinstance(modes, dict):
            raise ValueError("greedy numerical case is missing")
        for mode in ("eager", "graph"):
            result = modes.get(mode)
            if not isinstance(result, dict) or any(
                result.get(key) is not True
                for key in (
                    "passed",
                    "probabilities_exact",
                    "tokens_exact",
                    "selected_probability_exact",
                    "rng_unchanged",
                )
            ):
                raise ValueError("greedy numerical evidence is missing or failed")


def verify_case(check, operator, *, rows, variant="default", mtp=False):
    if not isinstance(check, dict) or not set(CHECKS[operator]) <= check.keys():
        raise ValueError("required numerical or guard evidence is missing")
    if any(check[key] is not True for key in FLAGS[operator]):
        raise ValueError("required boolean guard is missing or failed")
    for key in METRICS[operator]:
        if (operator, variant, key) not in {
            ("fa_attention", "kv_only", "output"),
            ("mtp_draft", "query_only", "input"),
        }:
            verify_metric(
                check[key],
                tolerance_for(operator, (key,), mtp, rows=rows, variant=variant),
            )
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
            verify_metric(
                value,
                state_tolerance_for(operator, rows=rows, mtp=mtp, variant=variant),
            )
    if operator == "gdn_attention" and any(
        check["activation_quantization"].get(key) is not True
        for key in ("codes_exact", "scales_exact")
    ):
        raise ValueError("FP8 quantization evidence is missing or failed")
    if operator == "sample":
        verify_greedy(check["greedy"])
    if operator in {"mtp_context", "mtp_draft"}:
        verify_case(
            check["attention"],
            "fa_attention",
            rows=rows,
            variant="kv_only" if operator == "mtp_context" else variant,
            mtp=operator == "mtp_draft",
        )
        if operator == "mtp_context":
            verify_metric(check["endpoints"].get("reference"))
        else:
            verify_case(check["sample"], "sample", rows=rows, mtp=True)
    if (
        operator == "sample"
        and variant == "verify"
        and check.get("acceptance_state_selection") is not True
    ):
        raise ValueError("MTP acceptance evidence is missing or failed")
    if (
        check.get("passed") is not True
        or verify_checks(check, operator, rows=rows, mtp=mtp, variant=variant) == 0
    ):
        raise ValueError("numerical check is missing or did not pass")
