import argparse
import csv
import json
import os
import subprocess
from pathlib import Path

from oh_my_vllm.models.deepseek_v41.tokenizer import Tokenization
from oh_my_vllm.models.deepseek_v41.trace import execution_modes
from oh_my_vllm.operators.runtime.native import file_sha256

from . import deepseek_accounting as accounting
from . import deepseek_accounting_floor as accounting_floor
from . import deepseek_accounting_packed as accounting_packed
from . import deepseek_accounting_roofline as accounting_roofline
from .benchmark import ROOT, source_identity
from .deepseek_benchmark_config import (
    effective_cache_configuration as effective_cache_configuration,
)
from .deepseek_benchmark_config import (
    effective_service_configuration as effective_service_configuration,
)
from .deepseek_benchmark_config import service_config as service_config
from .deepseek_benchmark_config import (
    workload_cache_requirements as workload_cache_requirements,
)
from .deepseek_benchmark_identity import (
    DEEPSEEK_NATIVE_MODULES as DEEPSEEK_NATIVE_MODULES,
)
from .deepseek_benchmark_identity import (
    _checkpoint_shard_assignments as _checkpoint_shard_assignments,
)
from .deepseek_benchmark_identity import (
    _checkpoint_shard_manifest as _checkpoint_shard_manifest,
)
from .deepseek_benchmark_identity import (
    checkpoint_descriptor_identity as checkpoint_descriptor_identity,
)
from .deepseek_benchmark_identity import checkpoint_identity as checkpoint_identity
from .deepseek_benchmark_identity import (
    checkpoint_shard_identity as checkpoint_shard_identity,
)
from .deepseek_benchmark_identity import native_identity as native_identity
from .deepseek_benchmark_identity import (
    numeric_runtime_identity as numeric_runtime_identity,
)
from .deepseek_benchmark_identity import topology_identity as topology_identity
from .deepseek_benchmark_reference import _configure_floor as _configure_floor
from .deepseek_benchmark_reference import (
    reference_configuration as reference_configuration,
)
from .deepseek_benchmark_reference import runtime_reference as runtime_reference
from .deepseek_benchmark_service import (
    DEFAULT_REQUEST_TIMEOUT_SECONDS as DEFAULT_REQUEST_TIMEOUT_SECONDS,
)
from .deepseek_benchmark_service import SERVED_MODEL as SERVED_MODEL
from .deepseek_benchmark_service import SHUTDOWN_GRACE_SECONDS as SHUTDOWN_GRACE_SECONDS
from .deepseek_benchmark_service import Service as Service
from .deepseek_benchmark_service import exact_prompt as exact_prompt
from .deepseek_benchmark_service import post as post
from .deepseek_benchmark_service import request_body as request_body
from .deepseek_matrix import HTTP_CASES, case_id
from .device import selected_uuids


def artifact_identity(arguments, tokenizer, uuids, *, workload):
    cache_configuration = effective_cache_configuration(arguments)
    service_configuration = effective_service_configuration(arguments)
    identity = {
        "source": source_identity(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "binary": file_sha256(arguments.binary),
        "native": native_identity(),
        "harness_numeric_runtime": numeric_runtime_identity(),
        "checkpoint": checkpoint_descriptor_identity(arguments.model),
        "tokenizer": tokenizer.identity,
        "topology": topology_identity(uuids),
        "workload": workload,
        "configuration": {
            "cache_context": arguments.cache_context,
            "request_timeout_seconds": getattr(
                arguments, "request_timeout_seconds", DEFAULT_REQUEST_TIMEOUT_SECONDS
            ),
            **cache_configuration,
            **service_configuration,
            "rank_cache_requirements": getattr(arguments, "cache_requirements", {}),
            "rank_cache_configuration": {
                strategy: effective_cache_configuration(arguments, strategy)
                for strategy in getattr(arguments, "cache_requirements", {})
            },
            "native_sparse": os.environ.get("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", "1"),
            "execution_modes": {
                strategy: execution_modes(strategy)
                for strategy in ("tp4", "tep4", "dep4")
            },
        },
    }
    if workload.get("kind") == "http":
        identity["reference"] = reference_configuration(
            getattr(arguments, "reference_version", accounting.VERSION)
        )
    return identity


def verify_artifact_identity(arguments, tokenizer, uuids, expected, *, workload):
    actual = artifact_identity(
        arguments,
        tokenizer,
        uuids,
        workload=workload,
    )
    if actual != expected:
        raise RuntimeError("DeepSeek benchmark inputs changed during measurement")


def selected_cases(values):
    if not values:
        return list(HTTP_CASES)
    selected = set(values)
    cases = [case for case in HTTP_CASES if case_id(case) in selected]
    if len(cases) != len(selected):
        raise ValueError("each selected case must match the DeepSeek matrix")
    return cases


def write_csv(path, results):
    fields = (
        "mode",
        "strategy",
        "batch",
        "input_tokens",
        "output_tokens",
        "effective_strategy",
        "accounting_version",
        "source_trace_version",
        "source_trace_rate_profile",
        "rate_profile",
        "assumptions_sha256",
        "reference_scope",
        "reference_conditions",
        "reference_zero_credit",
        "reference_diagnostics",
        "communication_model",
        "profile_digest",
        "communication_rule_version",
        "dep4_communication_modeled",
        "dep4_communication_measured",
        "seconds",
        "request_count",
        "output_tokens_per_second",
        "reference_seconds",
        "reference_acceptance",
        "efficiency_percent",
        "accepted",
        "proposed",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                field: json.dumps(result[field])
                if isinstance(result[field], (dict, list, tuple))
                else result[field]
                for field in fields
            }
            for result in results
        )


def request_timeout_seconds(value):
    seconds = int(value)
    if seconds < 1:
        raise argparse.ArgumentTypeError("request timeout must be positive seconds")
    return seconds


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--case", action="append")
    parser.add_argument("--prompt-seed", default="deepseek-v41-nightly")
    parser.add_argument("--cache-context", type=int, default=1048576)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--mamba-blocks", type=int)
    parser.add_argument(
        "--reference-version",
        choices=(
            accounting.VERSION,
            accounting_floor.VERSION,
            accounting_packed.VERSION,
            accounting_roofline.VERSION,
        ),
        default=accounting.VERSION,
        help=(
            "HTTP accountant; v6 is the fixed-algorithm resource roofline "
            "(default: v3.1)"
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=request_timeout_seconds,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="timeout for both HTTP client and service requests (default: 7200)",
    )
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    return parser


def _print_result(result):
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "mode",
                    "strategy",
                    "batch",
                    "input_tokens",
                    "output_tokens",
                    "seconds",
                    "accounting_version",
                    "source_trace_version",
                    "rate_profile",
                    "assumptions_sha256",
                    "reference_scope",
                    "reference_seconds",
                    "reference_acceptance",
                    "efficiency_percent",
                )
            }
        ),
        flush=True,
    )


def _run_cases(arguments, cases, tokenizer, identity):
    results = []
    for strategy, mode in sorted({(case["strategy"], case["mode"]) for case in cases}):
        selected = [
            case
            for case in cases
            if case["strategy"] == strategy and case["mode"] == mode
        ]
        service = Service(arguments, mode, strategy)
        try:
            ready = service.ready()
            (arguments.artifacts / f"{mode}-{strategy}-ready.json").write_text(
                json.dumps(ready, indent=2) + "\n"
            )
            for case in selected:
                print(json.dumps({"event": "start", **case}), flush=True)
                result = service.run(
                    tokenizer, case, arguments.prompt_seed, identity=identity
                )
                results.append(result)
                name = case_id(case).replace(":", "-")
                (arguments.artifacts / f"{name}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                _print_result(result)
        finally:
            service.close()
    return results


def main():
    parser = argument_parser()
    arguments = parser.parse_args()
    arguments.model = arguments.model.resolve()
    arguments.artifacts = arguments.artifacts.resolve()
    arguments.binary = arguments.binary.resolve()
    arguments.csv = arguments.csv.resolve()
    if arguments.artifacts.is_relative_to(ROOT):
        parser.error("put raw artifacts external to the repository")
    if arguments.cache_context < 1:
        parser.error("cache context must be positive")
    try:
        cases = selected_cases(arguments.case)
        uuids = selected_uuids()
    except ValueError as error:
        parser.error(str(error))
    if len(uuids) != 4:
        parser.error("DeepSeek benchmark requires four selected GPUs")
    arguments.cache_requirements = workload_cache_requirements(HTTP_CASES)
    if any(
        row["maximum_total_tokens"] > arguments.cache_context
        for row in arguments.cache_requirements.values()
    ):
        parser.error("cache context must cover every approved prompt and output")
    try:
        effective_cache_configuration(arguments)
    except ValueError as error:
        parser.error(str(error))
    arguments.artifacts.mkdir(parents=True, exist_ok=False)
    (arguments.artifacts / "selection.json").write_text(
        json.dumps({"case_ids": [case_id(case) for case in cases]}, indent=2) + "\n"
    )
    tokenizer = Tokenization(arguments.model)
    workload = {
        "kind": "http",
        "case_ids": [case_id(case) for case in HTTP_CASES],
        "prompt_seed": arguments.prompt_seed,
    }
    identity = artifact_identity(
        arguments,
        tokenizer,
        uuids,
        workload=workload,
    )
    (arguments.artifacts / "identity.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )
    print(json.dumps({"event": "reference", **identity["reference"]}), flush=True)
    results = _run_cases(arguments, cases, tokenizer, identity)
    verify_artifact_identity(
        arguments,
        tokenizer,
        uuids,
        identity,
        workload=workload,
    )
    write_csv(arguments.csv, results)
    (arguments.artifacts / "summary.json").write_text(
        json.dumps({"cases": len(results)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
