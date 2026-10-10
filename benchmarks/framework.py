"""Independent workload and phase statistics for semantic latency acceptance."""

from __future__ import annotations

import math
import statistics

WORKLOADS = tuple(
    {"mode": mode, "input_len": length, "output_len": 4096, "batch_size": batch}
    for mode, length in (
        ("ordinary", 32768),
        ("mtp4", 32768),
        ("prefix", 32768),
        ("ordinary", 131072),
        ("dspark", 32768),
    )
    for batch in (1, 2, 4)
)


def require_raw(artifact):
    """Formal verification needs complete original records."""
    if not isinstance(artifact, dict) or "artifact_kind" in artifact:
        raise ValueError("formal verification requires original raw evidence")


def phase_times(run, workload):
    """Aggregate each request's own intervals, including mixed-phase batches."""
    expected = {
        "batch_size": workload["batch_size"],
        "input_len": workload["input_len"],
        "output_len": workload["output_len"],
        "speculative_mode": {
            "ordinary": "none",
            "prefix": "none",
            "mtp4": "mtp",
            "dspark": "dspark",
        }[workload["mode"]],
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("observed workload identity differs")
    phases = run["request_phases"]
    batch = workload["batch_size"]
    if len(phases) != batch or len({p["request_id"] for p in phases}) != batch:
        raise ValueError("incomplete or duplicate request phase boundaries")
    if run["output_tokens"] != batch * workload["output_len"] or run["preemptions"]:
        raise ValueError("incomplete or preempted output")
    prefix = 32144 * batch if workload["mode"] == "prefix" else 0
    if run["initial_prefix_hit_tokens"] != prefix:
        raise ValueError("incorrect initial prefix reuse")
    prefill, decode = [], []
    for phase in phases:
        submit, first, last = (
            phase[k] for k in ("submitted_s", "first_token_s", "last_token_s")
        )
        if not all(math.isfinite(t) for t in (submit, first, last)):
            raise ValueError("invalid timestamp")
        if not 0 <= submit < first < last <= run["elapsed_s"]:
            raise ValueError("unordered phase boundaries")
        if not 1 <= phase["first_step"] < phase["last_step"] <= run["steps"]:
            raise ValueError("unordered step boundaries")
        prefill.append(first - submit)
        decode.append(last - first)
    return {"prefill": max(prefill), "decode": max(decode)}


def phase_verdict(wall, lower, *, mode="ordinary", phase="prefill"):
    """Report both checks; prefix prefill gates only on wall-time spread."""
    if mode not in {w["mode"] for w in WORKLOADS} or phase not in {
        "prefill",
        "decode",
    }:
        raise ValueError("unknown phase or inference mode")
    if len(wall) != 5 or len(lower) != 5:
        raise ValueError("one full set requires five measurements")
    if any(not math.isfinite(t) or t <= 0 for t in (*wall, *lower)):
        raise ValueError("invalid phase duration or theoretical lower bound")
    median = statistics.median(wall)
    bound = statistics.median(lower)
    spread = (max(wall) - min(wall)) / median
    latency_gate = not (mode == "prefix" and phase == "prefill")
    latency_passed = median <= 3 * bound
    spread_passed = spread <= 0.10
    return {
        "wall_median_s": median,
        "lower_bound_median_s": bound,
        "latency_ratio": median / bound,
        "wall_spread": spread,
        "latency_gate": latency_gate,
        "latency_passed": latency_passed,
        "spread_passed": spread_passed,
        "passed": (latency_passed or not latency_gate) and spread_passed,
    }


def validate_identities(artifact):
    """Reject incomplete or conflicting original measurement identities."""
    import hashlib
    import re

    from benchmarks.common import _parse_bench_config, _parse_cuda_provenance
    from benchmarks.evidence import loaded_draft_provenance

    def digest(value, length=64):
        return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value)

    source = artifact["source"]
    if not digest(source.get("git_commit"), 40) or any(
        not digest(source.get(key))
        for key in ("git_diff_sha256", "binary_sha256", "rust_sources_sha256")
    ):
        raise ValueError("incomplete measured source or binary identity")
    python_sources = source.get("python_sources_sha256")
    if (
        not isinstance(python_sources, dict)
        or not python_sources
        or any(not digest(value) for value in python_sources.values())
    ):
        raise ValueError("incomplete measured Python source identity")
    trace = artifact["trace"]
    contract = trace.get("runtime_contract", {})
    if any(
        python_sources.get("python/oh_my_vllm/" + name) != value
        for name, value in contract.get("sources", {}).items()
        if name.endswith(".py")
    ):
        raise ValueError("worker Python contract differs from measured source")
    from pathlib import Path

    directory = Path(__file__).resolve().parents[1] / "python/oh_my_vllm/performance"
    if any(
        python_sources.get("python/oh_my_vllm/performance/" + path.name)
        != hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.glob("*.py")
    ):
        raise ValueError("phase model differs from measured source")
    hardware = artifact.get("hardware", {})
    device = trace.get("device", {})
    from oh_my_vllm.performance.dag import b200, retention_bytes

    b200(device.get("sm_count"), device.get("max_clock_hz"), retention_bytes(device))
    if hardware.get("max_sm_clock_hz") != device.get("max_clock_hz"):
        raise ValueError(
            "worker clock ceiling differs from collector hardware identity"
        )
    uuid = hardware.get("gpu")
    info = str(hardware.get("gpu_info", "")).split(",")
    if (
        not isinstance(uuid, str)
        or not re.fullmatch(r"GPU-[0-9a-f-]{36}", uuid)
        or device.get("uuid") != uuid
        or len(info) != 4
        or info[0].strip() != uuid
        or info[1].strip() != device.get("name")
        or "B200" not in info[1]
        or not info[2].strip()
        or not info[3].strip()
        or not hardware.get("cpu_affinity")
    ):
        raise ValueError("incomplete or conflicting GPU hardware identity")
    runtime = artifact.get("runtime", {})
    policy = {"multi_stream": True, "pdl": True}
    if (
        runtime.get("runner") != "oh_my_vllm.worker.model_runner.OhMyVllmWorker"
        or runtime.get("kernel_backend") != "cuda"
        or runtime.get("vllm_importable") is not False
        or runtime.get("execution_policy") != policy
        or trace.get("execution_policy") != policy
        or not runtime.get("python")
        or not runtime.get("python_version")
        or not runtime.get("allocator_config")
        or not trace.get("allocator_backend")
        or not contract.get("versions")
        or any(
            runtime.get("packages", {}).get(k) != v
            for k, v in contract["versions"].items()
        )
    ):
        raise ValueError("incomplete or conflicting independent execution identity")
    checkpoints = artifact.get("checkpoints")
    expected = (
        {"target", "draft"} if artifact["workload"]["mode"] == "dspark" else {"target"}
    )
    if not isinstance(checkpoints, dict) or set(checkpoints) != expected:
        raise ValueError("missing checkpoint identities")
    for checkpoint in checkpoints.values():
        if not isinstance(checkpoint, dict) or not checkpoint.get("config"):
            raise ValueError("missing checkpoint configuration")
        files = checkpoint.get("files", {})
        if "config.json" not in files or not any(
            n.endswith(".safetensors") for n in files
        ):
            raise ValueError("missing checkpoint source files")
        tree = hashlib.sha256()
        for name, row in sorted(files.items()):
            if (
                not digest(row.get("sha256"))
                or type(row.get("bytes")) is not int
                or row["bytes"] < 0
                or name.startswith("/")
                or ".." in Path(name).parts
            ):
                raise ValueError("invalid checkpoint file identity")
            encoded = name.encode()
            tree.update(len(encoded).to_bytes(8, "little"))
            tree.update(encoded)
            tree.update(bytes.fromhex(row["sha256"]))
        if any(
            row["bytes"] == 0
            for name, row in files.items()
            if name == "config.json" or name.endswith(".safetensors")
        ):
            raise ValueError("empty checkpoint configuration or weight file")
        if tree.hexdigest() != checkpoint.get("tree_sha256"):
            raise ValueError("checkpoint manifest tree differs")
    config = checkpoints["target"]["config"]
    text = config.get("text_config", config)
    geometry = {
        "hidden_size": 5120,
        "num_hidden_layers": 64,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_conv_kernel_dim": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "rms_norm_eps": 1e-6,
    }
    layers = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)]
    if (
        any(text.get(k) != v for k, v in geometry.items())
        or text.get("layer_types") != layers
    ):
        raise ValueError("checkpoint geometry differs from phase equations")
    if checkpoints != artifact.get("checkpoints_after"):
        raise ValueError("checkpoint identity changed")
    stdout = artifact.get("stdout", "")
    cuda = _parse_cuda_provenance(stdout)
    if (
        cuda != artifact.get("cuda_provenance")
        or cuda != trace.get("cuda_provenance")
        or cuda.get("pdl") is not True
    ):
        raise ValueError("loaded CUDA module or PDL identity differs")
    if "draft" in checkpoints and artifact.get(
        "loaded_draft"
    ) != loaded_draft_provenance(stdout, checkpoints["draft"]):
        raise ValueError("loaded DSpark checkpoint identity differs")
    capacities = {"fa": 1400, "mamba": 128}
    if any(
        artifact.get(key) != capacities
        for key in ("worker_capacities", "scheduler_capacities")
    ):
        raise ValueError("missing or conflicting cache capacities")
    settings = {
        "block_size": 784,
        "max_model_len": 262144,
        "max_num_batched_tokens": 32768,
        "num_speculative_tokens": {"ordinary": 0, "prefix": 0, "mtp4": 4, "dspark": 7}[
            artifact["workload"]["mode"]
        ],
    }
    if sum("BENCH_CONFIG" in line for line in stdout.splitlines()) != 1 or any(
        _parse_bench_config(stdout, k, int) != v for k, v in settings.items()
    ):
        raise ValueError("observed scheduler configuration differs")
    audit = artifact.get("measurement_audit", {})
    if (
        audit.get("log_sha256") != hashlib.sha256(stdout.encode()).hexdigest()
        or not audit.get("cache_tree_sha256")
        or any(not digest(v) for v in audit["cache_tree_sha256"].values())
    ):
        raise ValueError("missing or conflicting steady-state evidence hashes")


def validate_repetitions(artifact):
    """Each warmup and measurement must describe a distinct complete batch."""
    import re

    from oh_my_vllm.performance.analysis import request_boundaries, validate_progress

    from benchmarks.common import parse_rows

    warmups, runs = artifact["warmups"], artifact["runs"]
    if len(warmups) != 2 or len(runs) != 5:
        raise ValueError("require two complete warmups and five measurements")
    stdout = artifact["stdout"]
    if (
        parse_rows(stdout, "WARMUP_RESULT ") != warmups
        or parse_rows(stdout, "BENCH_RESULT ") != runs
    ):
        raise ValueError("measurement records differ from original output")
    markers = [
        re.search(r"BENCH_PHASE iteration=(\d+) measured=(true|false)", line)
        for line in stdout.splitlines()
        if "BENCH_PHASE" in line
    ]
    if len(markers) != 7 or any(
        marker is None or marker.groups() != (str(i), "true" if i >= 2 else "false")
        for i, marker in enumerate(markers)
    ):
        raise ValueError("warmup and measurement phase order differs")
    seen, previous_end = set(), -1
    for run in [*warmups, *runs]:
        phase_times(run, artifact["workload"])
        ids = {p["request_id"] for p in run["request_phases"]}
        if seen & ids:
            raise ValueError("repeated request IDs across measurement batches")
        seen.update(ids)
        validate_progress(artifact["trace"], run)
        boundaries = request_boundaries(artifact["trace"], run)
        begin, end = (
            min(b[0] for b in boundaries.values()),
            max(b[2] for b in boundaries.values()),
        )
        if begin <= previous_end:
            raise ValueError("measurement trace intervals overlap or change order")
        previous_end = end


def validate_bound_consistency(times, bounds):
    """A lower bound cannot exceed measured time beyond timestamp precision."""
    for measured, bound in zip(times, bounds, strict=True):
        if any(
            bound[phase] > measured[phase] + 1e-6 for phase in ("prefill", "decode")
        ):
            raise ValueError("theoretical bound exceeds observed time")


def assess(artifact):
    """Recompute theory from the original trace, never from supplied timings.

    The reviewed project model supplies all phase-bound mappings.
    GPU event times remain diagnostic fields.
    """
    require_raw(artifact)
    if artifact.get("diagnostic") or artifact.get("status") != "collected":
        raise ValueError("formal verification requires completed formal evidence")
    source = artifact.get("source", {})
    if not source or source.get("git_status") or source != artifact.get("source_after"):
        raise ValueError("source identity is dirty, missing, or changed")
    if artifact["workload"] not in WORKLOADS:
        raise ValueError("unknown workload")
    from oh_my_vllm.performance.analysis import analyze_trace
    from oh_my_vllm.performance.coverage import validate_inventory

    rust_sha = artifact.get("rust_build_sha256")
    if (
        not isinstance(rust_sha, str)
        or len(rust_sha) != 64
        or set(rust_sha) - set("0123456789abcdef")
        or rust_sha != source.get("rust_sources_sha256")
    ):
        raise ValueError("binary Rust build inputs differ from measured source")
    validate_identities(artifact)
    validate_inventory(artifact["trace"])
    validate_repetitions(artifact)
    workload = artifact["workload"]
    for run in artifact["warmups"]:
        phase_times(run, workload)
    if not artifact.get("measurement_audit", {}).get("steady_state_verified"):
        raise ValueError("missing or failed steady-state audit")
    times = [phase_times(run, workload) for run in artifact["runs"]]
    bounds = analyze_trace(artifact["trace"], artifact["runs"])
    if len(bounds) != 5:
        raise ValueError("incomplete semantic model result")
    validate_bound_consistency(times, bounds)
    phases = {
        phase: phase_verdict(
            [t[phase] for t in times],
            [b[phase] for b in bounds],
            mode=workload["mode"],
            phase=phase,
        )
        for phase in ("prefill", "decode")
    }
    import hashlib
    from pathlib import Path

    from oh_my_vllm.performance.analysis import MODEL_VERSION

    directory = Path(__file__).resolve().parents[1] / "python/oh_my_vllm/performance"
    return {
        "phases": phases,
        "bounds": bounds,
        "model_version": MODEL_VERSION,
        "model_sources_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.glob("*.py"))
        },
        "passed": all(p["passed"] for p in phases.values()),
    }


def main():
    """Collect one independently specified workload, preserving every attempt."""
    import argparse
    import json
    import os
    import shutil
    import subprocess
    import tempfile
    from pathlib import Path

    from benchmarks.common import (
        ROOT,
        _observed_pool_capacities,
        _parse_bench_config,
        _parse_cuda_provenance,
        parse_rows,
        run_engine,
        runtime_identity,
        source_identity,
    )
    from benchmarks.context_boundary import _reap_on_termination
    from benchmarks.evidence import checkpoint_identity, loaded_draft_provenance
    from benchmarks.measurement import audit, hardware_identity

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("ordinary", "mtp4", "prefix", "dspark"), required=True
    )
    parser.add_argument("--batch-size", type=int, choices=(1, 2, 4), required=True)
    parser.add_argument("--input-len", type=int, choices=(32768, 131072), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OH_MY_VLLM_MODEL"))
    parser.add_argument(
        "--draft-model", default=os.environ.get("OH_MY_VLLM_DRAFT_MODEL")
    )
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="allow uncommitted source; no formal acceptance verdict",
    )
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("keep original evidence outside the repository")
    workload = {
        "mode": args.mode,
        "batch_size": args.batch_size,
        "input_len": args.input_len,
        "output_len": 4096,
    }
    if workload not in WORKLOADS:
        parser.error("select one of the fifteen specified workload rows")
    if not args.model or (args.mode == "dspark" and not args.draft_model):
        parser.error("supply the required target and draft checkpoints")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the result before model execution. Existing attempts are immutable.
    with args.output.open("x") as out:
        json.dump({"passed": False, "status": "started", "workload": workload}, out)
    log = args.output.with_suffix(".log")
    trace = args.output.with_suffix(".trace.json")
    if log.exists() or trace.exists():
        raise FileExistsError("use new paths for each complete attempt")
    binary = ROOT / "target/release/oh-my-vllm-zmq-worker"
    artifact = {
        "schema": 2,
        "passed": False,
        "workload": workload,
        "diagnostic": args.diagnostic,
    }
    try:
        source = source_identity(binary)
        artifact.update(
            source=source, hardware=hardware_identity(), runtime=runtime_identity()
        )
        checkpoints = {"target": checkpoint_identity(args.model)}
        if args.mode == "dspark":
            checkpoints["draft"] = checkpoint_identity(args.draft_model)
        artifact["checkpoints"] = checkpoints
        if source["git_status"] and not args.diagnostic:
            raise ValueError("formal collection requires committed unchanged source")
        with tempfile.TemporaryDirectory(prefix="oh-my-vllm-framework-") as tmp:
            snapshot = Path(tmp) / binary.name
            shutil.copy2(binary, snapshot)
            if source_identity(snapshot)["binary_sha256"] != source["binary_sha256"]:
                raise ValueError("binary changed before collection")
            command = [
                str(snapshot),
                "--model",
                args.model,
                "--socket",
                str(Path(tmp) / "worker.ipc"),
                "--num-gpu-blocks",
                "4200",
                "--mamba-blocks",
                "128",
                "--max-model-len",
                "262144",
            ]
            if args.mode == "mtp4":
                command.extend(["--num-speculative-tokens", "4"])
            elif args.mode == "dspark":
                command.extend(
                    ["--speculative-mode", "dspark", "--draft-model", args.draft_model]
                )
            command.extend(
                [
                    "bench",
                    "--batch-size",
                    str(args.batch_size),
                    "--input-len",
                    str(args.input_len),
                    "--output-len",
                    "4096",
                    "--warmup",
                    "2",
                    "--repetitions",
                    "5",
                ]
            )
            if args.mode == "prefix":
                command.append("--prefix-hit")
            os.environ["OH_MY_VLLM_SEMANTIC_TRACE"] = str(trace.resolve())
            artifact["command"] = command
            with _reap_on_termination() as cancelled:
                run_engine(
                    command,
                    timeout=14400,
                    stderr=subprocess.STDOUT,
                    log_path=log,
                    cancelled=cancelled,
                )
        stdout = log.read_text()
        artifact["warmups"] = parse_rows(stdout, "WARMUP_RESULT ")
        artifact["runs"] = parse_rows(stdout, "BENCH_RESULT ")
        artifact["trace"] = json.loads(trace.read_text())
        artifact["cuda_provenance"] = _parse_cuda_provenance(stdout)
        artifact["rust_build_sha256"] = _parse_bench_config(
            stdout, "rust_sources_sha256", str
        )
        artifact["worker_capacities"], artifact["scheduler_capacities"] = (
            _observed_pool_capacities(stdout, 1400, 128)
        )
        artifact["measurement_audit"] = audit(
            log,
            [
                os.environ.get("FLASHINFER_WORKSPACE_BASE"),
                os.environ.get("TRITON_CACHE_DIR"),
                os.environ.get("TILELANG_CACHE_DIR"),
                os.environ.get("TVM_FFI_CACHE_DIR"),
            ],
        )
        artifact["source_after"] = source_identity(binary)
        if artifact["source_after"] != source:
            raise ValueError("source or binary changed during collection")
        after = {"target": checkpoint_identity(args.model)}
        if args.mode == "dspark":
            after["draft"] = checkpoint_identity(args.draft_model)
            artifact["loaded_draft"] = loaded_draft_provenance(
                stdout, checkpoints["draft"]
            )
        artifact["checkpoints_after"] = after
        if after != checkpoints:
            raise ValueError("checkpoint changed during collection")
        for run in artifact["warmups"] + artifact["runs"]:
            phase_times(run, workload)
        artifact["status"] = "collected"
        artifact["stdout"] = stdout
        if not args.diagnostic:
            artifact["assessment"] = assess(artifact)
            artifact["passed"] = artifact["assessment"]["passed"]
        else:
            artifact["acceptance_pending"] = "diagnostic collection"
    except BaseException as error:
        artifact["status"] = "failed"
        artifact["error"] = repr(error)
        raise
    finally:
        if log.exists():
            artifact["stdout"] = log.read_text()
        args.output.write_text(json.dumps(artifact, indent=2) + "\n")
    if not args.diagnostic and not artifact["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
