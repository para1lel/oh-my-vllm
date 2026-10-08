"""Compare current-source DSpark and native MTP4 EngineCore measurements.

This protocol reports paired TPS, spread, and confidence bounds. Existing vLLM
regression gates stay unchanged. Raw records are required for a valid decision.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.compare_vllm import (
    parse_rows,
    run_engine,
    runtime_identity,
    source_identity,
)
from benchmarks.measurement import audit, hardware_identity
from scripts.export_evidence import derive

INPUT_LEN = 32768
OUTPUT_LEN = 4096
BATCH_SIZES = (1, 2, 4)
MODES = ("mtp", "dspark")
ROUNDS = 3
PAIRS = 5
WARMUPS = 2
MAX_SPREAD = 0.10


def audit_paired(log_path: Path, cache_roots: list[str | None]) -> dict:
    """Audit every measured interval while keeping later warmups independent.

    The shared audit supplies full log/cache hashes and rejects unset roots.
    Here its first-measurement window is refined with all BENCH_PHASE markers.
    Cache mtimes and explicit compilation/capture messages use the same windows.
    """
    result = audit(log_path, cache_roots)
    events, windows, problems = [], [], []
    state = False
    begin = None
    activity = re.compile(r"captur(?:e|ing).*graph|compil(?:ing|ation started)", re.I)
    timestamp = re.compile(r"\d{4}-\d\d-\d\dT[\d:.]+(?:Z|\+00:00)")
    expected = []
    for number in range(ROUNDS):
        slots = [
            ("warmup", index, 0, mode) for mode in MODES for index in range(WARMUPS)
        ]
        for pair in range(PAIRS):
            order = MODES if (number + pair) % 2 == 0 else tuple(reversed(MODES))
            slots.extend(
                ("measured", pair, slot, mode) for slot, mode in enumerate(order)
            )
        expected.extend(
            {
                "round": str(number),
                "phase": phase,
                "index": str(index),
                "slot": str(slot),
                "speculative_mode": mode,
                "measured": str(phase == "measured").lower(),
            }
            for phase, index, slot, mode in slots
        )
    for line in log_path.read_text().splitlines():
        if "BENCH_PHASE" in line:
            match, measured = (
                timestamp.search(line),
                re.search(r"\bmeasured=(true|false)\b", line),
            )
            if match is None or measured is None:
                problems.append("paired phase has no timestamp or measured flag")
                continue
            fields = {
                key: value or quoted
                for key, value, quoted in re.findall(
                    r'\b(\w+)=(?:([^\s"]+)|"([^\"]*)")', line
                )
            }
            if len(events) >= len(expected) or any(
                fields.get(key) != value for key, value in expected[len(events)].items()
            ):
                problems.append("paired phase order or identity differs")
            moment = datetime.datetime.fromisoformat(
                match[0].replace("Z", "+00:00")
            ).timestamp()
            if events and moment < events[-1][0]:
                problems.append("paired phase timestamps are not monotone")
            if state:
                windows.append((begin, moment))
            state = measured[1] == "true"
            begin = moment if state else None
            events.append((moment, state))
        elif state and activity.search(line):
            problems.append(line)
    if state:
        windows.append((begin, math.inf))
    expected_phases = ROUNDS * (len(MODES) * WARMUPS + len(MODES) * PAIRS)
    if len(events) != expected_phases or sum(flag for _, flag in events) != (
        ROUNDS * len(MODES) * PAIRS
    ):
        problems.append("paired audit requires every warmup and measured phase")
    for problem in result["problems"]:
        prefix = "cache file changed during measurements: "
        if problem.startswith(prefix):
            path = Path(problem.removeprefix(prefix))
            modified = path.stat().st_mtime_ns / 1e9
            if any(start <= modified < end for start, end in windows):
                problems.append(problem)
        elif not activity.search(problem):
            problems.append(problem)
    result.update(
        {
            "steady_state_verified": not problems,
            "problems": problems,
            "phase_count": len(events),
            "measured_intervals": windows,
        }
    )
    # JSON records must keep finite values. None means through worker shutdown.
    result["measured_intervals"] = [
        [start, None if math.isinf(end) else end] for start, end in windows
    ]
    return result


def checkpoint_identity(directory: str | Path) -> dict:
    """Hash checkpoint content, with relative names and configuration bytes."""
    directory = Path(directory).resolve(strict=True)
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError("checkpoint must be a directory with config.json")
    files = {}
    tree = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        files[relative] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
        encoded = relative.encode()
        tree.update(len(encoded).to_bytes(8, "little"))
        tree.update(encoded)
        tree.update(digest.digest())
    if not any(name.endswith(".safetensors") for name in files):
        raise ValueError("checkpoint has no safetensors weights")
    return {
        "path": str(directory),
        "files": files,
        "tree_sha256": tree.hexdigest(),
        "config": json.loads((directory / "config.json").read_text()),
    }


def validate_measurement(row: dict, mode: str, batch_size: int) -> None:
    """Require kept output, actual verified drafts, cold cache, and finite times."""
    expected = {
        "speculative_mode": mode,
        "batch_size": batch_size,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "output_tokens": batch_size * OUTPUT_LEN,
        "preemptions": 0,
        "prefix_hit_tokens": 0,
    }
    if mode not in MODES or batch_size not in BATCH_SIZES:
        raise ValueError("unknown paired workload")
    for key, value in expected.items():
        if row.get(key) != value:
            raise ValueError(f"measurement {key}: expected {value}, got {row.get(key)}")
    for key in ("elapsed_s", "output_tps"):
        value = row.get(key)
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"invalid {key}")
    if not math.isclose(
        row["output_tps"], row["output_tokens"] / row["elapsed_s"], rel_tol=1e-6
    ):
        raise ValueError("TPS does not use the complete engine elapsed time")
    times = row.get("ttft_s")
    if (
        not isinstance(times, list)
        or len(times) != batch_size
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or value > row["elapsed_s"]
            for value in times
        )
    ):
        raise ValueError("invalid per-request TTFT")
    verified = row.get("verified_draft_tokens")
    accepted = row.get("accepted_draft_tokens")
    if type(verified) is not int or verified <= 0:
        raise ValueError("measurement has no actual verified drafts")
    if type(accepted) is not int or not 0 <= accepted <= verified:
        raise ValueError("accepted drafts exceed actual verified drafts")
    if type(row.get("steps")) is not int or row["steps"] <= 0:
        raise ValueError("invalid engine step count")


def compare_rounds(rounds: list[dict], batch_size: int, *, bootstrap_samples=10000):
    """Report per-round spread and speed, plus a paired gain confidence bound.

    Resample rounds and then pairs inside each selected round. The statistic is
    the mean paired DSpark-minus-MTP TPS gain. TTFT is reported without a gate.
    """
    if len(rounds) != ROUNDS or bootstrap_samples < 1000:
        raise ValueError("require three rounds and at least 1000 bootstrap resamples")
    summaries, differences = [], []
    for number, data in enumerate(rounds):
        warmups, pairs = data.get("warmups", {}), data.get("pairs", [])
        if len(pairs) != PAIRS or set(warmups) != set(MODES):
            raise ValueError("each round requires five pairs and two mode warmup lists")
        for mode in MODES:
            if len(warmups[mode]) != WARMUPS:
                raise ValueError("each round requires two full warmups per mode")
            for row in warmups[mode]:
                validate_measurement(row, mode, batch_size)
        for index, pair in enumerate(pairs):
            order = list(MODES if (number + index) % 2 == 0 else reversed(MODES))
            if pair.get("order") != order:
                raise ValueError("paired measurements must alternate execution order")
            for mode in MODES:
                validate_measurement(pair.get(mode, {}), mode, batch_size)
        result = {}
        for mode in MODES:
            rates = [pair[mode]["output_tps"] for pair in pairs]
            times = [max(pair[mode]["ttft_s"]) for pair in pairs]
            median = statistics.median(rates)
            verified = sum(pair[mode]["verified_draft_tokens"] for pair in pairs)
            accepted = sum(pair[mode]["accepted_draft_tokens"] for pair in pairs)
            result[mode] = {
                "output_tps_median": median,
                "output_tps_spread": (max(rates) - min(rates)) / median,
                "ttft_s_median": statistics.median(times),
                "ttft_s_spread": (max(times) - min(times)) / statistics.median(times),
                "verified_draft_tokens": verified,
                "accepted_draft_tokens": accepted,
                "acceptance_rate": accepted / verified,
            }
        result["tps_spread_exceeds_10_percent"] = {
            mode: result[mode]["output_tps_spread"] > MAX_SPREAD for mode in MODES
        }
        result["dspark_median_faster"] = (
            result["dspark"]["output_tps_median"] > result["mtp"]["output_tps_median"]
        )
        summaries.append(result)
        differences.append(
            [pair["dspark"]["output_tps"] - pair["mtp"]["output_tps"] for pair in pairs]
        )
    rng = random.Random(784)
    estimates = []
    for _ in range(bootstrap_samples):
        means = []
        for _ in differences:
            selected = rng.choice(differences)
            means.append(statistics.mean(rng.choices(selected, k=len(selected))))
        estimates.append(statistics.mean(means))
    estimates.sort()
    lower = estimates[int(0.05 * bootstrap_samples)]
    return {
        "rounds": summaries,
        "paired_mean_gain_tps": statistics.mean(map(statistics.mean, differences)),
        "gain_lower_95_tps": lower,
        "method": "hierarchical paired bootstrap, one-sided 95%, seed 784",
        "bootstrap_samples": bootstrap_samples,
        "spread_report_threshold": MAX_SPREAD,
        "ttft_gated": False,
        "paired_gain_positive": lower > 0,
        "additional_tps_gate": False,
        "passed": True,
    }


def read_service_evidence(
    path: Path, offset: int, mode: str, *, request_ids=None, expected=None
) -> list[dict]:
    """Match new completed requests to the configured speculative method."""
    if mode not in MODES:
        raise ValueError("service evidence requires mtp or dspark mode")
    with path.open("rb") as source:
        if source.seek(0, 2) < offset:
            raise ValueError("server log was truncated during acceptance")
        source.seek(0)
        full = source.read().decode("utf-8")
        source.seek(offset)
        tail = source.read().decode("utf-8")
    configured = [
        re.search(r'\bspeculative_mode="?(mtp|dspark|none)"?\b', line)
        for line in full.splitlines()
        if "BENCH_CONFIG" in line
    ]
    if len(configured) != 1 or configured[0] is None or configured[0][1] != mode:
        raise ValueError("server speculative mode identity is missing or differs")
    rows = []
    seen = set()
    for line in tail.splitlines():
        if "request completed" not in line:
            continue
        fields = {}
        for key in ("request_id", "verified_draft_tokens", "accepted_draft_tokens"):
            match = re.search(rf"\b{key}=(\d+)\b", line)
            if match is None:
                raise ValueError(f"completed request has no {key}")
            fields[key] = int(match[1])
        rid = fields["request_id"]
        if rid in seen:
            raise ValueError("duplicate completed request identity")
        seen.add(rid)
        if fields["accepted_draft_tokens"] > fields["verified_draft_tokens"]:
            raise ValueError("accepted drafts exceed actual verified drafts")
        rows.append(fields)
    if expected is not None and len(rows) != expected:
        raise ValueError(f"expected {expected} completed requests, found {len(rows)}")
    if request_ids is not None and seen != set(request_ids):
        raise ValueError("completed requests do not match client response identities")
    if not rows or sum(row["verified_draft_tokens"] for row in rows) <= 0:
        raise ValueError("service has no actual verified drafts")
    return rows


def write_artifact(path: Path, artifact: dict, portable_output: Path | None = None):
    """Keep a complete raw artifact and an explicitly derived portable envelope."""
    if portable_output is not None and portable_output.resolve() == path.resolve():
        raise ValueError("portable output must not replace raw evidence")
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(artifact, indent=2, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    if portable_output is not None:
        portable_output.parent.mkdir(parents=True, exist_ok=True)
        portable_output.write_text(json.dumps(derive(raw, path.name), indent=2) + "\n")


def organize_records(records: list[dict], batch_size: int) -> list[dict]:
    """Check physical measurement order, phases, and every round/pair identity."""
    remaining = iter(records)
    rounds = []
    for number in range(ROUNDS):
        data = {"warmups": {mode: [] for mode in MODES}, "pairs": []}
        slots = [
            ("warmup", index, 0, mode) for mode in MODES for index in range(WARMUPS)
        ]
        for pair in range(PAIRS):
            order = list(MODES if (number + pair) % 2 == 0 else reversed(MODES))
            data["pairs"].append({"order": order})
            slots.extend(
                ("measured", pair, slot, mode) for slot, mode in enumerate(order)
            )
        for phase, index, slot, mode in slots:
            row = next(remaining, None)
            if row is None or any(
                row.get(key) != value
                for key, value in {
                    "round": number,
                    "phase": phase,
                    "index": index,
                    "slot": slot,
                    "speculative_mode": mode,
                }.items()
            ):
                raise ValueError("missing or reordered paired measurement record")
            validate_measurement(row, mode, batch_size)
            if phase == "warmup":
                data["warmups"][mode].append(row)
            else:
                data["pairs"][index][mode] = row
        rounds.append(data)
    if next(remaining, None) is not None:
        raise ValueError("unexpected extra paired measurement record")
    return rounds


def _source(binary: Path) -> dict:
    result = source_identity(binary)
    root = Path(__file__).resolve().parents[1]
    result["runtime_sources_sha256"] = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for directory in ("crates", "python", "benchmarks", "scripts")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file() and path.suffix in (".rs", ".py", ".sh", ".cu")
    }
    return result


def _configuration(log: str) -> dict:
    lines = [line for line in log.splitlines() if "BENCH_CONFIG" in line]
    if len(lines) != 1:
        raise ValueError("expected one observed BENCH_CONFIG record")
    return {
        key: value.strip('"')
        for key, value in re.findall(r'(\w+)=("[^"]*"|\S+)', lines[0])
    }


def _provenance(log: str) -> dict:
    rows = parse_rows(log, "CUDA_BUILD_PROVENANCE ")
    if len(rows) != 1:
        raise ValueError("expected one loaded CUDA provenance record")
    row = rows[0]
    if not all(row.get(key) for key in ("nvcc_path", "nvcc_version", "so_path")) or (
        row.get("compiler_identity_source") not in ("live", "manifest")
        or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("so_sha256", "")))
        or str(row["nvcc_version"]).startswith("nvcc-unavailable:")
        or row["nvcc_version"] == "nvcc-changed-during-build"
    ):
        raise ValueError("loaded CUDA provenance is incomplete")
    return row


def _memory(log: str, total: int) -> dict:
    rows = []
    for line in log.splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and row.get("message") == "GPU memory high water":
            rows.append(row)
    if len(rows) != 1:
        raise ValueError("expected one worker peak memory record")
    row = rows[0]
    allocated, reserved = row.get("max_allocated_bytes"), row.get("max_reserved_bytes")
    if (
        type(allocated) is not int
        or type(reserved) is not int
        or not 0 < allocated <= reserved <= total
    ):
        raise ValueError("invalid worker peak GPU memory")
    return row


def loaded_draft_provenance(log: str, expected: dict) -> dict:
    """Match the worker's loaded tensors to the checkpoint's actual source bytes."""
    rows = []
    prefix = "DSPARK_CHECKPOINT_PROVENANCE "
    for line in log.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            message = line
        else:
            message = record.get("message", "") if isinstance(record, dict) else ""
        if prefix in message:
            rows.append(json.loads(message.split(prefix, 1)[1]))
    if len(rows) != 1:
        raise ValueError("expected one actually loaded DSpark checkpoint provenance")
    row = rows[0]
    if Path(row.get("path", "")).resolve() != Path(expected["path"]).resolve():
        raise ValueError("loaded DSpark checkpoint path differs")
    config_hash = expected["files"].get("config.json", {}).get("sha256")
    weights = row.get("weights_sha256", {})
    if (
        row.get("config_sha256") != config_hash
        or not weights
        or any(
            name not in expected["files"] or digest != expected["files"][name]["sha256"]
            for name, digest in weights.items()
        )
    ):
        raise ValueError("loaded DSpark configuration or weight hashes differ")
    return row


def validate_artifact(artifact: dict) -> list[dict]:
    """Validate raw identities and audit evidence before a paired decision."""
    if not isinstance(artifact, dict) or "artifact_kind" in artifact:
        raise ValueError("formal comparison requires original raw evidence")
    source = artifact.get("source", {})
    if (
        not source.get("git_commit")
        or not source.get("runtime_sources_sha256")
        or (
            source != artifact.get("source_end")
            or not re.fullmatch(r"[0-9a-f]{64}", str(source.get("binary_sha256", "")))
            or not source.get("python_sources_sha256")
        )
    ):
        raise ValueError(
            "paired measurements have missing or changed source identities"
        )
    checkpoints = artifact.get("checkpoints", {})
    if set(checkpoints) != {"target", "draft"} or checkpoints != artifact.get(
        "checkpoints_end"
    ):
        raise ValueError("paired measurements have missing or changed checkpoints")
    for checkpoint in checkpoints.values():
        if not checkpoint.get("files") or not re.fullmatch(
            r"[0-9a-f]{64}", str(checkpoint.get("tree_sha256", ""))
        ):
            raise ValueError("checkpoint content hashes are missing")
    hardware = artifact.get("hardware", {})
    if not hardware.get("gpu", "").startswith("GPU-") or not hardware.get(
        "cpu_affinity"
    ):
        raise ValueError("UUID-pinned hardware identity is missing")
    if hardware != artifact.get("hardware_end") or not artifact.get("runtime", {}).get(
        "python"
    ):
        raise ValueError("hardware changed or Python runtime identity is missing")
    rows = artifact.get("rows", [])
    if [row.get("batch_size") for row in rows] != list(BATCH_SIZES):
        raise ValueError("paired acceptance requires all three batch sizes in order")
    provenance = None
    decisions = []
    cutoff = artifact.get("protocol", {}).get("dspark_confidence_threshold")
    if (
        not isinstance(cutoff, (int, float))
        or not math.isfinite(cutoff)
        or not 0 <= cutoff < 1
    ):
        raise ValueError("requested DSpark confidence threshold is missing or invalid")
    for row in rows:
        if row.get("hardware") != hardware or row.get("source") != source:
            raise ValueError(
                "row execution identity differs from the shared source or GPU"
            )
        config = row.get("config", {})
        expected = {
            "max_num_seqs": "32",
            "max_num_batched_tokens": "32768",
            "block_size": "784",
            "max_model_len": "65536",
            "worker_fa_pool_blocks": str(artifact["requested_capacities"]["fa"]),
            "worker_gdn_pool_blocks": str(artifact["requested_capacities"]["mamba"]),
            "fa_pool_blocks": str(artifact["requested_capacities"]["fa"]),
            "gdn_pool_blocks": str(artifact["requested_capacities"]["mamba"]),
        }
        if (
            any(config.get(key) != value for key, value in expected.items())
            or config.get("comparison") != "true"
            or config.get("speculative_mode") != "comparison"
        ):
            raise ValueError("observed comparison configuration or capacities differ")
        if float(config.get("dspark_confidence_threshold", "nan")) != cutoff:
            raise ValueError("observed DSpark confidence threshold differs")
        measured_audit = row.get("measurement_audit", {})
        if (
            not measured_audit.get("steady_state_verified")
            or measured_audit.get("phase_count") != 42
            or len(measured_audit.get("measured_intervals", [])) != 30
        ):
            raise ValueError(
                "measurement compilation/capture audit is missing or failed"
            )
        current = row.get("cuda_build_provenance", {})
        if not current.get("so_sha256") or (
            provenance is not None and current != provenance
        ):
            raise ValueError("loaded CUDA module identity is missing or changed")
        provenance = current
        loaded_draft_provenance(
            "DSPARK_CHECKPOINT_PROVENANCE " + json.dumps(row.get("loaded_draft", {})),
            checkpoints["draft"],
        )
        if not row.get("memory", {}).get("max_reserved_bytes") or not row.get(
            "log_sha256"
        ):
            raise ValueError("memory or original log identity is missing")
        original_rounds = organize_records(row.get("records", []), row["batch_size"])
        if original_rounds != row.get("rounds"):
            raise ValueError("organized rounds differ from original sample records")
        decisions.append(compare_rounds(original_rounds, row["batch_size"]))
    return decisions


def collect(args) -> dict:
    from benchmarks.context_boundary import _gpu_total_bytes, _reap_on_termination

    if os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") == "1":
        raise ValueError("disable eager diagnosis for production paired acceptance")
    if args.raw_dir.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("raw paired logs must stay outside the repository")
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="speculative-attempt-", dir=args.raw_dir))
    hardware = hardware_identity()
    artifact = {
        "schema": 1,
        "engine": "oh-my-vllm",
        "hardware": hardware,
        "runtime": runtime_identity(),
        "rows": [],
        "passed": False,
        "run_id": os.environ.get("OH_MY_VLLM_RUN_ID"),
        "requested_capacities": {
            "fa": args.num_gpu_blocks // 3,
            "mamba": args.mamba_blocks,
        },
        "protocol": {
            "input_len": INPUT_LEN,
            "output_len": OUTPUT_LEN,
            "batch_sizes": list(BATCH_SIZES),
            "rounds": ROUNDS,
            "warmups_per_mode_per_round": WARMUPS,
            "pairs_per_round": PAIRS,
            "temperature": 0,
            "ignore_eos": True,
            "prefix_state": "cold",
            "tokens": "((i + request * 997) % 32000) + 1",
            "tps_clock": (
                "before registration through complete batch cleanup; includes prefill"
            ),
            "ttft_gated": False,
            "shared_target_and_physical_cache": True,
            "dspark_confidence_threshold": args.dspark_confidence_threshold,
        },
    }
    try:
        artifact["checkpoints"] = {
            "target": checkpoint_identity(args.model),
            "draft": checkpoint_identity(args.draft_model),
        }
        with tempfile.TemporaryDirectory(prefix="oh-my-vllm-speculative-") as temporary:
            snapshot = Path(temporary) / args.binary.name
            shutil.copy2(args.binary, snapshot)
            artifact["source"] = _source(snapshot)
            for batch_size in BATCH_SIZES:
                log_path = directory / f"batch-{batch_size}.log"
                command = [
                    str(snapshot),
                    "--model",
                    args.model,
                    "--draft-model",
                    args.draft_model,
                    "--socket",
                    str(Path(temporary) / f"worker-{batch_size}.ipc"),
                    "--num-gpu-blocks",
                    str(args.num_gpu_blocks),
                    "--mamba-blocks",
                    str(args.mamba_blocks),
                    "--max-model-len",
                    "65536",
                    "--dspark-confidence-threshold",
                    str(args.dspark_confidence_threshold),
                    "spec-bench",
                    "--batch-size",
                    str(batch_size),
                ]
                row = {
                    "batch_size": batch_size,
                    "command": command,
                    "log_path": str(log_path),
                }
                artifact["rows"].append(row)
                with _reap_on_termination() as cancelled:
                    log = run_engine(
                        command,
                        timeout=args.timeout,
                        stderr=subprocess.STDOUT,
                        log_path=log_path,
                        cancelled=cancelled,
                    )
                row.update(
                    {
                        "source": _source(snapshot),
                        "hardware": hardware_identity(),
                        "config": _configuration(log),
                        "cuda_build_provenance": _provenance(log),
                        "memory": _memory(log, _gpu_total_bytes(hardware)),
                        "loaded_draft": loaded_draft_provenance(
                            log, artifact["checkpoints"]["draft"]
                        ),
                        "log_sha256": hashlib.sha256(log.encode()).hexdigest(),
                        "records": parse_rows(log, "SPEC_BENCH_RESULT "),
                        "measurement_audit": audit_paired(
                            log_path,
                            [
                                os.environ.get(key)
                                for key in (
                                    "FLASHINFER_WORKSPACE_BASE",
                                    "TRITON_CACHE_DIR",
                                    "TILELANG_CACHE_DIR",
                                    "TVM_FFI_CACHE_DIR",
                                )
                            ],
                        ),
                    }
                )
                row["rounds"] = organize_records(row["records"], batch_size)
                write_artifact(args.output, artifact, args.portable_output)
            artifact["source_end"] = _source(args.binary)
        artifact["checkpoints_end"] = {
            "target": checkpoint_identity(args.model),
            "draft": checkpoint_identity(args.draft_model),
        }
        artifact["hardware_end"] = hardware_identity()
        artifact["comparison"] = validate_artifact(artifact)
        artifact["passed"] = all(row["passed"] for row in artifact["comparison"])
    except BaseException as error:
        artifact["error"] = {"type": type(error).__name__, "message": str(error)}
        for row in artifact["rows"]:
            path = Path(row["log_path"])
            if path.is_file():
                row["log_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_artifact(args.output, artifact, args.portable_output)
        raise
    write_artifact(args.output, artifact, args.portable_output)
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--model", default=os.environ.get("OH_MY_VLLM_MODEL"))
    parser.add_argument(
        "--draft-model", default=os.environ.get("OH_MY_VLLM_DRAFT_MODEL")
    )
    parser.add_argument(
        "--binary",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "target/release/oh-my-vllm-zmq-worker",
    )
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--portable-output", type=Path)
    parser.add_argument("--num-gpu-blocks", type=int, default=4200)
    parser.add_argument("--mamba-blocks", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=14400)
    parser.add_argument("--dspark-confidence-threshold", type=float, default=0.2)
    args = parser.parse_args()
    if args.output.exists() or (args.portable_output and args.portable_output.exists()):
        parser.error("keep every attempt; choose unused output paths")
    if args.input:
        artifact = json.loads(args.input.read_bytes())
        artifact["comparison"] = validate_artifact(artifact)
        artifact["passed"] = all(row["passed"] for row in artifact["comparison"])
        write_artifact(args.output, artifact, args.portable_output)
    else:
        if not args.model or not args.draft_model or not args.raw_dir:
            parser.error(
                "collection requires --model, --draft-model, and --raw-dir "
                "(or model environment variables)"
            )
        if args.num_gpu_blocks < 6 or args.mamba_blocks < 2 or args.timeout <= 0:
            parser.error("invalid physical cache capacity or timeout")
        if not math.isfinite(args.dspark_confidence_threshold) or not (
            0 <= args.dspark_confidence_threshold < 1
        ):
            parser.error("DSpark confidence threshold must be finite in [0, 1)")
        artifact = collect(args)
    print(json.dumps(artifact.get("comparison", artifact.get("error")), indent=2))
    raise SystemExit(0 if artifact["passed"] else 1)


if __name__ == "__main__":
    main()
