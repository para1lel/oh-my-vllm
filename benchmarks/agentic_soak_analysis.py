"""Strict request binding and memory checks for optional real-agent soak runs."""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from pathlib import Path

from benchmarks.agentic_soak_budget import validate_length_metric
from benchmarks.agentic_soak_transport import completed_transport_close

METRIC_MARKER = "SERVING_REQUEST_METRIC "
IDLE_COUNTS = (
    "active_requests",
    "pending_admissions",
    "scheduler_running",
    "scheduler_waiting",
    "queued_reservations",
    "live_reservations",
    "completed_stream_drains",
)


def finite_number(value, name: str, *, minimum=0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < minimum:
        raise ValueError(f"{name} is outside its finite range")
    return number


def device_used_bytes(sample: dict) -> int:
    """Include driver/vendor HBM outside Torch; exclusivity is a separate check."""
    gpu = (sample.get("worker") or {}).get("gpu", {})
    total, free = (gpu.get(name) for name in ("total_bytes", "free_bytes"))
    if (
        not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (total, free)
        )
        or not 0 <= free <= total
        or total <= 0
    ):
        raise ValueError("device HBM free/total bytes are outside their range")
    return total - free


def task_budget(args, phase: str) -> float:
    return args.warmup_task_timeout if phase == "warmup" else args.task_timeout


def source_citations(answer: str, reads: list[dict]) -> list[dict]:
    """Describe answer citations; actual source reads are verified upstream."""
    ranges, basenames = {}, {}
    for read in reads:
        path = read["path"]
        ranges.setdefault(path, set()).add(
            tuple(read["line_range"]) if read.get("line_range") else None
        )
        basenames.setdefault(Path(path).name, set()).add(path)
    references = []
    for match in re.finditer(
        r"(?<![\w./-])(?P<path>(?:[\w.-]+/)*[\w.-]+\.(?:rs|py))"
        r"(?![\w/-]|\.[\w])",
        answer,
    ):
        suffix = re.match(
            r"`?\s*(?::\s*|\(?\s*lines?\s+)([1-9][0-9]*)"
            r"\s*[-\u2013]\s*([1-9][0-9]*)",
            answer[match.end() :],
        )
        parseable = suffix is not None and all(
            len(value) <= 20 for value in suffix.groups()
        )
        references.append(
            {
                "reference": match["path"],
                "line_range": list(map(int, suffix.groups())) if parseable else None,
                "raw_line_range": suffix.group(0) if suffix else None,
                "line_range_parse_status": (
                    "parsed" if parseable else "unparseable" if suffix else "absent"
                ),
            }
        )
    unmatched = [
        reference
        for reference in references
        if Path(reference["reference"]).name not in basenames
    ]
    citations = []
    for path, actual_ranges in ranges.items():
        name = Path(path).name
        observed = []
        for reference in references:
            reported = reference["reference"]
            if Path(reported).name != name:
                continue
            qualified = reported == path
            shortened = reported == name and len(basenames[name]) == 1
            cited_range = reference["line_range"]
            observed.append(
                reference
                | {
                    "identity_matches_read": qualified or shortened,
                    "line_range_matches_read": (
                        tuple(cited_range) in actual_ranges
                        if cited_range is not None and None not in actual_ranges
                        else None
                    ),
                    "style": (
                        "full_path"
                        if qualified
                        else "unique_basename"
                        if shortened
                        else "ambiguous_basename"
                        if reported == name
                        else "different_path"
                    ),
                }
            )
        matched = next((r for r in observed if r["identity_matches_read"]), None)
        first = matched or (observed[0] if observed else None)
        citations.append(
            {
                "path": path,
                "actual_line_ranges": [
                    list(value) if value else None
                    for value in sorted(
                        actual_ranges, key=lambda value: value or (0, 0)
                    )
                ],
                "citation_present": matched is not None,
                "references": observed,
                "unmatched_answer_references": unmatched,
                "reference": first["reference"] if first else None,
                "line_range": first["line_range"] if first else None,
                "line_range_matches_read": (
                    first["line_range_matches_read"] if first else None
                ),
                "style": first["style"] if first else "missing",
            }
        )
    return citations


def probe_outcomes(probes: list[dict], metrics: list[dict]) -> dict:
    """Retain completion races, require genuine cancellations, reject failures."""
    by_id = {metric["request_id"]: metric for metric in metrics}
    cancelled = {api: 0 for api in ("chat", "responses")}
    outcomes = []
    for probe in probes:
        metric = by_id.get(probe["request_id"])
        if metric is None or metric["api"] != probe["api"]:
            raise ValueError("intentional probe has no matching server API metric")
        status = metric["status"]
        if probe["kind"] == "cancel" and status in ("cancelled", "completed"):
            outcome = (
                "cancelled" if status == "cancelled" else "completed_before_cancel"
            )
            cancelled[probe["api"]] += status == "cancelled"
        elif probe["kind"] == "slow" and status == "completed":
            outcome = "completed"
        else:
            raise ValueError(
                f"intentional {probe['kind']} probe {probe['request_id']} "
                f"has unexpected server status {status}"
            )
        outcomes.append(
            {
                "request_id": probe["request_id"],
                "api": probe["api"],
                "kind": probe["kind"],
                "outcome": outcome,
            }
        )
    if not all(cancelled.values()):
        raise ValueError("at least one genuine cancellation per API is required")
    return {"genuine_cancellations": cancelled, "outcomes": outcomes}


def request_metric(record: dict) -> dict:
    """Validate exact server timestamps, without replacing them with client time."""
    rid = record.get("request_id")
    if isinstance(rid, bool) or not isinstance(rid, int) or rid < 0:
        raise ValueError("request_id must be a nonnegative integer")
    if record.get("api") not in ("chat", "responses"):
        raise ValueError("request metric has an unknown API")
    expected = "chatcmpl_" if record["api"] == "chat" else "resp_"
    response_id = record.get("response_id")
    if not isinstance(response_id, str) or not response_id.startswith(expected):
        raise ValueError("request metric has an invalid response ID")
    if int(response_id.rsplit("_", 1)[1]) != rid:
        raise ValueError("request and response identities differ")
    status = record.get("status")
    if status not in ("completed", "cancelled", "failed"):
        raise ValueError("request metric has an unknown status")
    output = record.get("output_tokens")
    if isinstance(output, bool) or not isinstance(output, int) or output < 0:
        raise ValueError("output token count must be a nonnegative integer")
    received = finite_number(record.get("received_unix_s"), "received timestamp")
    ended = finite_number(
        record.get("generation_completed_unix_s"), "completion timestamp"
    )
    queued = finite_number(record.get("queue_seconds"), "queue duration")
    duration = finite_number(record.get("processing_seconds"), "processing duration")
    started = record.get("processing_started_unix_s")
    if started is None:
        if status == "completed" or output or duration:
            raise ValueError("an unprocessed request cannot have generation work")
    else:
        started = finite_number(started, "processing timestamp")
        # Duration uses the server monotonic clock. UTC can step; do not silently
        # derive throughput from two wall timestamps or impose equality on them.
        if duration <= 0 or (status == "completed" and output <= 0):
            raise ValueError("a processed request needs positive duration/output")
    return dict(
        record,
        received_unix_s=received,
        processing_started_unix_s=started,
        generation_completed_unix_s=ended,
        queue_seconds=queued,
        processing_seconds=duration,
    )


def read_metrics(path: Path, offset: int = 0) -> list[dict]:
    records, seen = [], set()
    with path.open("rb") as source:
        if source.seek(0, 2) < offset:
            raise ValueError("server log was truncated")
        source.seek(offset)
        for line in source:
            decoded = line.decode("utf-8")
            if METRIC_MARKER not in decoded:
                continue
            record = request_metric(json.loads(decoded.split(METRIC_MARKER, 1)[1]))
            identity = record["request_id"]
            if identity in seen:
                raise ValueError(f"duplicate server request metric {identity}")
            seen.add(identity)
            records.append(record)
    return records


def bind_points(
    metrics: list[dict], tasks: list[dict], measured_start: float
) -> list[dict]:
    """Keep every normal model request and reject missing or ambiguous evidence."""
    by_response = {}
    for metric in metrics:
        key = metric["response_id"]
        if key in by_response:
            raise ValueError("duplicate metric response identity")
        by_response[key] = metric
    points, seen = [], set()
    for task in tasks:
        if not task.get("passed"):
            raise ValueError("normal agent task failed; do not exclude its requests")
        for request in task["requests"]:
            if request.get("method") != "POST":
                continue
            if request.get("response_status") != 200:
                raise ValueError("normal model request failed")
            identities = request.get("response_ids", [])
            if len(identities) != 1:
                raise ValueError("normal request needs exactly one response identity")
            key = identities[0]
            if key in seen:
                raise ValueError("normal response appears in multiple requests")
            seen.add(key)
            metric = by_response.get(key)
            if metric is None or metric["status"] != "completed":
                raise ValueError(
                    f"normal response {key} has no completed server metric"
                )
            if metric["api"] != task["api"]:
                raise ValueError("client API differs from server metric")
            if request.get("error") and not completed_transport_close(
                request, key, task["api"]
            ):
                raise ValueError(
                    "normal model request failed: transport completion differs"
                )
            length_completion = validate_length_metric(request, metric, task["api"])
            points.append(
                {
                    **metric,
                    "task": task["task_id"],
                    "lane": task["lane"],
                    "post_terminal_transport_close": bool(request.get("error")),
                    "controlled_length_completion": length_completion,
                    "elapsed_minutes": (
                        metric["processing_started_unix_s"] - measured_start
                    )
                    / 60,
                    "output_tps": metric["output_tokens"]
                    / metric["processing_seconds"],
                }
            )
    return sorted(
        points,
        key=lambda value: (value["processing_started_unix_s"], value["request_id"]),
    )


def resource_checks(sample: dict, *, require_idle: bool = False) -> list[dict]:
    """Check public bounded counters, including each independent graph cache."""
    checks = []

    def check(name, passed, actual=None, maximum=None):
        checks.append(
            {"name": name, "passed": bool(passed), "actual": actual, "maximum": maximum}
        )

    check("worker_running", sample.get("status") == "running", sample.get("status"))
    check("no_worker_restart", sample.get("restarts") == 0, sample.get("restarts"), 0)
    limits = sample.get("limits", {})
    pairs = (
        ("active_requests", "live"),
        ("pending_admissions", "live"),
        ("queued_reservations", "queued"),
        ("live_reservations", "live"),
        ("stored_responses", "response_capacity"),
        ("stored_response_bytes", "response_max_bytes"),
    )
    for key, cap_key in pairs:
        actual, maximum = sample.get(key), limits.get(cap_key)
        valid = (
            isinstance(actual, int)
            and isinstance(maximum, int)
            and 0 <= actual <= maximum
        )
        check(key, valid, actual, maximum)
    actual, maximum = (
        sample.get("request_state_bytes"),
        sample.get("request_state_max_bytes"),
    )
    check(
        "request_state_bytes",
        isinstance(actual, int) and isinstance(maximum, int) and 0 <= actual <= maximum,
        actual,
        maximum,
    )
    worker = sample.get("worker") or {}
    check("worker_snapshot_present", bool(worker))
    check("graph_counters_present", bool(worker.get("graphs")))
    for name, graph in worker.get("graphs", {}).items():
        if graph is None:
            continue  # A mode does not own the other draft model's graphs.
        for counter, cap in (
            ("resident", "capacity"),
            ("probation", "probation_capacity"),
            ("capture_history", "capture_history_capacity"),
        ):
            actual, maximum = graph.get(counter), graph.get(cap)
            check(
                "graph_" + name + "_" + counter,
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and isinstance(maximum, int)
                and not isinstance(maximum, bool)
                and 0 <= actual <= maximum,
                actual,
                maximum,
            )
    compiled = worker.get("compile", {})
    for name in ("selection", "rewrite", "code_cache", "module"):
        actual, maximum = (
            compiled.get(name + "_entries"),
            compiled.get(name + "_capacity"),
        )
        check(
            "compile_" + name,
            isinstance(actual, int)
            and isinstance(maximum, int)
            and 0 <= actual <= maximum,
            actual,
            maximum,
        )
    for name in ("fx_source", "fake_tensor", "fx_profiler"):
        for suffix, cap_suffix in (
            ("entries", "capacity"),
            ("bytes", "byte_capacity"),
        ):
            actual, maximum = (
                compiled.get(name + "_" + suffix),
                compiled.get(name + "_" + cap_suffix),
            )
            check(
                "compile_" + name + "_" + suffix,
                isinstance(actual, int)
                and not isinstance(actual, bool)
                and isinstance(maximum, int)
                and not isinstance(maximum, bool)
                and maximum > 0
                and 0 <= actual <= maximum,
                actual,
                maximum,
            )
    actual, maximum = compiled.get("code_objects"), compiled.get("code_object_capacity")
    check(
        "compile_code_objects",
        isinstance(actual, int) and isinstance(maximum, int) and 0 <= actual <= maximum,
        actual,
        maximum,
    )
    counts, capacity = (
        compiled.get("compiled_graph_counts"),
        compiled.get("unit_capacity"),
    )
    check("compile_unit_counts_present", isinstance(counts, dict))
    if isinstance(counts, dict):
        check(
            "compile_unit_label_capacity",
            isinstance(capacity, int) and len(counts) <= capacity,
            len(counts),
            capacity,
        )
        for unit, actual in counts.items():
            check(
                "compile_unit_" + unit, isinstance(actual, int) and actual >= 0, actual
            )
    gpu = worker.get("gpu", {})
    allocated, reserved, total = (
        gpu.get(key) for key in ("allocated_bytes", "reserved_bytes", "total_bytes")
    )
    check(
        "gpu_allocation",
        all(isinstance(x, int) for x in (allocated, reserved, total))
        and 0 <= allocated <= reserved <= total,
        allocated,
        total,
    )
    try:
        used = device_used_bytes(sample)
        check("gpu_device_usage", True, used, total)
    except ValueError:
        check("gpu_device_usage", False, gpu.get("free_bytes"), total)
    kv = sample.get("kv", {})
    check("kv_snapshot_present", bool(kv))
    for pool in ("fa", "gdn"):
        total = kv.get(pool + "_blocks")
        for counter in ("free", "cached"):
            actual = kv.get(pool + "_" + counter)
            check(
                "kv_" + pool + "_" + counter,
                isinstance(actual, int)
                and isinstance(total, int)
                and 0 <= actual <= total,
                actual,
                total,
            )
    if require_idle:
        for key in IDLE_COUNTS:
            check("idle_" + key, sample.get(key) == 0, sample.get(key), 0)
        # Cached blocks are also allocatable; only reserved null block 0 stays
        # outside the free queue once every request reference has been released.
        for pool in ("fa", "gdn"):
            actual, total = kv.get(pool + "_free"), kv.get(pool + "_blocks")
            check(
                "idle_kv_" + pool + "_references",
                isinstance(total, int) and total > 0 and actual == total - 1,
                actual,
                total - 1 if isinstance(total, int) else None,
            )
        requests = worker.get("requests", {})
        check("worker_request_counters_present", bool(requests))
        for key, value in requests.items():
            check("idle_worker_" + key, value == 0, value, 0)
        preparation = worker.get("preparation", {})
        check("worker_preparation_counters_present", bool(preparation))
        for key, value in preparation.items():
            check("idle_preparation_" + key, value == 0, value, 0)
    return checks


def late_idle_samples(samples: list[dict], started: float, ended: float) -> list[dict]:
    """Select the last temporal half of the complete normal monotonic window."""
    started = finite_number(started, "normal start")
    ended = finite_number(ended, "normal end", minimum=started)
    cutoff = started + (ended - started) / 2
    observations = {}
    for sample in samples:
        if sample.get("phase") != "normal" or sample.get("observation") != "drained":
            continue
        collected = finite_number(sample.get("collected_monotonic_s"), "idle time")
        snapshot = finite_number(
            sample.get("worker_snapshot_unix_s"), "worker snapshot"
        )
        key = sample.get("worker_pid"), sample.get("epoch"), snapshot
        if (
            key not in observations
            or collected < observations[key]["collected_monotonic_s"]
        ):
            observations[key] = sample
    return sorted(
        (
            sample
            for sample in observations.values()
            if cutoff <= sample["collected_monotonic_s"] <= ended
        ),
        key=lambda sample: sample["collected_monotonic_s"],
    )


def plateau(
    samples: list[dict],
    path: tuple[str, ...],
    *,
    absolute_bytes: int,
    normal_started: float,
    normal_ended: float,
    fraction: float = 0.05,
) -> dict:
    """Report late idle medians and an explicit bounded-growth criterion."""
    rows = []
    late_samples = late_idle_samples(samples, normal_started, normal_ended)
    for sample in late_samples:
        value = sample
        for key in path:
            value = value.get(key, {}) if isinstance(value, dict) else None
        if not isinstance(value, int) or value < 0:
            raise ValueError("idle memory counter missing: " + ".".join(path))
        rows.append((sample["collected_monotonic_s"], value))
    window = {
        "samples": len(samples),
        "late_samples": len(rows),
        "normal_started_monotonic_s": normal_started,
        "normal_ended_monotonic_s": normal_ended,
        "late_cutoff_monotonic_s": normal_started + (normal_ended - normal_started) / 2,
        "late_first_monotonic_s": rows[0][0] if rows else None,
        "late_last_monotonic_s": rows[-1][0] if rows else None,
        "late_span_s": rows[-1][0] - rows[0][0] if rows else 0,
    }
    if len(rows) < 6:
        return {
            "passed": False,
            "reason": "normal last half needs at least six drained snapshots",
            **window,
        }
    baseline = statistics.median(value for _, value in rows[: len(rows) // 2])
    last = statistics.median(value for _, value in rows[len(rows) // 2 :])
    growth = last - baseline
    allowance = max(absolute_bytes, fraction * baseline)
    slopes = [
        (right[1] - left[1]) / (right[0] - left[0])
        for index, left in enumerate(rows)
        for right in rows[index + 1 :]
        if right[0] > left[0]
    ]
    return {
        "passed": growth <= allowance,
        **window,
        "initial_late_median_bytes": baseline,
        "final_late_median_bytes": last,
        "growth_bytes": growth,
        "allowance_bytes": allowance,
        "theil_sen_bytes_per_minute": 60 * statistics.median(slopes) if slopes else 0,
        "late_range_bytes": max(value for _, value in rows)
        - min(value for _, value in rows),
    }


def memory_audit(samples: list[dict], started: float, ended: float) -> dict:
    """Use one fixed normal interval for every allocator and host process."""
    metrics = (
        ("gpu_allocated", ("worker", "gpu", "allocated_bytes"), 64 << 20),
        ("gpu_reserved", ("worker", "gpu", "reserved_bytes"), 64 << 20),
        ("gpu_device_used", ("worker", "gpu", "device_used_bytes"), 64 << 20),
        ("worker_rss", ("worker", "cpu", "rss_bytes"), 128 << 20),
        ("http_rss", ("cpu", "rss_bytes"), 128 << 20),
    )
    return {
        name: plateau(
            samples,
            path,
            absolute_bytes=limit,
            normal_started=started,
            normal_ended=ended,
        )
        for name, path, limit in metrics
    }


def normal_run_complete(
    server_span_s: float,
    monotonic_elapsed_s: float,
    completed_requests: int,
    duration_s: float,
    minimum_requests: int,
    *,
    idle_samples: list[dict] | None = None,
    normal_started: float = 0,
) -> bool:
    """A UTC jump cannot replace a complete monotonic-duration collection."""
    return (
        math.isfinite(server_span_s)
        and math.isfinite(monotonic_elapsed_s)
        and server_span_s >= duration_s
        and monotonic_elapsed_s >= duration_s
        and completed_requests >= minimum_requests
        and (
            idle_samples is None
            or len(
                late_idle_samples(
                    idle_samples, normal_started, normal_started + monotonic_elapsed_s
                )
            )
            >= 6
        )
    )


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def plot_points(points: list[dict], mode: str, directory: Path):
    """Export independent scatter figures; never substitute synthetic points."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not points:
        raise ValueError("cannot plot a soak without completed normal requests")
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(figsize=(10, 4.2), layout="constrained")
        for purpose, color in (("task", "#21845c"), ("compaction", "#4667ac")):
            selected = [
                point
                for point in points
                if (point.get("purpose", "task") == "task") == (purpose == "task")
            ]
            if not selected:
                continue
            axes.scatter(
                [p["elapsed_minutes"] for p in selected],
                [p["output_tps"] for p in selected],
                s=8,
                alpha=0.55,
                linewidths=0,
                color=color,
                label="Task" if purpose == "task" else "Compaction",
            )
        axes.set(
            xlabel="Request processing start (minutes since measured start)",
            ylabel="Output tokens / processing time (tokens/s)",
            title=f"{'MTP4' if mode == 'mtp' else 'DSpark'} agentic serving: "
            f"{len(points)} completed requests",
        )
        axes.spines[["top", "right"]].set_visible(False)
        axes.grid(axis="y", alpha=0.2)
        axes.set_axisbelow(True)
        if any(point.get("purpose", "task") != "task" for point in points):
            axes.legend(frameon=False)
        for suffix in ("svg", "png"):
            figure.savefig(
                directory / f"{mode}-agentic-soak.{suffix}",
                dpi=160,
                metadata={"Date": None} if suffix == "svg" else {},
            )
        plt.close(figure)


def plot_memory(samples: list[dict], mode: str, directory: Path, started: float):
    """Retain active and drained observations with distinct allocator scopes."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    selected = [
        s
        for s in samples
        if s.get("worker") and s.get("worker_snapshot_unix_s", 0) >= started
    ]
    if not selected:
        raise ValueError("cannot plot memory without measured worker snapshots")
    selected.sort(key=lambda sample: sample["worker_snapshot_unix_s"])
    with plt.rc_context({"font.size": 10, "svg.fonttype": "none"}):
        figure, axes = plt.subplots(
            2, 1, figsize=(10, 6), layout="constrained", sharex=True
        )
        minutes = [(s["worker_snapshot_unix_s"] - started) / 60 for s in selected]
        mib = 1 << 20
        gpu = [s["worker"]["gpu"] for s in selected]
        for label, values in (
            ("Device used (total - free)", [device_used_bytes(s) for s in selected]),
            ("Torch allocated", [g["allocated_bytes"] for g in gpu]),
            ("Torch reserved", [g["reserved_bytes"] for g in gpu]),
        ):
            axes[0].plot(
                minutes, [value / mib for value in values], label=label, linewidth=1
            )
        for label, values in (
            ("GPU worker RSS", [s["worker"]["cpu"]["rss_bytes"] for s in selected]),
            ("HTTP server RSS", [s["cpu"]["rss_bytes"] for s in selected]),
        ):
            axes[1].plot(
                minutes, [value / mib for value in values], label=label, linewidth=1
            )
        drained = [s for s in selected if s.get("observation") == "drained"]
        axes[0].scatter(
            [(s["worker_snapshot_unix_s"] - started) / 60 for s in drained],
            [device_used_bytes(s) / mib for s in drained],
            s=15,
            color="#21845c",
            label="Drained snapshot",
        )
        axes[0].set(
            title=f"{'MTP4' if mode == 'mtp' else 'DSpark'} memory observations",
            ylabel="HBM (MiB)",
        )
        axes[1].set(
            xlabel="Worker snapshot (minutes since measured start)",
            ylabel="Host RSS (MiB)",
        )
        for axis in axes:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=0.2)
            axis.legend(frameon=False)
        for suffix in ("svg", "png"):
            figure.savefig(
                directory / f"{mode}-agentic-memory.{suffix}",
                dpi=160,
                metadata={"Date": None} if suffix == "svg" else {},
            )
        plt.close(figure)
