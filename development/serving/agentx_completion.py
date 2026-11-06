import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .agentx_data import phase_totals, utc_time

TREE_SLOT_TEARDOWN = re.compile(
    r"Session-tree slots for phase profiling: peak_open=\d+ "
    r"\(target concurrency \d+\); "
    r"released (\d+) still-open at teardown; late_events=\d+"
)


@dataclass
class TreeCompletionEvidence:
    status: Literal["incomplete", "unknown"] = "unknown"
    reasons: list[str] = field(
        default_factory=lambda: ["MISSING_COMPLETE_TREE_INVENTORY"]
    )
    scope: dict = field(default_factory=dict)
    observed_root_sessions: dict = field(
        default_factory=lambda: dict(
            sent=None, completed=None, cancelled=None, distinct_ids_in_raw=None
        )
    )
    tree_counts: dict = field(
        default_factory=lambda: dict(admitted=None, completed=None)
    )
    pending_children: int | None = None
    pending_joins: dict = field(default_factory=lambda: dict(active=None, future=None))
    truncated_children: int | None = None
    suppressed_joins: int | None = None
    forced_tree_slots: int | None = None
    facts: list[dict] = field(default_factory=list)
    source_digests: dict = field(default_factory=dict)
    strict_target_passed: Literal[False] = False

    def record(self, kind, source, *, incomplete=False, **details):
        self.facts.append(dict(kind=kind, source=source, **details))
        if incomplete:
            self.status = "incomplete"
            if kind not in self.reasons:
                self.reasons.append(kind)


def source_digest(path):
    try:
        with path.open("rb") as source:
            return dict(
                path=str(path), sha256=hashlib.file_digest(source, "sha256").hexdigest()
            )
    except OSError as error:
        return dict(path=str(path), sha256=None, error=str(error))


def read_summary(path, evidence):
    try:
        summary = json.loads(path.read_text())
        if not isinstance(summary, dict):
            raise ValueError("summary must be an object")
        return summary
    except (OSError, ValueError) as error:
        evidence.reasons.append("UNAVAILABLE_SUMMARY")
        evidence.record("UNAVAILABLE_SUMMARY", "summary", detail=str(error))
        return {}


def profiling_window(lines, path, summary, evidence):
    try:
        totals = phase_totals(path, summary)
    except (ValueError, KeyError, TypeError) as error:
        evidence.reasons.append("UNBOUND_PROFILING_LOG")
        evidence.record("UNBOUND_PROFILING_LOG", "harness_log", detail=str(error))
        return []
    marker = " - PhaseRunner - NOTICE - Phase profiling (profiling) "
    lifecycle = [index for index, line in enumerate(lines) if marker in line]
    start, _, end = lifecycle
    if any(
        "Starting AIPerf System" in line
        or re.search(r" - PhaseRunner - NOTICE - Phase .+ started(?: \||$)", line)
        or (
            "Session-tree slots for phase " in line
            and "for phase profiling:" not in line
        )
        for line in lines[start + 1 : end + 1]
    ):
        evidence.reasons.append("AMBIGUOUS_PHASE_BOUNDARY")
        return []
    stop = end
    completed_at = utc_time(lines[end].split(" - ", 1)[0])
    for index in range(end + 1, len(lines)):
        line = lines[index]
        try:
            seconds_after = (
                utc_time(line.split(" - ", 1)[0]) - completed_at
            ).total_seconds()
        except ValueError:
            break
        if not 0 <= seconds_after <= 1:
            break
        if (
            "Starting AIPerf System" in line
            or re.search(r" - PhaseRunner - NOTICE - Phase .+ started \|", line)
            or "All credits completed" in line
        ):
            break
        if "Session-tree slots for phase " in line:
            parts = line.split(" - ", 3)
            if (
                len(parts) == 4
                and parts[1] == "PhaseRunner"
                and TREE_SLOT_TEARDOWN.fullmatch(parts[3])
            ):
                stop = index
            break
    evidence.scope.update(
        phase_index=0,
        profiling_index=0,
        log_start_line=start + 1,
        log_complete_line=end + 1,
        log_complete_time=lines[end].split(" - ", 1)[0],
        log_end_line=stop + 1,
        log_start_time=lines[start].split(" - ", 1)[0],
        log_end_time=lines[stop].split(" - ", 1)[0],
        teardown_bound=stop > end,
    )
    if stop == end:
        evidence.reasons.append("MISSING_PROFILING_TEARDOWN_BOUNDARY")
    evidence.record("REQUEST_PHASE_TOTALS", "harness_log", line=end + 1, values=totals)
    for key, reason in (
        ("cancelled", "CANCELLED_REQUESTS"),
        ("errors", "REQUEST_ERRORS"),
    ):
        if totals[key]:
            evidence.record(
                reason, "harness_log", incomplete=True, line=end + 1, count=totals[key]
            )
    if totals["grace_period_timeout"]:
        evidence.record(
            "GRACE_PERIOD_TIMEOUT", "harness_log", incomplete=True, line=end + 1
        )
    return list(enumerate(lines[start : stop + 1], start + 1))


def log_fact(line, number, evidence):
    parts = line.split(" - ", 3)
    if len(parts) != 4:
        return
    timestamp, logger, _, message = parts
    try:
        bound = (
            utc_time(evidence.scope["log_start_time"])
            <= utc_time(timestamp)
            <= utc_time(evidence.scope["log_end_time"])
        )
    except ValueError:
        bound = False
    if not bound:
        evidence.reasons.append("UNBOUND_LOG_FACT")
        evidence.record("UNBOUND_LOG_FACT", "harness_log", line=number)
        return
    sessions = re.search(r" \| sessions: (.+?)(?: \||$)", message)
    if (
        logger == "PhaseRunner"
        and message.startswith("Phase profiling (profiling) ")
        and sessions
    ):
        values = {
            key: int(value.replace(",", ""))
            for key, value in re.findall(
                r"(sent|completed|cancelled)=([0-9][0-9,]*)", sessions[1]
            )
        }
        evidence.observed_root_sessions.update(values)
        evidence.record(
            "ROOT_SESSION_COUNTERS", "harness_log", line=number, values=values
        )
        if values.get("cancelled", 0):
            evidence.record(
                "CANCELLED_ROOT_SESSIONS",
                "harness_log",
                incomplete=True,
                line=number,
                count=values["cancelled"],
            )
    if logger == "PhaseRunner" and "Forcing phase completion." in message:
        evidence.record(
            "FORCED_PHASE_COMPLETION", "harness_log", incomplete=True, line=number
        )
    cleanup_bound = (
        evidence.scope["teardown_bound"]
        and evidence.scope["log_complete_line"]
        < number
        <= evidence.scope["log_end_line"]
        and utc_time(evidence.scope["log_complete_time"]) <= utc_time(timestamp)
    )
    slots = TREE_SLOT_TEARDOWN.fullmatch(message)
    if (
        logger == "PhaseRunner"
        and slots
        and cleanup_bound
        and number == evidence.scope["log_end_line"]
    ):
        evidence.forced_tree_slots = int(slots[1])
        evidence.record(
            "TREE_SLOT_TEARDOWN",
            "harness_log",
            line=number,
            count=int(slots[1]),
            incomplete=int(slots[1]) > 0,
        )
    if logger == "aiperf.timing.branch_orchestrator":
        if cleanup_bound:
            branch_fact(message, number, evidence)
        else:
            evidence.reasons.append("UNBOUND_BRANCH_CLEANUP")
            evidence.record("UNBOUND_BRANCH_CLEANUP", "harness_log", line=number)


def branch_fact(message, number, evidence):
    leaked = re.fullmatch(
        r"BranchOrchestrator leaked state at cleanup: "
        r"(\d+) active_joins, (\d+) future_joins, "
        r"(\d+) tracked children, (\d+) parents with descendants",
        message,
    )
    if leaked:
        active, future, children, parents = map(int, leaked.groups())
        evidence.pending_children = children
        evidence.pending_joins.update(active=active, future=future)
        evidence.record(
            "PENDING_BRANCH_STATE",
            "harness_log",
            line=number,
            incomplete=any((active, future, children, parents)),
            active_joins=active,
            future_joins=future,
            children=children,
            parents=parents,
        )
    if message.startswith("Abandoned pending join for parent "):
        evidence.record(
            "ABANDONED_JOIN",
            "harness_log",
            incomplete=True,
            line=number,
            message=message,
        )
    if message.startswith("BranchOrchestrator stats: "):
        match = re.search(r"\bjoins_suppressed=(\d+)\b", message)
        if match:
            evidence.suppressed_joins = int(match[1])
            evidence.record(
                "SUPPRESSED_JOINS",
                "harness_log",
                line=number,
                incomplete=int(match[1]) > 0,
                count=int(match[1]),
            )


def summary_facts(summary, evidence):
    stats = summary.get("branch_stats", {})
    if not isinstance(stats, dict):
        evidence.reasons.append("INVALID_BRANCH_STATS")
        stats = {}
    for key, target, reason in (
        ("children_truncated", "truncated_children", "TRUNCATED_CHILDREN"),
        ("joins_suppressed", "suppressed_joins", "SUPPRESSED_JOINS"),
    ):
        value = stats.get(key)
        if type(value) is int and value >= 0:
            setattr(evidence, target, value)
            evidence.record(
                reason,
                "summary",
                pointer=f"/branch_stats/{key}",
                count=value,
                incomplete=value > 0,
            )
    for key in ("children_errored", "parents_failed_due_to_child_error"):
        value = stats.get(key)
        if type(value) is int and value > 0:
            evidence.record(
                "BRANCH_ERRORS",
                "summary",
                incomplete=True,
                pointer=f"/branch_stats/{key}",
                count=value,
            )
    for key in (
        "context_overflow_count",
        "skipped_context_overflow_count",
        "error_request_count",
    ):
        metric = summary.get(key, {})
        if not isinstance(metric, dict):
            evidence.reasons.append("INVALID_SUMMARY_METRIC")
            continue
        value = metric.get("avg")
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value > 0
        ):
            evidence.record(
                key.upper(),
                "summary",
                incomplete=True,
                pointer=f"/{key}/avg",
                count=value,
            )
    if summary.get("was_cancelled") is True:
        evidence.record(
            "RUN_CANCELLED", "summary", incomplete=True, pointer="/was_cancelled"
        )


def bound_raw_metadata(line, number, evidence):
    try:
        record = json.loads(line)
        if not isinstance(record, dict) or not isinstance(record.get("metadata"), dict):
            raise ValueError("raw record must have a metadata object")
        metadata = record["metadata"]
    except ValueError as error:
        evidence.reasons.append("INVALID_RAW_RECORD")
        evidence.record("INVALID_RAW_RECORD", "raw", line=number, detail=str(error))
        return None
    if metadata.get("benchmark_phase") == "warmup":
        return {}
    if metadata.get("benchmark_phase") != "profiling" or any(
        type(metadata.get(key)) is not int or metadata[key] != 0
        for key in ("phase_index", "profiling_index")
    ):
        evidence.reasons.append("UNBOUND_RAW_PHASE")
        evidence.record("UNBOUND_RAW_PHASE", "raw", line=number)
        return None
    try:
        start = utc_time(evidence.scope["start_time"]).timestamp() * 1e9
        end = utc_time(evidence.scope["end_time"]).timestamp() * 1e9
        request_start, request_end = (
            metadata.get(key) for key in ("request_start_ns", "request_end_ns")
        )
        bound = (
            type(request_start) is int
            and type(request_end) is int
            and start <= request_start <= request_end <= end
        )
    except (ValueError, TypeError):
        bound = False
    if not bound:
        evidence.reasons.append("UNBOUND_RAW_TIME")
        evidence.record("UNBOUND_RAW_TIME", "raw", line=number)
        return None
    return metadata


def raw_observations(path, evidence):
    roots, missing, observed = set(), False, False
    try:
        source = path.open()
    except OSError as error:
        evidence.reasons.append("MISSING_RAW_RECORDS")
        evidence.record("MISSING_RAW_RECORDS", "raw", detail=str(error))
        return
    with source:
        try:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                metadata = bound_raw_metadata(line, number, evidence)
                if metadata is None:
                    missing = True
                    continue
                if not metadata:
                    continue
                observed = True
                root = metadata.get("root_correlation_id")
                if isinstance(root, str) and root:
                    roots.add(root)
                else:
                    missing = True
                for key in ("was_cancelled", "context_overflow_skip"):
                    if metadata.get(key) is True:
                        evidence.record(
                            key.upper(), "raw", incomplete=True, line=number
                        )
        except (OSError, UnicodeError) as error:
            missing = True
            evidence.reasons.append("UNREADABLE_RAW_RECORDS")
            evidence.record("UNREADABLE_RAW_RECORDS", "raw", detail=str(error))
    if observed and not missing:
        evidence.observed_root_sessions["distinct_ids_in_raw"] = len(roots)
    else:
        evidence.reasons.append("MISSING_ROOT_SESSION_IDS")


def write_tree_completion(artifacts, output, harness_log=None):
    artifacts, output = Path(artifacts), Path(output)
    sources = dict(
        summary=artifacts / "profile_export_aiperf.json",
        raw=artifacts / "profile_export_raw.jsonl",
        harness_log=artifacts / "logs/aiperf.log"
        if harness_log is None
        else Path(harness_log),
    )
    evidence = TreeCompletionEvidence()
    evidence.source_digests = {
        key: source_digest(path) for key, path in sources.items()
    }
    summary = read_summary(sources["summary"], evidence)
    evidence.scope = dict(
        target="all_admitted_session_trees",
        benchmark_id=summary.get("benchmark_id"),
        phase="profiling",
        phase_index=None,
        profiling_index=None,
        start_time=summary.get("start_time"),
        end_time=summary.get("end_time"),
        log_start_line=None,
        log_end_line=None,
        teardown_bound=False,
    )
    summary_facts(summary, evidence)
    raw_observations(sources["raw"], evidence)
    log = sources["harness_log"]
    try:
        lines = log.read_text().splitlines()
    except (OSError, UnicodeError) as error:
        lines = []
        evidence.reasons.append("UNREADABLE_HARNESS_LOG")
        evidence.record("UNREADABLE_HARNESS_LOG", "harness_log", detail=str(error))
    for number, line in profiling_window(lines, log, summary, evidence):
        log_fact(line, number, evidence)
    evidence.reasons = sorted(set(evidence.reasons))
    result = asdict(evidence)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")
    except OSError as error:
        result["reasons"].append("TREE_REPORT_WRITE_FAILED")
        print(f"Cannot save AgentX tree evidence: {error}", file=sys.stderr)
    return result
