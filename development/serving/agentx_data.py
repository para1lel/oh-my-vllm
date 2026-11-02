import argparse
import collections
import csv
import datetime
import json
import math
import re
from pathlib import Path


def read_jsonl(path):
    with Path(path).open() as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def service_metrics(path):
    result = {}
    decoder = json.JSONDecoder()
    with Path(path).open() as source:
        for line in source:
            marker = "SERVING_REQUEST_METRIC "
            if marker not in line:
                continue
            record, _ = decoder.raw_decode(line.split(marker, 1)[1])
            response_id = record["response_id"]
            if response_id in result:
                raise ValueError(f"duplicate service response ID: {response_id}")
            result[response_id] = record
    return result


def stream_result(record):
    ids, usages, finishes = set(), [], []
    done = False
    for response in record["responses"]:
        for packet in response.get("packets", []):
            if packet["name"] != "data":
                continue
            if packet["value"] == "[DONE]":
                done = True
                continue
            chunk = json.loads(packet["value"])
            if chunk.get("error"):
                return None, None, False
            if chunk.get("id"):
                ids.add(chunk["id"])
            if chunk.get("usage"):
                usages.append(chunk["usage"])
            finishes.extend(
                choice["finish_reason"]
                for choice in chunk.get("choices", [])
                if choice.get("finish_reason")
            )
    if len(ids) != 1:
        raise ValueError("a stream must identify exactly one full response ID")
    if not done or not finishes:
        return next(iter(ids)), None, False
    if len(usages) != 1:
        raise ValueError("a completed stream must supply exactly one usage record")
    return next(iter(ids)), usages[0], True


def finite_number(value, name, minimum=0):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
    ):
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def token_count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def completed_row(record, usage, metric, profile_start, max_context=262144):
    try:
        prompt = token_count(usage["prompt_tokens"], "prompt tokens", 1)
        output = token_count(usage["completion_tokens"], "output tokens", 1)
        total = token_count(usage["total_tokens"], "total tokens", 1)
        reasoning = token_count(
            usage["completion_tokens_details"]["reasoning_tokens"], "reasoning tokens"
        )
        cached = token_count(
            usage["prompt_tokens_details"]["cached_tokens"], "cached tokens"
        )
        seconds = finite_number(metric["processing_seconds"], "processing seconds")
        started = finite_number(metric["processing_started_unix_s"], "start time")
        payload = record["payload"]
        requested = payload.get("max_tokens", payload.get("max_completion_tokens"))
        if (
            seconds <= 0
            or started < profile_start
            or cached > prompt
            or total != prompt + output
            or reasoning != 0
            or metric["status"] != "completed"
            or metric["output_tokens"] != output
            or metric["prefix_hit_tokens"] != cached
            or requested != output
            or payload.get("ignore_eos") is not True
            or prompt + requested > max_context
        ):
            raise ValueError("service metrics, usage, or requested lengths disagree")
        metadata = record["metadata"]
        client_seconds = (
            max(response["perf_ns"] for response in record["responses"])
            - record["start_perf_ns"]
        ) / 1e9
        finite_number(client_seconds, "client seconds")
        return {
            "response_id": metric["response_id"],
            "client_request_id": metadata["x_request_id"],
            "conversation_id": metadata.get("conversation_id", ""),
            "source_trace_id": metadata.get("source_trace_id", ""),
            "turn_index": metadata.get("turn_index", ""),
            "processing_started_unix_s": started,
            "profile_started_unix_s": profile_start,
            "processing_start_s": started - profile_start,
            "processing_seconds": seconds,
            "output_tokens": output,
            "output_tokens_per_second": output / seconds,
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "cache_hit_ratio": cached / prompt,
            "queue_seconds": finite_number(metric["queue_seconds"], "queue seconds"),
            "client_seconds": client_seconds,
        }
    except KeyError as error:
        raise ValueError(f"missing required measurement field: {error}") from error


def collect_rows(records, metrics, profile_start, max_context=262144):
    finite_number(profile_start, "profile start")
    counts = collections.Counter(
        dict(
            warmup=0,
            completed=0,
            failed=0,
            cancelled=0,
            incomplete=0,
            skipped_context_overflow=0,
        )
    )
    rows, seen, client_ids = [], set(), set()
    for record in records:
        metadata = record["metadata"]
        phase = metadata["benchmark_phase"]
        if phase == "warmup":
            counts["warmup"] += 1
            continue
        if phase != "profiling":
            raise ValueError(f"unknown benchmark phase: {phase}")
        client_id = metadata["x_request_id"]
        if not client_id or client_id in client_ids:
            raise ValueError(f"duplicate or missing client ID: {client_id}")
        client_ids.add(client_id)
        if metadata.get("context_overflow_skip"):
            counts["skipped_context_overflow"] += 1
            continue
        if metadata.get("was_cancelled"):
            counts["cancelled"] += 1
            continue
        if record.get("error") or record.get("status") != 200:
            counts["failed"] += 1
            continue
        response_id, usage, complete = stream_result(record)
        if not complete:
            counts["incomplete"] += 1
            continue
        if response_id in seen or response_id not in metrics:
            raise ValueError(f"duplicate or unmatched response ID: {response_id}")
        seen.add(response_id)
        metric = metrics[response_id]
        if metric["response_id"] != response_id:
            raise ValueError("service metric key differs from its response ID")
        rows.append(completed_row(record, usage, metric, profile_start, max_context))
        counts["completed"] += 1
    return sorted(rows, key=lambda row: row["processing_start_s"]), dict(counts)


def utc_time(value):
    result = datetime.datetime.fromisoformat(value)
    return result.replace(tzinfo=datetime.UTC) if result.tzinfo is None else result


def phase_totals(path, summary):
    if summary.get("aiperf_version") != "1.0.7":
        raise ValueError("phase log accounting requires agentx-harness==1.0.7")
    number = r"(?:0|[1-9][0-9]{0,2}(?:,[0-9]{3})*)"
    patterns = (
        r"started \| phase_index=0 \| profiling_index=0(?: \| target: [^|]+)?",
        rf"sending complete \| sent=({number}), completed=({number}), "
        rf"in_flight=({number})(?: \| sessions: sent={number}, completed={number})?"
        r"(?: \| timeout_triggered=True)?",
        rf"complete \| completed=({number}), cancelled=({number}), errors=({number})"
        rf"(?: \| sessions: completed={number}, cancelled={number})?"
        r" \| elapsed=([0-9]+\.[0-9]{2})s( \| grace_period_timeout=True)?",
    )
    events = []
    try:
        with Path(path).open() as source:
            for line in source:
                marker = " - PhaseRunner - NOTICE - Phase profiling (profiling) "
                if marker not in line:
                    continue
                timestamp, message = line.rstrip().split(marker, 1)
                index = len(events)
                match = re.fullmatch(patterns[index], message) if index < 3 else None
                if match is None:
                    raise ValueError("invalid or duplicate profiling phase log")
                events.append((utc_time(timestamp), match.groups()))
    except OSError as error:
        raise ValueError("authoritative AgentX phase totals unavailable") from error
    if len(events) != 3:
        raise ValueError("authoritative AgentX phase totals unavailable")
    (start, _), (sending_end, sent_fields), (end, returned_fields) = events
    sent, early_completed, in_flight = (
        int(value.replace(",", "")) for value in sent_fields
    )
    returned, cancelled, errors = (
        int(value.replace(",", "")) for value in returned_fields[:3]
    )
    elapsed = float(returned_fields[3])
    if (
        not start <= sending_end <= end
        or abs((start - utc_time(summary["start_time"])).total_seconds()) > 1
        or abs((end - utc_time(summary["end_time"])).total_seconds()) > 1
        or abs((end - start).total_seconds() - elapsed) > 0.02
        or early_completed > returned
        or early_completed + in_flight > sent
        or sent != returned + cancelled
        or errors > returned
    ):
        raise ValueError("inconsistent AgentX phase totals or timestamps")
    return {
        "sent": sent,
        "returned": returned,
        "cancelled": cancelled,
        "errors": errors,
        "profiling_seconds": (sending_end - start).total_seconds(),
        "drain_seconds": (end - sending_end).total_seconds(),
        "phase_seconds": elapsed,
        "grace_period_timeout": returned_fields[4] is not None,
    }


def reconcile_counts(counts, totals):
    errors = counts["failed"] + counts["incomplete"]
    # Harness "completed" credits include errors and skips, but not cancellations.
    returned = counts["completed"] + errors + counts["skipped_context_overflow"]
    if (
        returned != totals["returned"]
        or errors != totals["errors"]
        or counts["cancelled"] > totals["cancelled"]
    ):
        raise ValueError("raw export differs from authoritative AgentX phase totals")
    # The pinned harness can return cancelled credits without exporting raw records.
    counts["cancelled_without_raw"] = totals["cancelled"] - counts["cancelled"]
    counts.update({key: value for key, value in totals.items() if key != "errors"})
    counts["accounting_complete"] = True
    counts["accounting_source"] = "agentx-harness-1.0.7-phase-log"


def export_run(
    artifacts, output, max_context=262144, service_log=None, harness_log=None
):
    artifacts, output = Path(artifacts), Path(output)
    summary = json.loads((artifacts / "profile_export_aiperf.json").read_text())
    if summary.get("metadata", {}).get("submission_valid") is not True or summary.get(
        "was_cancelled"
    ):
        raise ValueError("AgentX did not produce a valid completed measurement")
    # The launcher sets TZ=UTC because upstream exports a naive local datetime.
    start = utc_time(summary["start_time"])
    totals = phase_totals(
        artifacts / "logs/aiperf.log" if harness_log is None else harness_log,
        summary,
    )
    rows, counts = collect_rows(
        read_jsonl(artifacts / "profile_export_raw.jsonl"),
        service_metrics(
            artifacts.parent / "mtp4-service.log"
            if service_log is None
            else service_log
        ),
        start.timestamp(),
        max_context,
    )
    if not rows:
        raise ValueError("no completed profiling requests")
    expected = summary["request_count"]["avg"]
    if len(rows) != expected:
        raise ValueError(f"raw export lost requests: {len(rows)} != {expected}")
    skipped = summary.get("skipped_context_overflow_count", {}).get("avg", 0)
    if counts["skipped_context_overflow"] != skipped:
        raise ValueError("raw export context overflow count differs from the summary")
    expected_errors = summary.get("error_request_count", {}).get("avg", 0)
    if (
        sum(counts[name] for name in ("failed", "cancelled", "incomplete"))
        != expected_errors
    ):
        raise ValueError("raw export error count differs from the summary")
    counts["context_overflow"] = summary.get("context_overflow_count", {}).get("avg", 0)
    reconcile_counts(counts, totals)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (artifacts.parent / "request-counts.json").write_text(
        json.dumps(counts, indent=2) + "\n"
    )
    return rows, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--harness-log", type=Path)
    arguments = parser.parse_args()
    _, counts = export_run(
        arguments.artifacts, arguments.csv, harness_log=arguments.harness_log
    )
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
