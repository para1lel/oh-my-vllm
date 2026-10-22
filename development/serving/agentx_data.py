import argparse
import collections
import csv
import datetime
import json
import math
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


def completed_row(record, usage, metric, profile_start):
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
            or prompt + requested > 262144
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


def collect_rows(records, metrics, profile_start):
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
        rows.append(completed_row(record, usage, metric, profile_start))
        counts["completed"] += 1
    return sorted(rows, key=lambda row: row["processing_start_s"]), dict(counts)


def export_run(artifacts, output):
    artifacts, output = Path(artifacts), Path(output)
    summary = json.loads((artifacts / "profile_export_aiperf.json").read_text())
    if summary.get("metadata", {}).get("submission_valid") is not True or summary.get(
        "was_cancelled"
    ):
        raise ValueError("AgentX did not produce a valid completed measurement")
    # The launcher sets TZ=UTC because upstream exports a naive local datetime.
    start = datetime.datetime.fromisoformat(summary["start_time"])
    if start.tzinfo is None:
        start = start.replace(tzinfo=datetime.UTC)
    rows, counts = collect_rows(
        read_jsonl(artifacts / "profile_export_raw.jsonl"),
        service_metrics(artifacts.parent / "mtp4-service.log"),
        start.timestamp(),
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
    arguments = parser.parse_args()
    _, counts = export_run(arguments.artifacts, arguments.csv)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
