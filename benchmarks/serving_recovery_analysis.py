"""Inputs and strict evidence checks for target allocator OOM recovery."""

import math
import re
from pathlib import Path

from benchmarks.agentic_soak_analysis import resource_checks
from benchmarks.common import _process_record


def body(api: str, prompt: str, *, streaming=False, maximum=32):
    value = {"model": "qwen3.8-27b-fp8", "temperature": 0, "stream": streaming}
    if api == "chat":
        value.update(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=maximum,
            reasoning_effort="off",
        )
    elif api == "responses":
        value.update(
            input=prompt,
            max_output_tokens=maximum,
            reasoning={"effort": "off"},
            store=False,
        )
    else:
        raise ValueError("unknown recovery API")
    return value


def fresh_snapshot(sample, after):
    timestamp = sample.get("worker_snapshot_unix_s")
    return (
        isinstance(timestamp, (int, float))
        and not isinstance(timestamp, bool)
        and math.isfinite(timestamp)
        and timestamp >= after
    )


def cached_tokens(result: dict) -> int:
    details = result.get("usage", {}).get("prompt_tokens_details", {})
    value = details.get("cached_tokens")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Chat response has no valid prefix-hit count")
    return value


def validate_arithmetic_answer(text: str):
    """Allow whitespace and one terminal period around the expected integer."""
    if text.strip().removesuffix(".").strip() != "25":
        raise ValueError("post-recovery API did not return exactly 25")


def validate_fault_stream(record: dict):
    if (
        record.get("status") != 200
        or not isinstance(record.get("request_id"), int)
        or isinstance(record.get("request_id"), bool)
        or record["request_id"] <= 0
        or not record.get("armed")
        or not record.get("marker_recreated")
        or record.get("explicit_errors") != 1
        or record.get("success_events") != 0
        or (record["stage"] != "graph_capture" and not record.get("visible_output"))
    ):
        raise ValueError(
            "fault stream did not fail explicitly once without replay/completion"
        )


def idle_checks(sample: dict, restarts: int) -> list[dict]:
    checks = [
        check
        for check in resource_checks(sample, require_idle=True)
        if check["name"] != "no_worker_restart"
    ]
    checks.append(
        {
            "name": "expected_worker_restarts",
            "passed": sample.get("restarts") == restarts,
        }
    )
    checks.append(
        {"name": "idle_request_bytes", "passed": sample.get("request_state_bytes") == 0}
    )
    return checks


def live_group(group: int) -> list[dict]:
    result = []
    for path in Path("/proc").glob("[0-9]*"):
        record = _process_record(int(path.name))
        if record and record["pgid"] == group and record["state"] not in ("Z", "X"):
            result.append(record)
    return result


def validate_replacement(
    before: dict, after: dict, old_process: dict, live: list[dict]
):
    worker = _process_record(after["worker_pid"])
    if (
        before["server_pid"] != after["server_pid"]
        or after["status"] != "running"
        or after["epoch"] != before["epoch"] + 1
        or after["restarts"] != before["restarts"] + 1
        or after["worker_pid"] == before["worker_pid"]
        or worker is None
        or worker["start_ticks"] <= old_process["start_ticks"]
        or live
    ):
        raise ValueError(
            "worker replacement identity/count or old process-group cleanup failed"
        )
    for pool in ("fa", "gdn"):
        if (
            after["kv"][pool + "_cached"] != 0
            or after["kv"][pool + "_free"] != after["kv"][pool + "_blocks"] - 1
        ):
            raise ValueError("replacement logical KV cache is not empty")


def validate_oom(log: str):
    if (
        "Traceback (most recent call last)" not in log
        or "worker_faults.py" not in log
        or "torch.empty(free + cached" not in log
        or not re.search(r"(?:OutOfMemoryError|out of memory).*", log, re.IGNORECASE)
        or "test-owned OOM" in log
    ):
        raise ValueError(
            "no actual allocator OOM traceback from the owned fault fixture"
        )


def validate_configuration(observed, mode, source):
    expected = {
        "speculative_mode": mode,
        "block_size": "784",
        "max_model_len": "262144",
        "num_speculative_tokens": "4" if mode == "mtp" else "7",
        # The physical allocation option counts three FA layer groups per
        # logical page. Serving reports the resulting logical FA capacity.
        "worker_fa_pool_blocks": "1400",
        "fa_pool_blocks": "1400",
        "worker_gdn_pool_blocks": "128",
        "gdn_pool_blocks": "128",
        "rust_sources_sha256": source["rust_sources_sha256"],
    }
    if any(observed.get(key) != value for key, value in expected.items()):
        raise ValueError("loaded binary configuration differs from source or mode")
