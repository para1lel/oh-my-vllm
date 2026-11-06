import copy
import hashlib
import json
import time

from .qwen_fp8_evidence import Blocked, digest, require

KEY_FIELDS = ("worker_epoch", "step_id", "request_id", "lease_id")


def wire_digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def key(value):
    result = tuple(value[name] for name in KEY_FIELDS)
    require(
        all(type(item) is int and item > 0 for item in result), "invalid ExecutionKey"
    )
    return result


def decode_trace(data):
    events = [json.loads(line) for line in data.splitlines()]
    require(
        all(
            isinstance(event, dict) and isinstance(event.get("operation"), str)
            for event in events
        ),
        "malformed trace event",
    )
    return events


def trace_events(previous, current):
    require(
        current.startswith(previous), "execution trace changed before case boundary"
    )
    identities = [
        event
        for event in decode_trace(previous)
        if event["operation"] in ("identity", "rust_identity")
    ]
    return identities + decode_trace(current[len(previous) :])


def read_trace(path):
    return path.read_bytes() if path.exists() else b""


def wait_for_completions(path, previous, request_ids, timeout=2.0):
    deadline = time.monotonic() + timeout
    while True:
        current = read_trace(path)
        if not current.startswith(previous):
            return current
        lines = current[len(previous) :].splitlines()
        if current and not current.endswith(b"\n"):
            lines = lines[:-1]
        completed = {
            row["id"]
            for line in lines
            for event in [json.loads(line)]
            if event["operation"] == "complete"
            for row in event["requests"]
        }
        if request_ids <= completed or time.monotonic() >= deadline:
            return current
        time.sleep(0.01)


def loaded_identity(events, rust, artifacts):
    loaded = [row for row in events if row["operation"] == "identity"]
    require(len(loaded) == 1, "requires exactly one worker loaded identity")
    actual = copy.deepcopy(loaded[0])
    require(
        not actual.get("identity_error"),
        f"loaded identity observation failed: {actual.get('identity_error')}",
    )
    bindings = [row for row in rust if row["operation"] == "rust_identity"]
    require(len(bindings) == 1, "requires exactly one Rust loaded identity")
    binding = bindings[0]
    require(
        binding["worker_epoch"] == actual["config"]["worker_epoch"],
        "Rust and worker epochs differ",
    )
    require(
        binding["binary"] == actual["binary"] == artifacts["binary"],
        "running Rust binary differs",
    )
    require(
        binding["artifact_id"] == actual["artifact_id"] == digest(artifacts),
        "Rust and worker run bindings differ",
    )
    require(
        binding["rust_sources_sha256"] == artifacts["rust_sources_sha256"],
        "running Rust build sources differ",
    )
    require(actual["run_manifest"] == artifacts, "worker run manifest differs")
    actual["accounting"] = actual["run_manifest"]["accounting"]
    require(
        actual["cache_layout"] == binding["cache_layout"],
        "Rust Ready cache layout differs from worker layout",
    )
    actual["cache_memory"] = actual["cache_layout"]
    return actual


def wire_calls(events):
    scheduled, results, dispatches = {}, {}, {}
    for event in events:
        operation = event["operation"]
        if operation not in ("worker_scheduled", "worker_result", "dispatch"):
            continue
        at = (event["worker_epoch"], event["step_id"])
        if operation == "dispatch":
            dispatches.setdefault(at, []).append(event)
            continue
        target = scheduled if operation == "worker_scheduled" else results
        require(at not in target, "duplicate worker call")
        target[at] = event
    require(
        scheduled.keys() == results.keys(), "worker scheduled/result inventory differs"
    )
    require(dispatches.keys() <= scheduled.keys(), "orphan dispatch epoch or step")
    for at, event in scheduled.items():
        forward = any(row["action"] == "forward" for row in event["scheduled"])
        require(
            bool(dispatches.get(at)) == forward,
            "missing forward dispatch" if forward else "dispatch for zero-row call",
        )
    return scheduled, results, dispatches


def prepare_key(row):
    values = tuple(row[name] for name in ("worker_epoch", "rpc_id", "request_id"))
    require(
        all(type(value) is int and value > 0 for value in values), "invalid prepare key"
    )
    return values


def prepared_calls(events, epoch):
    observed = {}
    for event in events:
        if event["operation"] == "prepared":
            at = prepare_key(event)
            require(at[0] == epoch, "prepare came from a different loaded worker epoch")
            require(at not in observed, "duplicate worker prepare observation")
            observed[at] = event
    return observed


def join_prepared(event, observed, requests, used):
    at = prepare_key(event)
    require(at not in used and at in observed, "unmatched or duplicate Rust prepare")
    require(at[2] in requests, "prepare includes another case")
    actual = observed[at]
    require(
        actual["prompt_token_sha256"] == event["prompt_token_sha256"],
        "Rust and worker prepared prompt differs",
    )
    intent = {
        name: event[name]
        for name in (
            "normalized_sampling_f32_bits",
            "seed",
            "repetition_guard",
            "min_tokens",
            "max_tokens",
        )
    }
    used.add(at)
    return actual | dict(operation="prepared", rust_intent=intent)


def join_events(worker_events, rust_events, manifest, artifacts):
    identity = loaded_identity(worker_events, rust_events, artifacts)
    scheduled, results, dispatches = wire_calls(worker_events)
    joined, used = [identity], set()
    requests = {row["id"] for row in manifest["requests"]}
    prepares = prepared_calls(worker_events, identity["config"]["worker_epoch"])
    used_prepares = set()
    for event in rust_events:
        operation = event["operation"]
        if operation == "rust_identity":
            continue
        if operation == "prepare_validated":
            joined.append(join_prepared(event, prepares, requests, used_prepares))
        elif operation == "validated":
            keys = [key(row) for row in event["executions"]]
            require(
                keys and len(set(keys)) == len(keys), "duplicate Rust execution key"
            )
            at = keys[0][:2]
            require(
                all(value[:2] == at for value in keys), "mixed Rust execution steps"
            )
            require(at not in used and at in scheduled, "unmatched Rust validation")
            rows, output = scheduled[at], results[at]
            proof = event["wire"]
            require(
                (proof["worker_epoch"], proof["step_id"]) == at,
                "Rust wire proof has a different execution step",
            )
            require(
                proof["scheduled_sha256"] == wire_digest(rows["scheduled"])
                and proof["outputs_sha256"] == wire_digest(output["outputs"]),
                "archived wire differs from Rust-validated bytes",
            )
            require(rows["rpc_id"] == output["rpc_id"], "worker RPC differs")
            expected = {
                (*at, row["request_id"], row["lease_id"]) for row in rows["scheduled"]
            }
            received = {
                (*at, row["request_id"], row["lease_id"]) for row in output["outputs"]
            }
            require(
                set(keys) == expected == received, "validated batch inventory differs"
            )
            require(
                all(value[2] in requests for value in keys),
                "trace includes another case",
            )
            used.add(at)
            joined.append(
                dict(
                    operation="accounting_call",
                    worker_epoch=at[0],
                    step_id=at[1],
                    validated=True,
                    scheduled=rows["scheduled"],
                    outputs=output["outputs"],
                    dispatches=dispatches.get(at, []),
                )
            )
        elif operation in ("accepted", "complete"):
            require(
                all(row["id"] in requests for row in event["requests"]),
                "lifecycle includes another case",
            )
            joined.append(event)
        else:
            raise Blocked(f"unsupported Rust trace fact: {operation}")
    nonempty = {at for at, event in scheduled.items() if event["scheduled"]}
    require(used == nonempty, "missing Rust validation for worker execution")
    require(used_prepares == prepares.keys(), "missing Rust prepare validation")
    require(
        len(used_prepares) == len(requests)
        and {at[2] for at in used_prepares} == requests,
        "missing prepare inventory",
    )
    return joined


def prompt_tokens(events, request_id):
    try:
        values = [
            row["prefill_token_ids"]
            for event in events
            if event["operation"] == "worker_scheduled"
            for row in event["scheduled"]
            if row["request_id"] == request_id
            and row.get("prefill_token_ids") is not None
        ]
    except (KeyError, TypeError, AttributeError) as error:
        raise Blocked(f"malformed prepared prompt evidence: {error}") from error
    require(len(values) == 1, "missing or repeated prepared prompt token IDs")
    return values[0]


def rust_source_identity(root):
    paths = {root / "Cargo.toml", root / "Cargo.lock"}
    paths.update(
        path
        for path in (root / "crates").rglob("*")
        if path.is_file() and (path.suffix == ".rs" or path.name == "Cargo.toml")
    )
    result = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix().encode()
        result.update(len(relative).to_bytes(8, "little"))
        result.update(relative)
        result.update(hashlib.sha256(path.read_bytes()).digest())
    return result.hexdigest()
