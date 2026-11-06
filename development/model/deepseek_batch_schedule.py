import json


def execute_message(rows, *, rpc_id, evicted=()):
    return dict(
        type="execute",
        rpc_id=rpc_id,
        scheduled=rows,
        finished_request_ids=[],
        preempted_request_ids=list(evicted),
    )


def reset_graphs(worker, torch):
    torch.cuda.synchronize(worker.device)
    for graph in worker.decode_graphs.values():
        graph.close()
    worker.decode_graphs.clear()
    for bindings in worker.decode_bindings.values():
        bindings.close()
    worker.decode_bindings.clear()
    worker.decode_engram.clear()


def continuation_rows(worker, outputs, make_row, step):
    keys = [key for key in outputs if not worker.requests[key].finish]
    if keys and step >= 5:
        offset = (step - 4) % len(keys)
        keys = keys[offset:] + keys[:offset]
        if step % 2:
            keys.reverse()
    rows = []
    for key in keys:
        request = worker.requests[key]
        start = request.state_position
        if len(request.history) - start != 1:
            raise AssertionError("next target must start at the uncomputed output")
        drafts = outputs[key]["new_draft_token_ids"]
        if len(drafts) > min(5, max(0, request.limit - request.generated - 1)):
            raise AssertionError(
                "returned proposal exceeds the remaining output budget"
            )
        rows.append(make_row(key, [request.history[start], *drafts], start))
    return rows


def execute_scheduled(worker, rows, step, reference, evicted):
    if reference:
        if evicted:
            worker.execute(execute_message([], rpc_id=1000 + step, evicted=evicted))
        return [
            worker.execute(execute_message([item], rpc_id=2000 + step))["outputs"][0]
            for item in rows
        ]
    return worker.execute(execute_message(rows, rpc_id=2000 + step, evicted=evicted))[
        "outputs"
    ]


def execute_single_round(worker, rows, *, reference, batch_decode):
    if reference:
        return {item["request_id"]: worker._execute_row(item) for item in rows}
    if batch_decode:
        return worker._execute_decode_group(rows)
    outputs = worker.execute(execute_message(rows, rpc_id=3000))["outputs"]
    if len(outputs) != len(rows):
        raise AssertionError("scheduled request count differs from the worker outputs")
    return {item["request_id"]: item for item in outputs}


def require_scheduled_requests(rows, outputs, targets):
    expected = [item["request_id"] for item in rows]
    returned = [item["request_id"] for item in outputs]
    if (
        len(expected) != len(set(expected))
        or sorted(returned) != sorted(expected)
        or set(targets) != set(expected)
        or any(len(records) != 1 for records in targets.values())
    ):
        raise AssertionError(
            "scheduled requests differ from outputs or target execution"
        )


def require_single_execution(execution, calls):
    required = "graph" if execution == "graph" else "eager"
    if not calls or any(call["execution"] != required for call in calls):
        raise AssertionError(
            f"requested {required} target execution differs from captured calls"
        )


def update_boundaries(calls, observed, crossings):
    for call in calls:
        if call["batch_size"] != len(call["requests"]):
            raise AssertionError("target batch size differs from the captured requests")
        for key, signature in zip(call["requests"], call["signatures"], strict=True):
            seen = observed.setdefault(key, {})
            values = dict(
                bucket=signature["bucket"],
                window=signature["window_block"],
                **{
                    f"compress/{ratio}": group
                    for ratio, group in signature["compressed_groups"].items()
                },
            )
            for name, value in values.items():
                seen.setdefault(name, set()).add(value)
                crossings[name] |= len(seen[name]) > 1


def require_multiround_gates(
    mode,
    execution,
    *,
    continued,
    max_batch,
    crossings,
    require_verify_continuation=True,
    require_boundaries=False,
    require_batch=True,
    scheduled_requests=None,
    required_requests=None,
):
    if required_requests is not None and scheduled_requests != required_requests:
        raise AssertionError("candidate did not schedule the full request batch")
    if mode.startswith("dspark") and require_verify_continuation and not continued:
        raise AssertionError("verification did not consume the preceding proposal")
    if execution != "reference" and require_batch and max_batch <= 1:
        raise AssertionError("candidate did not execute a multi-request target group")
    if mode != "remaining" and require_boundaries and not all(crossings.values()):
        raise AssertionError(
            f"actual target boundary coverage is incomplete: {crossings}"
        )


def write_round_report(args, rank, report):
    with (args.output / f"multiround-rank{rank}.jsonl").open("a") as output:
        output.write(json.dumps(report) + "\n")
    print(json.dumps(report), flush=True)
