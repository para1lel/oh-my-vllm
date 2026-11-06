"""Validate and apply Rust schedules without assigning request owners."""

from ...worker.wire import validate_execute, validate_result


def local_execute(message, rank):
    plan = message.get("rank_plan")
    if not isinstance(plan, list) or len(plan) != 4:
        raise ValueError("DEP4 needs four explicit Rust rank schedules")
    if [row.get("rank") for row in plan] != list(range(4)):
        raise ValueError("DEP4 rank schedules must use rank order")
    for field, source in (
        ("request_ids", [row["request_id"] for row in message["scheduled"]]),
        ("finished_request_ids", message["finished_request_ids"]),
        ("preempted_request_ids", message["preempted_request_ids"]),
    ):
        ids = [request for row in plan for request in row[field]]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(source):
            raise ValueError("DEP4 rank schedule is not an exact request partition")
    local = dict(message)
    schedule = plan[rank]
    by_id = {row["request_id"]: row for row in message["scheduled"]}
    local["scheduled"] = [by_id[request] for request in schedule["request_ids"]]
    local["finished_request_ids"] = schedule["finished_request_ids"]
    local["preempted_request_ids"] = schedule["preempted_request_ids"]
    local["num_batched_tokens"] = sum(
        len(row["token_ids"]) for row in local["scheduled"]
    )
    return local, schedule


def handle_rank(worker, message, rank):
    kind = message["type"]
    if kind in {"prepare", "register"}:
        owner = message.get("owner_rank")
        if type(owner) is not int or owner not in range(4):
            raise ValueError("DEP4 registration requires a Rust owner")
        if owner != rank:
            return None
    if kind == "execute":
        local, schedule = local_execute(message, rank)
        reply = worker.dep4_executor.execute(lambda: worker.handle(local))
        reply["rank_completion"] = {
            "rank": rank,
            "step_id": message["step_id"],
            "worker_epoch": message["worker_epoch"],
            "schedule": schedule,
        }
        return reply
    if kind == "abort":
        worker.handle(message)
        device = next(worker.model.parameters()).device
        if device.type == "cuda":
            import torch

            torch.cuda.synchronize(device)
        return dict(
            type="abort_result",
            rpc_id=message["rpc_id"],
            request_id=message["request_id"],
            rank=rank,
        )
    return worker.handle(message)


def merge_replies(message, replies):
    kind = message["type"]
    if kind in {"prepare", "register"}:
        owner = message.get("owner_rank")
        if type(owner) is not int or owner not in range(4):
            raise ValueError("DEP4 registration requires a Rust owner")
        if any(
            replies[rank].get("type") != "ack" for rank in range(4) if rank != owner
        ):
            raise ValueError("DEP4 non-owner returned a preparation result")
        return replies[owner]
    if kind == "abort":
        for rank in range(4):
            if replies[rank] != dict(
                type="abort_result",
                rpc_id=message["rpc_id"],
                request_id=message["request_id"],
                rank=rank,
            ):
                raise ValueError("DEP4 rank did not acknowledge request abort")
        return dict(
            type="abort_result",
            rpc_id=message["rpc_id"],
            request_id=message["request_id"],
            ranks=list(range(4)),
        )
    if kind == "execute":
        return _merge_execute(message, replies)
    if kind == "stats":
        reply = dict(replies[0])
        reply["stats"] = dict(reply["stats"])
        reply["stats"]["ranks"] = [replies[rank]["stats"] for rank in range(4)]
        return reply
    return replies[0]


def _require_four_replies(replies):
    if set(replies) != set(range(4)):
        raise ValueError("DeepSeek execution requires all four rank replies")


def merge_replicated_execute(message, replies):
    _require_four_replies(replies)
    expected = validate_execute(message, message["worker_epoch"])
    reference = None
    fields = (
        "token_ids",
        "num_accepted_draft_tokens",
        "new_draft_token_ids",
        "text",
        "finish_reason",
        "reasoning_tokens",
        "computed_end",
        "accepted_end",
        "validated_exports",
        "committed_page_valid_ends",
        "mtp_accepted_endpoint",
    )
    for rank in range(4):
        reply = replies[rank]
        if reply.get("type") == "error":
            raise RuntimeError(
                f"DeepSeek rank {rank} returned an error: {reply.get('message')}"
            )
        validate_result(message, reply, expected)
        outputs = {}
        for row in reply["outputs"]:
            if row.get("error") is not None:
                raise RuntimeError(
                    f"DeepSeek rank {rank} request {row['request_id']} "
                    f"failed: {row['error']}"
                )
            outputs[row["request_id"]] = tuple(row[field] for field in fields)
        if reference is not None and outputs != reference:
            raise ValueError(
                f"DeepSeek rank {rank} execution data differs from rank zero"
            )
        reference = outputs
    return replies[0]


def _merge_execute(message, replies):
    _require_four_replies(replies)
    outputs = {}
    completions = []
    for rank in range(4):
        _local, schedule = local_execute(message, rank)
        reply = replies[rank]
        if (
            reply.get("type") != "execute_result"
            or reply.get("rpc_id") != message["rpc_id"]
            or reply.get("step_id") != message["step_id"]
            or reply.get("worker_epoch") != message["worker_epoch"]
        ):
            raise ValueError("DEP4 rank returned the wrong execute reply")
        completion = reply.get("rank_completion")
        if completion != {
            "rank": rank,
            "step_id": message["step_id"],
            "worker_epoch": message["worker_epoch"],
            "schedule": schedule,
        }:
            raise ValueError("DEP4 rank did not acknowledge its Rust schedule")
        ids = [row["request_id"] for row in reply["outputs"]]
        if len(set(ids)) != len(ids) or sorted(ids) != sorted(schedule["request_ids"]):
            raise ValueError("DEP4 rank returned unowned or missing output rows")
        outputs.update((row["request_id"], row) for row in reply["outputs"])
        completions.append(completion)
    return {
        "type": "execute_result",
        "rpc_id": message["rpc_id"],
        "step_id": message["step_id"],
        "worker_epoch": message["worker_epoch"],
        "outputs": [outputs[row["request_id"]] for row in message["scheduled"]],
        "rank_completions": completions,
    }
