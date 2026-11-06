EXECUTION_PROTOCOL_VERSION = 2


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def uint(value, name, bits=64):
    require(type(value) is int and 0 <= value < 1 << bits, f"invalid {name}")
    return value


def validate_init(config):
    require(
        config.get("protocol_version") == EXECUTION_PROTOCOL_VERSION,
        "unsupported or missing execution protocol version",
    )
    uint(config.get("worker_epoch"), "worker_epoch")
    require("host_cache_bytes" not in config, "removed host cache configuration")


def validate_execute(message, epoch):
    require(message["worker_epoch"] == epoch, "stale execution epoch")
    uint(message["rpc_id"], "rpc_id")
    uint(message["step_id"], "step_id")
    expected = {}
    rows = 0
    for row in message["scheduled"]:
        request = uint(row["request_id"], "request_id")
        uint(row["lease_id"], "lease_id")
        require(request not in expected, "duplicate scheduled request")
        expected[request] = row
        validate_row(row)
        rows += len(row["token_ids"])
    require(rows == message["num_batched_tokens"], "forward row total differs")
    return expected


def validate_row(row):
    require(row["purpose"] in ("advance", "finalize"), "invalid work purpose")
    action = row["action"]
    require(action in ("forward", "head_only", "export_only"), "invalid work action")
    start = uint(row["forward_start"], "forward_start", 32)
    end = uint(row["forward_end"], "forward_end", 32)
    uint(row["accepted_end"], "accepted_end", 32)
    require(
        end >= start and end - start == len(row["token_ids"]), "forward range differs"
    )
    require(row["endpoint"] == end, "endpoint differs from action")
    for token in row["token_ids"]:
        uint(token, "token", 32)
    sampling = row["sampling"]
    kind = sampling["kind"]
    require(kind in ("none", "target", "verify"), "invalid sampling mode")
    if kind == "verify":
        require(uint(sampling["drafts"], "drafts", 32) > 0, "verify has no drafts")
    if row["purpose"] == "finalize":
        require(kind == "none", "finalize cannot sample")
    if action == "forward":
        require(end > start, "forward requires token rows")
    else:
        require(end == start, "endpoint action contains rows")
        require(
            kind == ("target" if action == "head_only" else "none"),
            "invalid endpoint sampling",
        )
    require(
        row["model_state"]["kind"] in ("qwen_radix_v1", "deepseek_resident_v1"),
        "unsupported model state tag",
    )


def validate_result(message, result, expected):
    require(result["type"] == "execute_result", "wrong execution reply type")
    for name in ("rpc_id", "step_id", "worker_epoch"):
        require(result[name] == message[name], f"execution reply changed {name}")
    received = set()
    for output in result["outputs"]:
        request = output["request_id"]
        require(
            request in expected and request not in received,
            "unknown or duplicate result",
        )
        received.add(request)
        validate_output(expected[request], output)
    require(received == set(expected), "missing execution result")
    for rank in result.get("rank_completions", []):
        require(
            rank["worker_epoch"] == message["worker_epoch"]
            and rank["step_id"] == message["step_id"],
            "rank execution identity differs",
        )


def validate_output(row, output):
    for name in ("request_id", "lease_id", "purpose", "action"):
        require(output[name] == row[name], f"execution reply changed {name}")
    if output.get("error") is not None:
        require(
            not output["token_ids"]
            and not output["new_draft_token_ids"]
            and output["num_accepted_draft_tokens"] == 0
            and not output["validated_exports"]
            and not output["committed_page_valid_ends"]
            and output["mtp_accepted_endpoint"] is None,
            "error returned successful state",
        )
        return
    accepted = uint(output["num_accepted_draft_tokens"], "accepted drafts", 32)
    kind = row["sampling"]["kind"]
    if kind == "verify":
        require(accepted <= row["sampling"]["drafts"], "too many accepted drafts")
        count = accepted + 1
    else:
        require(accepted == 0, "non-verify accepted drafts")
        count = int(kind == "target")
    require(len(output["token_ids"]) == count, "sampling result count differs")
    if kind == "none":
        require(not output["new_draft_token_ids"], "non-sampling work returned drafts")
    if row["purpose"] == "finalize":
        require(
            not output["text"]
            and output["finish_reason"] is None
            and output["reasoning_tokens"] == 0,
            "finalize advanced generation state",
        )
    end = row["forward_end"]
    if kind == "verify":
        end += accepted - row["sampling"]["drafts"]
    require(output["computed_end"] == end, "computed endpoint differs")
    uint(output["accepted_end"], "accepted_end", 32)
