"""Content identities and cache evidence for archived Qwen execution."""

import hashlib
import json
import math
import struct

from .qwen_fp8_math import ba_precision, integer
from .workload import SAMPLING_FIELDS

VERSION = "qwen-fp8-http-reference-v1"
LAYOUT = "qwen_radix_gdn8192_fp8kv128_v1"
FORMAT = "qwen_fp8_e4m3fn_rne_fp32scale_v1"


class Blocked(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise Blocked(reason)


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tokens(value):
    require(isinstance(value, list), "missing archived token IDs")
    for token in value:
        integer(token, "token ID")
        require(token < 248320, "token ID exceeds vocabulary")
    return value


def execution_key(row, epoch, step):
    return (
        epoch,
        step,
        integer(row["request_id"], "request ID", 1),
        integer(row["lease_id"], "lease ID", 1),
    )


def sha256(value):
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value),
        "missing SHA256 content identity",
    )


def validate_content_identities(artifacts):
    for name in ("source", "binary"):
        sha256(artifacts[name])
    checkpoint = artifacts["checkpoint"]
    require(
        {"config", "index"} < checkpoint.keys(),
        "checkpoint lacks loaded shard identities",
    )
    for value in checkpoint.values():
        sha256(value)
    tokenizer = artifacts["tokenizer"]
    require(
        {"tokenizer.json", "generation_config.json", "chat_template.jinja"}
        <= tokenizer.keys(),
        "loaded tokenizer assets are incomplete",
    )
    for value in tokenizer.values():
        sha256(value)
    for name in ("device", "ffn", "fa_fp8", "gdn_radix", "mtp_fp8", "sample", "vision"):
        metadata = artifacts["native"][name]
        for field in ("source_id", "source_fingerprint", "binary_id"):
            sha256(metadata[field])
    require(artifacts["device"]["uuid"].startswith("GPU-"), "missing device UUID")


def validate_identity(events, manifest, artifacts):
    identities = [event for event in events if event.get("operation") == "identity"]
    require(len(identities) == 1, "requires exactly one loaded identity")
    actual = identities[0]
    validate_content_identities(artifacts)
    for name in (
        "source",
        "binary",
        "native",
        "checkpoint",
        "tokenizer",
        "device",
        "execution",
        "accounting",
    ):
        require(
            name in artifacts
            and artifacts[name]
            and actual.get(name) == artifacts[name],
            f"loaded {name} identity differs or is missing",
        )
    require(
        manifest.get("artifact_id") == digest(artifacts) == actual.get("artifact_id"),
        "run, manifest and loaded identity are not bound",
    )
    require(
        artifacts.get("accounting", {}).get("version") == VERSION,
        "accounting version is not the Qwen FP8 reference",
    )
    config = actual["config"]
    frozen = dict(
        tensor_parallel_size=1,
        max_num_seqs=4,
        max_num_batched_tokens=32768,
        max_model_len=262144,
        block_size=128,
    )
    require(
        all(
            type(config[name]) is int and config[name] == value
            for name, value in frozen.items()
        ),
        "service configuration differs from the frozen schedule",
    )
    integer(config["worker_epoch"], "loaded worker epoch", 1)
    require(
        config["num_speculative_tokens"] == (4 if manifest["mode"] == "mtp4" else 0),
        "trace mode differs from HTTP mode",
    )
    layout = actual["cache_memory"]
    require(
        layout["layout_id"] == LAYOUT and layout["format_fingerprint"] == FORMAT,
        "unknown cache format identity",
    )
    require(
        integer(layout["usable_pages"], "usable pages", 1)
        == 64 * integer(layout["usable_checkpoints"], "usable checkpoints", 1),
        "cache capacities violate P=64*C",
    )
    timer = manifest["timer"]
    require(timer["clock"] == "perf_counter", "requires the full HTTP clock")
    values = (timer["started"], timer["finished"])
    require(
        all(type(v) in (int, float) and math.isfinite(v) for v in values)
        and values[1] > values[0],
        "invalid full HTTP timer",
    )
    return actual


def forward_precisions(event, mtp):
    epoch, step = event["worker_epoch"], event["step_id"]
    forward = {
        execution_key(row, epoch, step): row
        for row in event["scheduled"]
        if row["action"] == "forward"
    }
    precisions, seen = {}, set()
    dispatches = event["dispatches"]
    require(isinstance(dispatches, list), "missing actual forward dispatch evidence")
    for dispatch in dispatches:
        require(
            dispatch["operation"] == "dispatch"
            and dispatch["worker_epoch"] == epoch
            and dispatch["step_id"] == step,
            "dispatch belongs to another execution",
        )
        require(
            dispatch["execution"] in ("eager", "cuda_graph"),
            "unknown dispatch execution mode",
        )
        keys = [execution_key(row, epoch, step) for row in dispatch["requests"]]
        require(
            keys
            and len(set(keys)) == len(keys)
            and not (set(keys) & seen)
            and set(keys) <= forward.keys(),
            "dispatch forward inventory differs",
        )
        rows = [forward[key] for key in keys]
        counts = [
            integer(row["forward_end"] - row["forward_start"], "forward rows", 1)
            for row in rows
        ]
        total, maximum = sum(counts), max(counts)
        verify = dispatch["verify"]
        require(
            type(verify) is bool
            and all((row["sampling"]["kind"] == "verify") == verify for row in rows),
            "dispatch verification mode differs",
        )
        require(
            integer(dispatch["rows"], "dispatch rows", 1) == total
            and integer(dispatch["max_queries"], "dispatch maximum", 1) == maximum
            and total <= (32764 if mtp else 32768)
            and len(keys) <= (4 if mtp else 20),
            "actual dispatch shape differs",
        )
        precision = dispatch["gdn_ba_precision"]
        require(
            precision == ba_precision(total, maximum, verify),
            "actual dispatch BA precision differs",
        )
        precisions.update((key[2], precision) for key in keys)
        seen.update(keys)
    require(seen == forward.keys(), "missing actual forward dispatch evidence")
    return precisions


def f32_bits(value):
    try:
        return struct.pack(">f", value).hex()
    except (struct.error, OverflowError) as error:
        raise Blocked("sampling value has no finite float32 representation") from error


def validate_prepared_request(actual, row, effective):
    require(
        actual["prompt_token_sha256"] == digest(row["prompt_token_ids"]),
        "prepared sampling prompt identity differs",
    )
    bits = {
        name: f32_bits(effective[name])
        for name in SAMPLING_FIELDS
        if name != "repetition_guard"
    }
    require(
        actual["sampling_f32_bits"] == bits, "prepared effective sampling bits differ"
    )
    intent = actual["rust_intent"]
    require(
        isinstance(intent, dict)
        and intent["normalized_sampling_f32_bits"]
        == {"presence_penalty": bits["presence_penalty"]},
        "Rust prepared sampling intent differs from the approved workload",
    )
    expected = dict(
        seed=integer(row["body"]["seed"], "request seed"),
        min_tokens=row["body"]["min_tokens"],
        max_tokens=row["body"]["max_tokens"],
        repetition_guard=effective["repetition_guard"],
    )
    for name, value in expected.items():
        require(
            type(actual[name]) is type(value)
            and actual[name] == value
            and type(intent[name]) is type(value)
            and intent[name] == value,
            f"prepared sampling {name} differs from Rust intent or HTTP request",
        )


def validate_prepared(events, manifest, effective):
    requests = {row["id"]: row for row in manifest["requests"]}
    epoch = next(event for event in events if event["operation"] == "identity")[
        "config"
    ]["worker_epoch"]
    observed, rpcs = set(), set()
    for event in events:
        if event["operation"] == "prepared":
            request = integer(event["request_id"], "prepared request ID", 1)
            rpc = integer(event["rpc_id"], "prepared RPC", 1)
            require(
                integer(event["worker_epoch"], "prepared worker epoch", 1) == epoch
                and request in requests
                and request not in observed
                and rpc not in rpcs,
                "prepared sampling inventory, RPC or epoch differs",
            )
            validate_prepared_request(event, requests[request], effective)
            observed.add(request)
            rpcs.add(rpc)
        elif event["operation"] == "accounting_call":
            require(
                all(row["request_id"] in observed for row in event["scheduled"]),
                "missing prepared sampling evidence before execution",
            )
    require(observed == requests.keys(), "missing prepared sampling inventory")


def validate_assignment(row, identity):
    state = row["model_state"]
    capacity = identity["cache_memory"]["usable_checkpoints"]
    private_end = capacity + 6 * identity["config"]["max_num_seqs"] + 1
    private = state["private_checkpoints"]
    refs = {
        (
            integer(p["slot"], "private slot", 1),
            integer(p["version"], "private version"),
        )
        for p in private
    }
    require(
        len(private) == len(refs) == len({p[0] for p in refs}) == 6,
        "a Qwen request requires six distinct private checkpoints",
    )
    require(
        all(capacity < slot < private_end for slot, _ in refs),
        "private checkpoint lies outside its physical pool",
    )
    exports = state["exports"]
    require(len({p["slot"] for p in exports}) == len(exports), "duplicate export slot")
    for export in exports:
        _, slot, version = endpoint_tuple(export)
        if export["role"] == "private_endpoint":
            require(
                (slot, version) in refs, "private export has no matching request lease"
            )
        else:
            require(
                export["role"] in ("periodic", "candidate_tail")
                and 0 < slot <= capacity,
                "persistent export lies outside its physical pool",
            )
    source = state["endpoint_source"]
    require(source["kind"] in ("forward", "saved_hidden"), "unknown endpoint source")
    require(
        set(source)
        == (
            {"kind"}
            if source["kind"] == "forward"
            else {"kind", "end", "slot", "version"}
        ),
        "endpoint source fields differ from the runtime contract",
    )
    require(
        row["action"] != "forward" or source["kind"] == "forward",
        "forward work contradicts its endpoint source",
    )
    copies = state["page_copies"]
    require(isinstance(copies, list), "missing page copy evidence")
    require(
        len({c["destination_slot"] for c in copies}) == len(copies),
        "aliased page copy destinations",
    )
    for copy in copies:
        for name in ("source_slot", "destination_slot"):
            require(
                integer(copy[name], name, 1)
                <= identity["cache_memory"]["usable_pages"],
                "page copy lies outside its physical pool",
            )
        integer(copy["source_version"], "source page version")
        integer(copy["destination_version"], "destination page version")
        require(
            copy["source_slot"] != copy["destination_slot"]
            and 0 < integer(copy["valid_tokens"], "copy valid tokens") < 128,
            "invalid FP8 KV prefix copy",
        )


def validate_checkpoint_sources(row, endpoints, last_result):
    read_end, read_slot, read_version = endpoint_tuple(row["model_state"]["resume"])
    require(
        (read_end == read_slot == 0)
        or endpoints.get(read_slot) == (read_version, read_end),
        "resume names an absent, overwritten or relabelled checkpoint",
    )
    if row["action"] == "forward":
        return
    source = row["model_state"]["endpoint_source"]
    if source["kind"] == "saved_hidden":
        end, slot, version = endpoint_tuple(source)
        require(
            slot > 0
            and end == row["endpoint"]
            and endpoints.get(slot) == (version, end),
            "saved hidden names an absent, overwritten or relabelled checkpoint",
        )
    else:
        require(
            last_result is not None
            and any(
                value["end"] == row["endpoint"]
                and endpoints.get(value["slot"]) == (value["version"], value["end"])
                for value in last_result["validated_exports"]
            ),
            "forward endpoint source has no current completed hidden state",
        )
    require(
        row["forward_start"] == row["forward_end"] == row["endpoint"],
        "endpoint-only work has inconsistent bounds",
    )


def validate_model_state(row, first, identity):
    state = row["model_state"]
    validate_assignment(row, identity)
    require(
        state["kind"] == "qwen_radix_v1" and state["layout_fingerprint"] == LAYOUT,
        "scheduled model state has an unknown layout",
    )
    if first:
        require(
            state["checkpoint_hit_end"]
            == state["kv_contiguous_end"]
            == state["resume"]["end"]
            == 0,
            "independent prefix restoration requires AgentX state evidence",
        )
    require(
        state["resume"]["end"] == row["forward_start"],
        "forward does not use its actual checkpoint",
    )
    pages = {page["page_index"]: page for page in state["page_table"]}
    require(len(pages) == len(state["page_table"]), "duplicate logical page")
    require(len({p["slot"] for p in pages.values()}) == len(pages), "aliased KV pages")
    cursor = row["forward_start"]
    for span in state["kv_spans"]:
        require(
            span["start"] == cursor and cursor < span["end"] <= row["forward_end"],
            "KV spans do not partition forwarded tokens",
        )
        require(span["mode"] in ("write", "reuse"), "unknown KV span mode")
        for position in range(cursor, span["end"]):
            page = pages[position // 128]
            require(
                integer(page["slot"], "page slot", 1) and type(page["version"]) is int,
                "invalid page identity",
            )
            require(
                (span["mode"] == "write" and page["access"] == "write")
                or (span["mode"] == "reuse" and page["valid_tokens"] > position % 128),
                "KV write ownership or reuse validity differs",
            )
        cursor = span["end"]
    require(cursor == row["forward_end"], "missing KV span coverage")
    return pages


def endpoint_tuple(value):
    return tuple(
        integer(value[name], f"checkpoint {name}")
        for name in ("end", "slot", "version")
    )


def validate_cache_result(row, result):
    planned = [endpoint_tuple(value) for value in row["model_state"]["exports"]]
    actual = [endpoint_tuple(value) for value in result["validated_exports"]]
    require(
        len(set(actual)) == len(actual) and set(actual) == set(planned),
        "validated checkpoint exports differ from the reservation",
    )
    for end, slot, _ in actual:
        require(
            slot > 0
            and (
                row["forward_start"] < end <= row["forward_end"]
                if row["action"] == "forward"
                else end == row["endpoint"]
            ),
            "checkpoint export lies outside executed work",
        )
    expected = set()
    for page in row["model_state"]["page_table"]:
        start = page["page_index"] * 128
        end = max(start, min(result["computed_end"], start + 128))
        if row["purpose"] != "finalize":
            end = max(end, start + page["valid_tokens"])
        expected.add((page["page_index"], page["slot"], page["version"], end))
    actual_pages = [
        tuple(
            integer(page[name], f"page {name}")
            for name in ("page_index", "slot", "version", "valid_end")
        )
        for page in result["committed_page_valid_ends"]
    ]
    require(
        len(set(actual_pages)) == len(actual_pages) and set(actual_pages) == expected,
        "committed KV valid ends differ from accepted work",
    )
    return actual
