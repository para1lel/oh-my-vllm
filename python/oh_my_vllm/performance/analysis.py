"""Map committed request IDs to phase-contained canonical GPU work.

The first complete worker response belongs to prefill, including its proposals.
Decode starts with that response's proposals and committed states available.
Whole-batch resource work bounds a span; subtract phase-start skew to bound
the longest individual request interval. GPU timings never supply model costs.
"""

from collections import defaultdict
from copy import deepcopy

from .coverage import validate_inventory
from .dag import b200, retention_bytes
from .memory import WriteLedger
from .model import Model

MODEL_VERSION = "qwen38-dspark-b200-semantic-v1"


def _validate_draft_step(operations, scheduled, outputs, kept, run, mtp_cursor):
    """Tie each draft invocation to accepted progress and the output budget."""
    per_request = defaultdict(list)
    for op in operations:
        kind = op["kind"]
        if kind not in {
            "dspark_inject",
            "dspark_backbone",
            "mtp_forward",
            "mtp_logits",
            "mtp_proposal_graph",
        }:
            continue
        requests = op["requests"]
        if not isinstance(requests, list) or len(requests) != len(set(requests)):
            raise ValueError("duplicate draft invocation request")
        if kind in {"mtp_logits", "mtp_proposal_graph", "dspark_inject"}:
            rows = sum(op["queries"]) if kind == "dspark_inject" else len(requests)
            if type(op["rows"]) is not int or op["rows"] != rows:
                raise ValueError("draft invocation rows disagree with requests")
        fields = {
            "dspark_inject": ("queries", "contexts"),
            "dspark_backbone": (
                "contexts",
                "incoming_contexts",
                "tables",
                "anchors",
                "candidates",
            ),
            "mtp_forward": (
                "queries",
                "contexts",
                "tables",
                "unique_token_ids_per_request",
            ),
            "mtp_proposal_graph": ("contexts", "tables", "tokens"),
            "mtp_logits": (),
        }[kind]
        if kind == "mtp_forward" and "incoming_contexts" in op:
            raise ValueError("raw MTP trace contains derived incoming ranges")
        if any(len(op[field]) != len(requests) for field in fields):
            raise ValueError("draft invocation metadata length differs")
        for index, rid in enumerate(requests):
            if rid in scheduled:
                per_request[rid].append((kind, op, index))
            elif rid in kept:
                raise ValueError("draft invocation has no target execution")

    for rid, target in scheduled.items():
        mode = run["speculative_mode"]
        records = per_request[rid]
        out = outputs[rid]
        context = target["context"]
        computed = context + out["computed"]
        remaining = run["output_len"] - kept[rid] - out["kept"] - 1
        if mode == "none":
            if records:
                raise ValueError("ordinary execution contains draft work")
            continue
        if mode == "dspark":
            expected = ["dspark_inject"]
            eligible = out["kept"] > 0 and min(remaining, 262144 - computed - 1) > 0
            if eligible:
                expected.append("dspark_backbone")
            if [kind for kind, _, _ in records] != expected:
                raise ValueError("DSpark invocation count disagrees with progress")
            _, op, i = records[0]
            if op["queries"][i] != out["computed"] or op["contexts"][i] != context:
                raise ValueError("DSpark injection differs from accepted rows")
            if eligible:
                _, op, i = records[1]
                if (
                    op["rows_per_request"] != 7
                    or op["contexts"][i] != computed
                    or op["incoming_contexts"][i] != context
                    or op["tables"][i] != target["fa_block_table"]
                    or op["greedy"] is not True
                    or len(op["candidates"][i]) != 7
                    or any(
                        type(t) is not int or not 0 <= t < 248320
                        for t in [op["anchors"][i], *op["candidates"][i]]
                    )
                ):
                    raise ValueError("DSpark backbone geometry differs from progress")
            continue

        limit = min(len(target["fa_block_table"]) * 784, 262144)
        previous = mtp_cursor.get(rid, max(1, context))
        if previous not in (max(1, context), context + 1):
            raise ValueError("MTP persistent cursor differs from target progress")
        if context > 0 and previous == context and context % 784:
            raise ValueError("MTP state restoration is away from a page boundary")
        first = context if context > 0 and previous == context else context + 1
        end = min(computed + 1, limit)
        if first >= end:
            mtp_cursor[rid] = first
            if records:
                raise ValueError("MTP invocation exceeds allocated state")
            continue
        mtp_cursor[rid] = end
        if not records or records[0][0] != "mtp_forward":
            raise ValueError("missing persistent MTP invocation")
        _, op, i = records[0]
        if (
            op["persistent"] is not True
            or op["queries"][i] != end - first
            or op["contexts"][i] != first
            or op["tables"][i] != target["fa_block_table"]
        ):
            raise ValueError("persistent MTP geometry differs from progress")
        budget = (
            min(4, remaining, limit - computed)
            if out["kept"] and end == computed + 1
            else 0
        )
        proposal = records[1:]
        if proposal and proposal[0][0] == "mtp_proposal_graph":
            _, op, i = proposal[0]
            if (
                len(proposal) != 1
                or budget != 4
                or op["draft_steps"] != 3
                or op["contexts"][i] != computed
                or op["tables"][i] != target["fa_block_table"]
                or len(op["tokens"][i]) != 4
                or any(
                    type(t) is not int or not 0 <= t < 248320 for t in op["tokens"][i]
                )
            ):
                raise ValueError("MTP proposal graph differs from legal budget")
        else:
            expected = ["mtp_logits"] if budget > 0 else []
            expected += ["mtp_forward", "mtp_logits"] * max(0, budget - 1)
            if [kind for kind, _, _ in proposal] != expected:
                raise ValueError("MTP proposal invocation count differs from budget")
            for position, (kind, op, i) in enumerate(proposal):
                if kind == "mtp_forward" and (
                    op["persistent"] is not False
                    or op["queries"][i] != 1
                    or op["contexts"][i] != computed + (position + 1) // 2
                    or op["tables"][i] != target["fa_block_table"]
                ):
                    raise ValueError(
                        "ephemeral MTP geometry differs from proposal position"
                    )
        for kind, op, i in records:
            if kind == "mtp_forward":
                tokens = op["unique_token_ids_per_request"][i]
                if len(tokens) > op["queries"][i] or any(
                    type(t) is not int or not 0 <= t < 248320 for t in tokens
                ):
                    raise ValueError("MTP embedding rows exceed invocation geometry")


def validate_progress(trace, run):
    """Effective GPU rows must agree with accepted history and worker commits."""
    ids = {p["request_id"] for p in run["request_phases"]}
    prefix = run["initial_prefix_hit_tokens"] // run["batch_size"]
    contexts = dict.fromkeys(ids, prefix)
    kept = dict.fromkeys(ids, 0)
    mtp_cursor = {}
    state_sources = {}
    draft_limit = {"none": 0, "mtp": 4, "dspark": 7}[run["speculative_mode"]]
    for step in trace["steps"]:
        scheduled, outputs = {}, {}
        for op in step["operations"]:
            if op["kind"] == "target":
                if sum(r["query"] for r in op["requests"]) > 32768:
                    raise ValueError("effective target rows exceed scheduler budget")
                for r in op["requests"]:
                    rid = r["request_id"]
                    if rid in ids:
                        if rid in scheduled:
                            raise ValueError("duplicate target request in one step")
                        scheduled[rid] = r
            elif op["kind"] == "commit":
                for row in op["outputs"]:
                    rid = row["request_id"]
                    if rid in ids:
                        if rid in outputs:
                            raise ValueError("duplicate commit request in one step")
                        outputs[rid] = row
        if scheduled.keys() != outputs.keys():
            raise ValueError("target execution and commit requests differ")
        _validate_draft_step(
            step["operations"], scheduled, outputs, kept, run, mtp_cursor
        )
        expected_copies = []
        for rid, r in scheduled.items():
            query, context, drafts, samples = (
                r[k] for k in ("query", "context", "drafts", "sample_rows")
            )
            history = run["input_len"] + kept[rid]
            if "incoming_context" in r:
                raise ValueError("raw target trace contains derived incoming ranges")
            if (
                any(type(v) is not int for v in (query, context, drafts, samples))
                or query <= 0
                or context != contexts[rid]
                or not context < history
                or context + query > 262144
                or kept[rid] >= run["output_len"]
            ):
                raise ValueError("effective query or context disagrees with history")
            known = min(query, history - context)
            expected_drafts = query - known
            expected_samples = query - known + 1 if context + query >= history else 0
            if (
                drafts != expected_drafts
                or not 0 <= drafts <= draft_limit
                or (drafts and known != 1)
                or samples != expected_samples
                or r["prefill"] is not (query > 1 and drafts == 0)
                or len(r["state_row_writes"]) != query
                or len(r["fa_block_table"]) < (context + query + 783) // 784
                or any(
                    type(p) is not int or not 0 < p < 1400 for p in r["fa_block_table"]
                )
                or any(
                    type(p) is not int or not 0 <= p < 128
                    for p in r["mamba_block_table"]
                )
                or len(r["unique_token_ids"]) > query
                or any(
                    type(token) is not int or not 0 <= token < 248320
                    for token in r["unique_token_ids"]
                )
            ):
                raise ValueError("effective sampling, draft, or cache rows disagree")
            base = (context + query + 783) // 784 - 1
            table = r["mamba_block_table"]
            destinations = table[base : base + (1 if r["prefill"] else query)]
            expected_source = state_sources.get(
                rid, 0 if context == 0 else table[(context - 1) // 784]
            )
            expected_writes = (
                [-1] * (query - 1) + destinations if r["prefill"] else destinations
            )
            if (
                type(r["state_source"]) is not int
                or r["state_source"] != expected_source
                or (context > 0 and expected_source <= 0)
                or any(type(d) is not int or not 0 < d < 128 for d in destinations)
                or len(set(destinations)) != len(destinations)
                or r["state_row_writes"] != expected_writes
                or r["state_destinations"] != destinations
            ):
                raise ValueError("recurrent state rows disagree with allocated table")
            out = outputs[rid]
            count, computed, accepted = (
                out[k] for k in ("kept", "computed", "accepted")
            )
            if (
                any(type(v) is not int for v in (count, computed, accepted))
                or not 0 <= count <= samples
                or bool(count) != bool(samples)
                or accepted != max(0, count - 1)
                or computed != query - drafts + accepted
                or kept[rid] + count > run["output_len"]
            ):
                raise ValueError("commit progress disagrees with effective input rows")
            if out["state_source"] != expected_writes[computed - 1]:
                raise ValueError("committed recurrent source differs from accepted row")
            state_sources[rid] = out["state_source"]
            for boundary in range(
                (context // 784 + 1) * 784, context + computed + 1, 784
            ):
                destination = table[boundary // 784 - 1]
                source = expected_writes[boundary - context - 1]
                if destination and destination != source:
                    expected_copies.append([source, destination])
            contexts[rid] += computed
            kept[rid] += count
        if scheduled:
            copies = [
                copy
                for op in step["operations"]
                if op["kind"] == "commit"
                for copy in op["copies"]
            ]
            if sorted(copies) != sorted(expected_copies):
                raise ValueError(
                    "recurrent checkpoint copies differ from accepted boundaries"
                )
    if any(count != run["output_len"] for count in kept.values()):
        raise ValueError("request progress has incomplete output")


def request_boundaries(trace, run):
    """Correlate batch-local clocks with worker commits, not global step IDs."""
    boundaries = {}
    for phase in run["request_phases"]:
        rid = phase["request_id"]
        scheduled, output = [], []
        for index, step in enumerate(trace["steps"]):
            for operation in step["operations"]:
                if operation["kind"] == "target" and any(
                    r["request_id"] == rid for r in operation["requests"]
                ):
                    scheduled.append(index)
                if operation["kind"] == "commit":
                    for row in operation["outputs"]:
                        if row["request_id"] == rid and row["kept"]:
                            output.append((index, row["kept"]))
        if (
            not scheduled
            or not output
            or sum(n for _, n in output) != run["output_len"]
        ):
            raise ValueError("request trace is incomplete")
        first, last = output[0][0], output[-1][0]
        if first < scheduled[0] or last > scheduled[-1]:
            raise ValueError("commit precedes execution")
        boundaries[rid] = (scheduled[0], first, last)
    batch_start = min(begin for begin, _, _ in boundaries.values())
    for phase in run["request_phases"]:
        _, first, last = boundaries[phase["request_id"]]
        if (phase["first_step"], phase["last_step"]) != (
            first - batch_start + 1,
            last - batch_start + 1,
        ):
            raise ValueError("Rust output steps disagree with worker execution trace")
    return boundaries


def select_operation(operation, ids):
    """Restrict effective rows to requests in this phase, retaining shared work."""
    operation = deepcopy(operation)
    kind = operation["kind"]
    if kind == "target":
        operation["requests"] = [
            r for r in operation["requests"] if r["request_id"] in ids
        ]
        return operation if operation["requests"] else None
    if kind == "commit":
        return None
    requests = operation.get("requests")
    if requests is None:
        raise ValueError(f"missing request binding for {kind}")
    indices = [i for i, rid in enumerate(requests) if rid in ids]
    if not indices:
        return None
    for field in (
        "requests",
        "queries",
        "contexts",
        "incoming_contexts",
        "tables",
        "anchors",
        "unique_token_ids_per_request",
        "tokens",
        "candidates",
    ):
        if field in operation:
            if len(operation[field]) != len(requests):
                raise ValueError("operation request metadata disagree")
            operation[field] = [operation[field][i] for i in indices]
    if kind == "mtp_forward":
        operation["unique_token_ids"] = sorted(
            set().union(*map(set, operation["unique_token_ids_per_request"]))
        )
    if "rows" in operation:
        operation["rows"] = (
            sum(operation["queries"]) if kind == "dspark_inject" else len(indices)
        )
    return operation


def phase_operations(step, ids, boundaries, index, phase):
    """Exclude endpoint GPU tails without adding a timing synchronization."""
    completed = step.get("completed_operations")
    endpoint = 1 if phase == "prefill" else 2
    if completed is None and all(index < boundaries[rid][endpoint] for rid in ids):
        completed = 0
    if type(completed) is not int or not 0 <= completed <= len(step["operations"]):
        raise ValueError("missing completed GPU feedback boundary")
    result = []
    for position, operation in enumerate(step["operations"]):
        contained = {
            rid
            for rid in ids
            if index < boundaries[rid][endpoint] or position < completed
        }
        selected = select_operation(operation, contained)
        if selected is not None:
            result.append(selected)
    return result


def step_work(hardware, weights, operations, *, phase=None, angle_seen=None):
    """Count target, fixed-weight draft, sampling, and persistent updates once."""
    model = Model(hardware, weights, angle_seen=angle_seen)
    model.phase = phase
    parent = None
    for operation in operations:
        kind = operation["kind"]
        if kind == "target":
            if parent is not None:
                raise ValueError("duplicate target invocation")
            parent = model.target(operation["requests"], operation["state_dtype"])
        elif parent is None:
            raise ValueError("draft work precedes target invocation")
        elif kind == "mtp_forward":
            parent = model.mtp_forward(operation, parent)
        elif kind == "mtp_logits":
            parent = model.mtp_logits(parent, operation["rows"])
        elif kind == "mtp_proposal_graph":
            parent = model.proposal_chain(operation, parent)
        elif kind == "dspark_inject":
            injected = model.dspark_inject(operation, model.features)
            parent = model.node("response:context", (parent, injected))
        elif kind == "dspark_backbone":
            parent = model.dspark_backbone(operation, parent)
        else:
            raise ValueError(f"unsupported semantic step: {kind}")
    if parent is None:
        raise ValueError("missing target operation")
    return model.bounds(parent)


def phase_resource_bound(hardware, work, read_intervals, writes, start_skew, path):
    """Separate resource totals; prefetch can overlap arithmetic across cuts."""
    tensor = sum(
        work.get(k, 0) / hardware.rates[k] for k in ("tensor_fp8", "tensor_bf16")
    )
    compute = max(
        tensor,
        max(
            (
                v / hardware.rates[k]
                for k, v in work.items()
                if not k.startswith("tensor_")
            ),
            default=0.0,
        ),
    )
    read_bytes = sum(
        max(0, sum(interval.values()) - hardware.cache_bytes)
        for interval in read_intervals
    )
    write_bytes = max(0, writes - hardware.cache_bytes)
    hbm = (read_bytes + write_bytes) / hardware.hbm_bytes_per_s
    span = max(compute, hbm, path)
    return {
        "compute_resource_s": compute,
        "hbm_s": hbm,
        "critical_path_s": path,
        "phase_start_skew_s": start_skew,
        "resource_span_s": span,
        "lower_bound_s": max(0, span - start_skew),
        "compulsory_hbm_bytes": read_bytes + write_bytes,
    }


def analyze_trace(trace, runs):
    """Recalculate all measured phase bounds using reviewed source contracts."""
    validate_inventory(trace)
    device = trace["device"]
    hardware = b200(device["sm_count"], device["max_clock_hz"], retention_bytes(device))
    results = []
    for run in runs:
        validate_progress(trace, run)
        boundaries = request_boundaries(trace, run)
        mode = run["speculative_mode"]
        if mode not in ("none", "mtp", "dspark"):
            raise ValueError("unknown measured speculative mode")
        state_dtype = "torch.float32" if mode == "none" else "torch.bfloat16"
        begin = min(cuts[0] for cuts in boundaries.values())
        end = max(cuts[2] for cuts in boundaries.values())
        kinds = set()
        for step in trace["steps"][begin : end + 1]:
            if step["mode"] != mode:
                raise ValueError("Rust mode disagrees with worker execution trace")
            for operation in step["operations"]:
                kinds.add(operation["kind"])
                if (
                    operation["kind"] == "target"
                    and operation["state_dtype"] != state_dtype
                ):
                    raise ValueError("state precision disagrees with measured mode")
        if mode == "none" and any(k.startswith(("mtp_", "dspark_")) for k in kinds):
            raise ValueError("ordinary workload contains speculative execution")
        if mode == "mtp" and "mtp_forward" not in kinds:
            raise ValueError("MTP workload has no draft execution")
        if mode == "mtp" and any(k.startswith("dspark_") for k in kinds):
            raise ValueError("MTP workload contains DSpark execution")
        if mode == "dspark" and any(k.startswith("mtp_") for k in kinds):
            raise ValueError("DSpark workload contains MTP execution")
        if mode == "dspark" and not {"dspark_inject", "dspark_backbone"} <= kinds:
            raise ValueError("DSpark workload has no complete draft execution")
        phases = {}
        for phase in ("prefill", "decode"):
            work = defaultdict(float)
            relaxed_work = defaultdict(float)
            intervals = []
            prefill_reads = {}
            writes = WriteLedger()
            angle_seen = set()
            path = 0
            first_index = min(begin for begin, _, _ in boundaries.values())
            last_index = max(
                cuts[1 if phase == "prefill" else 2] for cuts in boundaries.values()
            )
            for index, step in enumerate(trace["steps"]):
                if not first_index <= index <= last_index:
                    continue
                # Other-phase execution can consume or recycle a value created
                # here. Observe its invalidations without adding its work.
                for operation in step["operations"]:
                    if operation["kind"] == "target":
                        for request in operation["requests"]:
                            writes.invalidate(request)
                ids = {
                    rid
                    for rid, (begin, first, last) in boundaries.items()
                    if (
                        begin <= index <= first
                        if phase == "prefill"
                        else first < index <= last
                    )
                }
                if not ids:
                    continue
                committed = {
                    row["request_id"]: row["computed"]
                    for op in step["operations"]
                    if op["kind"] == "commit"
                    for row in op["outputs"]
                }
                operations = phase_operations(step, ids, boundaries, index, phase)
                if not operations or operations[0]["kind"] != "target":
                    raise ValueError("phase has no executed target rows")
                for request in operations[0]["requests"]:
                    rid = request["request_id"]
                    request["kept_rows"] = committed[rid]
                    request["terminal"] = index == boundaries[rid][2]
                    commit_index = next(
                        i
                        for i, op in enumerate(step["operations"])
                        if op["kind"] == "commit"
                    )
                    endpoint = boundaries[rid][1 if phase == "prefill" else 2]
                    request["checkpoint_copies_completed"] = (
                        index < endpoint or step["completed_operations"] > commit_index
                    )
                    request["mtp"] = any(
                        operation["kind"].startswith("mtp_")
                        for operation in operations[1:]
                    )
                # The write ledger uses physical addresses before incoming-read
                # relaxations suppress data produced earlier in this phase.
                writes.target(operations[0]["requests"], operations[0]["state_dtype"])
                writes.draft(operations[1:], operations[0]["requests"])
                for request in operations[0]["requests"]:
                    rid = request["request_id"]
                    if phase == "prefill" and index > boundaries[rid][0]:
                        # Data produced earlier in the same phase are internal.
                        # Giving them free retention weakens the HBM floor.
                        request["incoming_context"] = 0
                        request["state_source"] = 0
                if phase == "prefill":
                    for op in operations[1:]:
                        if op["kind"] in ("dspark_backbone", "mtp_forward"):
                            op["incoming_contexts"] = [0] * len(op["requests"])
                summary = step_work(
                    hardware,
                    trace["weights"],
                    operations,
                    phase=phase,
                    angle_seen=angle_seen,
                )
                for resource, count in summary["work"].items():
                    work[resource] += count
                for operation, count in summary["relaxed_native_operations"].items():
                    relaxed_work[operation] += count
                ranges = defaultdict(dict)
                for key, size in summary["incoming_ranges"].items():
                    epoch, name = key.split(":", 1)
                    ranges[epoch][name] = size
                    if phase == "prefill":
                        # Deduplicate physical parameter ranges across all
                        # proposal epochs for a weaker whole-prefill floor.
                        prefill_reads[name] = max(prefill_reads.get(name, 0), size)
                if phase == "decode":
                    intervals.extend(ranges.values())
                # Independent parameter prefetch can span worker invocations.
                # Full repeated service stays in phase resource totals.
                path = max(path, summary["critical_path_s"])
            if phase == "prefill":
                intervals = [prefill_reads]
            starts = [
                p["submitted_s" if phase == "prefill" else "first_token_s"]
                for p in run["request_phases"]
            ]
            if phase == "prefill":
                # Different first-token cuts do not imply simultaneous final
                # working versions. Keep only the last-cut versions.
                writes.working = {
                    rid: row
                    for rid, row in writes.working.items()
                    if boundaries[rid][1] == last_index
                }
            details = phase_resource_bound(
                hardware,
                work,
                intervals,
                writes.bytes(),
                max(starts) - min(starts),
                path,
            )
            phases[phase] = details["lower_bound_s"]
            details["recognized_relaxed_operations"] = dict(relaxed_work)
            phases[phase + "_model"] = details
        results.append(phases)
    return results
