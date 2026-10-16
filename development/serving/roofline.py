from development.kernels.roofline import Work
from development.kernels.work import add, work_for
from development.kernels.workload import Shape

from .trace import validate


def scale(work, count):
    return Work(*(value * count for value in vars(work).values()))


def operator_work(name, queries, *, variant="default", state_writes=None):
    shape = Shape(
        name,
        sum(count for _, count in queries),
        len(queries),
        max(count for _, count in queries),
        variant=variant,
    )
    return work_for(shape, queries=queries, state_writes=state_writes)


def target_work(rows, verify):
    queries = [(row["start"], row["count"]) for row in rows]
    writes = (
        sum(count for _, count in queries)
        if verify
        else sum(
            (start + count) // 784 - start // 784 + int((start + count) % 784 != 0)
            for start, count in queries
        )
    )
    return [
        (
            "gdn_attention",
            operator_work(
                "gdn_attention", queries, variant="embedding", state_writes=writes
            ),
            1,
            queries,
        ),
        (
            "gdn_attention",
            operator_work("gdn_attention", queries, state_writes=writes),
            47,
            queries,
        ),
        ("fa_attention", operator_work("fa_attention", queries), 16, queries),
        ("ffn", operator_work("ffn", queries), 64, queries),
    ]


def acceptance_work(rows, commits):
    accepted = {row["id"]: row["accepted"] for row in commits}
    rejected = sum(accepted[row["id"]] < row["count"] - 1 for row in rows)
    draws = sum(accepted.values()) + rejected + len(rows)
    return Work(
        len(rows) * (248320 * 4 + 4 * 12 + 16 + 8),
        fp32_flops=rejected * 248320,
        fp64_flops=draws * 2,
    )


def target_commits(events, request_ids):
    pending, commits, acceptance = {}, {}, {}
    for index, event in enumerate(events):
        for row in event.get("requests", []):
            request = row["id"]
            if request not in request_ids:
                continue
            if event["operation"] == "target":
                pending[request] = index
            elif event["operation"] == "commit":
                commits.setdefault(pending.pop(request), []).append(row)
                totals = acceptance.setdefault(
                    request, dict(tokens=0, accepted=0, proposed=0)
                )
                for name in totals:
                    totals[name] += row[name]
    return commits, acceptance


def event_work(event, rows, commits, mtp):
    operation = event["operation"]
    if operation not in {"target", "context", "draft"}:
        raise ValueError(f"unmodeled trace operation: {operation}")
    queries = (
        [(row["start"], row["count"]) for row in rows] if operation != "draft" else []
    )
    if operation == "target":
        result = [(*item, rows) for item in target_work(rows, event["verify"])]
        if mtp and event["verify"]:
            sample_rows = rows
            sample_queries = queries
        else:
            sample_rows = [
                row
                for row in rows
                if event["verify"] or row["start"] + row["count"] == row["prompt"]
            ]
            sample_queries = [(0, 1) for _ in sample_rows]
        if sample_queries:
            sample = operator_work("sample", sample_queries)
            if mtp and event["verify"]:
                sample = add(sample, acceptance_work(rows, commits))
            result.append(("sample", sample, 1, sample_queries, sample_rows))
        return result
    if operation == "context":
        return [
            ("mtp_context", operator_work("mtp_context", queries), 1, queries, rows)
        ]
    if operation == "draft":
        queries = [(row["position"], 1) for row in rows]
        variant = "full" if event["stage"] else "query_only"
        return [
            (
                "mtp_draft",
                operator_work("mtp_draft", queries, variant=variant),
                1,
                queries,
                rows,
            )
        ]
    raise ValueError(f"unmodeled trace operation: {operation}")


def lower_bound(events, request_ids, expected_tokens=None):
    mtp = validate(events, request_ids, expected_tokens)
    commits, acceptance = target_commits(events, request_ids)
    work, components = Work(0), []
    for index, event in enumerate(events):
        if event["operation"] in {"identity", "commit"}:
            continue
        rows = [row for row in event.get("requests", []) if row["id"] in request_ids]
        if not rows:
            continue
        for name, item, calls, queries, requests in event_work(
            event, rows, commits.get(index, []), mtp
        ):
            work = add(work, scale(item, calls))
            components.append(
                {
                    "event": index,
                    "trace_operation": event["operation"],
                    "stage": event.get("stage"),
                    "request_ids": [row["id"] for row in requests],
                    "operation": name,
                    "calls": calls,
                    "rows": sum(count for _, count in queries),
                    "queries": queries,
                    "seconds": calls * item.lower_bound_seconds(),
                    "work_per_call": vars(item),
                }
            )
    return {
        "seconds": sum(item["seconds"] for item in components),
        "global_resource_seconds": work.lower_bound_seconds(),
        "work": vars(work),
        "components": components,
        "acceptance": acceptance,
        "method": (
            "sum of serving-operator roofline times in the serial execution trace; "
            "preserve batches, prefill chunks, query lengths, and MTP acceptance; "
            "exclude padding, replay warmup, intermediate copies, and launch floors"
        ),
    }
