from development.kernels.roofline import Work
from development.kernels.work import add, linear

from .trace import validate


def scale(work, count):
    return Work(*(value * count for value in vars(work).values()))


def feed_forward(rows):
    return add(linear(rows, 34816, 5120, fp8=True), linear(rows, 5120, 17408, fp8=True))


def attention(rows, pairs, reads, *, columns=14336):
    return add(
        linear(rows, columns, 5120, fp8=True),
        linear(rows, 5120, 6144, fp8=True),
        Work(reads * 4096, bf16_flops=4 * pairs * 24 * 256),
    )


def vocabulary(rows):
    return linear(rows, 248320, 5120)


def target(rows, queries, *, prefill=False):
    pairs = sum(count * start + count * (count + 1) // 2 for start, count in queries)
    reads = sum(start + count for start, count in queries)
    recurrent = add(
        linear(rows, 16384, 5120, fp8=True),
        linear(rows, 96, 5120),
        linear(rows, 5120, 6144, fp8=True),
        Work(
            len(queries) * 2 * (48 * 128 * 128 + 10240 * 3) * 2,
            fp32_flops=rows * 48 * (7 * 128 * 128 + 3 * 128),
        ),
    )
    result = add(
        scale(recurrent, 48),
        scale(feed_forward(rows), 64),
        scale(attention(rows, pairs, reads), 16),
        Work(rows * 5120 * 2),
    )
    return add(result, vocabulary(len(queries) if prefill else rows))


def context(rows):
    return add(
        linear(rows, 5120, 10240), linear(rows, 2048, 5120, fp8=True), Work(rows * 4096)
    )


def draft(positions, stage):
    rows = len(positions)
    result = add(
        attention(
            rows,
            sum(positions),
            sum(positions),
            columns=12288 if stage == 0 else 14336,
        ),
        feed_forward(rows),
        vocabulary(rows),
    )
    return result if stage == 0 else add(result, linear(rows, 5120, 10240))


def lower_bound(events, request_ids, expected_tokens=None):
    mtp = validate(events, request_ids, expected_tokens)
    rounds, current, prefill, acceptance = {}, {key: -1 for key in request_ids}, {}, {}
    for event in events:
        operation = event["operation"]
        if operation == "identity":
            continue
        rows = [row for row in event.get("requests", []) if row["id"] in request_ids]
        for row in rows:
            request = row["id"]
            if operation == "target" and not event["verify"]:
                prefill[request] = row["prompt"]
                continue
            if operation == "target":
                current[request] += 1
            key = current[request]
            frame = rounds.setdefault(
                key,
                {
                    "target": [],
                    "counts": {},
                    "context": 0,
                    "draft": {},
                    "accept": Work(0),
                },
            )
            if operation == "target":
                frame["target"].append((row["start"], row["count"]))
                frame["counts"][request] = row["count"]
            elif operation == "context":
                frame["context"] += row["count"]
            elif operation == "draft":
                frame["draft"].setdefault(event["stage"], []).append(row["position"])
            elif operation == "commit":
                if mtp and request in frame["counts"]:
                    rejected = row["accepted"] < frame["counts"][request] - 1
                    draws = row["accepted"] + int(rejected) + 1
                    frame["accept"] = add(
                        frame["accept"],
                        Work(
                            0,
                            fp32_flops=(248320 if rejected else 0),
                            fp64_flops=draws * 2,
                        ),
                    )
                totals = acceptance.setdefault(
                    request, {"tokens": 0, "accepted": 0, "proposed": 0}
                )
                for name in totals:
                    totals[name] += row[name]
    if set(prefill) != set(request_ids):
        raise ValueError("missing uncached prefill evidence")
    prefill_work = target(
        sum(prefill.values()),
        [(0, length) for length in prefill.values()],
        prefill=True,
    )
    seconds = prefill_work.lower_bound_seconds()
    work = prefill_work
    components = [{"operation": "prefill", "seconds": seconds, **vars(prefill_work)}]
    for index, frame in sorted(rounds.items()):
        stages = []
        queries = frame["target"]
        if queries:
            stages.append(
                ("target", target(sum(count for _, count in queries), queries))
            )
        if frame["context"]:
            stages.append(("context", context(frame["context"])))
        if any(vars(frame["accept"]).values()):
            stages.append(("acceptance", frame["accept"]))
        stages.extend(
            (f"draft_{stage}", draft(positions, stage))
            for stage, positions in sorted(frame["draft"].items())
        )
        for operation, item in stages:
            seconds += item.lower_bound_seconds()
            work = add(work, item)
            components.append(
                {
                    "round": index,
                    "operation": operation,
                    "seconds": item.lower_bound_seconds(),
                    **vars(item),
                }
            )
    return {
        "seconds": work.lower_bound_seconds(),
        "sequential_regions_seconds": seconds,
        "work": vars(work),
        "components": components,
        "acceptance": acceptance,
        "method": (
            "one full prefill; per-request verification rounds combined; "
            "no padded rows, replay warmup, intermediate copies, or launch floor"
        ),
    }
