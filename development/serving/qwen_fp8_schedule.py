"""Canonical ordinal schedule of useful accepted inputs, independent of launches."""

from .qwen_fp8_trace import require


def prefill_allocation(pending, budget):
    allocations = dict.fromkeys((key for key, _, _ in pending), 0)
    while pending and budget:
        share, remainder = divmod(budget, len(pending))
        remaining = []
        for index, (key, start, end) in enumerate(pending):
            count = min(end - start - allocations[key], share + int(index < remainder))
            allocations[key] += count
            budget -= count
            if start + allocations[key] < end:
                remaining.append((key, start, end))
        pending = remaining
    return allocations


def accepted_steps(state):
    end = state.accepted_end
    generated, steps = state.manifest["input_tokens"], []
    for entry in state.steps:
        kept = min(len(entry["tokens"]), max(0, end - generated))
        if not kept:
            continue
        row = entry.get("forward", entry)["row"]
        correct = row["forward_start"]
        for position, token in enumerate(row["token_ids"], row["forward_start"]):
            if position >= end or state.history[position] != token:
                break
            correct = position + 1
        steps.append(
            dict(
                tokens=kept,
                dispatch=row["forward_end"] - row["forward_start"],
                verify=row["sampling"]["kind"] == "verify",
                correct_end=correct,
            )
        )
        generated += kept
    require(generated == end and steps, "missing accepted sampling dependencies")
    require(
        steps[0]["tokens"] == 1 and not steps[0]["verify"],
        "missing prompt completion sample",
    )
    return steps


def logical_row(key, start, end, dispatch, heads, context):
    return dict(
        id=key,
        start=start,
        end=end,
        dispatch_count=dispatch,
        head_count=heads,
        context_count=context,
    )


def closing_rows(finished, states, positions, contexts, mtp):
    closing = []
    for key in finished:
        end, start = states[key].accepted_end, positions[key]
        require(start <= end, "canonical useful work exceeded accepted history")
        if start < end:
            closing.append(
                logical_row(
                    key,
                    start,
                    end,
                    end - start,
                    0,
                    max(0, end - 1 - contexts[key]) if mtp else 0,
                )
            )
    return closing


def bind_forward_evidence(passes, states):
    if not any(hasattr(state, "calls") for state in states.values()):
        return passes
    precisions = {}
    for key, state in states.items():
        require(hasattr(state, "calls"), "missing recorded forward evidence")
        positions = {}
        for call in state.calls:
            row = call["row"]
            if row["action"] != "forward":
                continue
            precision = call["ba_precision"]
            require(
                precision in ("bf16", "fp32"), "missing recorded forward BA precision"
            )
            for at, token in enumerate(row["token_ids"], row["forward_start"]):
                if (
                    at >= min(state.accepted_end, call["result"]["computed_end"])
                    or token != state.history[at]
                ):
                    break
                positions.setdefault(at, precision)
        precisions[key] = positions
    for logical in passes:
        logical["requires_forward_evidence"] = True
        for row in logical["requests"]:
            spans = []
            for at in range(row["start"], row["end"]):
                precision = precisions[row["id"]].get(at)
                require(
                    precision is not None,
                    "useful token has no matching forward arithmetic",
                )
                if spans and spans[-1][2] == precision:
                    spans[-1] = spans[-1][0], at + 1, precision
                else:
                    spans.append((at, at + 1, precision))
            row["ba_spans"] = spans
    return passes


def canonical_passes(states, mode):
    mtp = mode == "mtp4"
    ordered = sorted(states, key=lambda key: states[key].manifest["ordinal"])
    steps = {key: accepted_steps(states[key]) for key in ordered}
    positions, indices, contexts = (dict.fromkeys(ordered, 0) for _ in range(3))
    pending, passes = list(ordered), []
    while pending:
        active = pending[:4]
        decode = [
            key
            for key in active
            if positions[key] >= states[key].manifest["input_tokens"]
            and indices[key] < len(steps[key])
        ]
        budget = (32764 if mtp else 32768) - sum(
            steps[key][indices[key]]["dispatch"] for key in decode
        )
        require(budget >= 0, "decode exceeds canonical budget")
        prefill = [
            key
            for key in active
            if positions[key] < states[key].manifest["input_tokens"]
        ]
        allocated = prefill_allocation(
            [
                (key, positions[key], states[key].manifest["input_tokens"])
                for key in prefill
            ],
            budget,
        )
        verified = [key for key in decode if steps[key][indices[key]]["verify"]]
        unverified = [key for key in decode if key not in verified]
        for keys, verify, phase in (
            (verified, True, "decode"),
            (unverified, False, "decode"),
            (prefill, False, "prefill"),
        ):
            rows = []
            for key in keys:
                start, prompt = positions[key], states[key].manifest["input_tokens"]
                step = steps[key][indices[key]]
                count = step["tokens"] if phase == "decode" else allocated[key]
                if not count:
                    continue
                end = start + count
                heads = count if phase == "decode" else int(end == prompt)
                dispatch = step["dispatch"] if phase == "decode" else count
                if phase == "decode" and indices[key] == len(steps[key]) - 1:
                    end = max(end, min(states[key].accepted_end, step["correct_end"]))
                context_end = max(0, end - 1) if mtp else 0
                rows.append(
                    logical_row(
                        key, start, end, dispatch, heads, context_end - contexts[key]
                    )
                )
                positions[key], contexts[key] = end, context_end
                if heads:
                    indices[key] += 1
            if rows:
                passes.append(dict(phase=phase, verify=verify, mtp=mtp, requests=rows))
        finished = [key for key in active if indices[key] == len(steps[key])]
        if finished:
            closing = closing_rows(finished, states, positions, contexts, mtp)
            for key in finished:
                pending.remove(key)
            if closing:
                passes.append(
                    dict(phase="exact_end", verify=False, mtp=mtp, requests=closing)
                )
        require(
            decode or any(allocated.values()) or finished,
            "canonical scheduler made no progress",
        )
    return bind_forward_evidence(passes, states)
