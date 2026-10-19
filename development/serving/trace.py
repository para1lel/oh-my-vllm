class RequestTrace:
    def __init__(self, request):
        self.prompt = request["input_tokens"]
        self.expected_tokens = request["output_tokens"]
        self.computed = self.tokens = self.context_end = self.proposed = 0
        self.pending = None
        self.drafts = []
        self.steps = []

    def target(self, row, verify):
        if self.pending is not None or row["start"] != self.computed:
            raise ValueError("recomputed, missing, or uncommitted target rows")
        if row["prompt"] != self.prompt or row["count"] < 1:
            raise ValueError(
                "trace prompt length or target shape differs from manifest"
            )
        if verify:
            if self.computed != self.prompt + self.tokens - 1 or self.tokens < 1:
                raise ValueError("verification does not follow accepted history")
            if row["count"] != self.proposed + 1:
                raise ValueError("verification must consume all supplied drafts")
        elif self.computed >= self.prompt or self.computed + row["count"] > self.prompt:
            raise ValueError("prefill repeats or exceeds the prompt")
        self.pending = (row, verify)
        self.drafts = []

    def context(self, row):
        if (
            self.pending is None
            or row["start"] != self.context_end + 1
            or row["count"] < 1
        ):
            raise ValueError("repeated or missing MTP context")
        self.context_end = row["start"] + row["count"] - 1

    def draft(self, row, stage):
        if self.pending is None or stage != len(self.drafts) or stage > 3:
            raise ValueError("repeated or missing proposal stage")
        if row["position"] != self.context_end + stage:
            raise ValueError("proposal must follow its context endpoint")
        self.drafts.append(row["position"])

    def commit(self, row, mtp):
        if self.pending is None:
            raise ValueError("commit without target computation")
        target, verify = self.pending
        accepted, tokens = row["accepted"], row["tokens"]
        if verify:
            if not 0 <= accepted < target["count"] or tokens != accepted + 1:
                raise ValueError("invalid accepted prefix")
            end = self.computed + accepted + 1
        else:
            end = self.computed + target["count"]
            if accepted or tokens != int(end == self.prompt):
                raise ValueError("invalid prefill output")
        if row["position"] != end:
            raise ValueError("cache state differs from accepted history")
        if row["proposed"] and (not mtp or not tokens):
            raise ValueError("proposal has no final target output")
        remaining = self.expected_tokens - self.tokens - tokens
        if not 0 <= row["proposed"] <= min(4, max(0, remaining - 1)):
            raise ValueError("unnecessary or invalid proposals after the output limit")
        if mtp and self.context_end != max(0, end - 1 + int(bool(row["proposed"]))):
            raise ValueError("MTP context differs from accepted history")
        if self.drafts != [end + index for index in range(row["proposed"])]:
            raise ValueError("proposal positions or counts differ from the context")
        if self.drafts and self.context_end != end:
            raise ValueError("first proposal has no cached endpoint")
        if tokens:
            self.steps.append(dict(target=target, commit=row, verify=verify))
        self.computed, self.tokens, self.proposed = (
            end,
            self.tokens + tokens,
            row["proposed"],
        )
        self.pending = None


def manifest_requests(manifest):
    requests = manifest.get("requests", [])
    if manifest.get("mode") not in {"ordinary", "mtp4"} or not requests:
        raise ValueError("manifest requires a mode and requests")
    fields = ("ordinal", "id", "input_tokens", "output_tokens")
    for request in requests:
        if any(type(request.get(name)) is not int for name in fields):
            raise ValueError("manifest request fields must be integers")
        if min(request["id"], request["input_tokens"], request["output_tokens"]) < 1:
            raise ValueError("manifest IDs and lengths must be positive")
        response_id = request.get("response_id", "")
        if (
            not isinstance(response_id, str)
            or not response_id.startswith("chatcmpl_")
            or response_id.rsplit("_", 1)[-1] != str(request["id"])
        ):
            raise ValueError("manifest response and trace IDs differ")
    if sorted(request["ordinal"] for request in requests) != list(range(len(requests))):
        raise ValueError("manifest ordinals must be unique and contiguous")
    if len({request["id"] for request in requests}) != len(requests):
        raise ValueError("manifest requires distinct request IDs")
    return sorted(requests, key=lambda request: request["ordinal"])


def validate(events, manifest):
    requests = manifest_requests(manifest)
    identities = [event for event in events if event["operation"] == "identity"]
    if len(identities) != 1 or identities[0]["config"][
        "num_speculative_tokens"
    ] not in (0, 4):
        raise ValueError("trace requires one ordinary or MTP4 identity")
    mtp = bool(identities[0]["config"]["num_speculative_tokens"])
    if mtp != (manifest["mode"] == "mtp4"):
        raise ValueError("trace identity differs from the requested service mode")
    states = {request["id"]: RequestTrace(request) for request in requests}
    fields = {
        "target": ("id", "start", "count", "prompt"),
        "context": ("id", "start", "count"),
        "draft": ("id", "position"),
        "commit": ("id", "accepted", "tokens", "proposed", "position"),
        "identity": (),
    }
    for event in events:
        operation = event["operation"]
        if operation not in fields:
            raise ValueError(f"unsupported trace operation: {operation}")
        if operation in {"context", "draft"} and not mtp:
            raise ValueError("ordinary trace cannot contain speculative operations")
        if operation == "draft" and type(event.get("stage")) is not int:
            raise ValueError("draft stage must be an integer")
        if operation == "target" and type(event.get("verify")) is not bool:
            raise ValueError("target verify flag must be boolean")
        rows = event.get("requests", [])
        row_ids = [row["id"] for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise ValueError("trace event repeats a request ID")
        for row in rows:
            if any(
                type(row.get(name)) is not int or row[name] < 0
                for name in fields[operation]
            ):
                raise ValueError("trace fields must be nonnegative integers")
            if row["id"] not in states:
                raise ValueError("trace request is absent from the manifest")
            state = states[row["id"]]
            if operation == "target":
                state.target(row, event["verify"])
            elif operation == "context":
                state.context(row)
            elif operation == "draft":
                state.draft(row, event["stage"])
            elif operation == "commit":
                state.commit(row, mtp)
    for state in states.values():
        if (
            state.pending is not None
            or not state.tokens
            or state.computed < state.prompt
        ):
            raise ValueError("incomplete inference trace")
        if state.tokens != state.expected_tokens:
            raise ValueError("trace output differs from HTTP usage")
        if state.proposed:
            raise ValueError("unused proposals at generation end")
    return states


def prefill_allocation(pending, budget):
    allocations = {request: 0 for request, _, _ in pending}
    while pending and budget:
        share, remainder = divmod(budget, len(pending))
        before = budget
        remaining = []
        for index, (request, start, length) in enumerate(pending):
            current = allocations[request]
            count = min(length - start, current + share + int(index < remainder))
            if start + count < length:
                count = max(current, (start + count) // 784 * 784 - start)
            extra = min(count - current, budget)
            allocations[request] += extra
            budget -= extra
            if start + allocations[request] < length:
                remaining.append((request, start, length))
        if budget == before:
            for request, start, length in pending:
                current = allocations[request]
                extra = min(784, length - start - current)
                if extra <= budget:
                    allocations[request] += extra
                    budget -= extra
            break
        pending = remaining
    return allocations


def canonical_passes(events: list[dict], manifest: dict) -> list[dict]:
    states = validate(events, manifest)
    ordered = manifest_requests(manifest)
    positions = dict.fromkeys(states, 0)
    contexts = dict.fromkeys(states, 0)
    steps = dict.fromkeys(states, 0)
    passes = []
    mtp = manifest["mode"] == "mtp4"
    while any(steps[key] < len(state.steps) for key, state in states.items()):
        active = [
            row["id"]
            for row in ordered
            if steps[row["id"]] < len(states[row["id"]].steps)
        ][:4]
        decode = [key for key in active if positions[key] >= states[key].prompt]
        prefill = [key for key in active if key not in decode]
        budget = 32768 - sum(
            states[key].steps[steps[key]]["target"]["count"] for key in decode
        )
        allocations = prefill_allocation(
            [(key, positions[key], states[key].prompt) for key in prefill], budget
        )
        for keys, verify in ((decode, True), (prefill, False)):
            rows = []
            for key in keys:
                state = states[key]
                recorded = state.steps[steps[key]]
                count = recorded["target"]["count"] if verify else allocations[key]
                if not count:
                    continue
                start = positions[key]
                end = recorded["commit"]["position"] if verify else start + count
                complete = verify or end == state.prompt
                commit = (
                    recorded["commit"]
                    if complete
                    else dict(accepted=0, proposed=0, tokens=0)
                )
                context_end = (
                    max(0, end - 1 + int(bool(commit["proposed"]))) if mtp else 0
                )
                rows.append(
                    dict(
                        id=key,
                        start=start,
                        count=count,
                        end=end,
                        prompt=state.prompt,
                        accepted=commit["accepted"],
                        proposed=commit["proposed"],
                        tokens=commit["tokens"],
                        context_start=contexts[key] + 1,
                        context_count=context_end - contexts[key],
                    )
                )
                positions[key], contexts[key] = end, context_end
                if complete:
                    steps[key] += 1
            if rows:
                passes.append(dict(verify=verify, mtp=mtp, requests=rows))
    return passes
