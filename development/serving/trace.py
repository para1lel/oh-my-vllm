class RequestTrace:
    def __init__(self, expected_tokens):
        self.prompt = None
        self.computed = self.tokens = self.context_end = self.proposed = 0
        self.pending = None
        self.drafts = []
        self.expected_tokens = expected_tokens

    def target(self, row, verify):
        if self.pending is not None or row["start"] != self.computed:
            raise ValueError("recomputed, missing, or uncommitted target rows")
        if self.prompt is None:
            self.prompt = row["prompt"]
        if row["prompt"] != self.prompt or row["count"] < 1:
            raise ValueError("invalid target shape")
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
        if self.expected_tokens is not None:
            remaining = self.expected_tokens - self.tokens - tokens
            if row["proposed"] > max(0, remaining - 1):
                raise ValueError("unnecessary proposals after the output limit")
        if mtp and self.context_end != max(0, end - 1 + int(bool(row["proposed"]))):
            raise ValueError("MTP context differs from accepted history")
        if self.drafts != [end + index for index in range(row["proposed"])]:
            raise ValueError("proposal positions or counts differ from the context")
        if self.drafts and self.context_end != end:
            raise ValueError("first proposal has no cached endpoint")
        self.computed, self.tokens, self.proposed = (
            end,
            self.tokens + tokens,
            row["proposed"],
        )
        self.pending = None


def validate(events, request_ids, expected_tokens):
    mtp = any(
        event["operation"] == "identity" and event["config"]["num_speculative_tokens"]
        for event in events
    )
    states = {request: RequestTrace(expected_tokens) for request in request_ids}
    for event in events:
        operation = event["operation"]
        for row in event.get("requests", []):
            if row["id"] not in states:
                continue
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
        if expected_tokens is not None and state.tokens != expected_tokens:
            raise ValueError("trace output differs from HTTP usage")
        if state.proposed:
            raise ValueError("unused proposals at generation end")
    return mtp
