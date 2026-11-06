from ...worker.wire import validate_execute, validate_result

_STATE_FIELDS = (
    "fa_block_table",
    "mamba_block_table",
    "fa_block_copies",
    "prefix_state_slot",
    "cache_checkpoint",
)


class ResidentWire:
    def _decode_execute(self, message):
        expected = validate_execute(message, self.worker_epoch)
        if message["step_id"] <= self.last_step_id:
            raise ValueError("DeepSeek execution step is stale")
        scheduled = []
        for row in message["scheduled"]:
            if row["purpose"] != "advance" or row["action"] != "forward":
                raise ValueError("DeepSeek resident work requires advance/forward")
            state = row["model_state"]
            if state["kind"] != "deepseek_resident_v1":
                raise ValueError("DeepSeek resident model state is required")
            if any(name in row for name in (*_STATE_FIELDS, "num_computed_tokens")):
                raise ValueError("DeepSeek model state must be nested")
            local = dict(row)
            local.update((name, state[name]) for name in _STATE_FIELDS)
            local["num_computed_tokens"] = row["forward_start"]
            scheduled.append(local)
        return {**message, "scheduled": scheduled}, expected

    def _validate_resident_sampling(self, message):
        for row in message["scheduled"]:
            self.cache._validate_row(row, end=row["forward_end"])
            request = self.requests[row["request_id"]]
            history = row.get("prefill_token_ids")
            history = request.history if history is None else history
            start, end = row["forward_start"], row["forward_end"]
            if row["accepted_end"] != len(history):
                raise ValueError("DeepSeek accepted history endpoint differs")
            known = min(len(row["token_ids"]), len(history) - start)
            if known <= 0 or row["token_ids"][:known] != history[start : start + known]:
                raise ValueError("DeepSeek forward inputs differ from accepted history")
            drafts = len(row["token_ids"]) - known
            sampling = row["sampling"]
            expected = (
                "none" if end < len(history) else "verify" if drafts else "target"
            )
            if sampling["kind"] != expected:
                raise ValueError("DeepSeek sampling differs from the accepted range")
            if drafts and (sampling["drafts"] != drafts or drafts > 5):
                raise ValueError("DeepSeek verification draft count differs")

    def _validate_resident_endpoint(self, row, request):
        end = request.state_position
        if request.cache_slot is None or end <= 0:
            raise ValueError("DeepSeek result has no resident accepted endpoint")
        state = row["model_state"]
        last = (end - 1) // self.cache.block_tokens
        fa = state["fa_block_table"]
        for index, block in enumerate(fa[: last + 1]):
            page = self.cache.fa_pages.get(block)
            required = min(
                self.cache.block_tokens, end - index * self.cache.block_tokens
            )
            if page is None or page.valid_tokens < required:
                raise ValueError("DeepSeek result has an incomplete accepted FA page")
        block = state["mamba_block_table"][last]
        page = self.cache.mamba_pages.get(block)
        if page is None or page.endpoint != end:
            raise ValueError("DeepSeek result has no exact accepted state export")
        checkpoint = state["cache_checkpoint"]
        if checkpoint is not None and row["forward_start"] < checkpoint[0] <= end:
            page = self.cache.mamba_pages.get(checkpoint[1])
            if page is None or page.endpoint != checkpoint[0]:
                raise ValueError("DeepSeek scheduled checkpoint export is missing")

    def _resident_output(self, row, output):
        request = self.requests[row["request_id"]]
        self._validate_resident_endpoint(row, request)
        return {
            **output,
            "lease_id": row["lease_id"],
            "purpose": row["purpose"],
            "action": row["action"],
            "error": None,
            "computed_end": request.state_position,
            "accepted_end": len(request.history),
            "validated_exports": [],
            "committed_page_valid_ends": [],
            "mtp_accepted_endpoint": None,
        }

    def _encode_execute(self, message, result, expected):
        outputs = [
            self._resident_output(expected[output["request_id"]], output)
            for output in result["outputs"]
        ]
        reply = {
            **result,
            "step_id": message["step_id"],
            "worker_epoch": self.worker_epoch,
            "outputs": outputs,
            "rank_completions": [],
        }
        validate_result(message, reply, expected)
        return reply
