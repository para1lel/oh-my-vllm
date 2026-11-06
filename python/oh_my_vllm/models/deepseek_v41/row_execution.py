import torch

from .batch import execute_decode_group
from .speculation import verify_greedy_drafts
from .trace import TargetTrace


class RowExecution:
    @staticmethod
    def _output(request, row, tokens, text, accepted=0, drafts=None):
        return {
            "request_id": row["request_id"],
            "token_ids": tokens,
            "num_accepted_draft_tokens": accepted,
            "new_draft_token_ids": [] if drafts is None else drafts,
            "text": text,
            "finish_reason": request.finish,
            "reasoning_tokens": request.thinking_count,
        }

    def _execute_row(self, row):
        request = self.requests[row["request_id"]]
        try:
            return self._execute_row_impl(row)
        except BaseException:
            self._free_slot(request)
            request.state_position = 0
            raise

    def _prepare_row(self, row):
        request = self.requests[row["request_id"]]
        if row.get("prefill_token_ids") is not None:
            request.history = list(row["prefill_token_ids"])
        slot_was_empty = request.cache_slot is None
        slot = self._slot(row["request_id"])
        token_ids = list(row["token_ids"])
        start_pos = int(row["num_computed_tokens"])
        try:
            if slot_was_empty:
                self.cache.restore(
                    row["request_id"],
                    slot,
                    row,
                    start_pos=start_pos,
                )
        except BaseException:
            if slot_was_empty:
                self._free_slot(request)
            raise
        return request, slot, token_ids, start_pos

    def _execute_prefill_group(self, rows, *, final=False):
        prepared = []
        try:
            for row in rows:
                request, slot, token_ids, start_pos = self._prepare_row(row)
                prepared.append((row, request, slot, token_ids, start_pos))
                self.cache.begin_step(
                    row,
                    slot,
                    start_pos=start_pos,
                    end_pos=start_pos + len(token_ids),
                    minimum_end=start_pos + len(token_ids),
                )
            start_pos = prepared[0][4]
            length = len(prepared[0][3])
            if any(item[4] != start_pos or len(item[3]) != length for item in prepared):
                raise ValueError("DeepSeek prefill batch rows are not aligned")
            cache_slots = torch.tensor(
                [item[2] for item in prepared],
                device=self.device,
                dtype=torch.long,
            )
            accounting = TargetTrace.for_requests(
                self, [(item[0]["request_id"], item[1].prompt_len) for item in prepared]
            )
            _, logits, main_hidden = self._target_batch(
                [item[3] for item in prepared],
                start_pos,
                cache_slots,
                compute_logits=final,
                **({"accounting": accounting} if accounting is not None else {}),
            )
            if accounting is not None:
                accounting.finish()
            mtp_state_updated = self._seed_mtp_batch(
                main_hidden,
                start_pos,
                cache_slots,
                source_length=self._mtp_source_length(length),
            )
            return self._finish_prefill_group(
                prepared,
                start_pos,
                length,
                logits,
                main_hidden,
                final,
                mtp_state_updated,
            )
        except BaseException:
            for _row, request, _slot, _token_ids, _start_pos in prepared:
                self._free_slot(request)
                request.state_position = 0
            raise

    def _finish_prefill_group(
        self, prepared, start_pos, length, logits, main_hidden, final, mtp_state_updated
    ):
        outputs = {}
        accepted_end = start_pos + length
        targets = logits.argmax(dim=-1).tolist() if final else []
        for index, (row, request, slot, token_ids, _) in enumerate(prepared):
            tokens, text, proposals = [], "", []
            if final:
                target = int(targets[index])
                tokens, text = request.append([target])
                proposals = self._next_proposals(
                    request,
                    tokens,
                    main_hidden[index : index + 1] if main_hidden is not None else None,
                    accepted_end,
                    slot,
                )
            self.cache.commit(
                row["request_id"],
                row,
                slot,
                start_pos=start_pos,
                accepted_end=accepted_end,
                include_mtp=mtp_state_updated,
            )
            request.state_position = accepted_end
            self.trace.record(
                "target",
                rank=self.rank,
                accepted=0,
                proposed=0,
                verify=False,
                execution=self._target_execution,
                requests=[
                    {
                        "id": row["request_id"],
                        "start": start_pos,
                        "count": len(token_ids),
                        "prompt": request.prompt_len,
                    }
                ],
            )
            outputs[row["request_id"]] = self._output(
                request,
                row,
                tokens,
                text,
                0,
                proposals,
            )
        return outputs

    def _execute_row_impl(self, row):
        request = self.requests[row["request_id"]]
        _, slot, token_ids, start_pos = self._prepare_row(row)
        if not token_ids:
            return self._output(request, row, [], "")

        known_count = max(0, min(len(token_ids), len(request.history) - start_pos))
        final = start_pos + len(token_ids) >= len(request.history)
        accounting = TargetTrace.for_requests(
            self, [(row["request_id"], request.prompt_len)]
        )
        trace_options = {} if accounting is None else {"accounting": accounting}
        if not final:
            result = self._run_prefill_row(
                row, slot, token_ids, start_pos, trace_options
            )
        else:
            known = token_ids[:known_count]
            if not known:
                raise ValueError("DeepSeek execution row has no accepted input")
            drafts = token_ids[known_count:]
            self.cache.begin_step(
                row,
                slot,
                start_pos=start_pos,
                end_pos=start_pos + len(token_ids),
                minimum_end=start_pos + known_count,
            )
            if drafts:
                result = self._verify_row(
                    request,
                    token_ids,
                    known_count,
                    drafts,
                    start_pos,
                    slot,
                    trace_options,
                )
            else:
                result = self._decode_row(
                    request, known, start_pos, slot, trace_options
                )
        return self._finish_row(
            request,
            row,
            slot,
            token_ids,
            start_pos,
            known_count,
            final,
            accounting,
            result,
        )

    def _run_prefill_row(self, row, slot, token_ids, start_pos, trace_options):
        self.cache.begin_step(
            row,
            slot,
            start_pos=start_pos,
            end_pos=start_pos + len(token_ids),
            minimum_end=start_pos + len(token_ids),
        )
        _, _, main_hidden = self._target(
            token_ids,
            start_pos,
            slot,
            compute_logits=False,
            **trace_options,
        )
        mtp_state_updated = self._seed_mtp(
            main_hidden,
            start_pos,
            slot,
            source_length=self._mtp_source_length(len(token_ids)),
        )
        accepted_end = start_pos + len(token_ids)
        tokens, text = [], ""
        accepted = 0
        proposals = []

        return accepted_end, mtp_state_updated, tokens, text, accepted, proposals

    def _verify_row(
        self, request, token_ids, known_count, drafts, start_pos, slot, trace_options
    ):
        _, logits, main_hidden = self._target(
            token_ids,
            start_pos,
            slot,
            full_logits=True,
            logit_start=known_count - 1,
            **trace_options,
        )
        if (
            not isinstance(logits, torch.Tensor)
            or logits.ndim != 3
            or logits.size(0) != 1
            or logits.size(1) != len(drafts) + 1
        ):
            raise ValueError("DeepSeek batched verification returned invalid logits")
        target_predictions = logits[0].argmax(dim=-1).tolist()
        verification = verify_greedy_drafts(drafts, target_predictions)
        emitted = list(verification.token_ids)
        tokens, text = request.append(emitted)
        accepted = min(
            verification.accepted_draft_tokens,
            max(0, len(tokens) - 1),
        )
        accepted_input_count = known_count + accepted
        hidden_start = start_pos
        hidden_count = accepted_input_count
        accepted_hidden = None
        if isinstance(main_hidden, torch.Tensor):
            hidden_start = start_pos + len(token_ids) - main_hidden.size(1)
            hidden_count = start_pos + accepted_input_count - hidden_start
            if not 0 < hidden_count <= main_hidden.size(1):
                raise ValueError("DeepSeek accepted hidden state is unavailable")
            accepted_hidden = main_hidden[:, :hidden_count]
        mtp_state_updated = self._seed_mtp(
            accepted_hidden,
            hidden_start,
            slot,
            source_length=hidden_count,
        )
        accepted_end = start_pos + accepted_input_count
        proposals = self._next_proposals(
            request, tokens, accepted_hidden, accepted_end, slot
        )

        return accepted_end, mtp_state_updated, tokens, text, accepted, proposals

    def _decode_row(self, request, known, start_pos, slot, trace_options):
        _, logits, main_hidden = self._target(known, start_pos, slot, **trace_options)
        mtp_state_updated = self._seed_mtp(
            main_hidden,
            start_pos,
            slot,
            source_length=self._mtp_source_length(len(known)),
        )
        target_predictions = [int(logits.argmax(dim=-1).item())]
        accepted_end = start_pos + len(known)
        tokens, text = request.append(target_predictions)
        accepted = 0
        proposals = self._next_proposals(
            request, tokens, main_hidden, accepted_end, slot
        )

        return accepted_end, mtp_state_updated, tokens, text, accepted, proposals

    def _finish_row(
        self,
        request,
        row,
        slot,
        token_ids,
        start_pos,
        known_count,
        final,
        accounting,
        result,
    ):
        accepted_end, mtp_state_updated, tokens, text, accepted, proposals = result
        self.cache.commit(
            row["request_id"],
            row,
            slot,
            start_pos=start_pos,
            accepted_end=accepted_end,
            include_mtp=mtp_state_updated,
        )
        if final and row["token_ids"][known_count:]:
            self.cache.restore(
                row["request_id"],
                slot,
                row,
                start_pos=accepted_end,
                use_prefix_state=False,
            )
        request.state_position = accepted_end
        if accounting is not None:
            accounting.finish(
                accepted=accepted, proposed=len(token_ids) - known_count if final else 0
            )
        self.trace.record(
            "target",
            rank=self.rank,
            accepted=accepted,
            proposed=len(token_ids) - known_count if final else 0,
            verify=bool(token_ids[known_count:]) if final else False,
            execution=self._target_execution,
            requests=[
                {
                    "id": row["request_id"],
                    "start": row["num_computed_tokens"],
                    "count": len(token_ids),
                    "prompt": request.prompt_len,
                }
            ],
        )
        return self._output(request, row, tokens, text, accepted, proposals)

    def _execute_decode_group(
        self, rows, *, plan=None, valid_rows=None, dispatch_combine=None
    ):
        return execute_decode_group(
            self,
            rows,
            plan=plan,
            valid_rows=valid_rows,
            dispatch_combine=dispatch_combine,
        )
