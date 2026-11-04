"""CPU checks for adapter control and cache state.

GPU checks cover real MTP history, batched model computation, and Graph execution.
"""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
from oh_my_vllm.models.deepseek_v41.inference import (
    DEFAULT_PREFILL_ROWS,
    MAX_PREFILL_ROWS,
    Worker,
    prefill_row_limit,
)
from oh_my_vllm.models.deepseek_v41.request import Request
from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts

from tests.test_deepseek_cache import FakeModel


class _PositionModel(FakeModel):
    def __init__(self, *, value_offset=1, predictions=None, batch_size=2):
        super().__init__()
        self.value_offset, self.predictions = value_offset, predictions
        self.engram_hash.cache = torch.zeros(batch_size, 32, dtype=torch.int64)
        attention = self.layers[0].attn
        attention.window_kv_cache = torch.zeros(batch_size, 4, 3)
        attention.compress_kv_cache = torch.zeros(batch_size, 16, 3)
        attention.indexer.k_cache = torch.zeros(batch_size, 16, 2)
        attention.compressor.kv_state = torch.zeros(batch_size, 2, 3)
        attention.compressor.score_state = torch.zeros(batch_size, 2, 3)
        self.mtp[0].attn.window_kv_cache = torch.zeros(batch_size, 4, 3)

    def __call__(self, inputs, *, start_pos, cache_slots_cpu, **kwargs):
        self._state_export.bind(cache_slots_cpu)
        attention = self.layers[0].attn
        positions = torch.arange(start_pos, start_pos + inputs.size(1))
        offsets = torch.tensor(cache_slots_cpu).view(-1, 1, 1) * 100
        values = (
            (positions + self.value_offset).float().view(1, -1, 1) + offsets
        ).expand(-1, -1, 3)
        self._state_export.window(attention.window_kv_cache, values, start_pos)
        self._state_export.tail(
            attention.compressor.kv_state,
            attention.compressor.score_state,
            values,
            values + 100,
            start_pos,
        )
        for index, slot in enumerate(cache_slots_cpu):
            attention.window_kv_cache[slot, positions[-4:].remainder(4)] = values[
                index, -4:
            ]
            self.engram_hash.cache[slot, positions] = positions
        logits = (
            None
            if self.predictions is None
            else torch.nn.functional.one_hot(
                torch.tensor([self.predictions]), 16
            ).float()
        )
        return None, logits, values

    def forward_spec_seed(self, hidden, *, start_pos, cache_slots):
        self._state_export.bind(cache_slots.tolist())
        window = self.mtp[0].attn.window_kv_cache
        self._state_export.window(window, hidden, start_pos)
        positions = torch.arange(start_pos, start_pos + hidden.size(1))
        for index, slot in enumerate(cache_slots.tolist()):
            window[slot, positions[-4:].remainder(4)] = hidden[index, -4:]

    def forward_spec(self, input_ids, hidden, *, start_pos, cache_slots):
        self.forward_spec_seed(hidden, start_pos=start_pos, cache_slots=cache_slots)
        return None


class _ChainModel(_PositionModel):
    """A CPU control fixture whose proposal does not read the MTP window."""

    def __init__(self, *, acceptance=5, batch_size=2):
        super().__init__(batch_size=batch_size)
        self.acceptance = acceptance
        self.targets, self.proposals = [], []
        self.engram_hash.cache = torch.zeros(batch_size, 64, dtype=torch.int64)
        self.layers[0].attn.compress_kv_cache = torch.zeros(batch_size, 32, 3)
        self.layers[0].attn.indexer.k_cache = torch.zeros(batch_size, 32, 2)

    @staticmethod
    def prediction(token, previous, position):
        return (token + 2 * previous + position) % 120 + 1

    def __call__(
        self,
        inputs,
        *,
        start_pos,
        cache_slots_cpu,
        full_logits=False,
        logit_start=0,
        routing_trace=None,
        **kwargs,
    ):
        hidden, predictions = [], []
        attention = self.layers[0].attn
        for row, slot in zip(inputs.tolist(), cache_slots_cpu, strict=True):
            self.targets.append(dict(start=start_pos, tokens=row, verify=full_logits))
            previous = (
                int(self.engram_hash.cache[slot, start_pos - 1]) if start_pos else 0
            )
            values, targets = [], []
            for position, token in enumerate(row, start_pos):
                values.append([position + 1, token, previous])
                targets.append(self.prediction(token, previous, position))
                self.engram_hash.cache[slot, position] = token
                previous = token
            values = torch.tensor([values], dtype=torch.float32)
            self._state_export.bind((slot,))
            self._state_export.window(attention.window_kv_cache, values, start_pos)
            kv, score = attention.compressor.kv_state, attention.compressor.score_state
            tail = torch.cat((kv[slot : slot + 1, : start_pos % 2], values), dim=1)
            self._state_export.tail(kv, score, tail, tail + 100, start_pos)
            positions = torch.arange(start_pos, start_pos + len(row))
            attention.window_kv_cache[slot, positions[-4:].remainder(4)] = values[
                0, -4:
            ]
            for offset, position in enumerate(positions.tolist()):
                if position % 2:
                    attention.compress_kv_cache[slot, position // 2] = values[0, offset]
                    attention.indexer.k_cache[slot, position // 2] = values[
                        0, offset, :2
                    ]
            kv[slot].zero_()
            score[slot].fill_(-torch.inf)
            if (start_pos + len(row)) % 2:
                kv[slot, 0] = values[0, -1]
                score[slot, 0] = values[0, -1] + 100
            hidden.append(values)
            predictions.append(targets)
        if routing_trace is not None:
            routing_trace.record(
                0, torch.arange(6, dtype=torch.int32).expand(inputs.numel(), -1)
            )
        logits = torch.nn.functional.one_hot(torch.tensor(predictions), 128).float()
        logits = logits[:, logit_start:] if full_logits else logits[:, -1]
        return None, logits, torch.cat(hidden)

    def forward_spec(self, input_ids, hidden, *, start_pos, cache_slots):
        if tuple(hidden.shape) != (1, 1, 3) or hidden[0, 0, 0] != start_pos + 1:
            raise ValueError("proposal did not receive the accepted boundary hidden")
        self.proposals.append(
            dict(start=start_pos, token=int(input_ids.item()), hidden=hidden.clone())
        )
        self.forward_spec_seed(hidden, start_pos=start_pos, cache_slots=cache_slots)
        if start_pos == 0:
            return None
        token, previous = int(input_ids.item()), int(hidden[0, 0, 1])
        drafts = []
        for position in range(start_pos + 1, start_pos + 6):
            predicted = self.prediction(token, previous, position)
            drafts.append(predicted)
            previous, token = token, predicted
        accepted = (
            self.acceptance
            if isinstance(self.acceptance, int)
            else self.acceptance[int(cache_slots.item())]
        )
        if accepted < 5:
            drafts[accepted] = (drafts[accepted] + 1) % 128
        return torch.tensor([[int(input_ids.item()), *drafts]]), None, None


def _chain_worker(*, prompts=((10, 11, 12),), acceptance=5, limit=24, **options):
    worker = Worker.__new__(Worker)
    worker.model = _ChainModel(acceptance=acceptance, batch_size=len(prompts) + 1)
    worker.device = torch.device("cpu")
    worker.dspark = True
    worker.args = SimpleNamespace(window_size=4)
    worker.slots = set(range(len(prompts) + 1))
    tokenizer = SimpleNamespace(
        valid_tokens=128,
        grammar=None,
        eos=set(),
        decode_bytes=lambda tokens: b"x" * len(tokens),
    )
    worker.requests = {
        key: Request(
            tokenizer, prompt, dict(max_tokens=limit, sampling={"seed": 0}, **options)
        )
        for key, prompt in enumerate(prompts, 1)
    }
    for request in worker.requests.values():
        request.cache_slot = None
    worker.trace = SimpleNamespace(record=Mock())
    worker.cache = DeepSeekCache(
        worker.model,
        block_tokens=4,
        max_seq_len=64,
        host_limit=1 << 20,
        requests=worker.requests,
        free_slots=worker.slots,
    )
    return worker


def _chain_row(worker, tokens, *, key=1, start=None):
    request = worker.requests[key]
    blocks = list(range(16 * (key - 1) + 1, 16 * key + 1))
    return dict(
        request_id=key,
        token_ids=list(tokens),
        num_computed_tokens=request.state_position if start is None else start,
        fa_block_table=blocks,
        mamba_block_table=list(blocks),
    )


def _chain_state(worker, key=1):
    cache, request = worker.cache, worker.requests[key]
    slot, end = request.cache_slot, request.state_position
    values = [value[slot] for value in (*cache._window, *cache._mtp_window)]
    values += [
        value[slot, : end // ratio]
        for value, ratio in (*cache._compressed, *cache._index)
    ]
    values += [value[slot] for tail in cache._tails for value in tail]
    values += [value[slot, :end] for value in cache._engram]
    return [value.clone() for value in values]


def _chain_batch(worker, tokens, positions, slots, **options):
    """Supply batch result metadata through separate CPU model calls."""
    results = [
        worker.model(
            torch.tensor([row]),
            start_pos=start,
            cache_slots_cpu=(slot,),
            full_logits=len(row) > 1,
        )
        for row, start, slot in zip(tokens, positions, slots, strict=True)
    ]
    return SimpleNamespace(
        logits=torch.cat([result[1] for result in results]),
        hidden=torch.cat([result[2] for result in results]),
        execution="eager",
        routes=None,
        bindings=None,
        call=1,
    )


def _verification_worker(predictions):
    worker = Worker.__new__(Worker)
    worker.model = _PositionModel(value_offset=0, predictions=predictions)
    worker.device = torch.device("cpu")
    worker.dspark = True
    worker.args = SimpleNamespace(window_size=4)
    worker.slots = {1}
    worker.requests = {
        1: SimpleNamespace(
            cache_slot=0,
            history=[10] * 20,
            prompt_len=20,
            state_position=0,
            generated=0,
            limit=8,
            finish=None,
            thinking_count=0,
            append=lambda tokens: (tokens, "ok"),
        )
    }
    worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
    worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=32)
    return worker, {
        "request_id": 1,
        "token_ids": [10] * 20 + [1, 2, 3, 4, 5],
        "num_computed_tokens": 0,
        "fa_block_table": [1, 2, 3, 4, 5, 6, 7],
        "mamba_block_table": [0, 0, 0, 3, 5, 6, 7],
    }


class DeepSeekSpeculation(unittest.TestCase):
    def _run_chain(self, mode, *, acceptance):
        worker = _chain_worker(acceptance=acceptance)
        request = worker.requests[1]
        next_proposals = worker._next_proposals
        if mode == "greedy":
            worker._next_proposals = lambda *_args: []
        elif mode == "legacy":
            worker._next_proposals = lambda *args: (
                [] if worker.model.targets[-1]["verify"] else next_proposals(*args)
            )
        output = worker._execute_row(_chain_row(worker, request.history))
        states = {request.state_position: _chain_state(worker)}
        rounds = []
        while not request.finish:
            drafts = output["new_draft_token_ids"]
            output = worker._execute_row(
                _chain_row(worker, [request.history[-1], *drafts])
            )
            rounds.append((len(drafts), output))
            self.assertEqual(request.state_position, len(request.history) - 1)
            states[request.state_position] = _chain_state(worker)
            if drafts:
                self.assertTrue(
                    torch.all(
                        worker.model.engram_hash.cache[
                            request.cache_slot, request.state_position :
                        ]
                        == 0
                    )
                )
        return worker, states, rounds

    def test_continuous_verification_matches_greedy_and_legacy_tokens_and_state(self):
        for accepted in (0, 2, 5):
            with self.subTest(accepted=accepted):
                greedy, states, _ = self._run_chain("greedy", acceptance=accepted)
                legacy, legacy_states, _ = self._run_chain(
                    "legacy", acceptance=accepted
                )
                continuous, continuous_states, rounds = self._run_chain(
                    "continuous", acceptance=accepted
                )
                for worker, actual in (
                    (legacy, legacy_states),
                    (continuous, continuous_states),
                ):
                    self.assertEqual(
                        worker.requests[1].history, greedy.requests[1].history
                    )
                    for end, values in actual.items():
                        for left, right in zip(values, states[end], strict=True):
                            torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertTrue(all(count == 5 for count, _ in rounds[:2]))
                self.assertLessEqual(
                    len(continuous.model.targets), len(legacy.model.targets)
                )
                if accepted:
                    self.assertLess(
                        len(continuous.model.targets), len(legacy.model.targets)
                    )
                self.assertEqual(len(continuous.model.targets), len(rounds) + 1)

    def test_verify_proposes_before_commit_and_restores_before_the_next_target(self):
        worker = _chain_worker(acceptance=2)
        request = worker.requests[1]
        output = worker._execute_row(_chain_row(worker, request.history))
        for _ in range(2):
            calls = Mock()
            with (
                patch.object(worker, "_target", wraps=worker._target) as target,
                patch.object(request, "append", wraps=request.append) as append,
                patch.object(worker, "_seed_mtp", wraps=worker._seed_mtp) as seed,
                patch.object(worker, "_proposal", wraps=worker._proposal) as proposal,
                patch.object(
                    worker.cache, "commit", wraps=worker.cache.commit
                ) as commit,
                patch.object(
                    worker.cache, "restore", wraps=worker.cache.restore
                ) as restore,
            ):
                for name, method in (
                    ("target", target),
                    ("append", append),
                    ("seed", seed),
                    ("proposal", proposal),
                    ("commit", commit),
                    ("restore", restore),
                ):
                    calls.attach_mock(method, name)
                output = worker._execute_row(
                    _chain_row(
                        worker, [request.history[-1], *output["new_draft_token_ids"]]
                    )
                )
                self.assertEqual(
                    [call[0] for call in calls.mock_calls],
                    ["target", "append", "seed", "proposal", "commit", "restore"],
                )
                token, hidden, start, slot = proposal.call_args.args
                self.assertEqual(
                    (token, start, slot),
                    (
                        output["token_ids"][-1],
                        request.state_position - 1,
                        request.cache_slot,
                    ),
                )
                self.assertEqual(hidden[0, 0, 0], request.state_position)
                self.assertEqual(
                    commit.call_args.kwargs["accepted_end"],
                    restore.call_args.kwargs["start_pos"],
                )
                self.assertFalse(restore.call_args.kwargs["use_prefix_state"])

    def test_batch_requests_keep_their_own_continuous_proposal_boundaries(self):
        worker = _chain_worker(
            prompts=((10, 11, 12, 13), (20, 21, 22, 23, 24, 25)), acceptance=(0, 2)
        )
        outputs = {
            key: worker._execute_row(_chain_row(worker, request.history, key=key))
            for key, request in worker.requests.items()
        }
        for _ in range(2):
            rows = [
                _chain_row(
                    worker,
                    [request.history[-1], *outputs[key]["new_draft_token_ids"]],
                    key=key,
                )
                for key, request in worker.requests.items()
            ]
            before = len(worker.model.targets)
            with patch(
                "oh_my_vllm.models.deepseek_v41.batch.target_decode_batch",
                side_effect=_chain_batch,
            ) as target:
                outputs = worker._execute_decode_group(rows)
                target.assert_called_once()
            self.assertEqual(len(worker.model.targets) - before, 2)
            for key, request in worker.requests.items():
                self.assertEqual(
                    outputs[key]["num_accepted_draft_tokens"], 2 * (key - 1)
                )
                self.assertEqual(len(outputs[key]["new_draft_token_ids"]), 5)
                self.assertEqual(request.state_position, len(request.history) - 1)
                proposal = worker.model.proposals[-3 + key]
                self.assertEqual(proposal["start"], request.state_position - 1)
                self.assertEqual(proposal["token"], request.history[-1])
                expected = _chain_state(worker, key)
                worker.cache.restore(
                    key,
                    request.cache_slot,
                    rows[key - 1],
                    start_pos=request.state_position,
                    use_prefix_state=False,
                )
                for left, right in zip(
                    _chain_state(worker, key), expected, strict=True
                ):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_continuous_state_survives_slot_release_host_transfer_and_restart(self):
        greedy, states, _ = self._run_chain("greedy", acceptance=2)
        for recovery in ("release", "host", "restart"):
            with self.subTest(recovery=recovery):
                worker = _chain_worker(acceptance=2)
                request = worker.requests[1]
                output = worker._execute_row(_chain_row(worker, request.history))
                output = worker._execute_row(
                    _chain_row(
                        worker, [request.history[-1], *output["new_draft_token_ids"]]
                    )
                )
                scheduled = _chain_row(
                    worker, [request.history[-1], *output["new_draft_token_ids"]]
                )
                if recovery == "release":
                    worker._free_slot(request)
                elif recovery == "host":
                    used = (request.state_position + 3) // 4
                    operation = dict(
                        transfer_id=1,
                        snapshot_id=7,
                        kind="store",
                        request_id=1,
                        fa_blocks=scheduled["fa_block_table"][:used],
                        mamba_blocks=scheduled["mamba_block_table"][:used],
                    )
                    self.assertEqual(
                        worker.cache.handle(dict(operations=[operation]))[0]["status"],
                        "complete",
                    )
                    self.assertIsNone(request.cache_slot)
                    worker.cache.fa_pages.clear()
                    worker.cache.mamba_pages.clear()
                    for name in ("fa_block_table", "mamba_block_table"):
                        scheduled[name] = [block + 32 for block in scheduled[name]]
                    operation.update(
                        transfer_id=2,
                        kind="load",
                        fa_blocks=scheduled["fa_block_table"][:used],
                        mamba_blocks=scheduled["mamba_block_table"][:used],
                    )
                    self.assertEqual(
                        worker.cache.handle(dict(operations=[operation]))[0]["status"],
                        "complete",
                    )
                if recovery == "restart":
                    scheduled = _chain_row(worker, request.history, start=0)
                    output = worker.execute(
                        dict(
                            rpc_id=1,
                            scheduled=[scheduled],
                            preempted_request_ids=[1],
                            finished_request_ids=[],
                        )
                    )["outputs"][0]
                else:
                    output = worker._execute_row(scheduled)
                for _ in range(2):
                    self.assertEqual(
                        request.history,
                        greedy.requests[1].history[: len(request.history)],
                    )
                    self.assertEqual(request.state_position, len(request.history) - 1)
                    for left, right in zip(
                        _chain_state(worker),
                        states[request.state_position],
                        strict=True,
                    ):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    self.assertEqual(
                        worker.model.proposals[-1]["start"], request.state_position - 1
                    )
                    scheduled.update(
                        token_ids=[request.history[-1], *output["new_draft_token_ids"]],
                        num_computed_tokens=request.state_position,
                    )
                    output = worker._execute_row(scheduled)

    def test_stop_and_length_use_the_visible_prefix_before_proposing(self):
        reference, _, _ = self._run_chain("greedy", acceptance=5)
        expected = reference.requests[1].history[3:]
        for accepted in (0, 2, 5):
            for finish in ("eos", "stop", "length"):
                with self.subTest(accepted=accepted, finish=finish):
                    kept = accepted + 1
                    options = {"limit": kept + 1} if finish == "length" else {}
                    worker = _chain_worker(acceptance=accepted, **options)
                    request = worker.requests[1]
                    output = worker._execute_row(_chain_row(worker, request.history))
                    if finish == "eos":
                        request.tokenizer.eos = {expected[kept]}
                    elif finish == "stop":
                        request.options["stop"] = ["x" * kept]
                    before = len(worker.model.proposals)
                    output = worker._execute_row(
                        _chain_row(
                            worker,
                            [request.history[-1], *output["new_draft_token_ids"]],
                        )
                    )
                    self.assertEqual(output["token_ids"], expected[1 : kept + 1])
                    self.assertEqual(output["num_accepted_draft_tokens"], accepted)
                    self.assertEqual(
                        output["finish_reason"],
                        "length" if finish == "length" else "stop",
                    )
                    self.assertEqual(output["new_draft_token_ids"], [])
                    self.assertEqual(len(worker.model.proposals), before)
                    self.assertEqual(request.state_position, len(request.history) - 1)

    def test_proposal_budget_skips_mtp_with_at_most_one_output_left(self):
        for limit, expected in ((1, 0), (2, 0), (3, 1)):
            for grouped in (False, True):
                with self.subTest(limit=limit, grouped=grouped):
                    worker = _chain_worker(limit=limit)
                    scheduled = _chain_row(worker, worker.requests[1].history)
                    output = (
                        worker._execute_prefill_group([scheduled], final=True)[1]
                        if grouped
                        else worker._execute_row(scheduled)
                    )
                    self.assertEqual(len(output["new_draft_token_ids"]), expected)
                    self.assertEqual(len(worker.model.proposals), int(expected > 0))
        worker = _chain_worker(prompts=((10,),))
        output = worker._execute_row(_chain_row(worker, [10]))
        self.assertEqual(output["new_draft_token_ids"], [])
        self.assertEqual(worker.model.proposals[0]["start"], 0)

    def test_first_request_failure_before_commit_keeps_pages_unpublished(self):
        for batched in (False, True):
            for failure in ("seed", "proposal", "export", "commit"):
                with self.subTest(batched=batched, failure=failure):
                    worker = _chain_worker(prompts=((10,),))
                    scheduled = _chain_row(worker, [10, 1, 2, 3, 4, 5])
                    owner, name = {
                        "seed": (worker.model, "forward_spec_seed"),
                        "proposal": (worker.model, "forward_spec"),
                        "export": (worker.cache.exports, "window_state"),
                        "commit": (worker.cache, "commit"),
                    }[failure]
                    with (
                        patch.object(owner, name, side_effect=ValueError(failure)),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.batch.target_decode_batch",
                            side_effect=_chain_batch,
                        ),
                        self.assertRaisesRegex(ValueError, failure),
                    ):
                        worker._execute_decode_group(
                            [scheduled]
                        ) if batched else worker._execute_row(scheduled)
                    self.assertEqual(worker.cache.fa_pages, {})
                    self.assertTrue(
                        all(
                            page.endpoint == 0
                            for page in worker.cache.mamba_pages.values()
                        )
                    )
                    self.assertEqual(worker.cache.exports.steps, {})
                    self.assertIsNone(worker.requests[1].cache_slot)
                    self.assertEqual(worker.requests[1].state_position, 0)
                    self.assertEqual(worker.slots, {0, 1})

    def test_later_batch_proposal_failure_discards_the_group_after_an_earlier_commit(
        self,
    ):
        from oh_my_vllm.models.deepseek_v41.launcher import GroupWorker

        worker = _chain_worker(prompts=((10, 11, 12), (20, 21, 22)), acceptance=2)
        worker.batch_decode_enabled, worker.decode_cycle = True, 2
        outputs = {
            key: worker._execute_row(_chain_row(worker, request.history, key=key))
            for key, request in worker.requests.items()
        }
        rows = [
            _chain_row(
                worker,
                [request.history[-1], *outputs[key]["new_draft_token_ids"]],
                key=key,
            )
            for key, request in worker.requests.items()
        ]
        histories = {
            key: list(request.history) for key, request in worker.requests.items()
        }
        failed_slot = worker.requests[2].cache_slot
        proposal = worker._proposal

        def fail_second(token, hidden, start, slot):
            if slot == failed_slot:
                raise RuntimeError("second proposal failed")
            return proposal(token, hidden, start, slot)

        message = dict(
            type="execute",
            rpc_id=3,
            scheduled=rows,
            finished_request_ids=[],
            preempted_request_ids=[],
        )
        bridge = GroupWorker.__new__(GroupWorker)
        bridge.commands = []
        bridge.close = Mock()
        bridge._collect = Mock(
            side_effect=lambda _timeout: {0: worker.execute(message)}
        )
        with (
            patch.object(worker, "_proposal", side_effect=fail_second) as propose,
            patch.object(worker.cache, "commit", wraps=worker.cache.commit) as commit,
            patch(
                "oh_my_vllm.models.deepseek_v41.batch.target_decode_batch",
                side_effect=_chain_batch,
            ),
            self.assertRaisesRegex(RuntimeError, "second proposal failed"),
        ):
            bridge.handle(message)
        self.assertEqual(propose.call_count, 2)
        commit.assert_called_once()
        self.assertEqual(commit.call_args.args[0], 1)
        bridge.close.assert_called_once()
        # Rust must receive no successful Execute result from this discarded group.
        self.assertEqual(worker.cache.fa_pages[1].valid_tokens, 4)
        self.assertEqual(worker.cache.mamba_pages[2].endpoint, 6)
        self.assertEqual(worker.cache.fa_pages[17].valid_tokens, 3)
        self.assertEqual(worker.cache.mamba_pages[17].endpoint, 3)
        self.assertNotIn(18, worker.cache.fa_pages)
        for key, request in worker.requests.items():
            self.assertEqual(request.generated, 4)
            self.assertNotEqual(request.history, histories[key])
            self.assertIsNone(request.cache_slot)
            self.assertEqual(request.state_position, 0)
        self.assertEqual(worker.cache.exports.steps, {})
        self.assertEqual(worker.slots, {0, 1, 2})

    def test_stopped_verification_commits_only_the_visible_accepted_prefix(self):
        worker, scheduled = _verification_worker([1, 2, 3, 4, 5, 9])
        request = worker.requests[1]

        def stop(tokens):
            request.finish = "stop"
            return tokens[:2], "ok"

        request.append = stop
        result = worker._execute_row(scheduled)
        self.assertEqual(result["token_ids"], [1, 2])
        self.assertEqual(result["num_accepted_draft_tokens"], 1)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(request.state_position, 21)
        worker.cache.restore(1, 1, {**scheduled, "prefix_state_slot": 6}, start_pos=21)
        for window in (
            worker.model.layers[0].attn.window_kv_cache,
            worker.model.mtp[0].attn.window_kv_cache,
        ):
            torch.testing.assert_close(
                window[1, :, 0], torch.tensor([20.0, 17, 18, 19]), rtol=0, atol=0
            )

    def test_missing_mtp_export_fails_before_physical_page_publication(self):
        worker, scheduled = _verification_worker([1, 2, 9, 9, 9, 9])
        worker.model.forward_spec_seed = lambda *args, **kwargs: None
        with self.assertRaisesRegex(ValueError, "window state at endpoint"):
            worker._execute_row(scheduled)
        self.assertEqual(worker.cache.fa_pages, {})
        self.assertTrue(
            all(page.endpoint == 0 for page in worker.cache.mamba_pages.values())
        )
        self.assertEqual(worker.cache.exports.steps, {})
        self.assertIsNone(worker.requests[1].cache_slot)
        self.assertEqual(worker.slots, {0, 1})

    def test_prefill_subbatches_keep_different_checkpoint_plans_and_slot_order(self):
        worker = Worker.__new__(Worker)
        worker.model = _PositionModel(batch_size=9)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.prefill_rows = 8
        worker.args = SimpleNamespace(window_size=4)
        worker.slots = {8}
        worker.requests = {
            slot: SimpleNamespace(
                cache_slot=slot,
                history=list(range(1, 33)),
                prompt_len=32,
                state_position=0,
                finish=None,
                thinking_count=0,
            )
            for slot in range(8)
        }
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=32)
        scheduled = [
            {
                "request_id": slot,
                "token_ids": list(range(1, 17)),
                "num_computed_tokens": 0,
                "fa_block_table": list(range(slot * 4 + 1, slot * 4 + 5)),
                "mamba_block_table": [0, 0, 0, 20 + slot],
            }
            for slot in range(8)
        ]
        scheduled[0]["cache_checkpoint"] = (3, 50)
        scheduled[5]["cache_checkpoint"] = (7, 55)
        worker._execute_prefill_group(scheduled)
        for slot, endpoint, source, expected in (
            (0, 3, 50, [1.0, 2, 3, 0]),
            (5, 7, 55, [505.0, 506, 507, 504]),
        ):
            with self.subTest(slot=slot):
                worker.cache.restore(
                    slot,
                    8,
                    {**scheduled[slot], "prefix_state_slot": source},
                    start_pos=endpoint,
                )
                for window in (
                    worker.model.layers[0].attn.window_kv_cache,
                    worker.model.mtp[0].attn.window_kv_cache,
                ):
                    torch.testing.assert_close(
                        window[8, :, 0], torch.tensor(expected), rtol=0, atol=0
                    )

    def test_verification_preserves_lookback_for_both_mtp_checkpoint_and_tail(self):
        worker, scheduled = _verification_worker([1, 2, 9, 9, 9, 9])
        worker._execute_row(scheduled)
        for endpoint, source, expected in (
            (20, 5, [16.0, 17, 18, 19]),
            (22, 6, [20.0, 21, 18, 19]),
        ):
            with self.subTest(endpoint=endpoint):
                worker.cache.restore(
                    1, 1, {**scheduled, "prefix_state_slot": source}, start_pos=endpoint
                )
                actual = worker.model.mtp[0].attn.window_kv_cache[1, :, 0]
                torch.testing.assert_close(
                    actual, torch.tensor(expected), rtol=0, atol=0
                )

    def test_prefill_export_failure_releases_the_new_request_slot(self):
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache

        from tests.test_deepseek_cache import FakeModel

        worker = Worker.__new__(Worker)
        worker.requests = {
            1: SimpleNamespace(cache_slot=None, state_position=0, history=[1])
        }
        worker.slots = {0}
        worker.cache = DeepSeekCache(FakeModel(), block_tokens=4, max_seq_len=8)
        scheduled = {
            "request_id": 1,
            "token_ids": [1],
            "num_computed_tokens": 0,
            "fa_block_table": [1],
            "mamba_block_table": [0],
        }
        with self.assertRaisesRegex(ValueError, "endpoint"):
            worker._execute_prefill_group([scheduled])
        self.assertIsNone(worker.requests[1].cache_slot)
        self.assertEqual(worker.slots, {0})

    def test_long_prefill_exports_mtp_checkpoint_before_discarding_hidden_rows(self):
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache

        worker = Worker.__new__(Worker)
        worker.model = _PositionModel()
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.prefill_rows = 4
        worker.args = SimpleNamespace(window_size=4)
        worker.slots = {1}
        worker.requests = {
            1: SimpleNamespace(
                cache_slot=0,
                history=list(range(1, 33)),
                prompt_len=32,
                state_position=0,
                finish=None,
                thinking_count=0,
            )
        }
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=16)
        scheduled = {
            "request_id": 1,
            "token_ids": list(range(1, 17)),
            "num_computed_tokens": 0,
            "fa_block_table": [1, 2, 3, 4],
            "mamba_block_table": [0, 0, 0, 5],
            "cache_checkpoint": (3, 9),
        }
        worker._execute_prefill_group([scheduled])
        worker.cache.restore(1, 1, {**scheduled, "prefix_state_slot": 9}, start_pos=3)
        for window in (
            worker.model.layers[0].attn.window_kv_cache,
            worker.model.mtp[0].attn.window_kv_cache,
        ):
            torch.testing.assert_close(window[1, :, 0], torch.tensor([1.0, 2, 3, 0]))

    def test_long_verification_seeds_mtp_from_the_accepted_hidden_tail(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.args = SimpleNamespace(window_size=4)
        worker.slots = set()
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        request = SimpleNamespace(
            cache_slot=0,
            history=[10] * 20,
            prompt_len=20,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            append=lambda tokens: (tokens, "ok"),
        )
        worker.requests = {1: request}
        ring = torch.zeros(4)

        class Model:
            def __call__(self, inputs, *, start_pos, **kwargs):
                hidden = torch.arange(start_pos, start_pos + inputs.size(1)).float()
                predictions = torch.tensor([[1, 2, 9, 9, 9, 9]])
                return (
                    None,
                    torch.nn.functional.one_hot(predictions, 16).float(),
                    hidden.view(1, -1, 1),
                )

            def forward_spec_seed(self, hidden, *, start_pos, cache_slots):
                positions = torch.arange(start_pos, start_pos + hidden.size(1))
                ring[positions.remainder(4)] = hidden.flatten()

            def forward_spec(self, input_ids, hidden, *, start_pos, cache_slots):
                self.forward_spec_seed(
                    hidden, start_pos=start_pos, cache_slots=cache_slots
                )
                return None

        worker.model = Model()
        output = worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10] * 20 + [1, 2, 3, 4, 5],
                "num_computed_tokens": 0,
            }
        )
        self.assertEqual(output["num_accepted_draft_tokens"], 2)
        self.assertEqual(request.state_position, 22)
        torch.testing.assert_close(ring, torch.tensor([20.0, 21, 18, 19]))

    def test_worker_rejection_restores_rows_overwritten_by_future_drafts(self):
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache

        from tests.test_deepseek_cache import FakeModel

        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.model = FakeModel()
        request = SimpleNamespace(
            cache_slot=0,
            state_position=2,
            history=[10, 20, 30],
            prompt_len=3,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            append=lambda tokens: (tokens, "ok"),
        )
        worker.requests = {1: request}
        worker.slots = {1}
        worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
        worker.cache = DeepSeekCache(worker.model, block_tokens=4, max_seq_len=8)
        attention = worker.model.layers[0].attn
        attention.window_kv_cache[0, :2] = torch.tensor([[1.0], [2.0]])
        worker.cache.commit(
            1, {"fa_block_table": [1], "mamba_block_table": [3]}, 0, accepted_end=2
        )
        worker.cache.copy_prefix([(1, 7, 2)])

        def target(tokens, start_pos, slot, **kwargs):
            exports = worker.cache.exports
            exports.bind((slot,))
            values = torch.arange(3, 9).float().view(1, 6, 1).expand(1, 6, 3)
            exports.window(attention.window_kv_cache, values, start_pos)
            exports.tail(
                attention.compressor.kv_state,
                attention.compressor.score_state,
                values,
                values + 100,
                start_pos,
            )
            for position in range(2, 8):
                attention.window_kv_cache[slot, position % 4] = position + 1
            logits = torch.nn.functional.one_hot(torch.full((1, 6), 10), 16).float()
            return None, logits, None

        worker._target = target
        result = worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [30, 5, 6, 7, 8, 9],
                "num_computed_tokens": 2,
                "fa_block_table": [7, 2],
                "mamba_block_table": [5, 6],
                "prefix_state_slot": 3,
            }
        )
        self.assertEqual(result["token_ids"], [10])
        self.assertEqual(request.state_position, 3)
        torch.testing.assert_close(
            attention.window_kv_cache[0, :, 0], torch.tensor([1.0, 2, 3, 0])
        )
        torch.testing.assert_close(
            attention.compressor.kv_state[0, 0], torch.full((3,), 3.0)
        )
        self.assertEqual(worker.cache.mamba_pages[3].endpoint, 2)

    def test_execute_rejects_a_resident_position_mismatch(self):
        worker = Worker.__new__(Worker)
        worker.requests = {
            1: SimpleNamespace(cache_slot=0, state_position=4),
        }
        with self.assertRaisesRegex(ValueError, "position"):
            worker.execute(
                {
                    "finished_request_ids": [],
                    "preempted_request_ids": [],
                    "scheduled": [
                        {
                            "request_id": 1,
                            "token_ids": [],
                            "num_computed_tokens": 3,
                        }
                    ],
                    "rpc_id": 1,
                }
            )

    def test_execute_restarts_a_preempted_request_in_the_same_step(self):
        worker, scheduled = _verification_worker([1])
        request = worker.requests[1]
        request.state_position = 4
        attention = worker.model.layers[0].attn
        attention.window_kv_cache[0].fill_(99)
        scheduled["token_ids"] = [10, 10, 10]
        scheduled["mamba_block_table"] = [1, 2, 3, 4, 5, 6, 7]

        result = worker.execute(
            {
                "rpc_id": 7,
                "finished_request_ids": [],
                "preempted_request_ids": [1],
                "scheduled": [scheduled],
            }
        )

        self.assertEqual(result["outputs"][0]["token_ids"], [])
        self.assertEqual(request.state_position, 3)
        self.assertIsNotNone(request.cache_slot)
        torch.testing.assert_close(
            attention.window_kv_cache[request.cache_slot, :, 0],
            torch.tensor([0.0, 1, 2, 0]),
            rtol=0,
            atol=0,
        )

    def test_execute_rejects_finished_request_reuse_before_state_changes(self):
        row = {"request_id": 1, "token_ids": [], "num_computed_tokens": 4}
        for scheduled, preempted in (([row], []), ([], [1])):
            with self.subTest(preempted=preempted):
                worker = Worker.__new__(Worker)
                request = SimpleNamespace(cache_slot=0, state_position=4)
                worker.requests = {1: request}
                worker.slots = {1}
                with self.assertRaisesRegex(ValueError, "overlap"):
                    worker.execute(
                        {
                            "rpc_id": 7,
                            "finished_request_ids": [1],
                            "preempted_request_ids": preempted,
                            "scheduled": scheduled,
                        }
                    )
                self.assertIs(worker.requests[1], request)
                self.assertEqual(request.cache_slot, 0)
                self.assertEqual(worker.slots, {1})

    def test_execute_rejects_unknown_lifecycle_request(self):
        worker = Worker.__new__(Worker)
        worker.requests = {}
        with self.assertRaisesRegex(ValueError, "unknown request"):
            worker.execute(
                {
                    "finished_request_ids": [1],
                    "preempted_request_ids": [],
                    "scheduled": [],
                    "rpc_id": 1,
                }
            )

    def test_restore_failure_releases_the_new_request_slot(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(cache_slot=None, history=[10])
        worker.requests = {1: request}
        worker.slots = {0}
        worker.cache = SimpleNamespace(
            restore=Mock(side_effect=ValueError("bad state")),
            clear_slot=Mock(),
            ensure_mamba_row=Mock(),
            begin_step=Mock(),
        )
        with self.assertRaisesRegex(ValueError, "bad state"):
            worker._execute_row(
                {
                    "request_id": 1,
                    "token_ids": [10],
                    "num_computed_tokens": 0,
                }
            )
        self.assertIsNone(request.cache_slot)
        self.assertEqual(worker.slots, {0})

    def test_commit_failure_releases_the_request_slot(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(
            cache_slot=None,
            state_position=0,
            history=[10],
            prompt_len=1,
            limit=1,
            generated=0,
            finish=None,
            thinking_count=0,
            append=Mock(return_value=([11], "ok")),
        )
        worker.requests = {1: request}
        worker.slots = {0}
        worker.device = torch.device("cpu")
        worker.dspark = False
        worker.model = Mock(
            return_value=(
                None,
                torch.tensor([[1.0]]),
                None,
            )
        )
        worker.cache = SimpleNamespace(
            restore=Mock(),
            ensure_mamba_row=Mock(),
            begin_step=Mock(),
            commit=Mock(side_effect=ValueError("commit failed")),
            clear_slot=Mock(),
        )
        worker.trace = SimpleNamespace(record=Mock())
        with self.assertRaisesRegex(ValueError, "commit failed"):
            worker._execute_row(
                {
                    "request_id": 1,
                    "token_ids": [10],
                    "num_computed_tokens": 0,
                }
            )
        self.assertIsNone(request.cache_slot)
        self.assertEqual(worker.slots, {0})
        worker.cache.clear_slot.assert_called_once_with(0)

    def test_register_rejects_a_duplicate_request_identifier(self):
        worker = Worker.__new__(Worker)
        worker.requests = {1: object()}
        worker.runtime_max_context = 8
        with (
            patch("oh_my_vllm.models.deepseek_v41.inference.Request"),
            self.assertRaisesRegex(ValueError, "already owned"),
        ):
            worker.handle(
                {
                    "type": "register",
                    "request_id": 1,
                    "prompt_token_ids": [3],
                    "max_tokens": 1,
                }
            )

    def test_worker_emits_bonus_token_after_full_draft_acceptance(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(
            history=[10, 11],
            prompt_len=2,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            cache_slot=0,
            append=Mock(return_value=([5, 6, 7], "ok")),
        )
        worker.requests = {1: request}
        worker.dspark = True
        worker._slot = lambda _: 0
        worker._proposal = Mock(return_value=[])
        worker.trace = SimpleNamespace(record=Mock())
        logits = [
            torch.nn.functional.one_hot(torch.tensor([5]), 8).float(),
            torch.nn.functional.one_hot(torch.tensor([6]), 8).float(),
            torch.nn.functional.one_hot(torch.tensor([7]), 8).float(),
        ]
        worker._target = Mock(return_value=(None, torch.stack(logits, dim=1), None))

        result = worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 11, 5, 6],
                "num_computed_tokens": 0,
            }
        )

        self.assertEqual(result["token_ids"], [5, 6, 7])
        self.assertEqual(result["num_accepted_draft_tokens"], 2)
        worker._target.assert_called_once()
        self.assertTrue(worker._target.call_args.kwargs["full_logits"])
        self.assertEqual(worker._target.call_args.kwargs["logit_start"], 1)

    def test_worker_proposes_from_the_last_target_token_in_a_prefill_chunk(self):
        worker = Worker.__new__(Worker)
        hidden = torch.arange(6, dtype=torch.float32).view(1, 2, 3)
        request = SimpleNamespace(
            history=[10, 11],
            prompt_len=2,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            cache_slot=0,
            append=Mock(return_value=([5], "ok")),
        )
        worker.requests = {1: request}
        worker.dspark = True
        worker._slot = lambda _: 0
        worker._proposal = Mock(return_value=[6, 7])
        worker.trace = SimpleNamespace(record=Mock())
        worker._target = Mock(
            return_value=(
                None,
                torch.nn.functional.one_hot(torch.tensor([[5]]), 8).float(),
                hidden,
            )
        )

        worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 11],
                "num_computed_tokens": 0,
            }
        )

        worker._proposal.assert_called_once()
        token, proposal_hidden, proposal_start, slot = worker._proposal.call_args.args
        self.assertEqual(token, 5)
        torch.testing.assert_close(proposal_hidden, hidden[:, -1:])
        self.assertEqual(proposal_start, 1)
        self.assertEqual(slot, 0)

    def test_verification_seeds_mtp_state_for_accepted_target_positions(self):
        worker = Worker.__new__(Worker)
        request = SimpleNamespace(
            history=[10, 5],
            prompt_len=1,
            limit=8,
            generated=0,
            finish=None,
            thinking_count=0,
            cache_slot=0,
            append=Mock(return_value=([6, 7], "ok")),
        )
        hidden = torch.arange(9, dtype=torch.float32).view(1, 3, 3)
        worker.requests = {1: request}
        worker.dspark = True
        worker.device = torch.device("cpu")
        worker._slot = lambda _: 0
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        worker._proposal = Mock(return_value=[])
        worker.trace = SimpleNamespace(record=Mock())
        worker._target = Mock(
            return_value=(
                None,
                torch.stack(
                    [
                        torch.nn.functional.one_hot(torch.tensor([6]), 8).float(),
                        torch.nn.functional.one_hot(torch.tensor([6]), 8).float(),
                    ],
                    dim=1,
                ),
                hidden,
            )
        )

        worker._execute_row(
            {
                "request_id": 1,
                "token_ids": [10, 5, 6],
                "num_computed_tokens": 0,
            }
        )

        self.assertEqual(worker.model.forward_spec_seed.call_count, 1)
        self.assertEqual(
            [
                call.kwargs["start_pos"]
                for call in worker.model.forward_spec_seed.call_args_list
            ],
            [0],
        )
        self.assertEqual(
            worker.model.forward_spec_seed.call_args.args[0].shape[1],
            3,
        )

    def test_target_keeps_long_prefill_in_one_native_call(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        worker.model = Mock(
            side_effect=[
                ("last", "last logits", "last hidden"),
            ]
        )

        result = worker._target([1] * 385, 7, 3)

        self.assertEqual(result, ("last", "last logits", "last hidden"))
        self.assertEqual(worker.model.call_count, 1)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7],
        )

    def test_target_default_prefill_limit_bounds_model_chunks(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.model = Mock(
            side_effect=[
                ("first", None, None),
                ("last", None, None),
            ]
        )

        worker._target([1] * (DEFAULT_PREFILL_ROWS + 1), 7, 3, compute_logits=False)

        self.assertEqual(worker.model.call_count, 2)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7, 7 + DEFAULT_PREFILL_ROWS],
        )

    def test_prefill_row_limit_keeps_explicit_operator_bound(self):
        self.assertEqual(prefill_row_limit(None), DEFAULT_PREFILL_ROWS)
        self.assertEqual(prefill_row_limit(str(MAX_PREFILL_ROWS)), MAX_PREFILL_ROWS)
        with self.assertRaisesRegex(ValueError, "prefill row limit"):
            prefill_row_limit(str(MAX_PREFILL_ROWS + 1))

    def test_target_releases_completed_prefill_hidden_chunks(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        worker.args = SimpleNamespace(window_size=2)
        references = []

        def model(_inputs, **_):
            hidden = torch.ones(1, 1, 2)
            if references:
                gc.collect()
                self.assertIsNone(references[0]())
            references.append(weakref.ref(hidden))
            return torch.empty(1, dtype=torch.long), None, hidden

        worker.model = Mock(side_effect=model)
        result = worker._target(
            [1] * (MAX_PREFILL_ROWS + 1), 7, 3, compute_logits=False
        )

        self.assertEqual(tuple(result[2].shape), (1, 2, 2))
        self.assertEqual(worker.model.call_count, 2)

    def test_target_uses_the_bounded_prefill_row_limit(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = 128
        worker.model = Mock(
            side_effect=[
                ("first", None, None),
                ("second", None, None),
                ("last", None, None),
            ]
        )

        worker._target([1] * 257, 7, 3, compute_logits=False)

        self.assertEqual(worker.model.call_count, 3)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7, 135, 263],
        )

    def test_target_graph_requires_vocab_logits_and_one_token(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        worker.decode_graph_enabled = True
        worker.model = Mock(return_value=(None, None, None))
        with patch.object(worker, "_decode_graph_step", return_value=None) as graph:
            worker._target([1], 5, 3)
            self.assertEqual(graph.call_count, 1)
            worker._target([1], 0, 3)
            self.assertEqual(graph.call_count, 1)
            worker._target([1], 5, 3, compute_logits=False)
            self.assertEqual(graph.call_count, 1)
            worker._target([1, 2], 5, 3)
            self.assertEqual(graph.call_count, 1)
            worker._target_batch_group([[1]], 5, torch.tensor([3]))
            self.assertEqual(graph.call_count, 2)
            worker._target_batch_group(
                [[1]], 5, torch.tensor([3]), compute_logits=False
            )
            self.assertEqual(graph.call_count, 2)

    def test_target_batch_keeps_total_rows_in_one_native_call(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS

        def model(inputs, **_):
            self.assertLessEqual(inputs.size(0) * inputs.size(1), MAX_PREFILL_ROWS)
            return None, None, None

        worker.model = Mock(side_effect=model)
        token_batches = [[index] * 200 for index in range(4)]

        worker._target_batch(
            token_batches,
            start_pos=7,
            cache_slots=torch.arange(4),
        )

        self.assertEqual(worker.model.call_count, 1)
        self.assertEqual(
            [call.kwargs["start_pos"] for call in worker.model.call_args_list],
            [7],
        )

    def test_mtp_seed_keeps_total_rows_at_native_limit(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        hidden = torch.zeros(4, 200, 8)
        cache_slots = torch.arange(4)

        self.assertTrue(worker._seed_mtp_batch(hidden, 7, cache_slots))

        self.assertEqual(worker.model.forward_spec_seed.call_count, 3)
        self.assertEqual(
            [
                call.args[0].shape[1]
                for call in worker.model.forward_spec_seed.call_args_list
            ],
            [96, 96, 8],
        )
        self.assertEqual(
            [
                call.kwargs["start_pos"]
                for call in worker.model.forward_spec_seed.call_args_list
            ],
            [7, 103, 199],
        )

    def test_target_batch_uses_four_request_prefill_subbatches(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = DEFAULT_PREFILL_ROWS

        def model(inputs, **_):
            self.assertLessEqual(inputs.size(0), 4)
            self.assertLessEqual(inputs.size(0) * inputs.size(1), MAX_PREFILL_ROWS)
            return None, None, None

        worker.model = Mock(side_effect=model)
        token_batches = [[index] * 100 for index in range(8)]

        worker._target_batch(
            token_batches,
            start_pos=7,
            cache_slots=torch.arange(8),
        )

        self.assertEqual(worker.model.call_count, 2)

    def test_target_batch_uses_wide_prefill_subbatches(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.prefill_rows = MAX_PREFILL_ROWS
        calls = []

        def model(inputs, **kwargs):
            self.assertLessEqual(inputs.size(0), 16)
            self.assertLessEqual(inputs.size(0) * inputs.size(1), MAX_PREFILL_ROWS)
            calls.append((inputs[:, 0].tolist(), kwargs["cache_slots_cpu"]))
            return None, None, None

        worker.model = Mock(side_effect=model)
        token_batches = [[index] * 100 for index in range(32)]

        worker._target_batch(
            token_batches,
            start_pos=7,
            cache_slots=torch.arange(32),
        )

        self.assertEqual(worker.model.call_count, 2)
        self.assertEqual(
            calls,
            [
                (list(range(16)), list(range(16))),
                (list(range(16, 32)), list(range(16, 32))),
            ],
        )

    def test_mtp_seed_keeps_only_the_target_window(self):
        worker = Worker.__new__(Worker)
        worker.device = torch.device("cpu")
        worker.dspark = True
        worker.args = SimpleNamespace(window_size=8)
        worker.model = SimpleNamespace(forward_spec_seed=Mock())
        hidden = torch.zeros(1, 200, 8)

        self.assertTrue(
            worker._seed_mtp_batch(
                hidden,
                7,
                torch.tensor([0]),
                source_length=200,
            )
        )

        worker.model.forward_spec_seed.assert_called_once()
        call = worker.model.forward_spec_seed.call_args
        self.assertEqual(tuple(call.args[0].shape), (1, 8, 8))
        self.assertEqual(call.kwargs["start_pos"], 199)

    def test_execute_groups_complete_initial_prefills(self):
        worker = Worker.__new__(Worker)
        worker.requests = {
            1: SimpleNamespace(cache_slot=None, history=[10, 11, 12], prompt_len=3),
            2: SimpleNamespace(cache_slot=None, history=[20, 21, 22], prompt_len=3),
        }
        worker._execute_prefill_group = Mock(
            return_value={
                1: {"request_id": 1},
                2: {"request_id": 2},
            }
        )
        worker._execute_row = Mock()

        result = worker.execute(
            {
                "finished_request_ids": [],
                "preempted_request_ids": [],
                "scheduled": [
                    {
                        "request_id": 1,
                        "token_ids": [10, 11, 12],
                        "num_computed_tokens": 0,
                    },
                    {
                        "request_id": 2,
                        "token_ids": [20, 21, 22],
                        "num_computed_tokens": 0,
                    },
                ],
                "rpc_id": 1,
            }
        )

        worker._execute_prefill_group.assert_called_once()
        self.assertEqual(
            worker._execute_prefill_group.call_args.kwargs,
            {"final": True},
        )
        worker._execute_row.assert_not_called()
        self.assertEqual(result["outputs"], [{"request_id": 1}, {"request_id": 2}])

    def test_rejection_keeps_only_the_matching_prefix_and_target_correction(self):
        result = verify_greedy_drafts(
            drafts=[11, 12, 13],
            target_predictions=[11, 99, 14, 15],
        )

        self.assertEqual(result.accepted_draft_tokens, 1)
        self.assertEqual(result.token_ids, (11, 99))
        self.assertTrue(result.rejected)

    def test_full_acceptance_returns_the_target_bonus_token(self):
        result = verify_greedy_drafts(
            drafts=[11, 12, 13],
            target_predictions=[11, 12, 13, 17],
        )

        self.assertEqual(result.accepted_draft_tokens, 3)
        self.assertEqual(result.token_ids, (11, 12, 13, 17))
        self.assertFalse(result.rejected)

    def test_empty_draft_is_ordinary_target_decode(self):
        result = verify_greedy_drafts(
            drafts=[],
            target_predictions=[23],
        )

        self.assertEqual(result.accepted_draft_tokens, 0)
        self.assertEqual(result.token_ids, (23,))
        self.assertFalse(result.rejected)


if __name__ == "__main__":
    unittest.main()
