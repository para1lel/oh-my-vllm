import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import model_dspark

from tests.deepseek_speculation_failures import (
    DeepSeekSpeculationFailures as DeepSeekSpeculationFailures,
)
from tests.deepseek_speculation_fixtures import (
    DeepSeekSpeculationFixture,
    _chain_batch,
    _chain_row,
    _chain_state,
    _chain_worker,
    _worker,
)
from tests.deepseek_speculation_fixtures import _ChainModel as _ChainModel
from tests.deepseek_speculation_fixtures import _PositionModel as _PositionModel
from tests.deepseek_speculation_fixtures import (
    _verification_worker as _verification_worker,
)
from tests.deepseek_speculation_lifecycle import (
    DeepSeekSpeculationLifecycle as DeepSeekSpeculationLifecycle,
)
from tests.deepseek_speculation_target_batch import (
    DeepSeekSpeculationTargetBatch as DeepSeekSpeculationTargetBatch,
)


class DeepSeekSpeculation(DeepSeekSpeculationFixture, unittest.TestCase):
    def test_continuous_verification_matches_greedy_tokens_and_state(self):
        for accepted in (0, 2, 5):
            with self.subTest(accepted=accepted):
                greedy, states, _ = self._run_chain("greedy", acceptance=accepted)
                continuous, continuous_states, rounds = self._run_chain(
                    "continuous", acceptance=accepted
                )
                self.assertEqual(
                    continuous.requests[1].history, greedy.requests[1].history
                )
                for end, values in continuous_states.items():
                    for left, right in zip(values, states[end], strict=True):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                self.assertTrue(all(count == 5 for count, _ in rounds[:2]))
                self.assertEqual(len(continuous.model.targets), len(rounds) + 1)

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

    def test_proposal_requires_the_complete_five_row_position_range(self):
        from oh_my_vllm.models.deepseek_v41 import model

        for context in (32, 33, 34):
            with self.subTest(context=context):
                attention = object.__new__(model.DSparkAttention)
                torch.nn.Module.__init__(attention)
                attention.compress_ratio = 0
                attention.window_size = 4
                attention.rope_head_dim = attention.head_dim = 4
                attention.n_local_heads = attention.n_local_groups = 1
                attention.o_lora_rank = 4
                attention.softmax_scale = 0.5
                attention.freqs_cis = torch.ones(context, 2, dtype=torch.complex64)
                attention.window_kv_cache = torch.zeros(1, 4, 4)
                attention.attn_sink = torch.zeros(1)
                for name in ("wq_a", "q_norm", "wq_b", "kv_norm", "wkv", "wo_b"):
                    setattr(attention, name, torch.nn.Identity())
                attention.wo_a = torch.nn.Linear(4, 4, bias=False)
                attention._store_main_kv = Mock()
                worker = _worker()
                worker.dspark = True
                worker.runtime_max_context = context

                def proposal(token, hidden, start, slot, *, attention=attention):
                    attention(
                        torch.zeros(1, 5, 4),
                        start,
                        torch.zeros(1, 1, 4),
                        torch.tensor([slot]),
                    )
                    return [1, 2, 3, 4, 5]

                worker._proposal = Mock(side_effect=proposal)
                request = SimpleNamespace(limit=16, generated=14, finish=None)
                with (
                    patch.object(model_dspark, "act_quant"),
                    patch.object(
                        model_dspark,
                        "sparse_attn",
                        side_effect=lambda query, *_args, **_kwargs: torch.zeros_like(
                            query
                        ),
                    ),
                ):
                    result = worker._next_proposals(
                        request, [1], torch.zeros(1, 1, 4), 29, 0
                    )
                self.assertEqual(result, [1] if context == 34 else [])
                self.assertEqual(worker._proposal.call_count, int(context == 34))

    def test_no_draft_decode_does_not_enter_the_prefill_group(self):
        for batched in (False, True):
            with self.subTest(batched=batched):
                worker = _worker()
                worker.batch_decode_enabled = batched
                worker.decode_cycle = 2
                worker.requests = {
                    key: SimpleNamespace(
                        cache_slot=key - 1,
                        state_position=31,
                        prompt_len=16,
                        history=[key] * 32,
                    )
                    for key in (1, 2)
                }
                rows = [
                    dict(request_id=key, token_ids=[key], num_computed_tokens=31)
                    for key in (1, 2)
                ]
                worker._execute_prefill_group = Mock(
                    side_effect=AssertionError("decode was classified as prefill")
                )
                worker._execute_row = Mock(side_effect=lambda row: row)
                worker._execute_decode_group = Mock(
                    return_value={row["request_id"]: row for row in rows}
                )
                result = worker.execute(
                    dict(
                        rpc_id=1,
                        scheduled=rows,
                        finished_request_ids=[],
                        preempted_request_ids=[],
                    )
                )
                self.assertEqual(result["outputs"], rows)
                self.assertEqual(worker._execute_decode_group.call_count, int(batched))
                self.assertEqual(worker._execute_row.call_count, 0 if batched else 2)
                worker._execute_prefill_group.assert_not_called()


if __name__ == "__main__":
    unittest.main()
