import copy
import datetime
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from oh_my_vllm.models.deepseek_v41.dep4 import Dep4Executor
from oh_my_vllm.models.deepseek_v41.dispatch import destination_pairs
from oh_my_vllm.models.deepseek_v41.rank_plan import local_execute, merge_replies


class _Expert:
    def __init__(self, expert_id):
        self.factor = (expert_id + 1) / 8

    def __call__(self, rows, weights):
        return (rows.float() * weights * self.factor).to(torch.bfloat16)


class _MoE(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.dim = 4
        self.n_routed_experts = 8
        self.n_activated_experts = 6
        self.experts_start_idx = 2 * rank
        self.experts_end_idx = 2 * rank + 2
        self.experts = [_Expert(index) for index in range(8)]

    def _grouped_routed(self, *args):
        return None

    def forward(self, rows):
        indices = ((torch.arange(6)[None, :] + rows[:, :1].long()) % 8).to(torch.int32)
        weights = torch.full(indices.shape, 0.125)
        return self.dep4_exchange(self, rows, (weights, indices)).to(rows.dtype)


class _Model(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(2)])
        self.mtp = torch.nn.ModuleList([torch.nn.Module()])
        for layer in (*self.layers, *self.mtp):
            layer.ffn = _MoE(rank)

    def forward(self, rows):
        for layer in self.layers:
            rows = layer.ffn(rows)
        return rows

    def forward_spec(self, rows, main_hidden=None, start_pos=1):
        for layer in self.mtp:
            rows = layer.ffn(rows)
        return rows


def _reference(rows, layers):
    for _ in range(layers):
        ids = (torch.arange(6)[None, :] + rows[:, :1].long()) % 8
        total = torch.zeros_like(rows, dtype=torch.float32)
        for rank in range(4):
            local = torch.zeros_like(total)
            for expert in range(2 * rank, 2 * rank + 2):
                tokens = (ids == expert).any(dim=1).nonzero().flatten()
                local[tokens] += (
                    (rows[tokens].float() * (expert + 1) / 64)
                    .to(torch.bfloat16)
                    .float()
                )
            total += local
        rows = total.to(torch.bfloat16)
    return rows


def _peer(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model = _Model(rank)
        executor = Dep4Executor(model)
        rows = (torch.arange((rank + 1) * 4).view(-1, 4) + rank).to(torch.bfloat16)

        def run():
            if rank in (0, 2):
                actual = model(rows)
                torch.testing.assert_close(actual, _reference(rows, 2), rtol=0, atol=0)
            if rank == 0:
                actual = model(rows + 2)
                torch.testing.assert_close(
                    actual, _reference(rows + 2, 2), rtol=0, atol=0
                )
            if rank in (0, 1):
                actual = model.forward_spec(rows, start_pos=1)
                torch.testing.assert_close(actual, _reference(rows, 1), rtol=0, atol=0)

        executor.execute(run)
        assert executor.call == 2
        assert executor.target_active_ranks == 3
        # A fully empty step still reaches group completion and can be followed by work.
        executor.execute(lambda: None)
        executor.execute(lambda: model(rows) if rank == 3 else None)
        assert executor.call == 3
    finally:
        dist.destroy_process_group()


class DeepSeekDep4(unittest.TestCase):
    def test_four_rank_exchange_with_empty_rank_and_divergent_target_proposal_calls(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_peer, args=(str(Path(directory) / "store"),), nprocs=4, join=True)

    def test_dense_owner_keeps_mp4_partial_rounding_and_state_loading(self):
        from oh_my_vllm.models.deepseek_v41.model import RowParallelLinear

        with (
            patch("oh_my_vllm.models.deepseek_v41.model.world_size", 1),
            patch("oh_my_vllm.models.deepseek_v41.model.expert_world_size", 4),
            patch("oh_my_vllm.models.deepseek_v41.model.default_dtype", torch.bfloat16),
        ):
            layer = RowParallelLinear(128, 1)
            with torch.no_grad():
                layer.weight.zero_()
                layer.weight[0, [0, 32]] = 1
                layer.weight[0, [1, 33, 64]] = 1 / 256
            inputs = torch.ones((1, 128), dtype=torch.bfloat16)
            self.assertEqual(float(layer(inputs).item()), 2.0)
            layer.load_state_dict({"weight": torch.zeros_like(layer.weight)})
            self.assertEqual(float(layer(inputs).item()), 0.0)

    def test_destination_rows_are_distinct_pairs_instead_of_assignments(self):
        ids = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 4, 5, 6, 7]])
        tokens, destinations, counts = destination_pairs(ids, 2, 4)
        self.assertEqual(counts.tolist(), [2, 1, 2, 1])
        self.assertEqual(
            list(zip(tokens.tolist(), destinations.tolist(), strict=True)),
            [(0, 0), (1, 0), (0, 1), (0, 2), (1, 2), (1, 3)],
        )
        self.assertEqual(len(tokens), 6)
        self.assertEqual(ids.numel(), 12)

    def test_rust_rank_plan_preserves_output_order_and_rejects_missing_ack(self):
        plan = [
            dict(
                rank=rank,
                request_ids=[],
                finished_request_ids=[],
                preempted_request_ids=[],
            )
            for rank in range(4)
        ]
        plan[2]["request_ids"] = [11]
        message = dict(
            type="execute",
            rpc_id=4,
            step_id=3,
            rank_plan=plan,
            scheduled=[dict(request_id=11, token_ids=[2])],
            finished_request_ids=[],
            preempted_request_ids=[],
        )
        replies = {}
        for rank in range(4):
            local, schedule = local_execute(message, rank)
            replies[rank] = dict(
                type="execute_result",
                rpc_id=4,
                outputs=[
                    dict(request_id=row["request_id"]) for row in local["scheduled"]
                ],
                rank_completion=dict(rank=rank, step_id=3, schedule=schedule),
            )
        self.assertEqual(
            merge_replies(message, replies)["outputs"], [dict(request_id=11)]
        )
        missing = copy.deepcopy(replies)
        missing[0].pop("rank_completion")
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            merge_replies(message, missing)
        duplicate = copy.deepcopy(message)
        duplicate["rank_plan"][0]["request_ids"] = [11]
        with self.assertRaisesRegex(ValueError, "exact request partition"):
            local_execute(duplicate, 0)


if __name__ == "__main__":
    unittest.main()
