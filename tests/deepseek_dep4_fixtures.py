import datetime
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from oh_my_vllm.models.deepseek_v41.dep4 import Dep4Executor

from tests.deepseek_dep4_driver import _driver_failure


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

    def forward(self, rows, *, check_status=True):
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


def _fixed_input(rank, mode, device="cpu"):
    count = 0 if mode == "empty" else ((0, 4, 1, 3)[rank] if mode == "unequal" else 4)
    rows = (torch.arange(count * 4, device=device).view(count, 4) + rank).to(
        torch.bfloat16
    )
    ids = torch.arange(6, device=device)[None, :].expand(count, 6)
    if mode != "concentrated":
        ids = (ids * 4 + rank) % 24
    weights = (torch.arange(6, device=device).float() + 1).expand(count, 6) / 32
    return rows, ids, weights


class DeepSeekDep4Fixture:
    def _check_driver_failure(self, failure):
        import json
        import time
        from types import SimpleNamespace

        from development.model.check_deepseek_dep4 import RankGroupFailure, supervise

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            baseline.mkdir()
            if failure == "corrupt":
                (baseline / "endpoints.safetensors").write_bytes(b"invalid safetensors")
                (baseline / "chains.json").write_text("{}")
            args = SimpleNamespace(
                output=root / "result",
                mode="legacy",
                timeout_seconds=12,
                baseline=baseline if failure in ("missing", "corrupt") else None,
                failure=failure,
            )
            previous_children = {child.pid for child in mp.active_children()}
            started = time.monotonic()
            with self.assertRaises(RankGroupFailure):
                supervise(args, operation=_driver_failure)
            self.assertLess(time.monotonic() - started, 12)
            self.assertEqual(
                {child.pid for child in mp.active_children()}, previous_children
            )
            manifest = json.loads((args.output / "run.json").read_text())
            self.assertEqual(manifest["outcome"], "failed")
            self.assertEqual(manifest["remaining_child_pids"], [])
            self.assertTrue(
                all(code is not None for code in manifest["child_exitcodes"])
            )
            self.assertLess(manifest["failure_to_group_exit_seconds"], 3)
            print(
                json.dumps(
                    dict(
                        failure=failure,
                        shutdown_seconds=manifest["shutdown_seconds"],
                        failure_to_group_exit_seconds=manifest[
                            "failure_to_group_exit_seconds"
                        ],
                        remaining_child_pids=manifest["remaining_child_pids"],
                        child_exitcodes=manifest["child_exitcodes"],
                    )
                ),
                flush=True,
            )
            records = [
                json.loads((args.output / f"proof-rank{rank}.json").read_text())
                for rank in range(4)
            ]
            for record in records:
                self.assertEqual(record["run_id"], manifest["run_id"])
                self.assertEqual(record["outcome"], "failed")
                self.assertEqual(record["group_error"]["type"], "RankGroupFailure")
            if failure in ("legacy", "endpoint", "device", "initialize"):
                self.assertIn("injected", records[2]["error"]["message"])
            else:
                kind = (
                    "FileNotFoundError" if failure == "missing" else "SafetensorError"
                )
                self.assertTrue(
                    any(
                        record["error"] is not None and record["error"]["type"] == kind
                        for record in records
                    )
                )
