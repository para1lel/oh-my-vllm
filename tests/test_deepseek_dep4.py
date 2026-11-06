import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from oh_my_vllm.models.deepseek_v41.dep4 import Dep4Executor
from oh_my_vllm.models.deepseek_v41.dispatch import (
    DispatchStatus,
    ExpertExchange,
    FixedDispatchBuffers,
    FixedDispatchCapacity,
    PaddedExpertBuffers,
    destination_pairs,
)
from oh_my_vllm.models.deepseek_v41.rank_plan import (
    handle_rank,
    local_execute,
    merge_replies,
)

from tests.deepseek_dep4_cuda import (
    _check_cuda_memory_budget as _check_cuda_memory_budget,
)
from tests.deepseek_dep4_cuda import _cuda_fixed_peer
from tests.deepseek_dep4_cuda import _cuda_moe as _cuda_moe
from tests.deepseek_dep4_cuda import _cuda_multiwave_experts as _cuda_multiwave_experts
from tests.deepseek_dep4_driver import _driver_failure as _driver_failure
from tests.deepseek_dep4_exchange import _multiwave_peer, _shadow_peer
from tests.deepseek_dep4_failure import _FaultExchange as _FaultExchange
from tests.deepseek_dep4_failure import _round_failure_peer
from tests.deepseek_dep4_fixed import _check_fixed_failures as _check_fixed_failures
from tests.deepseek_dep4_fixed import _check_fixed_routes as _check_fixed_routes
from tests.deepseek_dep4_fixed import _fixed_peer
from tests.deepseek_dep4_fixtures import (
    DeepSeekDep4Fixture,
    _fixed_input,
    _Model,
    _peer,
)
from tests.deepseek_dep4_fixtures import _Expert as _Expert
from tests.deepseek_dep4_fixtures import _MoE as _MoE
from tests.deepseek_dep4_fixtures import _reference as _reference


class DeepSeekDep4(DeepSeekDep4Fixture, unittest.TestCase):
    def test_driver_owner_selection_keeps_default_and_allows_four_owners(self):
        from development.model.check_deepseek_dep4 import (
            argument_parser,
            request_owners,
        )

        parser = argument_parser()
        values = [
            "--model",
            "/unused",
            "--output",
            "/unused-output",
            "--mode",
            "legacy",
        ]
        self.assertEqual(parser.parse_args(values).lengths, [600, 300, 17])
        self.assertEqual(request_owners([600, 300, 17]), [0, 1, 2])
        lengths = parser.parse_args(
            [*values, "--lengths", "600", "300", "17", "33"]
        ).lengths
        self.assertEqual(request_owners(lengths), [0, 1, 2, 3])
        self.assertEqual(request_owners([17]), [0])
        for invalid in ([], [1, 2, 3, 4, 5]):
            with (
                self.subTest(lengths=invalid),
                self.assertRaisesRegex(ValueError, "one and four"),
            ):
                request_owners(invalid)

    def test_driver_rejects_invalid_owners_before_starting_ranks(self):
        from development.model.check_deepseek_dep4 import main

        values = [
            "check",
            "--model",
            "/unused",
            "--output",
            "/unused-output",
            "--mode",
            "legacy",
            "--lengths",
        ]
        for lengths in (("1", "2", "3", "4", "5"), ("0",), ("-1",)):
            with (
                self.subTest(lengths=lengths),
                patch("sys.argv", [*values, *lengths]),
                patch("development.model.check_deepseek_dep4.supervise") as start,
                self.assertRaises(SystemExit) as error,
            ):
                main()
            self.assertEqual(error.exception.code, 2)
            start.assert_not_called()

    def test_driver_terminates_peers_after_legacy_or_endpoint_exception(self):
        for failure in ("legacy", "endpoint"):
            with self.subTest(failure=failure):
                self._check_driver_failure(failure)

    def test_driver_records_baseline_device_and_initialization_failures(self):
        for failure in ("missing", "corrupt", "device", "initialize"):
            with self.subTest(failure=failure):
                self._check_driver_failure(failure)

    def test_driver_rejects_old_successful_output_directory(self):
        from types import SimpleNamespace

        from development.model.check_deepseek_dep4 import supervise

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            old = output / "proof-rank0.json"
            old.write_text('{"run_id":"old","outcome":"completed"}')
            with self.assertRaisesRegex(FileExistsError, "new or empty"):
                supervise(SimpleNamespace(output=output))
            self.assertEqual(old.read_text(), '{"run_id":"old","outcome":"completed"}')
            self.assertFalse((output / "run.json").exists())

    def test_path_proof_survives_failure_and_rejects_silent_fallback(self):
        import json

        from development.model.check_deepseek_dep4 import ExchangeEvidence

        for failure in (True, False):
            with (
                self.subTest(failure=failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                proof = ExchangeEvidence(directory, 0, mode="fixed")
                exchange = ExpertExchange()
                exchange.prepare_round(600, fixed=failure)

                def operation(_exchange, _moe, rows, _routing, *, fail=failure):
                    if fail:
                        raise ValueError("exchange failed")
                    return rows

                exception = ValueError if failure else AssertionError
                with (
                    patch.object(ExpertExchange, "__call__", operation),
                    self.assertRaises(exception),
                    proof.observe(),
                ):
                    exchange(None, torch.empty((0, 4)), None)
                record = json.loads(Path(directory, "proof-rank0.json").read_text())
                self.assertEqual(record["outcome"], "failed")
                self.assertEqual(record["counts"]["maximum_source_rows"], 600)
                self.assertEqual(record["counts"]["fixed"], int(failure))
                self.assertEqual(record["counts"]["legacy"], int(not failure))
                self.assertEqual(record["counts"]["fallback"], int(not failure))
                self.assertEqual(record["counts"]["failed"], int(failure))
                self.assertEqual(record["error"]["type"], exception.__name__)

    def test_shadow_rejects_all_ranks_and_keeps_comparison_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _shadow_peer,
                args=(str(Path(directory) / "store"),),
                nprocs=4,
                join=True,
            )

    def test_multiwave_exchange_matches_legacy_with_long_and_empty_owners(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _multiwave_peer,
                args=(str(Path(directory) / "store"),),
                nprocs=4,
                join=True,
            )

    @unittest.skipUnless(
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 4
        and os.environ.get("OH_MY_VLLM_GPU_COUNT") == "4",
        "four leased CUDA devices are required",
    )
    def test_cuda_fixed_experts_keep_bits_within_memory_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _cuda_fixed_peer,
                args=(str(Path(directory) / "store"),),
                nprocs=4,
                join=True,
            )

    def test_status_initialization_failure_releases_execute_ownership(self):
        executor = Dep4Executor(_Model(0), check_round_status=True)
        with (
            patch.object(
                executor.status,
                "begin_execute",
                side_effect=RuntimeError("reset failed"),
            ),
            self.assertRaisesRegex(RuntimeError, "reset failed"),
        ):
            executor.execute(lambda: self.fail("operation must not run"))
        self.assertFalse(executor.active)
        self.assertFalse(torch.is_inference_mode_enabled())

    def test_mid_round_status_rejects_before_cache_commit_or_rank_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _round_failure_peer,
                args=(str(Path(directory) / "store"),),
                nprocs=4,
                join=True,
            )

    def test_one_rank_sparse_failure_rejects_all_ranks_before_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _round_failure_peer,
                args=(str(Path(directory) / "store"), True),
                nprocs=4,
                join=True,
            )

    def test_fixed_capacity_exchange_keeps_prefixes_order_and_sticky_status(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(
                _fixed_peer, args=(str(Path(directory) / "store"),), nprocs=4, join=True
            )

    def test_fixed_capacity_budget_and_bad_device_counts_fail_explicitly(self):
        capacity = FixedDispatchCapacity(4, 4, 4, 6, 24)
        self.assertEqual(capacity.receive_rows, 16)
        self.assertEqual(capacity.routed_rows, 96)
        status = DispatchStatus("cpu")
        with self.assertRaisesRegex(ValueError, "buffer budget"):
            FixedDispatchBuffers(
                capacity,
                status,
                device="cpu",
                buffer_budget_bytes=capacity.tensor_bytes - 1,
            )
        for dimensions in ((257, 4, 4, 6, 24), (4, 3, 4, 6, 24), (4, 4, 4, 25, 24)):
            with self.assertRaises(ValueError):
                FixedDispatchCapacity(*dimensions)
        workspace = FixedDispatchBuffers(
            capacity, status, device="cpu", buffer_budget_bytes=capacity.tensor_bytes
        )
        allocated = sum(
            value.numel() * value.element_size() for value in workspace.storage.values()
        )
        allocated += sum(
            value.numel() * value.element_size()
            for value in (workspace.slots, workspace.ranks)
        )
        self.assertEqual(allocated, capacity.tensor_bytes)

        def exchange(output, value, **_options):
            if output.ndim == 1:
                output.copy_(torch.tensor([-1, 5, 0, 4], dtype=torch.int32))
            else:
                output.zero_()

        rows, ids, weights = _fixed_input(0, "concentrated")
        with self.assertRaisesRegex(RuntimeError, "requires a dispatch"):
            workspace.combine(workspace.partial)
        with (
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(dist, "all_to_all_single", side_effect=exchange),
        ):
            status.begin_execute()
            status.tensor.fill_(4)
            workspace.dispatch(rows, ids, weights)
            self.assertEqual(status.tensor.item(), 5)
            self.assertEqual(
                workspace.receive_valid.view(4, 4).sum(1).tolist(), [0, 4, 0, 4]
            )
            with self.assertRaisesRegex(RuntimeError, "before reuse"):
                workspace.dispatch(rows, ids, weights)
            workspace.partial.zero_()
            workspace.combine(workspace.partial)
            self.assertEqual(status.tensor.item(), 5)
        with self.assertRaisesRegex(RuntimeError, "rejected Execute"):
            status.finish_execute()
        status.begin_execute()
        status.finish_execute()

        capacity = FixedDispatchCapacity(256, 4, 5120, 6, 384)
        with (
            patch(
                "oh_my_vllm.models.deepseek_v41.kernel_native._projection_module"
            ) as module,
            patch.object(
                torch, "empty", side_effect=AssertionError("allocation before budget")
            ),
        ):
            module.return_value.grouped_workspace_size.return_value = 0
            with self.assertRaisesRegex(ValueError, "workspace budget"):
                PaddedExpertBuffers(capacity, 2304, "cpu", budget_bytes=340 << 20)

    def test_four_rank_exchange_with_empty_rank_and_divergent_target_proposal_calls(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_peer, args=(str(Path(directory) / "store"),), nprocs=4, join=True)

    def test_dense_owner_keeps_mp4_partial_rounding_and_state_loading(self):
        from oh_my_vllm.models.deepseek_v41.model import RowParallelLinear

        with (
            patch("oh_my_vllm.models.deepseek_v41.model_context.world_size", 1),
            patch("oh_my_vllm.models.deepseek_v41.model_context.expert_world_size", 4),
            patch(
                "oh_my_vllm.models.deepseek_v41.model_context.default_dtype",
                torch.bfloat16,
            ),
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
            worker_epoch=7,
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
                step_id=3,
                worker_epoch=7,
                outputs=[
                    dict(request_id=row["request_id"]) for row in local["scheduled"]
                ],
                rank_completion=dict(
                    rank=rank, step_id=3, worker_epoch=7, schedule=schedule
                ),
            )
        self.assertEqual(
            merge_replies(message, replies)["outputs"], [dict(request_id=11)]
        )
        for field in ("worker_epoch", "step_id"):
            stale = copy.deepcopy(replies)
            stale[0][field] += 1
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reply"):
                merge_replies(message, stale)
        stale = copy.deepcopy(replies)
        stale[3]["rank_completion"]["worker_epoch"] += 1
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            merge_replies(message, stale)
        incomplete = dict(replies)
        incomplete.pop(3)
        with self.assertRaisesRegex(ValueError, "four rank"):
            merge_replies(message, incomplete)
        missing = copy.deepcopy(replies)
        missing[0].pop("rank_completion")
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            merge_replies(message, missing)
        duplicate = copy.deepcopy(message)
        duplicate["rank_plan"][0]["request_ids"] = [11]
        with self.assertRaisesRegex(ValueError, "exact request partition"):
            local_execute(duplicate, 0)

    def test_wire_all_ranks_ack_epoch_and_finish_releases_only_the_owner(self):
        from tests.deepseek_speculation_fixtures import (
            _chain_row,
            _chain_worker,
            _wire_message,
        )

        workers = {}
        for rank in range(4):
            worker = _chain_worker()
            self.addCleanup(worker.cache.close)
            worker.rank = rank
            worker.dep4_executor = SimpleNamespace(
                execute=lambda operation: operation()
            )
            if rank != 2:
                worker.requests.clear()
            workers[rank] = worker
        owner = workers[2]
        message = _wire_message(
            owner,
            dict(
                rpc_id=4,
                scheduled=[_chain_row(owner, owner.requests[1].history)],
                finished_request_ids=[],
                preempted_request_ids=[],
            ),
        )
        message["rank_plan"] = [
            dict(
                rank=rank,
                request_ids=[1] if rank == 2 else [],
                finished_request_ids=[],
                preempted_request_ids=[],
            )
            for rank in range(4)
        ]
        replies = {
            rank: handle_rank(worker, message, rank) for rank, worker in workers.items()
        }
        result = merge_replies(message, replies)
        self.assertEqual(
            [len(replies[rank]["outputs"]) for rank in range(4)], [0, 0, 1, 0]
        )
        self.assertEqual(
            [ack["worker_epoch"] for ack in result["rank_completions"]], [7] * 4
        )
        self.assertEqual((result["step_id"], result["worker_epoch"]), (1, 7))
        message = _wire_message(
            owner,
            dict(
                rpc_id=5,
                scheduled=[],
                finished_request_ids=[1],
                preempted_request_ids=[],
            ),
        )
        message["rank_plan"] = [
            dict(
                rank=rank,
                request_ids=[],
                finished_request_ids=[1] if rank == 2 else [],
                preempted_request_ids=[],
            )
            for rank in range(4)
        ]
        replies = {
            rank: handle_rank(worker, message, rank) for rank, worker in workers.items()
        }
        result = merge_replies(message, replies)
        self.assertEqual(result["outputs"], [])
        self.assertEqual(len(result["rank_completions"]), 4)
        self.assertNotIn(1, owner.requests)
        self.assertEqual(owner.slots, {0, 1})


if __name__ == "__main__":
    unittest.main()
