import copy
import datetime
import os
import tempfile
import unittest
from pathlib import Path
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
    combine,
    destination_pairs,
    dispatch,
    local_expert_sum,
)
from oh_my_vllm.models.deepseek_v41.rank_plan import (
    handle_rank,
    local_execute,
    merge_replies,
)


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


def _fixed_peer(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=120),
    )
    try:
        capacity = FixedDispatchCapacity(4, 4, 4, 6, 24)
        status = DispatchStatus("cpu")
        workspace = FixedDispatchBuffers(
            capacity, status, device="cpu", buffer_budget_bytes=capacity.tensor_bytes
        )
        addresses = tuple(value.data_ptr() for value in workspace.storage.values())
        for mode in ("concentrated", "empty", "unequal", "spread", "concentrated"):
            status.begin_execute()
            rows, ids, weights = _fixed_input(rank, mode)
            with (
                patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("host scalar")
                ),
                patch.object(
                    torch.Tensor, "tolist", side_effect=AssertionError("host counts")
                ),
                patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("host copy")
                ),
            ):
                remote_rows, remote_ids, remote_weights = workspace.dispatch(
                    rows, ids, weights
                )
                routed_rows, routed_weights, counts, sorted_routes = (
                    workspace.local_routes(rank)
                )
            expected_routes = [
                token * 6 + slot
                for expert in range(rank * 6, rank * 6 + 6)
                for token in range(capacity.receive_rows)
                for slot in range(6)
                if workspace.receive_valid[token] and remote_ids[token, slot] == expert
            ]
            real = len(expected_routes)
            assert counts.dtype == torch.int32
            assert int(counts.sum()) == real
            assert sorted_routes[:real].tolist() == expected_routes
            expected_tokens = torch.tensor(expected_routes, dtype=torch.int64) // 6
            torch.testing.assert_close(
                routed_rows[:real], remote_rows[expected_tokens], rtol=0, atol=0
            )
            torch.testing.assert_close(
                routed_weights[:real],
                remote_weights.flatten()[
                    torch.tensor(expected_routes, dtype=torch.int64)
                ],
                rtol=0,
                atol=0,
            )
            for source in range(4):
                values, indices, scales = _fixed_input(source, mode)
                selected = (indices // 6 == rank).any(dim=1)
                count = int(selected.sum())
                assert int(workspace.receive_counts[source]) == count
                first = source * 4
                torch.testing.assert_close(
                    remote_rows[first : first + count], values[selected], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    remote_ids[first : first + count],
                    indices[selected].int(),
                    rtol=0,
                    atol=0,
                )
                torch.testing.assert_close(
                    remote_weights[first : first + count],
                    scales[selected],
                    rtol=0,
                    atol=0,
                )
            factors = (1e20, 1, -1e20, 3)
            workspace.partial.fill_(factors[rank])
            workspace.partial[~workspace.receive_valid] = float("nan")
            with (
                patch.object(
                    torch.Tensor, "item", side_effect=AssertionError("host scalar")
                ),
                patch.object(
                    torch.Tensor, "tolist", side_effect=AssertionError("host counts")
                ),
                patch.object(
                    torch.Tensor, "cpu", side_effect=AssertionError("host copy")
                ),
            ):
                actual = workspace.combine(workspace.partial)
            expected = torch.zeros_like(rows, dtype=torch.float32)
            for destination in range(4):
                expected[(ids // 6 == destination).any(dim=1)] += factors[destination]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            assert addresses == tuple(
                value.data_ptr() for value in workspace.storage.values()
            )
            status.finish_execute()
        for failure in ("negative", "large", "expert_overflow"):
            status.begin_execute()
            rows, ids, weights = _fixed_input(rank, "concentrated")
            ids = ids.clone()
            if rank == 2:
                if failure == "expert_overflow":
                    ids.fill_(0)
                else:
                    ids[0, 0] = -1 if failure == "negative" else 2**40
            workspace.dispatch(rows, ids, weights)
            workspace.local_routes(rank)
            workspace.partial.zero_()
            workspace.combine(workspace.partial)
            try:
                status.finish_execute()
            except RuntimeError as error:
                assert "rejected Execute" in str(error)
            else:
                raise AssertionError(f"{failure} must reject every rank")
        status.begin_execute()
        if rank == 2:
            status.tensor.fill_(-2147483648)
        try:
            status.finish_execute()
        except RuntimeError as error:
            assert "rejected Execute" in str(error)
        else:
            raise AssertionError("one-rank status failure must reject all ranks")
        status.begin_execute()
        status.finish_execute()
    finally:
        dist.destroy_process_group()


def _multiwave_peer(rank, store):
    from types import SimpleNamespace

    from oh_my_vllm.models.deepseek_v41.kernel import ordered_expert_sum

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        moe = _MoE(rank)
        status = DispatchStatus("cpu")
        fixed_exchange = ExpertExchange(status=status)
        legacy_exchange = ExpertExchange()
        pools = {}
        waves = []

        def project(moe, fixed, expert_rank):
            routed, scales, counts, routes = fixed.local_routes(expert_rank)
            live = int(counts.sum())
            down = torch.empty((live, moe.dim), dtype=torch.bfloat16)
            first = 0
            for expert, count in zip(
                range(moe.experts_start_idx, moe.experts_end_idx),
                counts.tolist(),
                strict=True,
            ):
                last = first + count
                down[first:last] = moe.experts[expert](
                    routed[first:last], scales[first:last, None]
                )
                first = last
            ordered_expert_sum(down, routes[:live], fixed.partial, 6)
            waves.append(tuple(fixed.receive_counts.tolist()))
            return fixed.partial

        def buffers(_moe, device):
            rows = min(256, 1 << (max(1, fixed_exchange.source_rows) - 1).bit_length())
            if rows not in pools:
                capacity = FixedDispatchCapacity(rows, 4, moe.dim, 6, 8)
                pools[rows] = FixedDispatchBuffers(
                    capacity,
                    status,
                    device=device,
                    buffer_budget_bytes=capacity.tensor_bytes,
                )
            return pools[rows], SimpleNamespace(run=project)

        with patch.object(fixed_exchange, "_buffers", side_effect=buffers):
            for lengths in (
                (600, 300, 17, 0),
                (257, 256, 0, 1),
                (0, 0, 0, 0),
                (1, 0, 0, 0),
            ):
                count = lengths[rank]
                values = torch.arange(count * moe.dim).reshape(count, moe.dim)
                rows = ((values % 31 - 15).float() / 8).to(torch.bfloat16)
                ids = (
                    (torch.arange(6)[None, :] + torch.arange(count)[:, None]) % 8
                ).int()
                weights = (torch.arange(6).float() + 1).expand(count, 6) / 32
                expected = legacy_exchange(moe, rows, (weights, ids))
                fixed_exchange.prepare_round(max(lengths), fixed=True)
                status.begin_execute()
                before = len(waves)
                actual = fixed_exchange(moe, rows, (weights, ids))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                assert len(waves) - before == (max(lengths) + 255) // 256
                status.finish_execute()
    finally:
        dist.destroy_process_group()


def _shadow_peer(rank, store):
    import json
    from types import SimpleNamespace

    from development.model.check_deepseek_dep4 import ExchangeEvidence

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        exchange = ExpertExchange()
        exchange.prepare_round(600, fixed=True)
        rows = torch.ones((0 if rank == 3 else 1, 4), dtype=torch.bfloat16)
        ids = torch.zeros((rows.size(0), 6), dtype=torch.int32)
        weights = torch.ones_like(ids, dtype=torch.float32)
        originals = [value.clone() for value in (rows, ids, weights)]
        moe = SimpleNamespace(layer_id=0)

        def operation(instance, _moe, values, _routing):
            result = values.float()
            if instance.fixed_round and rank == 2:
                result = result + 1
            return result

        proof = ExchangeEvidence(Path(store).parent, rank, mode="shadow")
        with patch.object(ExpertExchange, "__call__", operation):
            try:
                with proof.observe():
                    exchange(moe, rows, (weights, ids))
                    raise AssertionError("a divergent shadow must stop all ranks")
            except AssertionError as error:
                assert "expert sums differ" in str(error), error
        assert exchange.fixed_round
        for actual, expected in zip((rows, ids, weights), originals, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        record = json.loads(
            Path(store).parent.joinpath(f"proof-rank{rank}.json").read_text()
        )
        assert record["outcome"] == "failed"
        assert record["counts"]["fixed"] == record["counts"]["shadow_legacy"] == 1
        assert record["counts"]["failed"] == 1
        assert len(record["shadow_differences"]) == int(rank == 2)
    finally:
        dist.destroy_process_group()


def _driver_failure(args, proof):
    import json
    from types import SimpleNamespace

    from development.model.check_deepseek_dep4 import initialize_rank

    torch.set_num_threads(1)
    for peer in range(4):
        record = json.loads((args.output / f"proof-rank{peer}.json").read_text())
        assert record["run_id"] == proof.run_id
        assert record["outcome"] == "initializing"
    if args.failure in ("missing", "corrupt", "device", "initialize"):
        original_init = dist.init_process_group

        def device(rank):
            if args.failure == "device" and rank == 2:
                raise RuntimeError("injected device failure")

        def initialize(_backend, **options):
            if args.failure == "initialize" and proof.rank == 2:
                raise ValueError("injected initialization failure")
            return original_init("gloo", **options)

        with (
            patch.object(torch.cuda, "set_device", device),
            patch.object(dist, "init_process_group", initialize),
        ):
            initialize_rank(args, proof)
        return
    dist.init_process_group(
        "gloo",
        init_method=f"file://{args.output / 'rank-store'}",
        rank=proof.rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=240),
    )
    if args.failure == "endpoint":
        if proof.rank == 2:
            raise AssertionError("injected endpoint assertion")
        dist.all_reduce(torch.ones(1))
        raise AssertionError("a peer endpoint failure must terminate this collective")
    exchange = ExpertExchange()
    exchange.prepare_round(600, fixed=True)
    rows = torch.ones((1, 4), dtype=torch.bfloat16)

    def legacy(_exchange, _moe, values, _routing):
        if proof.rank == 2:
            raise ValueError("injected legacy failure")
        dist.all_reduce(torch.ones(1))
        return values.float()

    proof.compare_shadow(
        exchange, legacy, SimpleNamespace(layer_id=0), rows, None, rows.float()
    )
    raise AssertionError("a peer legacy failure must terminate this collective")


def _cuda_fixed_peer(rank, store):
    from oh_my_vllm.models.deepseek_v41 import model

    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{store}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=60),
    )
    try:
        with (
            torch.inference_mode(),
            patch.object(model, "world_size", 1),
            patch.object(model, "expert_world_size", 4),
            patch.object(model, "rank", rank),
        ):
            moe = model.MoE(
                0,
                model.ModelArgs(
                    dim=5120,
                    moe_inter_dim=2304,
                    n_routed_experts=384,
                    n_activated_experts=6,
                    expert_dtype="fp4",
                    parallel_strategy="DEP4",
                    swiglu_limit=10,
                ),
            ).to(device)
            for expert in moe.experts:
                if expert is not None:
                    for projection in (expert.w1, expert.w3, expert.w2):
                        projection.weight.view(torch.uint8).zero_()
                        projection.weight.view(torch.uint8)[0, 0] = 2
                        projection.scale.fill_(1)
            status = DispatchStatus(device)
            capacity = FixedDispatchCapacity(4, 4, moe.dim, 6, moe.n_routed_experts)
            fixed = FixedDispatchBuffers(
                capacity,
                status,
                device=device,
                buffer_budget_bytes=capacity.tensor_bytes,
            )
            projections = PaddedExpertBuffers(
                capacity, 2304, device, budget_bytes=512 << 20
            )
            for mode in ("concentrated", "spread", "empty", "unequal"):
                status.begin_execute()
                rows, ids, weights = _fixed_input(rank, mode, device)
                rows = torch.nn.functional.pad(rows, (0, moe.dim - rows.size(1)))
                ids = ids.div(6, rounding_mode="floor") * 96 + ids.remainder(6)
                (remote_rows, remote_ids, remote_weights), layout = dispatch(
                    rows, ids, weights, num_experts=moe.n_routed_experts
                )
                expected = combine(
                    local_expert_sum(moe, remote_rows, remote_ids, remote_weights),
                    layout,
                    num_rows=rows.size(0),
                )
                fixed.dispatch(rows, ids, weights)
                actual = fixed.combine(projections.run(moe, fixed, rank))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                status.finish_execute()
            del fixed, projections, actual, expected
            rows = torch.zeros((256, moe.dim), dtype=torch.bfloat16, device=device)
            rows[:, 0] = 1
            ids = torch.arange(6, device=device)[None, :].expand(256, 6)
            weights = torch.full((256, 6), 0.125, device=device)
            expected = torch.zeros((256, moe.dim), dtype=torch.float32, device=device)
            expected[:, 0] = 0.5625
            torch.cuda.synchronize(device)
            baseline = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            capacity = FixedDispatchCapacity(256, 4, moe.dim, 6, moe.n_routed_experts)
            fixed = FixedDispatchBuffers(
                capacity,
                status,
                device=device,
                buffer_budget_bytes=capacity.tensor_bytes,
            )
            projections = PaddedExpertBuffers(
                capacity, 2304, device, budget_bytes=512 << 20
            )
            status.begin_execute()
            fixed.dispatch(rows, ids, weights)
            actual = fixed.combine(projections.run(moe, fixed, rank))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            status.finish_execute()
            torch.cuda.synchronize(device)
            # CUDA allocator segments can exceed the tensor's byte size.
            assert (
                torch.cuda.max_memory_allocated(device) - baseline
                <= projections.reserved_bytes
                <= 512 << 20
            )
            del fixed, projections, actual, expected
            _cuda_multiwave_experts(moe, rank, device, status, store)
    finally:
        torch.cuda.synchronize(device)
        dist.destroy_process_group()


def _cuda_multiwave_experts(moe, rank, device, status, store):
    from development.model.check_deepseek_dep4 import ExchangeEvidence

    torch.manual_seed(901 + rank)
    for expert in moe.experts:
        if expert is not None:
            for projection in (expert.w1, expert.w3, expert.w2):
                projection.weight.view(torch.uint8).random_(0, 256)
                projection.scale.fill_(1 / 32)
    moe._routed_state.clear_weights()
    exchange = ExpertExchange(status=status)
    proof = ExchangeEvidence(Path(store).parent, rank, mode="fixed")
    with proof.observe():
        for lengths in (
            (600, 300, 17, 0),
            (257, 256, 0, 1),
            (0, 0, 0, 0),
            (1, 0, 0, 0),
        ):
            count = lengths[rank]
            positions = torch.arange(count, device=device)
            values = torch.arange(count * moe.dim, device=device).reshape(
                count, moe.dim
            )
            rows = ((values.remainder(31) - 15).float() / 128).to(torch.bfloat16)
            ids = (
                (
                    torch.arange(6, device=device)[None, :]
                    + positions[:, None] * 73
                    + rank * 97
                )
                % 384
            ).int()
            weights = (
                (positions[:, None] * 7 + torch.arange(6, device=device) * 3).remainder(
                    13
                )
                + 1
            ).float() / 128
            (remote_rows, remote_ids, remote_weights), layout = dispatch(
                rows, ids, weights, num_experts=moe.n_routed_experts
            )
            expected = combine(
                local_expert_sum(moe, remote_rows, remote_ids, remote_weights),
                layout,
                num_rows=count,
            )
            exchange.prepare_round(max(lengths), fixed=True)
            status.begin_execute()
            actual = exchange(moe, rows, (weights, ids))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            status.finish_execute()
            assert exchange.reserved_bytes <= 512 << 20


class _FaultExchange(ExpertExchange):
    def __init__(self, rank, status, *, sparse=False):
        super().__init__(status=status)
        self.rank, self.calls, self.sparse = rank, 0, sparse

    def __call__(self, moe, rows, routing):
        value = super().__call__(moe, rows, routing)
        if self.rank == 2 and self.calls == 0:
            if self.sparse:
                from oh_my_vllm.models.deepseek_v41.kernel import (
                    _sparse_status_accumulator,
                )

                _sparse_status_accumulator(rows.device).fill_(True)
            else:
                self.status.tensor.fill_(4)
        self.calls += 1
        return value


def _round_failure_peer(rank, store, sparse=False):
    from types import SimpleNamespace

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
        executor = Dep4Executor(model, check_round_status=True)
        executor.sparse_control = sparse
        executor.exchange = _FaultExchange(rank, executor.status, sparse=sparse)
        for layer in (*model.layers, *model.mtp):
            layer.ffn.dep4_exchange = executor.exchange
        commits = []
        message = dict(
            type="execute",
            rpc_id=8,
            step_id=2,
            scheduled=[dict(request_id=11, token_ids=[1])],
            finished_request_ids=[],
            preempted_request_ids=[],
            rank_plan=[
                dict(
                    rank=peer,
                    request_ids=[11] if peer == 0 else [],
                    finished_request_ids=[],
                    preempted_request_ids=[],
                )
                for peer in range(4)
            ],
        )

        def handle(local):
            for row in local["scheduled"]:
                model(torch.ones((1, 4), dtype=torch.bfloat16))
                commits.append(row["request_id"])
                model(torch.ones((1, 4), dtype=torch.bfloat16))
                commits.append(row["request_id"])
            return dict(type="execute_result", rpc_id=8, outputs=[])

        worker = SimpleNamespace(dep4_executor=executor, handle=handle)
        readbacks = []
        original_cpu = torch.Tensor.cpu

        def readback(value, *args, **kwargs):
            readbacks.append(value.numel())
            return original_cpu(value, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", readback):
            try:
                handle_rank(worker, message, rank)
            except RuntimeError as error:
                reason = (
                    "native sparse attention failure"
                    if sparse
                    else "invalid padded expert metadata"
                )
                assert reason in str(error)
                assert "no cache commit or result" in str(error)
            else:
                raise AssertionError("a failed round cannot produce rank completion")
        assert not commits
        assert executor.target_rounds == 1
        assert executor.active is False
        assert readbacks == [12, 2], readbacks
        errors = {peer: dict(type="error", rpc_id=8) for peer in range(4)}
        try:
            merge_replies(message, errors)
        except ValueError:
            pass
        else:
            raise AssertionError("failed ranks cannot publish an Execute result")
    finally:
        dist.destroy_process_group()


class DeepSeekDep4(unittest.TestCase):
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
            patch("oh_my_vllm.models.deepseek_v41.kernel._projection_module") as module,
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
