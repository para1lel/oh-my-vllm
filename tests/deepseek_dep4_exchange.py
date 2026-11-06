import datetime
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from oh_my_vllm.models.deepseek_v41.dispatch import (
    DispatchStatus,
    ExpertExchange,
    FixedDispatchBuffers,
    FixedDispatchCapacity,
)

from tests.deepseek_dep4_fixtures import _MoE


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
