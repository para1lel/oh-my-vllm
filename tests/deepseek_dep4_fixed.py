import datetime
from unittest.mock import patch

import torch
import torch.distributed as dist
from oh_my_vllm.models.deepseek_v41.dispatch import (
    DispatchStatus,
    FixedDispatchBuffers,
    FixedDispatchCapacity,
)

from tests.deepseek_dep4_fixtures import _fixed_input


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
            _check_fixed_routes(
                rank,
                capacity,
                workspace,
                remote_ids,
                remote_rows,
                remote_weights,
                counts,
                sorted_routes,
                routed_rows,
                routed_weights,
                mode,
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
        _check_fixed_failures(status, rank, workspace)
    finally:
        dist.destroy_process_group()


def _check_fixed_failures(status, rank, workspace):
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


def _check_fixed_routes(
    rank,
    capacity,
    workspace,
    remote_ids,
    remote_rows,
    remote_weights,
    counts,
    sorted_routes,
    routed_rows,
    routed_weights,
    mode,
):
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
        remote_weights.flatten()[torch.tensor(expected_routes, dtype=torch.int64)],
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
