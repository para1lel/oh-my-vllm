import datetime
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
from oh_my_vllm.models.deepseek_v41 import model_context
from oh_my_vllm.models.deepseek_v41.dispatch import (
    DispatchStatus,
    ExpertExchange,
    FixedDispatchBuffers,
    FixedDispatchCapacity,
    PaddedExpertBuffers,
    combine,
    dispatch,
    local_expert_sum,
)

from tests.deepseek_dep4_fixtures import _fixed_input


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
            patch.object(model_context, "world_size", 1),
            patch.object(model_context, "expert_world_size", 4),
            patch.object(model_context, "rank", rank),
        ):
            moe = _cuda_moe(model, device)
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
            _check_cuda_memory_budget(device, moe, status, rank)
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


def _check_cuda_memory_budget(device, moe, status, rank):
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
    projections = PaddedExpertBuffers(capacity, 2304, device, budget_bytes=512 << 20)
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


def _cuda_moe(model, device):
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
    return moe
