"""Context-only CUDA graph commits with changed slots and untouched KV canaries."""

import math

import pytest
import torch
import torch.nn.functional as F
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.kernels.dspark_attention import append, normalize_rope
from oh_my_vllm.worker.dspark_graph import DSparkContextGraph

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("rows", [1, 8, 32])
def test_context_graph_restores_capture_and_replays_new_committed_slots(rows):
    generator = torch.Generator(device="cuda").manual_seed(784 + rows)
    features = torch.randn(
        rows, 16, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    projections = [
        torch.randn(2048, 16, generator=generator, device="cuda", dtype=torch.bfloat16)
        for _ in range(5)
    ]
    weights = [
        torch.randn(128, generator=generator, device="cuda").bfloat16().float()
        for _ in range(5)
    ]
    frequencies = (
        1e7 ** (-torch.arange(64, device="cuda", dtype=torch.float32) / 64) / 32
    )
    caches = [
        torch.full((5, 2, 784, 8, 128), 123, device="cuda", dtype=torch.bfloat16)
        for _ in range(5)
    ]
    positions = torch.arange(783, 783 + rows, device="cuda")
    slots = positions + 784

    def inject(context, absolute, destinations, pools):
        for projection, weight, pool in zip(projections, weights, pools, strict=True):
            projected = F.linear(context, projection).view(rows, 2, 8, 128)
            key = normalize_rope(
                projected[:, 0], weight, absolute, frequencies, 1 + 0.1 * math.log(32)
            )
            append(pool, key, projected[:, 1].contiguous(), destinations)

    originals = [cache.clone() for cache in caches]
    unit = compile_forward(inject, unit="test_dspark_context_graph")
    graph = DSparkContextGraph(
        unit, caches, features, positions, slots, pool=torch.cuda.graph_pool_handle()
    )
    assert graph.outputs is None
    for pool, original in zip(caches, originals, strict=True):
        torch.testing.assert_close(pool, original, atol=0, rtol=0)
    expected = [cache.clone() for cache in caches]
    for step in range(2):
        current_features = features + step * 0.25
        current_positions = positions + step * 784
        current_slots = slots + step * 784
        inject(current_features, current_positions, current_slots, expected)
        assert graph.replay(current_features, current_positions, current_slots) is None
        for pool, reference in zip(caches, expected, strict=True):
            # Compare every physical row, including page zero, rejected suffix
            # slots and both sides of the retained rows at the page boundary.
            torch.testing.assert_close(pool, reference, atol=0.03, rtol=0.03)
        torch.testing.assert_close(graph.positions, current_positions, atol=0, rtol=0)
        torch.testing.assert_close(graph.slots, current_slots, atol=0, rtol=0)
