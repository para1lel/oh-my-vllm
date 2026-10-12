"""Context-only CUDA graph commits with changed slots and untouched KV canaries."""

import gc
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.kernels.dspark_attention import append, normalize_rope
from oh_my_vllm.models.dspark import DSparkConfig, DSparkModel, yarn_frequencies
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


@torch.inference_mode()
def test_real_context_projection_survives_unrelated_cublas_graph_retirement():
    """Exercise the production 29x25600 projection and all five KV writes.

    Warm on the exact capture stream outside its private pool. The pinned Torch
    used to embed that cached workspace and lose it on unrelated graph reset.
    Random weights exercise the real model method without checkpoint mutation.
    """
    generator = torch.Generator(device="cuda").manual_seed(121029)

    def random(*shape):
        return (
            torch.randn(
                *shape, generator=generator, device="cuda", dtype=torch.bfloat16
            )
            * 0.01
        )

    model = DSparkModel.__new__(DSparkModel)
    model.config = config = DSparkConfig()
    model.fc = random(config.hidden_size, config.feature_size)
    model.hidden_norm = torch.ones(
        config.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    model.inv_freq, model.attention_factor = yarn_frequencies(config, "cuda")
    model.layers = [
        SimpleNamespace(
            k=random(1024, config.hidden_size),
            v=random(1024, config.hidden_size),
            k_norm=torch.ones(128, device="cuda"),
        )
        for _ in range(config.num_hidden_layers)
    ]
    caches = [
        torch.full((5, 2, 784, 8, 128), 123, device="cuda", dtype=torch.bfloat16)
        for _ in model.layers
    ]
    originals = [cache.clone() for cache in caches]
    source = random(30, config.feature_size)
    features = torch.cat((source[:25], source[25:27], source[28:30]))
    positions = torch.tensor(
        [*range(38416, 38441), 32064, 32065, 33263, 33264], device="cuda"
    )
    slots = torch.tensor(
        [*range(784, 809), 2 * 784 + 704, 2 * 784 + 705, 3 * 784 + 335, 3 * 784 + 336],
        device="cuda",
    )
    budgets = (
        torch.backends.cuda.cublas_workspace_size(),
        torch.backends.cuda.cublaslt_workspace_size(),
    )
    # Constructing the public graph context initializes its default stream.
    unused = torch.cuda.graph(torch.cuda.CUDAGraph())
    capture_stream = torch.cuda.graph.default_capture_stream
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        warm_output = F.linear(features, model.fc)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()
    del warm_output, unused
    unit = compile_forward(model.inject, unit="test_dspark_workspace_lifetime")
    graph = DSparkContextGraph(
        unit, caches, features, positions, slots, pool=torch.cuda.graph_pool_handle()
    )
    for cache, original in zip(caches, originals, strict=True):
        torch.testing.assert_close(cache, original, atol=0, rtol=0)
    pages, offsets = slots // 784, slots % 784
    untouched = torch.ones((5, 784), device="cuda", dtype=torch.bool)
    untouched[pages, offsets] = False
    for step in range(3):
        if step:
            # Leave the survivor live while another graph on the same stream
            # releases its cuBLAS cache and allocator reservations.
            victim = torch.cuda.CUDAGraph()
            other_features = random(17, config.feature_size)
            with torch.cuda.graph(victim):
                other_output = F.linear(other_features, model.fc)
            victim.replay()
            torch.cuda.synchronize()
            victim.reset()
            del victim, other_output, other_features
            gc.collect()
            torch.cuda.empty_cache()
            pressure = torch.full((1 << 24,), 91, device="cuda", dtype=torch.float32)
            torch.cuda.synchronize()
            del pressure
        changed = features + step * 0.002
        expected = [cache.clone() for cache in originals]
        model.inject(changed, positions + step, slots, expected)
        graph.replay(changed, positions + step, slots)
        for cache, reference, original in zip(caches, expected, originals, strict=True):
            torch.testing.assert_close(cache, reference, atol=0.03, rtol=0.03)
            flat = cache.permute(0, 2, 1, 3, 4)
            original_flat = original.permute(0, 2, 1, 3, 4)
            torch.testing.assert_close(
                flat[untouched], original_flat[untouched], atol=0, rtol=0
            )
        assert (
            torch.backends.cuda.cublas_workspace_size(),
            torch.backends.cuda.cublaslt_workspace_size(),
        ) == budgets
