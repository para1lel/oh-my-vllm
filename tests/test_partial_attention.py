"""Context/query pruning preserves long-position preparation and mutable caches."""

import os
from types import SimpleNamespace

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir.partial_attention import prepare_context, prepare_query
from oh_my_vllm.kernels.attention_prepare import prepare_attention
from oh_my_vllm.kernels.cuda_backend import compiled
from oh_my_vllm.kernels.mtp_attention import MTPAttention
from oh_my_vllm.models.qwen import AttentionBatch, Checkpoint, Layer
from oh_my_vllm.worker.decode_graph import DecodeAttention
from oh_my_vllm.worker.mtp_context_graph import MTPContextGraph


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@torch.inference_mode()
def test_split_preparation_is_bitwise_equal_and_replays_changed_metadata(dtype):
    torch.manual_seed(784)
    packed = torch.randn(5, 14336, device="cuda", dtype=torch.bfloat16)
    qw, kw = (torch.randn(256, device="cuda") for _ in range(2))
    positions = torch.tensor([0, 1, 784, 131072, 262143], device="cuda", dtype=dtype)
    slots = torch.tensor([-1, 784, 785, 1568, 1569], device="cuda", dtype=dtype)
    full = torch.zeros(3, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    split = torch.zeros_like(full)
    kv, qg = packed[:, 12288:].contiguous(), packed[:, :12288].contiguous()

    def run():
        prepare_context(kv, kw, positions, split, slots)
        return prepare_query(qg, qw, positions)

    unit = compile_forward(run, unit="test_partial_prepare")
    unit()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        query = unit()
    for scale in (1, 2):
        packed.mul_(scale)
        kv.copy_(packed[:, 12288:])
        qg.copy_(packed[:, :12288])
        positions.add_(1)
        expected = prepare_attention(packed, qw, kw, positions, full, slots)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(query, expected, rtol=0, atol=0)
        torch.testing.assert_close(split, full, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("guard", ["dtype", "stride", "norm", "alias"])
def test_context_ffi_rejects_invalid_storage(guard):
    packed = torch.zeros(2, 2048, device="cuda", dtype=torch.bfloat16)
    weight = torch.ones(256, device="cuda")
    positions = torch.arange(2, device="cuda")
    cache = torch.zeros(1, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    slots = positions.clone()
    if guard == "dtype":
        packed = packed.float()
    elif guard == "stride":
        packed = torch.zeros(2, 4096, device="cuda", dtype=torch.bfloat16)[:, ::2]
    elif guard == "norm":
        weight = weight[:128]
    else:
        packed = cache.flatten()[:4096].view(2, 2048)
    with pytest.raises(Exception, match=r"requires|contiguous|overlap"):
        compiled().prepare_context(packed, weight, positions, cache, slots)


@pytest.fixture(scope="module")
def mtp_layer():
    path = os.environ.get("OH_MY_VLLM_MODEL")
    if not path:
        pytest.skip("set OH_MY_VLLM_MODEL for checkpoint integration")
    return Layer(Checkpoint(path), "mtp.layers.0", "full_attention")


@pytest.mark.gpu
@pytest.mark.parametrize("selected", [[128, 131], [131], []])
@torch.inference_mode()
def test_selected_mtp_outputs_preserve_context_and_graph_replay(mtp_layer, selected):
    torch.manual_seed(784)
    layer = mtp_layer
    hidden = torch.randn(132, 5120, device="cuda", dtype=torch.bfloat16)
    positions = [*range(1, 130), 784, 785, 786]
    tables = [[1], [2, 3]]
    slots = [784 + p for p in positions[:129]] + [
        3 * 784 + p % 784 for p in positions[129:]
    ]
    cache = torch.zeros(7, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    # A restored second-request prefix is immutable during both runs.
    cache[2].normal_()
    attention = MTPAttention(4096)
    attention.plan([0, 129, 132], tables, positions)
    batch = AttentionBatch(
        torch.tensor(positions, device="cuda"),
        torch.tensor(slots, device="cuda"),
        attention,
    )
    indices = torch.tensor(selected, device="cuda", dtype=torch.int64)
    picked = None
    if selected:
        picked = DecodeAttention(
            torch.tensor(
                [
                    ([1] if i < 129 else [2, 3]) + [0] * (6 - (1 if i < 129 else 2))
                    for i in selected
                ],
                device="cuda",
                dtype=torch.int32,
            ),
            torch.tensor(
                [positions[i] + 1 for i in selected], device="cuda", dtype=torch.int32
            ),
            4096,
        )
        picked.first = 1

    def draft_context(tokens, x, plan, pool, output_indices, output_attention):
        branch, residual = layer.forward_selected(
            x, plan, pool, output_indices, output_attention
        )
        return branch + residual

    model = SimpleNamespace(draft_context=draft_context)
    original = cache.clone()
    graph = MTPContextGraph(
        model,
        cache,
        torch.zeros(132, device="cuda", dtype=torch.int64),
        hidden,
        batch,
        indices,
        picked,
    )
    torch.testing.assert_close(cache, original, rtol=0, atol=0)
    for iteration, multiplier in enumerate((1, 0.5)):
        cache.copy_(original)
        if iteration:
            # Replay owns new physical pages and positions, with the same
            # query/context shapes and immutable-prefix semantics.
            tables = [[4], [5, 6]]
            cache[5].copy_(original[2])
            positions = [p + 1 for p in positions]
            batch.positions.copy_(torch.tensor(positions, device="cuda"))
            new_slots = [4 * 784 + p for p in positions[:129]] + [
                6 * 784 + p % 784 for p in positions[129:]
            ]
            batch.fa_slots.copy_(torch.tensor(new_slots, device="cuda"))
            attention.plan([0, 129, 132], tables, positions)
            if picked is not None:
                for row, index in enumerate(selected):
                    table = [4] if index < 129 else [5, 6]
                    picked.tables[row].zero_()
                    picked.tables[row, : len(table)] = torch.tensor(
                        table, device="cuda", dtype=torch.int32
                    )
                picked.lengths.copy_(
                    torch.tensor(
                        [positions[i] + 1 for i in selected],
                        device="cuda",
                        dtype=torch.int32,
                    )
                )
        hidden.mul_(multiplier)
        baseline = cache.clone()
        expected = layer(hidden, batch, baseline)[selected]
        actual = graph.replay(
            torch.zeros(132, device="cuda", dtype=torch.int64),
            hidden,
            batch,
            indices,
            picked,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(cache, baseline, rtol=0, atol=0)
        # Preserve the full-model hidden tolerance used by the existing compile
        # integration test. Native prefill and owned decode attention can differ
        # within their BF16 operator tolerance; the following MLP amplifies it.
        if selected:
            error = actual.float() - expected.float()
            reference = expected.float()
            nrmse = (
                error.square().mean().sqrt()
                / reference.square().mean().sqrt().clamp_min(1e-10)
            )
            relative_max = error.abs().max() / reference.abs().max().clamp_min(1e-10)
            assert nrmse < 0.1
            assert relative_max < 0.2
