"""Full model compile units inside the existing manual CUDA Graph lifecycle."""

import os

import pytest
import torch
from oh_my_vllm.ir import compile_forward, selected_implementations
from oh_my_vllm.kernels.attention import PagedAttention
from oh_my_vllm.models.qwen import AttentionBatch, Batch, Qwen
from oh_my_vllm.worker.decode_graph import (
    DecodeAttention,
    DecodeGraph,
    DraftGraph,
    ProposalGraph,
)

MODEL = os.environ.get("OH_MY_VLLM_MODEL")


def _clone_caches(caches):
    return [
        tuple(pool.clone() for pool in cache)
        if isinstance(cache, tuple)
        else cache.clone()
        for cache in caches
    ]


def _assert_numerically_close(actual, expected, *, name):
    """Bound accumulated BF16 model drift without ignoring gross state errors."""
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, name
    error = actual.float() - expected.float()
    reference = expected.float()
    nrmse = error.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(
        1e-10
    )
    relative_max = error.abs().amax() / reference.abs().amax().clamp_min(1e-10)
    assert nrmse < 0.1, f"{name}: NRMSE={nrmse.item():.5g}"
    assert relative_max < 0.2, f"{name}: relative max={relative_max.item():.5g}"


def _assert_written_caches(actual, expected, kinds, *, fa_tokens):
    for layer, (current, reference, kind) in enumerate(
        zip(actual, expected, kinds, strict=True)
    ):
        if kind == "full_attention":
            _assert_numerically_close(
                current[0, :, :fa_tokens],
                reference[0, :, :fa_tokens],
                name=f"FA layer {layer}",
            )
            assert not current[1].count_nonzero(), f"FA layer {layer} touched page 1"
        else:
            for pool_name, written, old in zip(
                ("conv", "GDN"), current, reference, strict=True
            ):
                _assert_numerically_close(
                    written[1], old[1], name=f"{pool_name} layer {layer}"
                )
                assert not written[0].count_nonzero(), (
                    f"{pool_name} layer {layer} touched source slot"
                )


@pytest.mark.gpu
@torch.inference_mode()
def test_compiled_target_graph_replays_changed_native_page_tables():
    class AttentionModel:
        def forward(self, tokens, batch, caches):
            query = torch.ones(
                len(tokens), 24, 256, device=tokens.device, dtype=torch.bfloat16
            )
            return batch.attention(query, caches[0])

        def logits(self, hidden):
            return hidden.sum(-1)

    cache = torch.zeros(2, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    cache[0, 1] = 1
    cache[1, 1] = 2
    tokens = torch.zeros(1, device="cuda", dtype=torch.int64)
    positions = torch.zeros_like(tokens)
    batch = Batch(
        positions=positions,
        fa_slots=positions.clone(),
        attention=None,
        starts=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        sequence_ids=positions.clone(),
        state_reads=positions.clone(),
        state_writes=positions.clone(),
        final_state_writes=positions.clone(),
        prefill_sequences=0,
        prefill_tokens=0,
    )
    first = torch.zeros(1, 2, device="cuda", dtype=torch.int32)
    graph = DecodeGraph(
        AttentionModel(), [cache], tokens, batch, first, 1568, compile_model=True
    )
    output, _ = graph.replay(tokens, batch, first)
    torch.testing.assert_close(output.float(), torch.ones_like(output.float()))
    second = torch.ones_like(first)
    output, _ = graph.replay(tokens, batch, second)
    torch.testing.assert_close(output.float(), torch.full_like(output.float(), 2))


@pytest.mark.gpu
@torch.inference_mode()
def test_full_model_prefill_target_and_mtp_compile_units():
    if not MODEL:
        pytest.skip("set OH_MY_VLLM_MODEL for full-model integration tests")
    model = Qwen(MODEL, mtp=True)
    caches = []
    for kind in model.kinds:
        if kind == "full_attention":
            caches.append(
                torch.zeros(2, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
            )
        else:
            caches.append(
                (
                    torch.zeros(2, 10240, 3, device="cuda", dtype=torch.bfloat16),
                    torch.zeros(2, 48, 128, 128, device="cuda", dtype=torch.bfloat16),
                )
            )

    attention = PagedAttention()
    attention.plan(
        torch.tensor([0, 7], dtype=torch.int32),
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
        24,
        4,
        256,
    )
    positions = torch.arange(7, device="cuda")
    batch = Batch(
        positions=positions,
        fa_slots=positions.clone(),
        attention=attention,
        starts=torch.tensor([0, 7], device="cuda", dtype=torch.int32),
        sequence_ids=torch.zeros(7, device="cuda", dtype=torch.int64),
        state_reads=torch.zeros(1, device="cuda", dtype=torch.int64),
        state_writes=torch.tensor([-1] * 6 + [1], device="cuda"),
        final_state_writes=torch.ones(1, device="cuda", dtype=torch.int64),
        prefill_sequences=1,
        prefill_tokens=7,
    )
    prefill_tokens = torch.ones(7, device="cuda", dtype=torch.int64)
    eager_caches = _clone_caches(caches)
    expected_prefill = model.forward(prefill_tokens, batch, eager_caches)
    prefill = compile_forward(model.forward)
    hidden = prefill(prefill_tokens, batch, caches)
    assert hidden.shape == (7, 5120)
    _assert_numerically_close(hidden, expected_prefill, name="prefill hidden")
    _assert_written_caches(caches, eager_caches, model.kinds, fa_tokens=7)

    tokens = torch.ones(1, device="cuda", dtype=torch.int64)
    target_position = torch.full((1,), 7, device="cuda", dtype=torch.int64)
    tables = torch.zeros(1, 6, device="cuda", dtype=torch.int32)
    target_batch = Batch(
        positions=target_position,
        fa_slots=target_position.clone(),
        attention=attention,
        starts=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
        sequence_ids=torch.zeros_like(target_position),
        state_reads=torch.ones_like(target_position),
        state_writes=torch.ones(1, device="cuda", dtype=torch.int64),
        final_state_writes=torch.ones(1, device="cuda", dtype=torch.int64),
        prefill_sequences=0,
        prefill_tokens=0,
    )
    eager_target_graph = DecodeGraph(
        model, eager_caches, tokens, target_batch, tables, 4096, compile_model=False
    )
    target_graph = DecodeGraph(
        model, caches, tokens, target_batch, tables, 4096, compile_model=True
    )
    target_hidden, target_logits = target_graph.replay(tokens, target_batch, tables)
    eager_hidden, eager_logits = eager_target_graph.replay(tokens, target_batch, tables)
    assert target_hidden.shape == (1, 5120)
    assert target_logits.shape == (1, 248320)
    _assert_numerically_close(target_hidden, eager_hidden, name="target hidden")
    _assert_numerically_close(target_logits, eager_logits, name="target logits")
    assert torch.equal(target_logits.argmax(-1), eager_logits.argmax(-1))
    _assert_written_caches(caches, eager_caches, model.kinds, fa_tokens=8)

    mtp_cache = torch.zeros(2, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
    mtp_position = torch.ones(1, device="cuda", dtype=torch.int64)
    mtp_tables = torch.ones(1, 6, device="cuda", dtype=torch.int32)
    mtp_attention = DecodeAttention(mtp_tables, mtp_position + 1, 4096)
    mtp_attention.first = 1
    mtp_batch = AttentionBatch(
        mtp_position,
        torch.full((1,), 785, device="cuda", dtype=torch.int64),
        mtp_attention,
    )
    eager_mtp_cache = mtp_cache.clone()
    eager_draft_graph = DraftGraph(
        model,
        eager_mtp_cache,
        tokens,
        target_hidden,
        mtp_batch,
        mtp_tables,
        4096,
        compile_model=False,
    )
    draft_graph = DraftGraph(
        model,
        mtp_cache,
        tokens,
        target_hidden,
        mtp_batch,
        mtp_tables,
        4096,
        compile_model=True,
    )
    draft_hidden = draft_graph.replay(tokens, target_hidden, mtp_batch, mtp_tables)
    eager_draft_hidden = eager_draft_graph.replay(
        tokens, target_hidden, mtp_batch, mtp_tables
    )
    assert draft_hidden.shape == (1, 5120)
    _assert_numerically_close(draft_hidden, eager_draft_hidden, name="draft hidden")
    _assert_numerically_close(
        mtp_cache[1, :, 1], eager_mtp_cache[1, :, 1], name="draft cache"
    )

    proposal_tables = mtp_tables.to(torch.int64)
    eager_proposal_graph = ProposalGraph(
        model,
        eager_mtp_cache,
        draft_hidden,
        mtp_position,
        proposal_tables,
        4096,
        compile_model=False,
    )
    proposal_graph = ProposalGraph(
        model,
        mtp_cache,
        draft_hidden,
        mtp_position,
        proposal_tables,
        4096,
        compile_model=True,
    )
    proposals = proposal_graph.replay(draft_hidden, mtp_position, proposal_tables)
    eager_proposals = eager_proposal_graph.replay(
        draft_hidden, mtp_position, proposal_tables
    )
    torch.cuda.synchronize()
    assert proposals.shape == (1, 4)
    torch.testing.assert_close(proposals, eager_proposals)
    _assert_numerically_close(
        mtp_cache[1, :, 2:5],
        eager_mtp_cache[1, :, 2:5],
        name="proposal cache",
    )

    changed_position = torch.full_like(mtp_position, 2)
    changed_tables = torch.zeros_like(mtp_tables)
    changed_batch = AttentionBatch(
        changed_position,
        torch.full_like(mtp_position, 2),
        mtp_attention,
    )
    changed_tokens = tokens + 1
    changed_draft = draft_graph.replay(
        changed_tokens, target_hidden, changed_batch, changed_tables
    )
    eager_changed_draft = eager_draft_graph.replay(
        changed_tokens, target_hidden, changed_batch, changed_tables
    )
    _assert_numerically_close(
        changed_draft, eager_changed_draft, name="changed draft hidden"
    )
    _assert_numerically_close(
        mtp_cache[0, :, 2], eager_mtp_cache[0, :, 2], name="changed draft cache"
    )
    changed_proposal_tables = changed_tables.to(torch.int64)
    changed_proposals = proposal_graph.replay(
        changed_draft, changed_position, changed_proposal_tables
    )
    eager_changed_proposals = eager_proposal_graph.replay(
        changed_draft, changed_position, changed_proposal_tables
    )
    torch.testing.assert_close(changed_proposals, eager_changed_proposals)
    _assert_numerically_close(
        mtp_cache[0, :, 3:6],
        eager_mtp_cache[0, :, 3:6],
        name="changed proposal cache",
    )
    assert {"prepare_attention", "planned_attention", "gdn_prefill"} <= {
        record.operation
        for record in selected_implementations()
        if record.phase == "compile"
    }
