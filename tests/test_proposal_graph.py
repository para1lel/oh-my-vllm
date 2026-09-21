"""Proposal capture is transactional and keeps every replay input alive."""

import pytest
import torch
from oh_my_vllm.models.qwen import AttentionBatch
from oh_my_vllm.worker.decode_graph import ProposalGraph

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class ToyProposal:
    def logits(self, hidden):
        return hidden

    def draft(self, tokens, hidden, batch, cache):
        value = torch.roll(hidden + tokens[:, None] + batch.positions[:, None], 1, 1)
        pages, offsets = batch.fa_slots // 784, batch.fa_slots % 784
        cache[pages, :, offsets] = value[:, :1, None, None]
        return value


@torch.inference_mode()
def test_proposal_capture_dynamic_tables_and_allocator_reuse():
    device = "cuda"
    model = ToyProposal()
    cache = torch.randn(5, 2, 784, 1, 1, device=device)
    hidden = torch.tensor([[1.0, 3.0, 2.0], [3.0, 1.0, 2.0]], device=device)
    positions = torch.tensor([782, 779], device=device)
    tables = torch.tensor([[1, 3], [2, 4]], device=device)
    before = cache.clone()
    graph = ProposalGraph(model, cache, hidden, positions, tables, 1568)
    torch.testing.assert_close(cache, before, rtol=0, atol=0)
    # Force reuse of small allocations; a local-only row index formerly dangled.
    pressure = [
        torch.full((2,), 99999, device=device, dtype=torch.int64) for _ in range(64)
    ]
    for changed in [False, True]:
        if changed:
            tables.copy_(torch.tensor([[2, 4], [1, 3]], device=device))
            positions.copy_(torch.tensor([44, 783], device=device))
            hidden.copy_(
                torch.tensor([[9.0, 1.0, 3.0], [2.0, 7.0, 1.0]], device=device)
            )
        expected_cache = cache.clone()
        current = hidden
        token = model.logits(current).argmax(-1)
        expected = [token]
        for step in range(1, 4):
            pos = positions + step
            slots = tables[torch.arange(2, device=device), pos // 784] * 784 + pos % 784
            batch = AttentionBatch(pos, slots, None)
            current = model.draft(token, current, batch, expected_cache)
            token = model.logits(current).argmax(-1)
            expected.append(token)
        actual = graph.replay(hidden, positions, tables)
        torch.testing.assert_close(actual, torch.stack(expected, -1), rtol=0, atol=0)
        torch.testing.assert_close(cache, expected_cache, rtol=0, atol=0)
    assert pressure[0][0].item() == 99999
