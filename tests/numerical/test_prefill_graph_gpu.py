"""Prefill captures restore mutations and replay new logical page addresses."""

import pytest
import torch
from oh_my_vllm.kernels.attention import PagedAttention
from oh_my_vllm.models.qwen import Batch
from oh_my_vllm.worker.prefill_graph import PrefillGraph

pytestmark = pytest.mark.gpu


class ToyTarget:
    def forward(self, tokens, batch, caches):
        conv, state = caches[0]
        update = tokens.float().sum().reshape(1, 1, 1)
        state.index_copy_(0, batch.final_state_writes, update.expand(1, 1, 1))
        conv.index_copy_(0, batch.final_state_writes, (update + 1).expand(1, 1, 1))
        cache = caches[1]
        pages, offsets = batch.fa_slots // 784, batch.fa_slots % 784
        values = tokens[:, None, None, None].to(torch.bfloat16).expand(-1, 2, 4, 256)
        cache[pages, :, offsets] = values
        query = torch.ones(
            len(tokens), 24, 256, device=tokens.device, dtype=torch.bfloat16
        )
        return batch.attention(query, cache)


@pytest.mark.parametrize("rows", [624, 1024])
@torch.inference_mode()
def test_prefill_graph_matches_eager_and_replays_changed_pages(rows):
    context = 8192
    total = context + rows
    pages = (total + 783) // 784
    cache = torch.zeros(
        2 * pages + 1, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16
    )
    cache[1 : pages + 1, 1] = 1
    cache[pages + 1 :, 1] = 2
    state = (torch.zeros(3, 1, 1, device="cuda"), torch.zeros(3, 1, 1, device="cuda"))
    caches = [state, cache]
    attention = PagedAttention()
    starts = torch.tensor([0, rows], dtype=torch.int32)
    page_starts = torch.tensor([0, pages], dtype=torch.int32)
    table = torch.arange(1, pages + 1, dtype=torch.int32)
    last = torch.tensor([(total - 1) % 784 + 1], dtype=torch.int32)
    plan = starts, page_starts, table, last
    attention.plan(*plan, 24, 4, 256)
    positions = torch.arange(context, total, device="cuda")
    tokens = torch.ones(rows, device="cuda", dtype=torch.int64)
    writes = torch.full_like(tokens, -1)
    writes[-1] = 2
    batch = Batch(
        positions=positions,
        fa_slots=table.to("cuda").long()[positions // 784] * 784 + positions % 784,
        attention=attention,
        starts=starts.to("cuda"),
        sequence_ids=torch.zeros_like(tokens),
        state_reads=torch.tensor([1], device="cuda"),
        state_writes=writes,
        final_state_writes=torch.tensor([2], device="cuda"),
        prefill_sequences=1,
        prefill_tokens=rows,
    )
    old = cache.clone(), state[0].clone(), state[1].clone()
    graph = PrefillGraph(ToyTarget(), caches, tokens, batch, plan)
    for current, original in zip((cache, *state), old, strict=True):
        torch.testing.assert_close(current, original, rtol=0, atol=0)
    for changed in (False, True):
        current_table = table + pages if changed else table
        current_plan = starts, page_starts, current_table, last
        batch.fa_slots = (
            current_table.to("cuda").long()[positions // 784] * 784 + positions % 784
        )
        tokens.fill_(2 if changed else 1)
        attention.plan(*current_plan, 24, 4, 256)
        expected_caches = [tuple(value.clone() for value in state), cache.clone()]
        expected = ToyTarget().forward(tokens, batch, expected_caches)
        actual, features = graph.replay(tokens, batch, current_plan)
        assert features is None
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for current, reference in zip(state, expected_caches[0], strict=True):
            torch.testing.assert_close(current, reference, rtol=0, atol=0)
        torch.testing.assert_close(cache, expected_caches[1], rtol=0, atol=0)
