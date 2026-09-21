"""Graph capture must be invisible to mutable state and dynamic addresses."""

import unittest

import torch
from oh_my_vllm.models.qwen import AttentionBatch, Batch
from oh_my_vllm.worker.decode_graph import DecodeGraph, DraftGraph


class ToyModel:
    def forward(self, tokens, batch, caches):
        conv, state = caches[0]
        value = state[batch.state_reads] + tokens[:, None].float()
        state[batch.state_writes] = value
        conv[batch.state_writes] = value + 1
        cache = caches[1]
        pages, offsets = batch.fa_slots // 784, batch.fa_slots % 784
        cache[pages, :, offsets] = value[:, None, :, None]
        return value + batch.positions[:, None] + batch.attention.tables[:, :1]

    def draft(self, tokens, hidden, batch, cache):
        value = hidden + tokens[:, None] + batch.positions[:, None]
        pages, offsets = batch.fa_slots // 784, batch.fa_slots % 784
        cache[pages, :, offsets] = value[:, None, :, None]
        return value + batch.attention.tables[:, :1]

    def logits(self, hidden):
        return hidden * 2


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class DecodeGraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_draft_capture_and_recurrent_output_alias(self):
        device = "cuda"
        cache = torch.zeros(4, 2, 784, 1, 1, device=device)
        tokens = torch.tensor([3], device=device)
        hidden = torch.tensor([[10.0]], device=device)
        batch = AttentionBatch(
            torch.tensor([7], device=device),
            torch.tensor([784 + 7], device=device),
            None,
        )
        tables = torch.tensor([[1]], device=device)
        graph = DraftGraph(ToyModel(), cache, tokens, hidden, batch, tables, 784)
        self.assertEqual(cache.count_nonzero().item(), 0)
        output = graph.replay(tokens, hidden, batch, tables)
        self.assertEqual(output.item(), 21)
        batch.positions.fill_(8)
        batch.fa_slots.fill_(784 + 8)
        # The next draft consumes the previous replay's output from this graph.
        output = graph.replay(tokens, output, batch, tables)
        self.assertEqual(output.item(), 33)
        self.assertEqual(cache[1, 0, 7].item(), 20)
        self.assertEqual(cache[1, 0, 8].item(), 32)

    @torch.inference_mode()
    def test_capture_restore_and_dynamic_replay(self):
        device = "cuda"
        pools = (torch.zeros(4, 1, device=device), torch.zeros(4, 1, device=device))
        pools[1][1] = 10
        fa = torch.zeros(4, 2, 784, 1, 1, device=device)
        caches = [pools, fa]
        tensor = lambda x: torch.tensor(x, device=device)  # noqa: E731
        batch = Batch(
            positions=tensor([7]),
            fa_slots=tensor([784 + 7]),
            attention=None,
            starts=tensor([0, 1]),
            sequence_ids=tensor([0]),
            state_reads=tensor([1]),
            state_writes=tensor([1]),
            final_state_writes=tensor([1]),
            prefill_sequences=0,
            prefill_tokens=0,
        )
        tokens, tables = tensor([3]), tensor([[1]])
        before = [x.clone() for x in (*pools, fa)]
        graph = DecodeGraph(ToyModel(), caches, tokens, batch, tables, 784)
        for actual, expected in zip((*pools, fa), before, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        hidden, logits = graph.replay(tokens, batch, tables)
        self.assertEqual(hidden.item(), 21)
        self.assertEqual(logits.item(), 42)
        self.assertEqual(pools[1][1].item(), 13)
        self.assertEqual(fa[1, 0, 7].item(), 13)
        batch.state_reads.fill_(1)
        batch.state_writes.fill_(2)
        batch.positions.fill_(11)
        batch.fa_slots.fill_(2 * 784 + 11)
        hidden, _ = graph.replay(tensor([4]), batch, tensor([[2]]))
        self.assertEqual(hidden.item(), 30)
        self.assertEqual(pools[1][1].item(), 13)
        self.assertEqual(pools[1][2].item(), 17)
        self.assertEqual(fa[1, 0, 7].item(), 13)
        self.assertEqual(fa[2, 0, 11].item(), 17)


if __name__ == "__main__":
    unittest.main()
