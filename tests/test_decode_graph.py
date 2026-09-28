"""Graph capture must be invisible to mutable state and dynamic addresses."""

import unittest

import pytest
import torch
from oh_my_vllm.models.qwen import AttentionBatch, Batch
from oh_my_vllm.worker.decode_graph import DecodeGraph, DraftGraph
from oh_my_vllm.worker.tensors import device_page_tables

pytestmark = pytest.mark.gpu


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
    def test_shared_draft_pool_copies_previous_hidden_before_other_replay(self):
        cache = torch.zeros(4, 2, 784, 1, 1, device="cuda")
        pool = torch.cuda.graph_pool_handle()

        def capture(extent, position):
            tokens = torch.tensor([3], device="cuda")
            hidden = torch.tensor([[10.0]], device="cuda")
            batch = AttentionBatch(
                torch.tensor([position], device="cuda"),
                torch.tensor([784 + position], device="cuda"),
                None,
            )
            tables = torch.tensor([[1]], device="cuda")
            graph = DraftGraph(
                ToyModel(), cache, tokens, hidden, batch, tables, extent, pool=pool
            )
            return graph, tokens, hidden, batch, tables

        first = capture(784, 7)
        second = capture(1568, 8)
        output = first[0].replay(*first[1:])
        # The copy into second.input_hidden is queued before second.graph.replay.
        # A Python reference to first's output alone would not enforce this.
        output = second[0].replay(second[1], output, second[3], second[4])
        output = first[0].replay(first[1], output, first[3], first[4])
        # No host read/sync is allowed between the three replays.
        torch.testing.assert_close(output, torch.tensor([[44.0]], device="cuda"))
        assert cache[1, 0, 7].item() == 43
        assert cache[1, 0, 8].item() == 32

    @torch.inference_mode()
    def test_shared_target_pool_keeps_live_outputs_across_reverse_replay(self):
        class Model:
            def forward(self, tokens, batch, caches):
                return tokens[:, None].float() + batch.positions[:, None]

            def logits(self, hidden):
                return hidden * 2

        def capture(count, pool):
            tokens = torch.full((count,), count, device="cuda")
            positions = torch.arange(count, device="cuda")
            batch = Batch(
                positions=positions,
                fa_slots=torch.zeros(count, device="cuda", dtype=torch.int64),
                attention=None,
                starts=torch.arange(count + 1, device="cuda", dtype=torch.int32),
                sequence_ids=torch.arange(count, device="cuda"),
                state_reads=torch.zeros(count, device="cuda", dtype=torch.int64),
                state_writes=torch.full((count,), -1, device="cuda"),
                final_state_writes=torch.full((count,), -1, device="cuda"),
                prefill_sequences=0,
                prefill_tokens=0,
            )
            tables = torch.ones((count, 1), device="cuda", dtype=torch.int32)
            graph = DecodeGraph(Model(), [], tokens, batch, tables, 784, pool=pool)
            return graph, tokens, batch, tables

        pool = torch.cuda.graph_pool_handle()
        one = capture(1, pool)
        two = capture(2, pool)
        hidden_one, logits_one = one[0].replay(*one[1:])
        saved_one = hidden_one.clone(), logits_one.clone()
        hidden_two, logits_two = two[0].replay(*two[1:])
        # A saved output can be read after a different graph replays. The
        # original static output is not promised to remain untouched.
        torch.testing.assert_close(saved_one[0], torch.tensor([[1.0]], device="cuda"))
        torch.testing.assert_close(saved_one[1], saved_one[0] * 2)
        torch.testing.assert_close(
            hidden_two, torch.tensor([[2.0], [3.0]], device="cuda")
        )
        torch.testing.assert_close(logits_two, hidden_two * 2)
        hidden_one, _ = one[0].replay(*one[1:])
        torch.testing.assert_close(hidden_one, saved_one[0], rtol=0, atol=0)

    @torch.inference_mode()
    def test_target_replay_copies_compact_expanded_page_tables(self):
        class TableModel:
            def forward(self, tokens, batch, caches):
                return batch.attention.tables[:, :1].float() + tokens[:, None]

            def logits(self, hidden):
                return hidden * 2

        device = "cuda"
        tokens = torch.tensor([1, 2, 3], device=device)
        batch = Batch(
            positions=torch.tensor([783, 784, 785], device=device),
            fa_slots=torch.zeros(3, device=device, dtype=torch.int64),
            attention=None,
            starts=torch.tensor([0, 2, 3], device=device, dtype=torch.int32),
            sequence_ids=torch.tensor([0, 0, 1], device=device),
            state_reads=torch.zeros(3, device=device, dtype=torch.int64),
            state_writes=torch.full((3,), -1, device=device),
            final_state_writes=torch.full((2,), -1, device=device),
            prefill_sequences=0,
            prefill_tokens=0,
        )
        tables = device_page_tables(
            [[11, 12], [21, 22]], 2, counts=[2, 1], device=device
        )
        graph = DecodeGraph(TableModel(), [], tokens, batch, tables, 1568)
        hidden, logits = graph.replay(tokens, batch, tables)
        torch.testing.assert_close(
            hidden[:, 0], torch.tensor([12, 13, 24.0], device=device)
        )
        torch.testing.assert_close(logits, hidden * 2)

        batch.starts.copy_(torch.tensor([0, 1, 3], device=device))
        batch.sequence_ids.copy_(torch.tensor([0, 1, 1], device=device))
        changed = device_page_tables(
            [[31, 32], [41, 42]], 2, counts=[1, 2], device=device
        )
        hidden, logits = graph.replay(tokens, batch, changed)
        torch.testing.assert_close(
            hidden[:, 0], torch.tensor([32, 43, 44.0], device=device)
        )
        torch.testing.assert_close(logits, hidden * 2)

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


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class GroupedDraftGraphTest(unittest.TestCase):
    @torch.inference_mode()
    def test_first_one_and_dynamic_regrouping(self):
        from oh_my_vllm.kernels.decode_attention import decode

        from tests.test_independent_decode_attention import reference

        class AttentionModel:
            def draft(self, tokens, hidden, batch, cache):
                return batch.attention(hidden.view(-1, 24, 256), cache).flatten(1)

        torch.manual_seed(94)
        cache = torch.randn(5, 2, 784, 4, 256, device="cuda", dtype=torch.bfloat16)
        hidden = torch.randn(5, 24 * 256, device="cuda", dtype=torch.bfloat16)
        tokens = torch.ones(5, device="cuda", dtype=torch.int64)
        tables = device_page_tables([[1, 3], [2, 4]], 2, counts=[2, 3], device="cuda")
        positions = torch.tensor([780, 781, 783, 784, 785], device="cuda")
        batch = AttentionBatch(positions, torch.arange(5, device="cuda"), None)
        starts = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
        graph = DraftGraph(
            AttentionModel(), cache, tokens, hidden, batch, tables, 1568, starts
        )
        for regroup in (False, True):
            if regroup:
                starts.copy_(torch.tensor([0, 4, 5], device="cuda"))
                tables.copy_(
                    device_page_tables(
                        [[2, 4], [1, 3]], 2, counts=[4, 1], device="cuda"
                    )
                )
                positions.copy_(torch.tensor([782, 783, 784, 785, 44], device="cuda"))
                hidden.mul_(0.5)
            actual = graph.replay(tokens, hidden, batch, tables, starts)
            expected = decode(
                hidden.view(-1, 24, 256),
                cache,
                tables,
                positions + 1,
                first=1,
                max_tokens=1568,
            ).flatten(1)
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
            absolute = reference(
                hidden.view(-1, 24, 256), cache, tables, positions + 1, first=1
            )
            torch.testing.assert_close(
                actual.cpu().double().view(-1, 24, 256),
                absolute,
                atol=0.03,
                rtol=0.03,
            )


if __name__ == "__main__":
    unittest.main()
