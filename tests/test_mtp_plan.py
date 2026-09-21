"""MTP boundary planning independently of model/GPU arithmetic."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.worker.batch_plan import PlannedRequest
from oh_my_vllm.worker.mtp import MTP
from oh_my_vllm.worker.protocol import RequestOutput, ScheduledRequest


class MTPPlanTest(unittest.TestCase):
    def setUp(self):
        model = SimpleNamespace(
            mtp=object(),
            embedding=torch.zeros(1),
            logits=lambda hidden: torch.zeros(len(hidden), 3),
        )
        with patch("oh_my_vllm.worker.mtp.MTPAttention"):
            self.mtp = MTP(model, 8, 4096)
        self.calls = []

        def run(tokens, hidden, starts, tables, positions):
            self.calls.append(
                (list(tokens), hidden.clone(), list(starts), tables, list(positions))
            )
            return torch.zeros(len(tokens), 5120, dtype=torch.bfloat16)

        self.mtp._run = run

    def step(self, start, count, pages, rid=1, history=None, output=True):
        history = history or list(range(start + count + 1))
        req = ScheduledRequest(rid, history[start : start + count], start, pages, [])
        plan = PlannedRequest(req, 0, [], [], [], False)
        hidden = torch.arange(start, start + count, dtype=torch.float32)[:, None]
        hidden = hidden.expand(-1, 5120).to(torch.bfloat16).contiguous()
        result = RequestOutput(rid, [history[start + count]] if output else [])
        return self.mtp.propose(
            [plan], [0, count], [count], hidden, {rid: history}, [result]
        )

    def test_full_page_defers_boundary_then_restores(self):
        for boundary, pages in [(784, [1]), (1568, [1, 2])]:
            self.setUp()
            self.assertEqual(self.step(0, boundary, pages, output=False), {1: []})
            self.assertEqual(self.calls[0][4], list(range(1, boundary)))
            self.assertEqual(self.mtp.next_position[1], boundary)
            saved = self.mtp.boundary_hidden[pages[-1]].clone()
            self.calls.clear()
            proposals = self.step(boundary, 1, [*pages, 3])
            self.assertEqual(len(proposals[1]), 4)
            self.assertEqual(self.calls[0][4], [boundary, boundary + 1])
            torch.testing.assert_close(self.calls[0][1][0], saved)
            self.assertEqual(self.mtp.next_position[1], boundary + 2)

    def test_same_prefix_different_suffix_does_not_write_shared_page(self):
        self.step(0, 784, [1], output=False)
        saved = self.mtp.boundary_hidden[1].clone()
        for rid, suffix, page in [(2, [10000, 10001], 2), (3, [20000, 20001], 3)]:
            self.calls.clear()
            self.step(784, 1, [1, page], rid=rid, history=list(range(784)) + suffix)
            self.assertEqual(self.calls[0][0], suffix)
            torch.testing.assert_close(self.calls[0][1][0], saved)
            for _, _, _, tables, positions in self.calls:
                self.assertTrue(all(tables[0][p // 784] == page for p in positions))
        torch.testing.assert_close(self.mtp.boundary_hidden[1], saved)

    def test_rejected_or_fully_accepted_input_advances_true_prefix_only(self):
        for count in (1, 5):
            self.mtp.next_position[1] = 101
            self.calls.clear()
            self.step(100, count, [1])
            self.assertEqual(self.calls[0][4], list(range(101, 101 + count)))
            # Speculative rows written by later draft steps do not advance this marker.
            self.assertEqual(self.mtp.next_position[1], 101 + count)

    def test_page_capacity_shortens_drafts(self):
        for computed, expected in [(781, 3), (783, 1), (784, 0)]:
            self.mtp.next_position[1] = computed
            self.calls.clear()
            proposals = self.step(computed - 1, 1, [1])
            self.assertEqual(len(proposals[1]), expected)
            for _, _, _, _, positions in self.calls:
                self.assertLess(max(positions), 784)


if __name__ == "__main__":
    unittest.main()
