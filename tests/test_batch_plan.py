"""CPU checks of the Rust allocation contract at hybrid/MTP boundaries."""

import unittest

from oh_my_vllm.worker.batch_plan import plan_request, validate_batch
from oh_my_vllm.worker.protocol import ScheduledRequest


def request(start, tokens, mamba, history=None, rid=1, fa_table=None):
    pages = (start + len(tokens) + 783) // 784
    fa = fa_table if fa_table is not None else list(range(1, pages + 1))
    return ScheduledRequest(rid, tokens, start, fa, mamba, history)


class BatchPlanTest(unittest.TestCase):
    def test_intermediate_prefill_and_final_sample(self):
        history = list(range(1700))
        first = request(0, history[:1568], [0, 7, 8, 9, 10, 11], history)
        plan = plan_request(first, history, None, 128, 4)
        self.assertEqual(plan.sample_indices, [])
        self.assertEqual(plan.writes, [-1] * 1567 + [7])
        self.assertEqual(plan.commit(0), (1568, 7, []))
        last = request(1568, history[1568:], [0, 7, 8, 9, 10, 11, 12])
        plan = plan_request(last, history, 7, 128, 4)
        self.assertEqual(plan.sample_indices, [131])
        self.assertEqual(plan.commit(1), (132, 8, []))

    def test_cross_boundary_all_acceptance_lengths(self):
        history = list(range(783))
        req = request(782, [782, 1000, 1001, 1002, 1003], [10, 20, 21, 22, 23, 24])
        plan = plan_request(req, history, 10, 128, 4)
        self.assertEqual(plan.drafts, [1000, 1001, 1002, 1003])
        for accepted in range(1, 6):
            count, source, copies = plan.commit(accepted)
            self.assertEqual(count, accepted)
            self.assertEqual(source, 19 + accepted)
            self.assertEqual(copies, [] if accepted == 1 else [(21, 10)])

    def test_prefix_hit_reads_shared_but_writes_private(self):
        history = list(range(785))
        req = request(784, [784], [10, 20, 21, 22, 23, 24], history)
        plan = plan_request(req, history, None, 128, 4)
        self.assertEqual(plan.source, 10)
        self.assertEqual(plan.writes, [20])
        self.assertEqual(plan.commit(1), (1, 20, []))
        req.mamba_block_table[1] = 10
        with self.assertRaisesRegex(ValueError, "shared prefix"):
            plan_request(req, history, None, 128, 4)

    def test_shared_reads_and_checkpoint_write_conflicts(self):
        history = list(range(785))
        a = plan_request(
            request(784, [784], [10, 20], history, 1, fa_table=[1, 2]),
            history,
            None,
            128,
            0,
        )
        b = plan_request(
            request(784, [784], [10, 30], history, 2, fa_table=[1, 3]),
            history,
            None,
            128,
            0,
        )
        validate_batch([a, b])
        b.source = 20
        with self.assertRaisesRegex(ValueError, "another request"):
            validate_batch([a, b])
        # A checkpoint copy is also a write, separate from candidate destinations.
        # Give crossing a distinct FA tail page (page 5) so it doesn't conflict
        # with request a's tail page (page 2) before the recurrent-state check.
        crossing = plan_request(
            request(782, [782, 1000, 1001], [10, 40, 41, 42], rid=3, fa_table=[1, 5]),
            list(range(783)),
            50,
            128,
            4,
        )
        with self.assertRaisesRegex(ValueError, "another request"):
            validate_batch([a, crossing])

    def test_invalid_admission_history_and_addresses(self):
        for mutate in (
            lambda r: setattr(r, "prefill_token_ids", None),
            lambda r: setattr(r, "fa_block_table", [0]),
            lambda r: setattr(r, "mamba_block_table", [128]),
            lambda r: setattr(r, "token_ids", [99]),
        ):
            req = request(0, [1], [2], [1])
            mutate(req)
            with self.assertRaises(ValueError):
                plan_request(req, [1], None, 128, 0)
        req = request(784, [784], [], list(range(785)))
        with self.assertRaisesRegex(ValueError, "prefix checkpoint"):
            plan_request(req, list(range(785)), None, 128, 4)

    def test_fa_page_sharing_detected_by_debug_assertion(self):
        # PY-01: two requests with the same writable FA tail page must raise.
        history = list(range(785))
        a = plan_request(
            request(784, [784], [10, 20], history, 1, fa_table=[1, 99]),
            history,
            None,
            128,
            0,
        )
        b = plan_request(
            request(784, [784], [10, 30], history, 2, fa_table=[1, 99]),
            history,
            None,
            128,
            0,
        )
        with self.assertRaises(AssertionError):
            validate_batch([a, b])

    def test_retained_output_count_is_bounded(self):
        history = [1]
        plan = plan_request(request(0, history, [2], history), history, None, 128, 0)
        for count in (0, 2):
            with self.assertRaises(ValueError):
                plan.commit(count)


if __name__ == "__main__":
    unittest.main()
