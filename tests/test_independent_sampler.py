"""Distribution-level and speculative-prefix tests; no GPU or vLLM required."""

import unittest

import numpy as np
import torch
from oh_my_vllm.worker.sampler import RequestSampler, probabilities
from oh_my_vllm.worker.sampling import SamplingParams


class SamplerTest(unittest.TestCase):
    def test_penalties_and_speculative_prefix(self):
        logits = torch.tensor([[4.0, -2, 3, 1]]).repeat(3, 1)
        params = SamplingParams(
            10, repetition_penalty=2, frequency_penalty=0.5, presence_penalty=1
        )
        probs = probabilities(logits, params, [0, 1], [1], [2, 2])
        expected = torch.tensor(
            [[2.0, -5.5, 3, 1], [2, -5.5, 0, 1], [2, -5.5, -0.5, 1]]
        ).softmax(-1)
        torch.testing.assert_close(probs, expected)
        torch.testing.assert_close(logits[0], torch.tensor([4.0, -2, 3, 1]))

    def test_top_p_keeps_crossing_token_and_top_k_ties(self):
        logits = torch.tensor([[0.5, 0.3, 0.2]]).log()
        probs = probabilities(logits, SamplingParams(10, top_p=0.6), [], [])
        torch.testing.assert_close(probs, torch.tensor([[0.625, 0.375, 0]]))
        ties = probabilities(
            torch.tensor([[2.0, 1, 1, 0]]), SamplingParams(10, top_k=2), [], []
        )
        self.assertGreater(float(ties[0, 2]), 0)
        self.assertEqual(float(ties[0, 3]), 0)

    def test_mask_signed_word_boundary_and_greedy(self):
        logits = torch.arange(35.0)[None, :]
        mask = np.array([[-2147483648, 1]], dtype=np.int32)
        probs = probabilities(
            logits, SamplingParams(10, temperature=0), [], [], bitmask=mask
        )
        self.assertEqual(int(probs.argmax()), 32)
        self.assertEqual(float(probs.sum()), 1)

    def test_first_mismatch_bonus_and_unreachable_mask(self):
        sampler = RequestSampler(SamplingParams(10, temperature=0), [], "cpu")
        logits = torch.tensor([[0.0, 2, 1], [0, 1, 2], [2, 0, 1]])
        self.assertEqual(sampler.sample(logits, [1, 2]), [1, 2, 0])
        self.assertEqual(sampler.sample(logits, [1, 0]), [1, 2])
        mask = np.array([[7], [7], [0]], dtype=np.int32)
        self.assertEqual(sampler.sample(logits, [0, 0], mask), [1])
        with self.assertRaisesRegex(RuntimeError, "no valid token"):
            sampler.sample(logits, [1, 2], mask)
        sampler.commit([1])
        self.assertEqual(sampler.generated, [1])

    def test_nonfinite_reachable_rows_rejected(self):
        for temperature in (0, 1):
            for invalid in (float("nan"), float("inf"), -float("inf")):
                with self.subTest(temperature=temperature, invalid=invalid):
                    sampler = RequestSampler(
                        SamplingParams(10, temperature=temperature), [], "cpu"
                    )
                    bad = torch.full((1, 3), invalid)
                    with self.assertRaisesRegex(RuntimeError, "no valid token"):
                        sampler.sample(bad)
                    # Rejection in the first row makes a later bad row unreachable.
                    good = torch.tensor([[0.0, -torch.inf, -torch.inf]])
                    self.assertEqual(sampler.sample(torch.cat((good, bad)), [1]), [0])
            for invalid in (float("nan"), float("inf")):
                with self.assertRaisesRegex(RuntimeError, "no valid token"):
                    sampler.sample(torch.tensor([[0.0, invalid, 1.0]]))

    def test_seed_is_local_to_request(self):
        params = SamplingParams(100, seed=314)
        first, second = (RequestSampler(params, [], "cpu") for _ in range(2))
        other = RequestSampler(SamplingParams(100, seed=999), [], "cpu")
        logits = torch.tensor([[0.0, 0.3, -0.2]])
        outputs = []
        for _ in range(50):
            a = first.sample(logits)
            other.sample(logits)
            b = second.sample(logits)
            self.assertEqual(a, b)
            first.commit(a)
            second.commit(b)
            outputs.extend(a)
        self.assertGreater(len(set(outputs)), 1)

    def test_stochastic_verification_matches_target_distribution(self):
        # Exhaustively combine target-row outcomes rather than a flaky histogram.
        target = torch.tensor([[0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
        draft = [1, 0]
        outcomes = {}
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    logits = torch.full((3, 2), -torch.inf)
                    logits[range(3), [a, b, c]] = 0
                    sampler = RequestSampler(SamplingParams(10), [], "cpu")
                    tokens = tuple(sampler.sample(logits, draft))
                    mass = float(target[0, a] * target[1, b] * target[2, c])
                    outcomes[tokens] = outcomes.get(tokens, 0) + mass
        for tokens, expected in {
            (0,): 0.2,
            (1, 1): 0.8 * 0.3,
            (1, 0, 0): 0.8 * 0.7 * 0.4,
            (1, 0, 1): 0.8 * 0.7 * 0.6,
        }.items():
            self.assertAlmostEqual(outcomes[tokens], expected, places=6)


if __name__ == "__main__":
    unittest.main()
