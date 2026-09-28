"""Distribution-level and speculative-prefix tests; no GPU or vLLM required."""

import unittest
from unittest.mock import patch

import numpy as np
import torch
from oh_my_vllm.worker.sampler import RequestSampler, probabilities, verify_rows
from oh_my_vllm.worker.sampling import SamplingParams


def check_seeded_drafts_mask_partial_commit(device: str) -> None:
    params = SamplingParams(
        10,
        temperature=0.8,
        top_k=5,
        top_p=0.9,
        repetition_penalty=1.1,
        frequency_penalty=0.1,
        seed=784,
    )
    prompt = [1, 2, 2, 5]
    drafts = [1, 2]
    mask = np.array([[255], [0b01111111], [255]], dtype=np.int32)
    logits = torch.tensor(
        [
            [0, 14, 0, 0, 0, 0, 0, 0],
            [0, 0, 14, 0, 0, 0, 0, 0],
            [0, 0, 0, 2, 2, 1, 0, 0],
        ],
        device=device,
        dtype=torch.float32,
    )
    sampler = RequestSampler(params, prompt, device, vocab_size=8)
    reference_generator = torch.Generator(device=device).manual_seed(784)
    committed_history = []
    for _ in range(5):
        expected_probs = probabilities(
            logits, params, prompt, committed_history, drafts, mask
        )
        noise = torch.empty_like(expected_probs).exponential_(
            generator=reference_generator
        )
        selected = (expected_probs / noise).argmax(-1).tolist()
        expected = verify_rows([[token, 1] for token in selected], drafts)
        actual = sampler.sample(logits, drafts, mask)
        assert actual == expected
        assert len(actual) == 3
        sampler.commit(actual[:1])
        committed_history.extend(actual[:1])
        assert sampler.generated == committed_history
        assert sampler._history_counts[1].sum().item() == len(committed_history)


class SamplerTest(unittest.TestCase):
    def test_seeded_drafts_mask_and_partial_commit_use_only_accepted_history(self):
        check_seeded_drafts_mask_partial_commit("cpu")

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

    def test_top_p_preserves_stable_ties_inside_and_at_top_k_cutoff(self):
        params = SamplingParams(10, top_k=3, top_p=0.75)
        inside = probabilities(
            torch.tensor([[4.0, 3, 3, 1, 0, -1, -2, -3]]), params, [], []
        )
        self.assertGreater(float(inside[0, 1]), 0)
        self.assertEqual(float(inside[0, 2]), 0)
        cutoff = probabilities(
            torch.tensor([[4.0, 3, 3, 1, 0, -1, -2, -3]]),
            SamplingParams(10, top_k=2, top_p=0.75),
            [],
            [],
        )
        torch.testing.assert_close(cutoff, inside)

    def test_top_k_top_p_matches_full_stable_sort_including_invalid_rows(self):
        generator = torch.Generator().manual_seed(784)
        logits = torch.randn(16, 64, generator=generator)
        logits[0, :3] = torch.tensor([4.0, 3, 3])
        logits[1, 0] = float("nan")
        logits[2, 0] = float("inf")
        logits[3].fill_(-torch.inf)
        params = SamplingParams(10, top_k=7, top_p=0.83)
        actual = torch.cat(
            [probabilities(row[None], params, [], []) for row in logits], dim=0
        )
        scores = logits.float().clone()
        threshold = scores.topk(params.top_k, dim=-1).values[:, -1:]
        scores.masked_fill_(scores < threshold, -torch.inf)
        sorted_scores, indices = scores.sort(dim=-1, descending=True, stable=True)
        sorted_probs = sorted_scores.softmax(-1)
        remove = sorted_probs.cumsum(-1) - sorted_probs >= params.top_p
        sorted_scores.masked_fill_(remove, -torch.inf)
        scores.scatter_(1, indices, sorted_scores)
        expected = scores.softmax(-1)
        torch.testing.assert_close(
            actual, expected, rtol=1e-6, atol=1e-8, equal_nan=True
        )

    def test_cached_penalties_match_legacy_across_commits_and_drafts(self):
        params = SamplingParams(
            10,
            temperature=0.7,
            repetition_penalty=1.2,
            frequency_penalty=0.3,
            presence_penalty=-0.1,
            top_k=5,
            top_p=0.8,
            seed=784,
        )
        prompt = [0, 1, 1, 2, 7] * 50
        sampler = RequestSampler(params, prompt, "cpu", vocab_size=8)
        logits = torch.tensor(
            [[3.0, -2, 1, 0, 2, 4, 1, -1], [2.0, -1, 4, 3, 0, 1, 2, -2]]
        )
        mask = np.array([[255], [255]], dtype=np.int32)
        for committed in ([], [2, 2], [6]):
            sampler.commit(committed)
            expected = probabilities(
                logits, params, prompt, sampler.generated, [1], mask
            )
            cached = probabilities(
                logits,
                params,
                prompt,
                sampler.generated,
                [1],
                mask,
                history_counts=sampler._history_counts,
            )
            torch.testing.assert_close(cached, expected, rtol=0, atol=0)
            assert sampler._history_counts[1][1].item() == 0

    def test_request_sampler_reuses_prompt_counts_and_seeded_draws(self):
        params = SamplingParams(10, seed=42, repetition_penalty=1.1, top_k=3, top_p=0.8)
        prompt = [0, 1, 2] * 100
        with patch("torch.bincount", wraps=torch.bincount) as bincount:
            sampler = RequestSampler(params, prompt, "cpu", vocab_size=8)
            self.assertEqual(bincount.call_count, 1)
            reference_generator = torch.Generator(device="cpu").manual_seed(42)
            logits = torch.tensor([[1.0, 4, 2, 3, 0, -1, -2, -3]])
            generated = []
            for _ in range(6):
                expected = probabilities(logits, params, prompt, generated)
                noise = torch.empty_like(expected).exponential_(
                    generator=reference_generator
                )
                token = verify_rows([[int((expected / noise).argmax()), 1]], [])
                self.assertEqual(sampler.sample(logits), token)
                sampler.commit(token)
                generated.extend(token)
                # The reference call above recomputes counts once per history.
                self.assertEqual(bincount.call_count, 1 + 2 * len(generated))

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
