import unittest

import numpy as np

from . import sampling_reference as reference


class SamplingReference(unittest.TestCase):
    def test_rejection_at_each_position_and_all_accept_commit_only_inputs(self):
        proposal = np.tile([0.75, 0.25, 0.0], (5, 1))
        target = np.tile([0.25, 0.25, 0.5], (6, 1))
        original = target.copy(), proposal.copy()
        for rejected in range(5):
            draws = np.zeros(5)
            draws[rejected] = 0.5
            result = reference.verify(target, proposal, [0] * 5, draws, [0.0] * 5, 0.0)
            self.assertEqual(result.accepted, rejected)
            self.assertEqual(result.tokens, (0,) * rejected + (2,))
            self.assertEqual(result.processed_rows, rejected + 1)
        result = reference.verify(
            target, proposal, [0] * 5, np.zeros(5), [0.0] * 5, 0.9
        )
        self.assertEqual(result.tokens, (0, 0, 0, 0, 0, 2))
        self.assertEqual(result.processed_rows, 6)
        np.testing.assert_array_equal(target, original[0])
        np.testing.assert_array_equal(proposal, original[1])
        result = reference.verify(target[:1], proposal[:0], [], [], [], 0.3)
        self.assertEqual(result.tokens, (1,))
        self.assertEqual(result.processed_rows, 1)
        greedy = reference.verify_greedy(
            np.asarray([[0.5, 0.5], [0.1, 0.9], [1.0, 0.0]]), [0, 0]
        )
        self.assertEqual(greedy.tokens, (0, 1))
        self.assertEqual(greedy.accepted, 1)

    def test_exact_residual_mass_and_zero_probability_boundaries(self):
        target = np.asarray([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
        proposal = np.asarray([[0.5, 0.4, 0.1]])
        # The residual is entirely token 2, not a scaled subtraction of q.
        for draw in (0.0, 0.5, np.nextafter(1.0, 0.0)):
            result = reference.verify(target, proposal, [0], [0.8], [draw], 0.0)
            self.assertEqual(result.tokens, (2,))
        probability = np.zeros(3)
        for token, q in enumerate(proposal[0]):
            acceptance = min(1, target[0, token] / q)
            probability[token] += q * acceptance
            if acceptance < 1:
                result = reference.verify(target, proposal, [token], [0.9], [0.5], 0.0)
                probability[result.tokens[0]] += q * (1 - acceptance)
        np.testing.assert_allclose(probability, target[0], rtol=0, atol=1e-15)
        zero_target = np.asarray([[0.0, 1.0], [1.0, 0.0]])
        self.assertEqual(
            reference.verify(
                zero_target, np.asarray([[1.0, 0.0]]), [0], [0.0], [0.0], 0.0
            ).tokens,
            (1,),
        )
        with self.assertRaisesRegex(ValueError, "proposal.*zero"):
            reference.verify(
                zero_target, np.asarray([[0.0, 1.0]]), [0], [0.5], [0.5], 0.0
            )
        for bad in (np.nan, -0.1, 1.0):
            with self.subTest(draw=bad), self.assertRaises(ValueError):
                reference.verify(target, proposal, [0], [bad], [0.5], 0.0)
        with self.assertRaisesRegex(ValueError, "sum"):
            reference.verify(target * 2, proposal, [0], [0.5], [0.5], 0.0)

    def test_philox_known_vectors_and_request_local_counter_addressing(self):
        self.assertEqual(
            reference.philox((0, 0, 0, 0), (0, 0)),
            (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8),
        )
        self.assertEqual(
            reference.philox((0xFFFFFFFF,) * 4, (0xFFFFFFFF,) * 2),
            (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD),
        )
        kwargs = {"seed": 17, "cycle": 23, "position": 400000}
        draws = {
            stream: reference.uniform(**kwargs, stream=stream)
            for stream in reference.STREAMS
        }
        self.assertEqual(len(set(draws.values())), 5)
        for stream, expected in reversed(list(draws.items())):
            self.assertEqual(reference.uniform(**kwargs, stream=stream), expected)
            self.assertTrue(0 <= expected < 1)
        self.assertNotEqual(
            reference.uniform(seed=17, cycle=24, position=400000, stream="proposal"),
            draws["proposal"],
        )
        for key, value in (("seed", -1), ("cycle", 1 << 32), ("position", 1 << 64)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                reference.uniform(**{**kwargs, key: value}, stream="proposal")
        target = np.asarray([[0.25, 0.25, 0.5], [0.25, 0.75, 0]])
        proposal = np.asarray([[0.75, 0.25, 0]])
        expected = reference.verify(
            target,
            proposal,
            [0],
            [reference.uniform(seed=17, cycle=23, position=400001, stream="accept")],
            [
                reference.uniform(
                    seed=17, cycle=23, position=400001, stream="correction"
                )
            ],
            reference.uniform(seed=17, cycle=23, position=400002, stream="bonus"),
        )
        self.assertEqual(
            reference.verify_seeded(
                target, proposal, [0], seed=17, cycle=23, pending_position=400000
            ),
            expected,
        )


if __name__ == "__main__":
    unittest.main()
