import unittest
from types import SimpleNamespace

import numpy as np
import torch

from . import fa, ffn, gdn, mtp
from .decisions import DecisionMargins, cdf_margin
from .random import philox


class ZeroWeights:
    def tensor(self, name):
        return torch.zeros(2, dtype=torch.float64)

    def linear(self, value, prefix):
        return torch.zeros_like(value)


class Formulas(unittest.TestCase):
    def test_analytical_operator_and_rng_contracts(self):
        self.assertEqual(
            [philox(0, index) for index in range(4)],
            [0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8],
        )
        value = torch.tensor([[1.0, -2.0]], dtype=torch.float64)
        residual = torch.tensor([[0.25, 0.5]], dtype=torch.float64)
        output, summed = ffn.compute(value, residual, ZeroWeights(), "")
        torch.testing.assert_close(output, torch.zeros_like(value), rtol=0, atol=0)
        torch.testing.assert_close(summed, value + residual, rtol=0, atol=0)
        query = torch.tensor([[[1.0, 2.0]]], dtype=torch.float64)
        key = query * 3
        values = query * -2
        attended = fa.attend(query, key, values, torch.tensor([0]), first_token=0)
        torch.testing.assert_close(attended, values, rtol=0, atol=0)
        state = torch.eye(2, dtype=torch.float64)[None]
        q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        k = torch.tensor([[0.0, 1.0]], dtype=torch.float64)
        v = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
        updated, observed = gdn.step(
            state, q, k, v, torch.tensor([0.5]), torch.tensor([0.0])
        )
        torch.testing.assert_close(
            updated,
            torch.tensor([[[1.0, 1.0], [0.0, 2.0]]], dtype=torch.float64),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(observed, q / 2**0.5, rtol=0, atol=0)
        scale = gdn.projection_scales(
            torch.tensor([[3.0, 4.0]], dtype=torch.float64),
            torch.tensor([[1.0, 1.0], [0.0, 2.0]], dtype=torch.float64),
            [0, 1],
        )
        self.assertEqual(scale, [[5.0, 8.0]])
        target = torch.zeros((1, 5, 8), dtype=torch.float64)
        target[0, 0, 3] = target[0, 1, 7] = 1
        proposal = torch.zeros((1, 4, 8), dtype=torch.float64)
        proposal[0, 0, 3] = 1
        accepted = mtp.accept(
            SimpleNamespace(shape={"batch": 1}),
            {
                "target": target,
                "proposal": proposal,
                "drafts": torch.tensor([[3, -1, -1, -1, 1]]),
                "private_slots": torch.tensor([[10, 11, 12, 13, 14]]),
            },
            {"rng": torch.tensor([[0, 0]], dtype=torch.uint64)},
            None,
        )
        np.testing.assert_array_equal(accepted["tokens"], [[3, 7, -1, -1, -1]])
        np.testing.assert_array_equal(accepted["counts"], [2])
        np.testing.assert_array_equal(accepted["candidate_slots"], [11])
        np.testing.assert_array_equal(accepted["rng"], [[0, 4]])
        accepted = mtp.accept(
            SimpleNamespace(shape={"batch": 1}),
            {
                "target": torch.tensor([[3, 7, 2, 1, 0]]),
                "proposal": proposal,
                "drafts": torch.tensor([[3, 6, 2, 1, 1]]),
                "private_slots": torch.tensor([[10, 11, 12, 13, 14]]),
            },
            {"rng": torch.tensor([[0, 0]], dtype=torch.uint64)},
            None,
        )
        np.testing.assert_array_equal(accepted["tokens"], [[3, 7, -1, -1, -1]])
        np.testing.assert_array_equal(accepted["counts"], [2])
        np.testing.assert_array_equal(accepted["rng"], [[0, 6]])
        with DecisionMargins() as margins:
            cdf_margin(torch.tensor([0.25, 0.75], dtype=torch.float64), 0.5)
        self.assertEqual(margins.records[0]["distance"], 0.25)
        with self.assertRaisesRegex(ValueError, "ambiguous"), DecisionMargins():
            cdf_margin(torch.tensor([0.25, 0.75], dtype=torch.float64), 0.250001)
