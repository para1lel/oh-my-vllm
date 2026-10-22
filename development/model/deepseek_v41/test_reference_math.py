import math
import unittest

import torch

from . import reference_math as reference


class ReferenceMathematics(unittest.TestCase):
    def test_hyper_connection_uses_source_to_destination_mixing(self):
        residual = torch.tensor([[[2.0, 3.0], [5.0, 7.0]]], dtype=torch.float64)
        value = torch.tensor([[11.0, 13.0]], dtype=torch.float64)
        post = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
        comb = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float64)
        actual = reference.hc_expand(value, residual, post, comb)
        torch.testing.assert_close(
            actual,
            torch.tensor([[[23.7, 28.4], [35.4, 42.4]]], dtype=torch.float64),
            rtol=1e-14,
            atol=1e-14,
        )
        torch.testing.assert_close(
            reference.hc_collapse(
                residual, torch.tensor([[1.0, 0.0]], dtype=torch.float64)
            ),
            torch.tensor([[2.0, 3.0]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            reference.rms(
                torch.tensor([[2.0, 6.0]], dtype=torch.float64),
                torch.tensor([2.0, 3.0], dtype=torch.float64),
                eps=5.0,
            ),
            torch.tensor([[0.8, 3.6]], dtype=torch.float64),
        )
        pre, post, comb = reference.hc_mixes(
            residual,
            torch.zeros((8, 4), dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            torch.zeros(8, dtype=torch.float64),
            iterations=1,
            eps=0.25,
            norm_eps=1e-20,
        )
        torch.testing.assert_close(pre, torch.full((1, 2), 0.75, dtype=torch.float64))
        torch.testing.assert_close(post, torch.ones((1, 2), dtype=torch.float64))
        torch.testing.assert_close(
            comb, torch.full((1, 2, 2), 3 / 7, dtype=torch.float64)
        )
        _, _, comb = reference.hc_mixes(
            residual,
            torch.zeros((8, 4), dtype=torch.float64),
            torch.ones(3, dtype=torch.float64),
            torch.tensor(
                [0, 0, 0, 0, 0, math.log(3), math.log(2), math.log(2)],
                dtype=torch.float64,
            ),
            iterations=2,
            eps=0.125,
            norm_eps=1e-20,
        )
        expected = [
            [
                [829504 / 2433331, 1339968 / 2545715],
                [1293760 / 2433331, 179136 / 509143],
            ]
        ]
        torch.testing.assert_close(
            comb, torch.tensor(expected, dtype=torch.float64), rtol=1e-14, atol=1e-14
        )

    def test_rotary_uses_adjacent_pairs_without_changing_inputs(self):
        value = torch.tensor([[[1.0, 0.0, 1.0, 0.0]]], dtype=torch.float64)
        original = value.clone()
        positions = torch.tensor([1], dtype=torch.int64)
        actual = reference.rotary(value, positions, base=100.0)
        expected = torch.tensor(
            [[[math.cos(1), math.sin(1), math.cos(0.1), math.sin(0.1)]]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(
            reference.rotary(actual, positions, base=100.0, inverse=True),
            original,
            rtol=1e-14,
            atol=1e-14,
        )
        torch.testing.assert_close(value, original, rtol=0, atol=0)
        yarn = reference.rotary(
            value,
            positions,
            base=100.0,
            original_context=16,
            factor=4.0,
            beta_fast=1.0,
            beta_slow=0.01,
        )
        torch.testing.assert_close(
            yarn,
            torch.tensor(
                [[[math.cos(1), math.sin(1), math.cos(0.075), math.sin(0.075)]]],
                dtype=torch.float64,
            ),
            rtol=1e-14,
            atol=1e-14,
        )
        with self.assertRaisesRegex(ValueError, "FP64"):
            reference.rotary(value.float(), positions, base=100.0)

    def test_attention_sink_has_zero_value_and_routing_bias_only_selects(self):
        query = torch.zeros((1, 2, 2), dtype=torch.float64)
        key_value = torch.tensor([[2.0, 6.0], [100.0, 200.0]], dtype=torch.float64)
        sink = torch.tensor([0.0, math.log(3.0)], dtype=torch.float64)
        output = reference.sparse_attention(query, key_value, sink, [[0]], scale=1.0)
        torch.testing.assert_close(
            output, torch.tensor([[[1.0, 3.0], [0.5, 1.5]]], dtype=torch.float64)
        )
        torch.testing.assert_close(
            reference.sparse_attention(query, key_value, sink, [[-1]], scale=1.0),
            torch.zeros_like(query),
        )
        with self.assertRaisesRegex(ValueError, "cache"):
            reference.sparse_attention(query, key_value, sink, [[2]], scale=1.0)
        ids, weights = reference.route(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.tensor([0.0, 1.0, 1.0], dtype=torch.float64),
            topk=2,
            scale=1.5,
        )
        torch.testing.assert_close(ids, torch.tensor([[1, 2], [1, 2]]))
        torch.testing.assert_close(
            weights, torch.full((2, 2), 0.75, dtype=torch.float64)
        )
        ids, weights = reference.route(
            torch.zeros((2, 3), dtype=torch.float64),
            torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64),
            topk=2,
            scale=1.5,
        )
        torch.testing.assert_close(ids, torch.tensor([[1, 2], [1, 2]]))
        torch.testing.assert_close(
            weights, torch.full((2, 2), 0.75, dtype=torch.float64)
        )


if __name__ == "__main__":
    unittest.main()
