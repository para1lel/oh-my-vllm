import unittest

import torch


class ReferenceCases(unittest.TestCase):
    def test_query_tiles_keep_the_complete_causal_key_range(self):
        from .reference_layers import attention

        torch.set_num_threads(1)
        rows = 257
        values = torch.arange(rows, dtype=torch.float64)[:, None].expand(-1, 5120)
        positions = torch.arange(rows, dtype=torch.int64)[None].expand(3, -1)

        class Weights:
            def tensor(self, name):
                return torch.zeros(256, dtype=torch.float64)

            def linear(self, value, name):
                if name.endswith("q_proj"):
                    return torch.zeros((len(value), 12288), dtype=torch.float64)
                if name.endswith("k_proj"):
                    return torch.zeros((len(value), 1024), dtype=torch.float64)
                if name.endswith("v_proj"):
                    return value[:, :1].expand(-1, 1024)
                if name.endswith("o_proj"):
                    return value[:, :5120]
                raise AssertionError(name)

        actual, state = attention(Weights(), "", values, None, positions, 0)
        expected = (torch.arange(rows, dtype=torch.float64) / 4)[:, None].expand(
            -1, 5120
        )
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(state[0].shape, (257, 4, 256))
