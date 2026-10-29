import unittest

import torch


class DeepSeekStateExports(unittest.TestCase):
    def test_model_write_boundaries_export_before_recurrent_updates(self):
        from oh_my_vllm.models.deepseek_v41.model import (
            Attention,
            Compressor,
            DSparkAttention,
            ModelArgs,
        )
        from oh_my_vllm.models.deepseek_v41.state import StateExports

        for kind in (Attention, DSparkAttention):
            with self.subTest(kind=kind):
                exports = StateExports()
                exports.begin(0, 4, (5, 8, 10))
                exports.bind((0,))
                attention = kind.__new__(kind)
                torch.nn.Module.__init__(attention)
                attention.window_size = 4
                attention.rope_head_dim = 2
                attention.wkv = attention.kv_norm = torch.nn.Identity()
                attention.window_kv_cache = torch.ones(1, 4, 32)
                attention.freqs_cis = torch.ones(16, 1, dtype=torch.complex64)
                attention._state_export = exports
                values = (2.0 ** torch.arange(1, 7)).view(1, 6, 1).expand(1, 6, 32)
                if kind is Attention:
                    attention._window_kv(
                        values.clone(), attention.freqs_cis[4:10], 4, torch.tensor([0])
                    )
                else:
                    attention._store_main_kv(values.clone(), 4, torch.tensor([0]))
                for endpoint, want in (
                    (5, (2, 1, 1, 1)),
                    (8, (2, 4, 8, 16)),
                    (10, (32, 64, 8, 16)),
                ):
                    torch.testing.assert_close(
                        exports.window_state(0, endpoint, attention.window_kv_cache)[
                            :, 0
                        ],
                        torch.tensor(want).float(),
                    )
        compressor = Compressor(ModelArgs(dim=32, head_dim=32, compress_ratios=(2,)), 0)
        compressor.wkv = compressor.wgate = torch.nn.Identity()
        compressor._state_export = exports
        compressor(torch.ones(1, 6, 32), 4, torch.tensor([0]))
        kv, score = exports.tail_state(0, 5, compressor.kv_state)
        torch.testing.assert_close(kv[0], torch.ones(32))
        torch.testing.assert_close(score[0], torch.ones(32))

    def test_verification_exports_keep_each_accepted_window_before_overwrite(self):
        from oh_my_vllm.models.deepseek_v41.state import StateExports

        exports = StateExports()
        exports.begin(1, 4, range(5, 11))
        exports.bind((1,))
        ring = torch.arange(4, dtype=torch.float32).view(1, 4, 1).repeat(2, 1, 1)
        values = torch.arange(4, 10, dtype=torch.float32).view(1, 6, 1)
        exports.window(ring, values, 4)
        for position in range(4, 10):
            ring[1, position % 4] = position
        values.fill_(-1)
        expected = (
            (4, 1, 2, 3),
            (4, 5, 2, 3),
            (4, 5, 6, 3),
            (4, 5, 6, 7),
            (8, 5, 6, 7),
            (8, 9, 6, 7),
        )
        for endpoint, want in zip(range(5, 11), expected, strict=True):
            with self.subTest(endpoint=endpoint):
                torch.testing.assert_close(
                    exports.window_state(1, endpoint, ring).flatten(),
                    torch.tensor(want, dtype=torch.float32),
                )
        with self.assertRaisesRegex(ValueError, "endpoint"):
            exports.window_state(0, 5, ring)
        exports.discard(1)
        with self.assertRaisesRegex(ValueError, "endpoint"):
            exports.window_state(1, 5, ring)

    def test_long_prefill_exports_only_endpoint_windows(self):
        from oh_my_vllm.models.deepseek_v41.state import StateExports

        exports = StateExports()
        exports.begin(0, 0, (3, 17))
        exports.bind((0,))
        ring = torch.zeros(1, 4, 1)
        values = torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1)
        exports.window(ring, values, 0)
        for endpoint, want in ((3, (1, 2, 3, 0)), (17, (17, 14, 15, 16))):
            snapshot = exports.window_state(0, endpoint, ring)
            torch.testing.assert_close(snapshot.flatten(), torch.tensor(want).float())
            self.assertEqual(snapshot.untyped_storage().nbytes(), 16)

    def test_compressor_exports_rebuild_only_the_live_partial_group(self):
        from oh_my_vllm.models.deepseek_v41.state import StateExports

        exports = StateExports()
        exports.begin(0, 5, range(6, 11))
        exports.bind((0,))
        kv = torch.full((1, 2, 1), -100.0)
        score = torch.full_like(kv, 100)
        values = torch.arange(4, 10, dtype=torch.float32).view(1, 6, 1)
        scores = values + 10
        exports.tail(kv, score, values, scores, 5)
        kv.fill_(1000)
        score.fill_(1000)
        values.fill_(1000)
        for endpoint, want in ((6, None), (7, 6), (8, None), (9, 8), (10, None)):
            with self.subTest(endpoint=endpoint):
                actual_kv, actual_score = exports.tail_state(0, endpoint, kv)
                expected_kv = torch.tensor([[0.0], [0.0]])
                expected_score = torch.full((2, 1), -torch.inf)
                if want is not None:
                    expected_kv[0] = want
                    expected_score[0] = want + 10
                torch.testing.assert_close(actual_kv, expected_kv)
                torch.testing.assert_close(actual_score, expected_score)

    def test_capture_rejects_an_unbound_batch_and_incomplete_endpoint(self):
        from oh_my_vllm.models.deepseek_v41.state import StateExports

        exports = StateExports()
        exports.begin(0, 0, (4,))
        ring = torch.zeros(1, 4, 1)
        with self.assertRaisesRegex(ValueError, "batch"):
            exports.window(ring, torch.ones(1, 4, 1), 0)
        exports.bind((0,))
        with self.assertRaisesRegex(ValueError, "endpoint"):
            exports.window_state(0, 4, ring)
