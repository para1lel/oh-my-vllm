import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel
from oh_my_vllm.models.deepseek_v41.decode import (
    DecodeGraph,
)
from oh_my_vllm.models.deepseek_v41.packed_cache import (
    PackedHistory,
    PackedView,
)
from oh_my_vllm.models.deepseek_v41.quantization import fp4_act_quant, fp4_dequantize

from development.model.check_deepseek_batch import (
    cache_state,
    differences,
    reset_graphs,
)
from tests.deepseek_packed_cache_fixtures import DeepSeekPackedCacheFixture, row
from tests.deepseek_packed_cache_fixtures import packed_model as packed_model
from tests.deepseek_packed_cache_leases import (
    DeepSeekPackedCacheLeases as DeepSeekPackedCacheLeases,
)


class DeepSeekPackedCache(DeepSeekPackedCacheFixture, unittest.TestCase):
    def test_producer_pages_and_active_views_share_one_allocation(self):
        schedule = row([1, 2], [3, 4])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=8)
        source = self.produce(0, 0, 4)
        tokens = self.model.engram_hash.cache
        tokens[0, :8] = torch.arange(100, 108)
        history = self.model.layers[0].attn.compress_kv_cache
        codes, scales = fp4_act_quant(source, 16, scale_dtype=torch.float8_e4m3fn)
        before = history.codes.clone()
        with patch.object(
            PackedHistory, "__getitem__", side_effect=AssertionError("history copy")
        ):
            self.cache.commit(7, schedule, 0, accepted_end=8)
        self.assertEqual(
            self.cache.fa_pages[1].compressed[0].codes.untyped_storage().data_ptr(),
            history.codes.untyped_storage().data_ptr(),
        )
        torch.testing.assert_close(
            self.cache.fa_pages[1].compressed[0].codes,
            codes.view(torch.uint8)[0, :2],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            self.cache.fa_pages[1].compressed[0].scales,
            scales.view(torch.uint8)[0, :2],
            atol=0,
            rtol=0,
        )
        self.cache.clear_slot(0)
        with patch.object(
            PackedHistory, "__getitem__", side_effect=AssertionError("history copy")
        ):
            self.cache.restore(7, 1, schedule, start_pos=8)
        torch.testing.assert_close(history.codes, before, atol=0, rtol=0)
        expected = fp4_dequantize(codes, scales, 16, dtype=history.dtype)
        torch.testing.assert_close(history[1, :4], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(
            tokens[1, :8], torch.arange(100, 108), atol=0, rtol=0
        )
        self.assertEqual(
            self.cache._fa_page_bytes(), 4 // 2 * (16 + 2 + 16 + 1) + 4 * 8
        )

    def test_checker_snapshots_keep_independent_codes_and_scales(self):
        schedule = row([1], [3])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=4)
        self.produce(0, 0, 2)
        self.cache.commit(7, schedule, 0, accepted_end=4)
        worker = SimpleNamespace(
            cache=self.cache,
            requests={7: SimpleNamespace(cache_slot=0, state_position=4)},
        )
        snapshot = cache_state(worker, torch)
        digest = cache_state(worker, torch, digest=True)
        saved = io.BytesIO()
        torch.save(snapshot, saved)
        saved.seek(0)
        restored = torch.load(saved, weights_only=True)
        self.assertEqual(differences(snapshot, restored, torch), [])
        key = "pages/fa/1"
        page = self.cache.fa_pages[1].compressed[0]
        stored = snapshot[key]["compressed"][0]
        self.assertEqual(set(stored), {"codes", "scales"})
        for name in ("codes", "scales"):
            with self.subTest(plane=name):
                tensor = getattr(page, name)
                self.assertEqual(stored[name].device.type, "cpu")
                self.assertEqual(stored[name].dtype, tensor.dtype)
                self.assertEqual(stored[name].stride(), tensor.stride())
                self.assertNotEqual(
                    stored[name].untyped_storage().data_ptr(),
                    tensor.untyped_storage().data_ptr(),
                )
                tensor[0, 0] ^= 1
                changed = cache_state(worker, torch)
                changed_digest = cache_state(worker, torch, digest=True)
                names = {item["name"] for item in differences(snapshot, changed, torch)}
                self.assertIn(f"endpoint/{key}/compressed/0/{name}", names)
                self.assertNotEqual(
                    digest[key]["compressed"][0][name],
                    changed_digest[key]["compressed"][0][name],
                )
                other = "scales" if name == "codes" else "codes"
                self.assertEqual(
                    digest[key]["compressed"][0][other],
                    changed_digest[key]["compressed"][0][other],
                )
                self.assertEqual(differences(snapshot, restored, torch), [])
                tensor.copy_(stored[name])

    def test_checker_reads_inference_status_without_leaking_the_context(self):
        schedule = row([1], [3])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=4)
        self.produce(0, 0, 2)
        self.cache.commit(7, schedule, 0, accepted_end=4)
        worker = SimpleNamespace(
            cache=self.cache,
            requests={7: SimpleNamespace(cache_slot=0, state_position=4)},
        )
        materialize = PackedView.materialize

        def native_status(view, *, shared=True):
            kernel._sparse_status_accumulator(view.device).logical_or_(
                torch.tensor(False)
            )
            return materialize(view, shared=shared)

        with patch.object(kernel, "_SPARSE_STATUS", {}):
            with torch.inference_mode():
                status = torch.zeros((), dtype=torch.bool, device="cpu")
            kernel._SPARSE_STATUS["cpu"] = status
            self.assertTrue(torch.is_inference(status))
            self.assertFalse(torch.is_inference_mode_enabled())
            with patch.object(PackedView, "materialize", native_status):
                view = self.model.layers[0].attn.compress_kv_cache.view(
                    torch.tensor([0]), 2
                )
                with self.assertRaisesRegex(RuntimeError, "outside InferenceMode"):
                    view.materialize(shared=False)
                for digest in (False, True):
                    with self.subTest(digest=digest):
                        result = cache_state(worker, torch, digest=digest)
                        self.assertIn("pages/fa/1", result)
                        self.assertFalse(torch.is_inference_mode_enabled())
            with (
                patch.object(
                    PackedView,
                    "materialize",
                    side_effect=ValueError("snapshot rejected"),
                ),
                self.assertRaisesRegex(ValueError, "snapshot rejected"),
            ):
                cache_state(worker, torch)
            self.assertFalse(torch.is_inference_mode_enabled())

    def test_checker_closes_graph_lease_before_resetting_the_cache(self):
        events = []
        graph = DecodeGraph.__new__(DecodeGraph)
        graph.bindings = SimpleNamespace(position=torch.empty(0))
        graph.closed = False
        graph.output = torch.ones(1)
        graph.retained, graph.routed_bindings = (self.cache.index_workspace.values,), ()
        graph.cache_lease = self.cache.page_table.acquire()
        graph.graph = SimpleNamespace(
            reset=Mock(side_effect=RuntimeError("reset failed"))
        )
        bindings = SimpleNamespace(
            close=Mock(
                side_effect=lambda: events.append(
                    ("bindings", self.cache.page_table.leases)
                )
            )
        )
        worker = SimpleNamespace(
            device=torch.device("cpu"),
            decode_graphs={"test": graph},
            decode_bindings={"test": bindings},
            decode_engram={"test": object()},
        )
        with patch("torch.cuda.synchronize"):
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                reset_graphs(worker, torch)
            self.assertIs(worker.decode_graphs["test"], graph)
            self.assertEqual(self.cache.page_table.leases, 1)
            bindings.close.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "graphs"):
                self.cache.reset()
            graph.graph.reset.side_effect = lambda: events.append(
                ("reset", self.cache.page_table.leases)
            )
            reset_graphs(worker, torch)
        self.assertEqual(events, [("reset", 1), ("bindings", 0)])
        self.assertTrue(graph.closed)
        self.assertEqual(self.cache.page_table.leases, 0)
        self.assertEqual(worker.decode_graphs, {})
        self.assertEqual(worker.decode_bindings, {})
        self.assertEqual(worker.decode_engram, {})
        self.cache.reset()

    def test_partial_prefix_copies_only_accepted_complete_groups_and_raw_bytes(self):
        schedule = row([1], [3])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=4)
        self.produce(0, 0, 2)
        self.model.engram_hash.cache[0, :4] = torch.arange(4)
        self.cache.commit(7, schedule, 0, accepted_end=4)
        source = self.cache.fa_pages[1]
        source.compressed[0].codes[1].fill_(0xAB)
        source.compressed[0].scales[1].fill_(0xCD)
        self.cache.copy_prefix([(1, 2, 3)])
        destination = self.cache.fa_pages[2]
        torch.testing.assert_close(
            destination.compressed[0].codes[:1],
            source.compressed[0].codes[:1],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            destination.compressed[0].scales[:1],
            source.compressed[0].scales[:1],
            atol=0,
            rtol=0,
        )
        self.assertEqual(destination.compressed[0].codes[1].count_nonzero(), 0)
        torch.testing.assert_close(
            destination.compressed[0].scales[1],
            torch.ones(2, dtype=torch.uint8),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            destination.engram[0][:3], source.engram[0][:3], atol=0, rtol=0
        )
        self.assertEqual(destination.valid_tokens, 3)
        destination.compressed[0].codes[0].fill_(0x11)
        self.assertFalse(
            torch.equal(
                destination.compressed[0].codes[0], source.compressed[0].codes[0]
            )
        )

    def test_rejected_suffix_is_unpublished_and_odd_endpoint_uses_the_tail(self):
        schedule = row([1, 2], [3, 4])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=8, minimum_end=3)
        self.produce(0, 0, 4)
        self.model.engram_hash.cache[0, :8] = torch.arange(8)
        export = self.cache.exports
        export.bind((0,))
        attention = self.model.layers[0].attn
        values = torch.arange(8).view(1, 8, 1).expand(1, 8, 32).float()
        export.window(attention.window_kv_cache, values, 0)
        export.tail(
            attention.compressor.kv_state,
            attention.compressor.score_state,
            values,
            values + 10,
            0,
        )
        self.cache.commit(7, schedule, 0, accepted_end=5)
        self.assertEqual(self.cache.fa_pages[2].valid_tokens, 1)
        self.cache.copy_prefix([(2, 5, 1)])
        self.assertEqual(self.cache.fa_pages[5].compressed[0].codes.count_nonzero(), 0)
        self.cache.restore(7, 1, schedule, start_pos=5, use_prefix_state=False)
        torch.testing.assert_close(
            attention.compressor.kv_state[1, 0], torch.full((32,), 4.0), atol=0, rtol=0
        )
        with self.assertRaisesRegex(ValueError, "endpoint"):
            self.cache.restore(7, 1, schedule, start_pos=6, use_prefix_state=False)
