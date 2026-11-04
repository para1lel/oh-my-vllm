import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel
from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
from oh_my_vllm.models.deepseek_v41.decode import (
    DecodeGraph,
    native_sparse_graph_eligible,
)
from oh_my_vllm.models.deepseek_v41.inference import Worker
from oh_my_vllm.models.deepseek_v41.model import Attention, ModelArgs, Transformer
from oh_my_vllm.models.deepseek_v41.packed_cache import (
    PackedHistory,
    PackedView,
    PageTable,
    TokenHistory,
)
from oh_my_vllm.models.deepseek_v41.quantization import fp4_act_quant, fp4_dequantize
from oh_my_vllm.operators.runtime import uploads

from development.model.check_deepseek_batch import (
    cache_state,
    differences,
    reset_graphs,
)


def packed_model():
    args = SimpleNamespace(max_batch_size=2, max_seq_len=16, lazy_cache=False)
    return SimpleNamespace(
        layers=[
            SimpleNamespace(
                attn=SimpleNamespace(
                    window_kv_cache=torch.zeros(2, 4, 32),
                    compress_ratio=2,
                    compress_kv_cache=PackedHistory(
                        args, 32, 2, 16, torch.float8_e4m3fn
                    ),
                    indexer=SimpleNamespace(
                        k_cache=PackedHistory(args, 32, 2, 32, torch.float8_e8m0fnu)
                    ),
                    compressor=SimpleNamespace(
                        kv_state=torch.zeros(2, 2, 32),
                        score_state=torch.zeros(2, 2, 32),
                    ),
                )
            )
        ],
        mtp=[],
        engram_hash=SimpleNamespace(cache=TokenHistory(args)),
    )


def row(fa, state):
    return {"fa_block_table": fa, "mamba_block_table": state}


class DeepSeekPackedCache(unittest.TestCase):
    def setUp(self):
        self.model = packed_model()
        self.cache = DeepSeekCache(
            self.model,
            block_tokens=4,
            max_seq_len=16,
            fa_capacity=16,
            host_limit=1 << 20,
        )
        self.addCleanup(self.cache.close)

    def produce(self, slot, first, count):
        values = (torch.arange(count * 32).reshape(1, count, 32).float() - 17) / 11
        slots = torch.tensor([slot])
        for history in (
            self.model.layers[0].attn.compress_kv_cache,
            self.model.layers[0].attn.indexer.k_cache,
        ):
            encoded, scales = fp4_act_quant(
                values, history.group_size, scale_dtype=history.scale_dtype
            )
            history.write(slots, first, encoded, scales)
        export = self.cache.exports
        export.bind((slot,))
        attention = self.model.layers[0].attn
        state_values = (
            torch.arange(first * 2, (first + count) * 2)
            .float()
            .view(1, count * 2, 1)
            .expand(1, count * 2, 32)
        )
        export.window(attention.window_kv_cache, state_values, first * 2)
        export.tail(
            attention.compressor.kv_state,
            attention.compressor.score_state,
            state_values,
            state_values + 10,
            first * 2,
        )
        return values

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
                status = kernel._sparse_status_accumulator(torch.device("cpu"))
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

    def test_host_roundtrip_retains_original_codes_scales_and_tokens(self):
        schedule = row([1], [3])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=4)
        self.produce(0, 0, 2)
        self.model.engram_hash.cache[0, :4] = torch.arange(100, 104)
        self.cache.commit(7, schedule, 0, accepted_end=4)
        source = self.cache.fa_pages[1]
        source.compressed[0].codes[0, 0] = 0x80
        source.index[0].scales[0, 0] = 0x04
        expected_codes = source.compressed[0].codes.clone()
        expected_scales = source.index[0].scales.clone()
        operations = [
            dict(
                kind="store",
                transfer_id=1,
                snapshot_id=1,
                fa_blocks=[1],
                mamba_blocks=[3],
            )
        ]
        stored = self.cache.handle({"operations": operations})
        loaded = self.cache.handle(
            {
                "operations": [
                    dict(
                        kind="load",
                        transfer_id=2,
                        snapshot_id=1,
                        fa_blocks=[2],
                        mamba_blocks=[4],
                    )
                ]
            }
        )
        target = self.cache.fa_pages[2]
        torch.testing.assert_close(
            target.compressed[0].codes, expected_codes, atol=0, rtol=0
        )
        torch.testing.assert_close(
            target.index[0].scales, expected_scales, atol=0, rtol=0
        )
        torch.testing.assert_close(target.engram[0], source.engram[0], atol=0, rtol=0)
        self.assertEqual(stored[0]["bytes"], loaded[0]["bytes"])
        self.assertEqual(
            stored[0]["bytes"],
            self.cache._fa_page_bytes() + self.cache._mamba_page_bytes(),
        )
        self.cache.restore(7, 1, row([2], [4]), start_pos=4)
        torch.testing.assert_close(
            self.model.engram_hash.cache[1, :4], torch.arange(100, 104), atol=0, rtol=0
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

    def test_graph_lease_prevents_pool_release_and_rejects_stale_generation(self):
        table = PageTable(2, 16, 4, 16, torch.device("cpu"))
        lease = table.acquire()
        with self.assertRaisesRegex(RuntimeError, "graphs"):
            table.close()
        lease.validate()
        lease.close()
        table.close()
        with self.assertRaisesRegex(RuntimeError, "bindings"):
            lease.validate()

    def test_repeated_page_table_uploads_keep_one_buffer_per_active_slot(self):
        constructed, streams = [], []

        class CPUUploadBuffer:
            def __init__(self, device, destination, source):
                self.destination = destination
                self.tensor = torch.empty_like(source)
                self.closed = False
                constructed.append(self)

            def copy(self, source, stream):
                self.tensor.copy_(source)
                self.destination.copy_(self.tensor)
                streams.append(stream)

            def close(self):
                self.closed = True
                self.tensor = None

        table = PageTable(3, 12, 4, 128, torch.device("cpu"))
        table.uploads = uploads.Uploads(SimpleNamespace())
        self.addCleanup(table.close)
        identities = tuple(id(value) for value in table.rows)
        addresses = tuple(value.data_ptr() for value in table.rows)
        with (
            patch.object(uploads, "_eager_stream", return_value=17),
            patch.object(uploads, "UploadBuffer", CPUUploadBuffer),
        ):
            for index in range(1024):
                slot = index % 3
                blocks = (
                    [] if index % 5 == 0 else [index % 120 + 1, (index + 1) % 120 + 1]
                )
                table.bind(slot, blocks)
                self.assertEqual(
                    table.values[slot].tolist(), blocks + [0] * (3 - len(blocks))
                )
                self.assertLessEqual(len(table.uploads.buffers), 3)
        self.assertEqual(len(constructed), 3)
        self.assertEqual(
            sum(buffer.tensor.nbytes for buffer in constructed), table.values.nbytes
        )
        self.assertEqual(tuple(id(value) for value in table.rows), identities)
        self.assertEqual(tuple(value.data_ptr() for value in table.rows), addresses)
        self.assertEqual(streams, [17] * 1024)
        table.close()
        self.assertEqual(table.uploads.buffers, {})
        self.assertTrue(all(buffer.closed for buffer in constructed))

    def test_reset_starts_empty_epoch_at_the_same_pool_addresses(self):
        schedule = row([1], [3])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=4)
        self.produce(0, 0, 2)
        self.model.engram_hash.cache[0, :4] = torch.arange(10, 14)
        self.cache.commit(7, schedule, 0, accepted_end=4)
        history = self.model.layers[0].attn.compress_kv_cache
        addresses = (
            history.codes.data_ptr(),
            history.scales.data_ptr(),
            self.cache.page_table.values.data_ptr(),
            self.cache.index_workspace.values.data_ptr(),
        )
        generation = self.cache.page_table.generation
        lease = self.cache.page_table.acquire()
        with self.assertRaisesRegex(RuntimeError, "graphs"):
            self.cache.reset()
        lease.close()
        self.cache.reset()
        self.assertEqual(
            addresses,
            (
                history.codes.data_ptr(),
                history.scales.data_ptr(),
                self.cache.page_table.values.data_ptr(),
                self.cache.index_workspace.values.data_ptr(),
            ),
        )
        self.assertEqual(self.cache.page_table.generation, generation + 1)
        self.assertEqual(self.cache.fa_pages, {})
        self.assertEqual(self.cache.mamba_pages, {})
        self.assertEqual(history.page(1).codes.count_nonzero(), 0)
        self.assertEqual(self.cache.page_table.values.count_nonzero(), 0)
        self.assertEqual(self.model.layers[0].attn.window_kv_cache.count_nonzero(), 0)
        self.assertTrue(
            self.model.layers[0].attn.compressor.score_state.isneginf().all()
        )
        with self.assertRaisesRegex(ValueError, "bind"):
            self.cache.require_active_slots((0,))
        self.cache.activate_row(schedule, 0, end_pos=4)
        self.cache.require_active_slots((0,))

    def test_missing_endpoint_state_and_wrong_host_layout_do_not_publish_pages(self):
        schedule = row([1], [3])
        self.cache.begin_step(schedule, 0, start_pos=0, end_pos=4)
        self.produce(0, 0, 2)
        self.cache.exports.steps[0].tails.clear()
        with self.assertRaisesRegex(ValueError, "compressor"):
            self.cache.commit(7, schedule, 0, accepted_end=4)
        self.assertEqual(self.cache.fa_pages[1].valid_tokens, 0)
        self.assertEqual(self.cache.mamba_pages[3].endpoint, 0)
        self.cache.exports.discard(0)
        self.cache.commit(7, schedule, 0, accepted_end=4)
        self.cache.handle(
            {
                "operations": [
                    dict(
                        kind="store",
                        transfer_id=1,
                        snapshot_id=1,
                        fa_blocks=[1],
                        mamba_blocks=[3],
                    )
                ]
            }
        )
        self.cache.snapshots[1].layout_version = 2
        with self.assertRaisesRegex(ValueError, "layout"):
            self.cache.handle(
                {
                    "operations": [
                        dict(
                            kind="load",
                            transfer_id=2,
                            snapshot_id=1,
                            fa_blocks=[2],
                            mamba_blocks=[4],
                        )
                    ]
                }
            )
        self.assertNotIn(2, self.cache.fa_pages)
        self.assertNotIn(4, self.cache.mamba_pages)

    def test_reference_sparse_graph_selection_stops_before_preparation(self):
        for disabled, failed in ((True, False), (False, True)):
            with (
                self.subTest(disabled=disabled, failed=failed),
                patch.object(kernel, "_SPARSE_NATIVE_DISABLED", disabled),
                patch.object(kernel, "_SPARSE_NATIVE_FAILED", failed),
                patch(
                    "oh_my_vllm.models.deepseek_v41.inference.DecodeBindings"
                ) as bindings,
            ):
                worker = SimpleNamespace(decode_graph_enabled=True, model=self.model)
                self.assertIsNone(
                    Worker._decode_graph_step(
                        worker, 1, 0, 0, collect_main_hidden=False
                    )
                )
                bindings.assert_not_called()
                graph = DecodeGraph.__new__(DecodeGraph)
                graph.model = self.model
                graph.closed, graph.cache_lease, graph.output = False, None, None
                graph._capture = Mock()
                with self.assertRaisesRegex(RuntimeError, "Eager"):
                    graph.capture()
                graph._capture.assert_not_called()
                self.assertEqual(self.cache.page_table.leases, 0)
        attention = Attention.__new__(Attention)
        torch.nn.Module.__init__(attention)
        attention.head_dim, attention.n_local_heads = 32, 4
        attention.window_kv_cache = torch.zeros(1, 4, 32)
        unsupported = SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
        self.assertFalse(native_sparse_graph_eligible(unsupported))

    def test_reference_capture_rejects_before_full_history_materialization(self):
        self.cache.activate_row(row([1], [3]), 0, end_pos=4)
        history = self.model.layers[0].attn.compress_kv_cache
        view = history.view(torch.tensor([0]), 2)
        query = torch.zeros(1, 6, 1, 32)
        indices = torch.zeros(1, 6, 1, dtype=torch.int32)
        window = torch.ones(1, 1, 32)
        for disabled in (False, True):
            with (
                self.subTest(disabled=disabled),
                patch.object(kernel, "_SPARSE_NATIVE_DISABLED", disabled),
                patch.object(kernel, "_SPARSE_NATIVE_FAILED", False),
                kernel.capture_cached_tensors(),
                patch.object(PackedView, "materialize") as materialize,
                patch("torch.cat") as concatenate,
            ):
                with self.assertRaisesRegex(RuntimeError, "Eager"):
                    kernel.sparse_attn(
                        query,
                        None,
                        torch.zeros(1),
                        indices,
                        1,
                        native_window=window,
                        native_compressed=view,
                    )
                materialize.assert_not_called()
                concatenate.assert_not_called()
        with patch.object(kernel, "_SPARSE_NATIVE_DISABLED", True):
            output = kernel.sparse_attn(
                query,
                None,
                torch.zeros(1),
                indices,
                1,
                native_window=window,
                native_compressed=view,
            )
        torch.testing.assert_close(output, torch.full_like(query, 0.5), atol=0, rtol=0)

    def test_graph_close_resets_executable_before_releasing_pool_lease(self):
        events = []
        graph = DecodeGraph.__new__(DecodeGraph)
        graph.model = self.model
        graph.bindings = SimpleNamespace(position=torch.empty(0))
        graph.closed, graph.cache_lease, graph.output = False, None, None
        graph.retained, graph.routed_bindings = (self.cache.index_workspace.values,), ()
        graph.graph = SimpleNamespace(
            reset=Mock(side_effect=lambda: events.append("reset"))
        )
        graph._capture = Mock(side_effect=ValueError("record failed"))
        with (
            patch(
                "torch.cuda.synchronize",
                side_effect=lambda *args: events.append("wait"),
            ),
            self.assertRaisesRegex(ValueError, "record failed"),
        ):
            graph.capture()
        self.assertEqual(events, ["wait", "reset"])
        self.assertTrue(graph.closed)
        self.assertEqual(self.cache.page_table.leases, 0)
        self.assertEqual(graph.retained, ())
        with self.assertRaisesRegex(RuntimeError, "only once"):
            graph.capture()

    def test_failed_graph_reset_retains_the_pool_and_scratch(self):
        graph = DecodeGraph.__new__(DecodeGraph)
        graph.bindings = SimpleNamespace(position=torch.empty(0))
        graph.closed = False
        graph.output = torch.ones(1)
        graph.cache_lease = self.cache.page_table.acquire()
        graph.retained, graph.routed_bindings = (self.cache.index_workspace.values,), ()
        retained = graph.retained
        graph.graph = SimpleNamespace(
            reset=Mock(side_effect=RuntimeError("reset failed"))
        )
        with patch("torch.cuda.synchronize"):
            with self.assertRaisesRegex(RuntimeError, "reset failed"):
                graph.close()
            self.assertIs(graph.retained, retained)
            self.assertEqual(self.cache.page_table.leases, 1)
            graph.cache_lease.validate()
            with self.assertRaisesRegex(RuntimeError, "graphs"):
                self.cache.close()
            graph.graph.reset.side_effect = None
            graph.close()
        self.assertEqual(self.cache.page_table.leases, 0)
        self.assertTrue(graph.closed)

    def test_actual_model_producers_and_full_index_math_match_bf16_history(self):
        previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous)
        torch.set_default_dtype(torch.bfloat16)
        args = ModelArgs(
            max_batch_size=2,
            max_seq_len=16,
            vocab_size=32,
            dim=64,
            n_layers=2,
            n_heads=4,
            head_dim=32,
            rope_head_dim=16,
            q_lora_rank=32,
            o_groups=4,
            o_lora_rank=16,
            window_size=4,
            compress_ratios=(2, 2),
            kv_source_layers=(0,),
            index_source_layers=(0, 1),
            index_n_heads=4,
            index_head_dim=32,
            index_topk=4,
            dtype="bf16",
            expert_dtype=None,
            n_routed_experts=2,
            n_activated_experts=1,
            moe_inter_dim=32,
        )
        torch.manual_seed(71)
        packed = Transformer(args)
        for parameter in packed.parameters():
            torch.nn.init.normal_(parameter, mean=0, std=0.02)
        reference = Transformer(args)
        reference.load_state_dict(packed.state_dict())
        reference.layers[0].attn.compress_kv_cache = torch.zeros(2, 8, 32)
        reference.layers[0].attn.indexer.k_cache = torch.zeros(2, 8, 32)
        cache = DeepSeekCache(packed, block_tokens=4, max_seq_len=16, fa_capacity=16)
        self.addCleanup(cache.close)
        schedule = row([1, 2], [3, 4])
        cache.begin_step(schedule, 0, start_pos=0, end_pos=8)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        actual = packed(tokens, full_logits=True)[1]
        expected = reference(tokens, full_logits=True)[1]
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertIsInstance(
            packed.layers[0].attn.compress_kv_cache.view(torch.tensor([0]), 4),
            PackedView,
        )
