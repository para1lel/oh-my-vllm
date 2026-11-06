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
    PackedView,
    PageTable,
)
from oh_my_vllm.operators.runtime import uploads

from tests.deepseek_packed_cache_fixtures import DeepSeekPackedCacheFixture, row


class DeepSeekPackedCacheLeases(DeepSeekPackedCacheFixture, unittest.TestCase):
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
                    "oh_my_vllm.models.deepseek_v41.target_execution.DecodeBindings"
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
        with self.assertRaisesRegex(RuntimeError, "bound.*conversion"):
            packed.float()
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
