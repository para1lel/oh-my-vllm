import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel_native
from oh_my_vllm.models.deepseek_v41.decode import (
    EngramStage,
)
from oh_my_vllm.models.deepseek_v41.model import (
    Engram,
    ModelArgs,
    ParallelEngramEmbedding,
    compressed_context_width,
    context_token_bucket,
    get_prefill_window_topk_idxs,
)

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelEngram(DeepSeekModelFixture, unittest.TestCase):
    def test_engram_accepts_a_staged_embedding(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            layout = SimpleNamespace(
                max_ngram_size=2,
                n_heads=1,
                head_dim=32,
                layer_ids=(0,),
                num_embeddings=(8,),
            )
            args = ModelArgs(dim=8, hc_mult=1, norm_eps=1e-20)
            engram = Engram(args, 0, layout)
            for parameter in engram.parameters():
                if parameter.dtype in (torch.float8_e4m3fn, torch.float8_e8m0fnu):
                    parameter.data.copy_(
                        torch.full(parameter.shape, 0.02, dtype=torch.float32).to(
                            parameter.dtype
                        )
                    )
                else:
                    torch.nn.init.normal_(parameter, mean=0.0, std=0.02)
            x = torch.randn(1, 1, 1, 8, dtype=torch.bfloat16)
            hash_ids = torch.randint(0, 8, (1, 1, 1))
            eager = engram(x, hash_ids)
            staged = engram.embed(hash_ids).flatten(-2)
            mixed = engram(x, None, embedding=staged)
            self.assertTrue(torch.equal(eager, mixed))
        finally:
            torch.set_default_dtype(previous_dtype)

    def test_compressed_context_width_covers_each_bucket(self):
        for ratio in (1, 2):
            capacity = 512
            widths = {}
            for end_pos in range(1, capacity * ratio + 1):
                width = compressed_context_width(end_pos, ratio, capacity)
                self.assertGreaterEqual(width, end_pos // ratio)
                self.assertLessEqual(width, capacity)
                extent = context_token_bucket(end_pos)
                widths.setdefault(extent, set()).add(width)
            for extent, values in widths.items():
                self.assertEqual(
                    len(values), 1, f"bucket {extent} ratio {ratio}: {values}"
                )
            self.assertEqual(
                compressed_context_width(capacity * ratio, ratio, capacity), capacity
            )
            self.assertEqual(
                compressed_context_width(capacity * ratio + 1, ratio, capacity),
                capacity,
            )

    def test_prefill_window_indices_include_previous_cache_and_causal_chunk(self):
        indices = get_prefill_window_topk_idxs(4, 1, 3, 4)
        self.assertEqual(tuple(indices.shape), (1, 3, 4))
        self.assertTrue(torch.equal(indices[0, 0], torch.tensor([1, 2, 3, 4])))
        self.assertTrue(torch.equal(indices[0, 1], torch.tensor([2, 3, 4, 5])))
        self.assertTrue(torch.equal(indices[0, 2], torch.tensor([3, 4, 5, 6])))

        indices = get_prefill_window_topk_idxs(8, 2, 6, 4)
        expected = torch.tensor(
            [
                [-1, -1, -1, 0, 1, 2, 3, 4],
                [-1, -1, 0, 1, 2, 3, 4, 5],
                [-1, 0, 1, 2, 3, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7, 8],
                [2, 3, 4, 5, 6, 7, 8, 9],
            ],
            dtype=torch.int32,
        )
        self.assertTrue(torch.equal(indices, expected.expand(2, -1, -1)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_table_stays_on_host_for_cuda_lookup(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            table = ParallelEngramEmbedding(8, 32)
            table.weight.data.copy_(
                torch.arange(8 * 32, device="cpu", dtype=torch.float32)
                .reshape(8, 32)
                .to(torch.float8_e4m3fn)
            )
            table.scale.data.fill_(1)

            result = table(torch.tensor([[1, 3]], device="cuda"))
            original = result.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                repeated = table(
                    torch.tensor([[3, 1]], dtype=torch.int32, device="cuda")
                )
            torch.cuda.current_stream().wait_stream(stream)
            self.assertTrue(torch.equal(repeated, original.flip(1)))
            self.assertTrue(torch.equal(result, original))

            self.assertEqual(table.weight.device.type, "cpu")
            self.assertFalse(table.weight.is_pinned())
            self.assertEqual(result.device.type, "cuda")
            expected = (
                torch.arange(32, 64, device="cpu", dtype=torch.float32)
                .to(torch.float8_e4m3fn)
                .float()
                .to("cuda")
            )
            self.assertTrue(torch.equal(result[0, 0].float(), expected))
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_staging_waits_before_third_chunk_and_keeps_tail(self):

        table = self._engram_staging_table()
        ids = (torch.arange(2 * 65536 + 7, device="cuda") % 10) - 1
        native = kernel_native._transform_module()
        events = []
        stages = []
        submissions = []
        event_type = torch.cuda.Event

        class TrackedEvent:
            def __init__(self):
                self.event = event_type()
                self.ready = False
                self.records = 0
                events.append(self)

            def synchronize(self):
                self.event.synchronize()
                self.ready = True

            def record(self, stream):
                self_index = events.index(self)
                submissions.append(("record", self_index))
                self.event.record(stream)
                self.records += 1
                self.ready = False

        def stage(weight, scales, indices, packed, start):
            event = events[len(stages) % 2]
            self.assertTrue(event.ready)
            if event.records:
                self.assertTrue(event.event.query())
            stages.append((packed.data_ptr(), indices.numel()))
            event.ready = False
            native.engram_lookup_stage(weight, scales, indices, packed, start)

        def decode(packed, output):
            native.engram_lookup_decode(packed, output)
            submissions.append(("decode", (len(stages) - 1) % 2))

        module = SimpleNamespace(
            engram_lookup_stage=stage,
            engram_lookup_decode=decode,
        )
        with (
            patch.object(torch.cuda, "Event", TrackedEvent),
            patch.object(kernel_native, "_transform_module", return_value=module),
        ):
            actual = table(ids)
        expected = self._engram_lookup_reference(table, ids)
        self.assertTrue(
            torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))
        )
        self.assertEqual([length for _, length in stages], [65536, 65536, 7])
        self.assertEqual(stages[0][0], stages[2][0])
        self.assertNotEqual(stages[0][0], stages[1][0])
        self.assertEqual([event.records for event in events], [2, 1])
        self.assertEqual(
            submissions,
            [
                ("decode", 0),
                ("record", 0),
                ("decode", 1),
                ("record", 1),
                ("decode", 0),
                ("record", 0),
            ],
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_staging_recovers_after_copy_or_decode_failure(self):

        native = kernel_native._transform_module()
        copy = torch.Tensor.copy_
        synchronize = torch.cuda.Stream.synchronize
        for failure in ("copy", "decode"):
            with self.subTest(failure=failure):
                table = self._engram_staging_table()
                ids = torch.tensor([1, 7, 3, -1], device="cuda")
                table(ids)
                synchronized = []

                def sync(stream, calls=synchronized):
                    synchronize(stream)
                    calls.append(stream.cuda_stream)

                def copy_then_fail(destination, source, *args, **kwargs):
                    result = copy(destination, source, *args, **kwargs)
                    if destination.is_cuda and destination.dtype == torch.uint8:
                        raise RuntimeError("injected Engram copy failure")
                    return result

                def decode_then_fail(packed, output):
                    native.engram_lookup_decode(packed, output)
                    raise RuntimeError("injected Engram decode failure")

                module = SimpleNamespace(
                    engram_lookup_stage=native.engram_lookup_stage,
                    engram_lookup_decode=decode_then_fail,
                )
                fault = (
                    patch.object(torch.Tensor, "copy_", copy_then_fail)
                    if failure == "copy"
                    else patch.object(
                        kernel_native, "_transform_module", return_value=module
                    )
                )
                fault_stream = torch.cuda.Stream()
                fault_stream.wait_stream(torch.cuda.current_stream())
                with (
                    torch.cuda.stream(fault_stream),
                    patch.object(torch.cuda.Stream, "synchronize", sync),
                    fault,
                    self.assertRaisesRegex(RuntimeError, "injected Engram"),
                ):
                    table(ids)
                self.assertEqual(synchronized, [fault_stream.cuda_stream])
                updated = ids.flip(0)
                actual = table(updated)
                expected = self._engram_lookup_reference(table, updated)
                self.assertTrue(
                    torch.equal(
                        actual.cpu().view(torch.int16), expected.view(torch.int16)
                    )
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_stage_refreshes_graph_replay_inputs(self):
        table = self._engram_staging_table()
        engram = SimpleNamespace(
            q_weight=torch.empty(1, device="cuda"),
            n_hash_cols=1,
            head_dim=32,
            layer_hash_index=0,
            embed=table,
        )
        staged_model = SimpleNamespace(
            layers=[SimpleNamespace(engram=engram)],
            engram_hash=lambda tokens, *args, **kwargs: tokens[:, :, None, None],
        )
        bindings = SimpleNamespace(
            batch=1,
            token_ids=torch.tensor([[1]], device="cuda"),
            cache_slots=None,
            position=None,
        )
        stage = EngramStage(staged_model, 1)
        stage.stage(bindings, 0)
        address = stage.buffers[0].data_ptr()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = stage.buffers[0].clone()
        torch.cuda.current_stream().wait_stream(stream)
        for token in (1, 6, 3):
            bindings.token_ids.fill_(token)
            stage.stage(bindings, token)
            graph.replay()
            expected = self._engram_lookup_reference(table, bindings.token_ids)
            self.assertEqual(stage.buffers[0].data_ptr(), address)
            self.assertTrue(
                torch.equal(output.cpu().view(torch.int16), expected.view(torch.int16))
            )
