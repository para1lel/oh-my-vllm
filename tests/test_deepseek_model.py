import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel_graph, kernel_weights, model
from oh_my_vllm.models.deepseek_v41.decode import (
    DecodeBindings,
)
from oh_my_vllm.models.deepseek_v41.model import (
    get_dspark_topk_idxs,
    get_prefill_window_topk_idxs,
    get_window_topk_idxs,
)

from tests.deepseek_model_cache import DeepSeekModelCache as DeepSeekModelCache
from tests.deepseek_model_decode import DeepSeekModelDecode as DeepSeekModelDecode
from tests.deepseek_model_engram import DeepSeekModelEngram as DeepSeekModelEngram
from tests.deepseek_model_fixtures import DeepSeekModelFixture
from tests.deepseek_model_moe import DeepSeekModelMoe as DeepSeekModelMoe
from tests.deepseek_model_moe_cuda import DeepSeekModelMoeCuda as DeepSeekModelMoeCuda
from tests.deepseek_model_prefill import DeepSeekModelPrefill as DeepSeekModelPrefill
from tests.deepseek_model_prefill_equivalence import (
    DeepSeekModelPrefillEquivalence as DeepSeekModelPrefillEquivalence,
)
from tests.deepseek_model_prefill_waves import (
    DeepSeekModelPrefillWaves as DeepSeekModelPrefillWaves,
)
from tests.deepseek_model_rank_weights import (
    DeepSeekModelRankWeights as DeepSeekModelRankWeights,
)
from tests.deepseek_model_target import DeepSeekModelTarget as DeepSeekModelTarget
from tests.deepseek_model_transformer import (
    DeepSeekModelTransformer as DeepSeekModelTransformer,
)


class DeepSeekModel(DeepSeekModelFixture, unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_decode_graph_owns_cached_inputs_after_lru_eviction(self):
        from oh_my_vllm.models.deepseek_v41.decode import DecodeGraph

        cached = (
            model.precompute_freqs_cis,
            get_window_topk_idxs,
            get_prefill_window_topk_idxs,
            get_dspark_topk_idxs,
        )

        class CachedModel:
            def __init__(self):
                attention = SimpleNamespace(
                    compress_ratio=0,
                    window_size=128,
                    window_kv_cache=torch.zeros((4, 128, 1)),
                )
                self.layers = [
                    SimpleNamespace(attn=attention, ffn=SimpleNamespace(), engram=None)
                ]
                self.references = ()

            def __call__(self, tokens, start_pos, **kwargs):
                values = (
                    model.precompute_freqs_cis(32, 32, 0, 10000, 1, 32, 1),
                    get_window_topk_idxs(128, 4, 6, 0),
                    get_prefill_window_topk_idxs(128, 4, 6, start_pos),
                    get_dspark_topk_idxs(128, 4, 6, start_pos),
                )
                self.references = tuple(weakref.ref(value) for value in values)
                output = torch.cat(
                    [value.contiguous().view(torch.uint8).flatten() for value in values]
                )
                return tokens[:, 0], output, None

        for operation in cached:
            operation.cache_clear()
        try:
            with torch.inference_mode(), torch.device("cuda"):
                bindings = DecodeBindings(4, (0, 1, 2, 3), "cuda", length=6)
                bindings.bind([[1] * 6] * 4, 2050)
                target = CachedModel()
                expected = target(bindings.token_ids, 2050)[1].clone()
                graph = DecodeGraph(target, bindings, 2050)
                graph.capture()
                references = target.references
                for length in (7, 8, 9):
                    model.precompute_freqs_cis(32, 32 + length, 0, 10000, 1, 32, 1)
                    get_window_topk_idxs(128, 1, length, 0)
                    get_prefill_window_topk_idxs(128, 1, length, 20)
                    get_dspark_topk_idxs(128, 1, length, 20)
                for operation in cached:
                    operation.cache_clear()
                gc.collect()
                self.assertTrue(
                    all(reference() is not None for reference in references)
                )
                for _ in range(2):
                    graph.replay()
                    self.assertTrue(torch.equal(graph.logits, expected))
                del graph
                gc.collect()
                self.assertTrue(all(reference() is None for reference in references))
                bindings.close()
        finally:
            for operation in cached:
                operation.cache_clear()

    def test_cached_tensor_capture_restores_owners_after_nested_failure(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        with kernel.capture_cached_tensors() as outer:
            first = get_prefill_window_topk_idxs(4, 1, 2, 2)
            with (
                self.assertRaisesRegex(RuntimeError, "capture failed"),
                kernel.capture_cached_tensors() as failed,
            ):
                second = get_window_topk_idxs(4, 1, 3, 0)
                raise RuntimeError("capture failed")
            get_prefill_window_topk_idxs(4, 1, 2, 2)
        third = get_dspark_topk_idxs(4, 1, 2, 2)
        self.assertEqual(set(outer), {id(first)})
        self.assertEqual(set(failed), {id(second)})
        self.assertNotIn(id(third), outer)
        self.assertEqual(get_prefill_window_topk_idxs.cache_parameters()["maxsize"], 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cached_projection_and_status_inputs_keep_tensor_owners(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        with (
            torch.inference_mode(),
            patch.dict(kernel_weights._PACKED_WEIGHTS, {}, clear=True),
            patch.dict(kernel._SPARSE_STATUS, {}, clear=True),
        ):
            weight = torch.zeros((128, 128), dtype=torch.float8_e4m3fn, device="cuda")
            scales = torch.full((4, 4), 127, dtype=torch.uint8, device="cuda").view(
                torch.float8_e8m0fnu
            )
            packed = kernel_weights._packed_weight(weight, scales, 8)
            with kernel.capture_cached_tensors() as owners:
                self.assertIs(kernel_weights._packed_weight(weight, scales, 8), packed)
                status = kernel._sparse_status_accumulator(weight.device)
            references = tuple(weakref.ref(value) for value in (*packed[:2], status))
            del packed, status
            kernel.invalidate_packed_weights(weight, scales)
            kernel._SPARSE_STATUS.clear()
            gc.collect()
            self.assertTrue(all(reference() is not None for reference in references))
            owners.clear()
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_prepared_expert_pointer_arrays_own_the_packed_targets(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        with (
            torch.inference_mode(),
            patch.dict(kernel_weights._PACKED_WEIGHTS, {}, clear=True),
        ):
            weight = torch.zeros((128, 64), dtype=torch.uint8, device="cuda").view(
                torch.float4_e2m1fn_x2
            )
            scales = torch.full((128, 4), 127, dtype=torch.uint8, device="cuda").view(
                torch.float8_e8m0fnu
            )
            pointers = kernel_weights._static_pointer_arrays(
                [(weight, scales)], weight.device
            )
            reference = weakref.ref(
                next(iter(kernel_weights._PACKED_WEIGHTS.values())).packed[1]
            )
            kernel.invalidate_packed_weights(weight, scales)
            gc.collect()
            self.assertIsNotNone(reference())
            del pointers
            gc.collect()
            self.assertIsNone(reference())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_capture_pointer_upload_retains_its_cached_host_snapshot(self):
        from oh_my_vllm.models.deepseek_v41 import kernel

        values = [16, 32]
        with (
            torch.inference_mode(),
            patch.dict(kernel_graph._CAPTURE_POINTER_HOSTS, {}, clear=True),
        ):
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            with (
                kernel.capture_cached_tensors() as owners,
                torch.cuda.graph(graph, stream=stream),
            ):
                output = kernel_graph._upload_pointer_array(
                    values, "ownership-test", torch.device("cuda")
                ).clone()
            host = next(iter(kernel_graph._CAPTURE_POINTER_HOSTS.values()))
            reference = weakref.ref(host)
            del host
            kernel_graph._CAPTURE_POINTER_HOSTS.clear()
            gc.collect()
            self.assertIsNotNone(reference())
            graph.replay()
            self.assertEqual(output.tolist(), values)
            del graph
            owners.clear()
            gc.collect()
            self.assertIsNone(reference())


if __name__ == "__main__":
    unittest.main()
