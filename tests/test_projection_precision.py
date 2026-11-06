import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel_weights, model, model_context
from oh_my_vllm.models.deepseek_v41.decode import DecodeBindings, DecodeGraph

from tests.projection_precision_bindings import (
    GroupedProjectionPrecisionBindings as GroupedProjectionPrecisionBindings,
)
from tests.projection_precision_collectives import (
    GroupedProjectionPrecisionCollectives as GroupedProjectionPrecisionCollectives,
)
from tests.projection_precision_cuda_padded import (
    ProjectionPrecisionCudaCudaPadded as ProjectionPrecisionCudaCudaPadded,
)
from tests.projection_precision_cuda_rounding import (
    ProjectionPrecisionCuda as ProjectionPrecisionCuda,
)
from tests.projection_precision_fixtures import (
    _RoutedDecodeModel,
    pack_weight_bytes,
    prepared_moe_literal,
)


class GroupedProjectionPrecision(unittest.TestCase):
    def test_decode_graph_rejects_rebound_expert_bindings_without_synchronization(self):
        for strategy in ("TP4", "TEP4"):
            for segmented in (False, True):
                for change in (
                    "expert",
                    "projection",
                    "weight",
                    "scale",
                    "load",
                    "apply",
                ):
                    with (
                        self.subTest(
                            strategy=strategy, segmented=segmented, change=change
                        ),
                        patch.object(model_context, "world_size", 1),
                    ):
                        moe, target, bindings, stream, native_graph = (
                            self._decode_rebinding_fixture(strategy)
                        )
                        with (
                            patch("torch.cuda.Stream", return_value=stream),
                            patch("torch.cuda.CUDAGraph", return_value=native_graph),
                            patch("torch.cuda.current_stream", return_value=stream),
                            patch(
                                "torch.cuda.stream",
                                side_effect=lambda *_: nullcontext(),
                            ),
                            patch(
                                "torch.cuda.graph",
                                side_effect=lambda *_args, **_kwargs: nullcontext(),
                            ),
                            patch("torch.cuda.synchronize") as synchronize,
                            patch(
                                "oh_my_vllm.models.deepseek_v41.decode.require_sparse_status"
                            ),
                        ):
                            graph = DecodeGraph(
                                target, bindings, 132, segmented=segmented
                            )
                            graph.capture()
                            synchronize.reset_mock()
                            graph.replay()
                            native_graph.replay.assert_called_once()
                            synchronize.assert_not_called()
                            native_graph.replay.reset_mock()
                            self._rebind_projection(moe, change)
                            with self.assertRaisesRegex(
                                RuntimeError, "bindings.*changed"
                            ):
                                graph.replay()
                            native_graph.replay.assert_not_called()
                            synchronize.assert_not_called()
                            # Repreparation cannot validate the old graph.
                            moe.prepare_routed_experts()
                            with self.assertRaisesRegex(
                                RuntimeError, "bindings.*changed"
                            ):
                                graph.replay()
                            native_graph.replay.assert_not_called()
                            synchronize.assert_not_called()
                            replacement_graph = DecodeGraph(
                                target, bindings, 132, segmented=segmented
                            )
                            replacement_graph.capture()
                            synchronize.reset_mock()
                            replacement_graph.replay()
                            native_graph.replay.assert_called_once()
                            synchronize.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_decode_graph_rebind_reject_and_recapture_match_eager_bits(self):
        with patch.object(model_context, "world_size", 1):
            moe, values, routing = prepared_moe_literal("cuda")
            target = _RoutedDecodeModel(moe, values, routing)
            bindings = DecodeBindings(1, (0,), "cuda")
            bindings.bind([1], 132)
            with torch.inference_mode():
                expected = target(bindings.token_ids)[1].clone()
                graph = DecodeGraph(target, bindings, 132).capture()
                graph.replay()
                self.assertTrue(
                    torch.equal(
                        graph.logits.view(torch.uint8), expected.view(torch.uint8)
                    )
                )
                for expert in moe.experts:
                    weight = expert.w2.weight.view(torch.uint8).clone()
                    weight[0, 0] = 4
                    expert.w2.weight = torch.nn.Parameter(
                        weight.view(torch.float4_e2m1fn_x2), requires_grad=False
                    )
                with self.assertRaisesRegex(RuntimeError, "bindings.*changed"):
                    graph.replay()
                expected = target(bindings.token_ids)[1].clone()
                wanted = torch.zeros_like(values)
                wanted[:, 0] = 3
                self.assertTrue(torch.equal(expected, wanted))
                with self.assertRaisesRegex(RuntimeError, "bindings.*changed"):
                    graph.replay()
                replacement = DecodeGraph(target, bindings, 132).capture()
                replacement.replay()
                self.assertTrue(
                    torch.equal(
                        replacement.logits.view(torch.uint8), expected.view(torch.uint8)
                    )
                )

    def test_packed_projection_refreshes_after_in_place_state_load(self):
        for inference in (False, True):
            with (
                self.subTest(inference=inference),
                torch.inference_mode(inference),
                patch.dict(kernel_weights._PACKED_WEIGHTS, {}, clear=True),
                patch.object(
                    kernel_weights,
                    "_projection_module",
                    return_value=SimpleNamespace(pack_weights=pack_weight_bytes),
                ),
            ):
                layer = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
                with torch.no_grad():
                    layer.weight.view(torch.uint8).fill_(18)
                    layer.scale.view(torch.uint8).fill_(127)
                alias_weight = layer.weight[8:24]
                alias_scale = layer.scale[8:24]
                self.assertNotEqual(alias_weight.data_ptr(), layer.weight.data_ptr())
                self.assertNotEqual(alias_scale.data_ptr(), layer.scale.data_ptr())
                alias_first = kernel_weights._packed_weight(
                    alias_weight, alias_scale, 4
                )
                first = kernel_weights._packed_weight(layer.weight, layer.scale, 4)
                self.assertIs(
                    first, kernel_weights._packed_weight(layer.weight, layer.scale, 4)
                )
                addresses = (layer.weight.data_ptr(), layer.scale.data_ptr())
                state = {
                    name: value.view(torch.uint8).clone().view(value.dtype)
                    for name, value in layer.state_dict().items()
                }
                state["weight"].view(torch.uint8).fill_(52)
                state["scale"].view(torch.uint8).fill_(129)
                layer.load_state_dict(state, assign=False)
                self.assertEqual(
                    addresses, (layer.weight.data_ptr(), layer.scale.data_ptr())
                )
                refreshed = kernel_weights._packed_weight(layer.weight, layer.scale, 4)
                self.assertTrue(
                    torch.equal(refreshed[0][:, :32], state["weight"].view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(
                        refreshed[1][:64], state["scale"].view(torch.uint8).flatten()
                    )
                )
                self.assertEqual(int(first[0][0, 0]), 18)
                self.assertEqual(int(first[1][0]), 127)
                alias_refreshed = kernel_weights._packed_weight(
                    alias_weight, alias_scale, 4
                )
                self.assertTrue(
                    torch.equal(
                        alias_refreshed[1][:32],
                        state["scale"][8:24].view(torch.uint8).flatten(),
                    )
                )
                self.assertEqual(int(alias_first[1][0]), 127)

    def test_packed_projection_tracks_direct_shared_view_mutations(self):
        with (
            patch.dict(kernel_weights._PACKED_WEIGHTS, {}, clear=True),
            patch.object(
                kernel_weights,
                "_projection_module",
                return_value=SimpleNamespace(pack_weights=pack_weight_bytes),
            ),
            torch.no_grad(),
        ):
            layer = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
            layer.weight.view(torch.uint8).fill_(18)
            layer.scale.view(torch.uint8).fill_(127)
            first = kernel_weights._packed_weight(layer.weight, layer.scale, 4)
            layer.scale.detach().view(torch.uint8).fill_(129)
            updated_scale = kernel_weights._packed_weight(layer.weight, layer.scale, 4)
            self.assertEqual(int(updated_scale[1][0]), 129)
            self.assertEqual(int(first[1][0]), 127)
            layer.weight.detach().view(torch.uint8).fill_(52)
            updated_weight = kernel_weights._packed_weight(layer.weight, layer.scale, 4)
            self.assertTrue(torch.all(updated_weight[0][:, :32] == 52))
            self.assertTrue(torch.all(updated_weight[0][:, 32:] == 0))
            self.assertIs(
                updated_weight,
                kernel_weights._packed_weight(layer.weight, layer.scale, 4),
            )
            self.assertEqual(len(kernel_weights._PACKED_WEIGHTS), 1)

    def _rebind_projection(self, moe, change):
        expert = moe.experts[0]
        if change == "expert":
            moe.experts[0] = model.Expert(32, 64, dtype=torch.float4_e2m1fn_x2)
        elif change == "projection":
            expert.w2 = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
        elif change in ("weight", "scale"):
            previous = getattr(expert.w2, change)
            replacement = previous.view(torch.uint8).clone().view(previous.dtype)
            setattr(
                expert.w2,
                change,
                torch.nn.Parameter(replacement, requires_grad=False),
            )
        elif change == "load":
            moe.load_state_dict(moe.state_dict(), assign=True)
        else:
            expert.w2._apply(lambda value: value)

    def _decode_rebinding_fixture(self, strategy):
        moe, values, routing = prepared_moe_literal("cpu", strategy)
        layer = SimpleNamespace(
            ffn=moe,
            engram=None,
            attn=SimpleNamespace(compress_ratio=0, window_size=128),
        )
        target = Mock()
        target._state_export = None
        target._physical_cache = None
        target.layers = [layer]
        target.modules.return_value = (moe,)

        def forward(*_args, moe=moe, values=values, routing=routing, **_kwargs):
            return (
                None,
                moe._forward_rows(values, routing=routing),
                None,
            )

        target.side_effect = forward
        bindings = DecodeBindings(1, (0,), "cpu")
        bindings.bind([1], 132)
        stream, native_graph = Mock(), Mock()
        return moe, target, bindings, stream, native_graph


if __name__ == "__main__":
    unittest.main()
