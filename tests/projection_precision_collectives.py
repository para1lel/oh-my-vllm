import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, model, model_context, model_expert


class GroupedProjectionPrecisionCollectives(unittest.TestCase):
    def test_prepare_weights_accepts_torch_uint8_storage(self):
        from oh_my_vllm.operators.projection import workspace as projection

        class Weights(SimpleNamespace):
            def __iter__(self):
                return iter((self.encoded, self.scales, self.encoded, self.scales, 8))

        calls = []
        device = SimpleNamespace(
            empty=lambda shape, dtype: torch.empty(shape, dtype=torch.uint8),
            synchronize_compute=lambda: None,
        )
        module = SimpleNamespace(pack_weights=lambda *args: calls.append(args))
        weights = Weights(
            encoded=torch.zeros((32, 64), dtype=torch.uint8),
            scales=torch.zeros((1, 2), dtype=torch.uint8),
        )
        with (
            patch.object(projection, "require_eager", lambda _: None),
            patch.object(projection, "load", return_value=(module, None, None)),
        ):
            result = projection.prepare_weights(device, weights, bits=8)
            self.assertEqual((result.width, result.bits, result.outputs), (64, 8, 32))
            self.assertTrue(calls)
            with self.assertRaisesRegex(ValueError, "raw weight type"):
                projection.prepare_weights(
                    device,
                    Weights(
                        encoded=torch.zeros((32, 64), dtype=torch.int8),
                        scales=torch.zeros((1, 2), dtype=torch.uint8),
                    ),
                    bits=8,
                )

    def test_channel_gather_keeps_rank_order_for_strided_and_empty_rows(self):
        values = torch.tensor([[0, -1, 1, -1], [2, -1, 3, -1]])[:, ::2]
        expected = torch.tensor(
            [[0, 1, 10, 11, 20, 21, 30, 31], [2, 3, 12, 13, 22, 23, 32, 33]]
        )

        def gather(output, value):
            output.copy_(torch.cat([value + offset for offset in (0, 10, 20, 30)]))

        with (
            patch.object(model_context, "world_size", 4),
            patch.object(
                torch.distributed, "all_gather_into_tensor", side_effect=gather
            ),
        ):
            for value in (values, values[:0], values):
                actual = model_expert._gather_channels(value)
                torch.testing.assert_close(
                    actual, expected[: value.size(0)], rtol=0, atol=0
                )
                self.assertTrue(actual.is_contiguous())
        with patch.object(model_context, "world_size", 1):
            torch.testing.assert_close(
                model_expert._gather_channels(values), values, rtol=0, atol=0
            )

    def test_adapter_output_dtype_keeps_unrounded_values_and_empty_shape(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            values = torch.zeros((2, 32), dtype=torch.bfloat16, device=device)
            values[:, :2] = torch.tensor([1, 1 / 256], device=device)
            weight = torch.full((32, 16), 0x22, dtype=torch.uint8, device=device)
            scales = torch.ones((32, 1), device=device).to(torch.float8_e8m0fnu)
            weights = [(weight.view(torch.float4_e2m1fn_x2), scales)] * 3
            counts = torch.tensor([1, 0, 1], dtype=torch.int32, device=device)
            for dtype, expected in ((torch.bfloat16, 1), (torch.float32, 1 + 1 / 256)):
                with self.subTest(device=device, dtype=dtype):
                    actual = kernel.grouped_fp4_gemm(
                        values, counts, weights, output_dtype=dtype
                    )
                    torch.testing.assert_close(
                        actual,
                        torch.full((2, 32), expected, dtype=dtype, device=device),
                        rtol=0,
                        atol=0,
                    )
                    empty = kernel.grouped_fp4_gemm(
                        values[:0], counts * 0, weights, output_dtype=dtype
                    )
                    self.assertEqual((empty.shape, empty.dtype), ((0, 32), dtype))
            default = kernel.grouped_fp4_gemm(values, counts, weights)
            self.assertEqual(default.dtype, torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "output dtype"):
                kernel.grouped_fp4_gemm(
                    values, counts, weights, output_dtype=torch.float16
                )

    def test_moe_tp4_rounds_complete_down_outputs_before_six_route_merge(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with (
                self.subTest(device=device),
                patch.object(model_context, "world_size", 4),
            ):
                expert = model.Expert(
                    128, 256, dtype=torch.float4_e2m1fn_x2, tensor_parallel=True
                ).to(device)
                self.assertEqual(expert.w2.in_features, 256)
                gate_raw = torch.zeros((64, 64), dtype=torch.uint8, device=device)
                gate_raw[0, 0] = 6
                up_raw = torch.zeros_like(gate_raw)
                up_raw[0, 0] = 2
                down_raw = torch.zeros((32, 128), dtype=torch.uint8, device=device)
                down_raw[:, 0] = down_raw[:, 32] = 2
                down_raw[:, 64] = 10
                down_scales = torch.ones((32, 8), device=device)
                down_scales[:, 0] = 1 / 4
                down_scales[:, 2] = 1 / 1024
                down_scales[:, 4] = 1 / 4096
                with torch.no_grad():
                    expert.w1.weight.copy_(gate_raw.view(torch.float4_e2m1fn_x2))
                    expert.w3.weight.copy_(up_raw.view(torch.float4_e2m1fn_x2))
                    expert.w2.weight.copy_(down_raw.view(torch.float4_e2m1fn_x2))
                    expert.w1.scale.fill_(1)
                    expert.w3.scale.fill_(1)
                    expert.w2.scale.copy_(down_scales.to(torch.float8_e8m0fnu))
                moe = SimpleNamespace(
                    experts=[expert] * 6,
                    experts_start_idx=0,
                    experts_end_idx=6,
                    n_routed_experts=6,
                    tensor_parallel=True,
                    expert_parallel=False,
                    gate=lambda x, mask, device=device: (
                        torch.ones((1, 6), device=device),
                        torch.arange(6, device=device).view(1, 6),
                    ),
                    shared_experts=torch.zeros_like,
                )
                moe.prepare_routed_experts = MethodType(
                    model.MoE.prepare_routed_experts, moe
                )
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.zeros((1, 128), dtype=torch.bfloat16, device=device)
                values[:, 0] = 1

                def gather(output, value):
                    output.copy_(value.repeat(4, 1))

                with patch.object(
                    torch.distributed, "all_gather_into_tensor", side_effect=gather
                ):
                    actual = model.MoE._forward_rows(moe, values)
                # Each complete expert gives 1 + 1/256 - 1/1024, rounded to 1.
                # Rounding only after the six-route merge would give 6.03125.
                torch.testing.assert_close(
                    actual, torch.full_like(values, 6), rtol=0, atol=0
                )

    def test_tp4_fallback_gathers_product_and_merged_output_channels(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with (
                self.subTest(device=device),
                patch.object(model_context, "world_size", 4),
            ):
                expert = model.Expert(
                    4, 128, dtype=torch.bfloat16, tensor_parallel=True
                ).to(device)
                self.assertEqual(expert.w2.in_features, 128)
                with torch.no_grad():
                    for parameter in expert.parameters():
                        parameter.zero_()
                    expert.w1.weight[0, 0] = expert.w3.weight[0, 0] = 1
                    expert.w2.weight[:, ::32] = 1
                moe = SimpleNamespace(
                    experts=[expert] * 4,
                    experts_start_idx=0,
                    experts_end_idx=4,
                    n_routed_experts=4,
                    tensor_parallel=True,
                    expert_parallel=False,
                    gate=lambda x, mask, device=device: (
                        torch.ones((2, 1), device=device),
                        torch.zeros((2, 1), dtype=torch.long, device=device),
                    ),
                    shared_experts=torch.zeros_like,
                )
                moe.prepare_routed_experts = MethodType(
                    model.MoE.prepare_routed_experts, moe
                )
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.zeros((2, 4), dtype=torch.bfloat16, device=device)
                values[:, 0] = torch.tensor([1, 2], device=device)

                def gather(output, value):
                    if value.dtype == torch.bfloat16:
                        output.copy_(value.repeat(4, 1))
                    else:
                        output.copy_(
                            torch.cat([value * factor for factor in (1, 2, 3, 4)])
                        )

                with patch.object(
                    torch.distributed, "all_gather_into_tensor", side_effect=gather
                ):
                    actual = model.MoE._forward_rows(moe, values)
                torch.testing.assert_close(
                    actual,
                    torch.tensor(
                        [
                            [2.921875, 5.84375, 8.75, 11.6875],
                            [14.0625, 28.125, 42.25, 56.25],
                        ],
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    rtol=0,
                    atol=0,
                )

    def test_grouped_tp4_gathers_distinct_channels_with_uneven_routes(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with (
                self.subTest(device=device),
                patch.object(model_context, "world_size", 4),
            ):
                up_raw = torch.zeros((64, 64), dtype=torch.uint8, device=device)
                up_raw[0, 0] = 2
                down_raw = torch.zeros((32, 128), dtype=torch.uint8, device=device)
                down_raw[:, ::32] = torch.tensor([2, 4, 5, 6], device=device)
                up = SimpleNamespace(
                    weight=up_raw.view(torch.float4_e2m1fn_x2),
                    scale=torch.ones((64, 4), device=device).to(torch.float8_e8m0fnu),
                )
                down = SimpleNamespace(
                    weight=down_raw.view(torch.float4_e2m1fn_x2),
                    scale=torch.ones((32, 8), device=device).to(torch.float8_e8m0fnu),
                )
                expert = SimpleNamespace(w1=up, w3=up, w2=down, swiglu_limit=0)
                moe = SimpleNamespace(
                    experts=[expert] * 4,
                    experts_start_idx=0,
                    experts_end_idx=4,
                    n_routed_experts=4,
                    tensor_parallel=True,
                    expert_parallel=False,
                    shared_experts=torch.zeros_like,
                )
                moe.prepare_routed_experts = MethodType(
                    model.MoE.prepare_routed_experts, moe
                )
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.ones((3, 128), dtype=torch.bfloat16, device=device)
                weights = torch.ones((3, 2), device=device)
                indices = torch.tensor([[0, 1], [0, 3], [0, 1]], device=device)

                def gather(output, value):
                    output.copy_(torch.cat([value * factor for factor in (1, 2, 3, 4)]))

                with patch.object(
                    torch.distributed, "all_gather_into_tensor", side_effect=gather
                ):
                    actual = model.MoE._forward_rows(
                        moe, values, routing=(weights, indices)
                    )
                # Quantized channel values are .75, 1.5, 2.25, 3; down weights
                # are 1, 2, 3, 4. Each expert gives 22.5 before two-route merging.
                expected = torch.cat(
                    [
                        torch.full((3, 32), value, dtype=torch.bfloat16, device=device)
                        for value in (45, 90, 135, 180)
                    ],
                    dim=-1,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
