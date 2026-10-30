import unittest
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, model


def fp4_literal(rows):
    encoded = torch.zeros((rows, 64), dtype=torch.float32, device="cuda")
    encoded[:, 8] = -1
    encoded[:, 32] = 64
    encoded[:, 48] = 32
    encoded[:, 63] = 3
    encoded = encoded.to(torch.float8_e4m3fn)
    raw = torch.zeros((128, 32), dtype=torch.uint8, device="cuda")
    raw[:, 4], raw[:, 16], raw[:, 24], raw[:, 31] = 2, 6, 4, 32
    row_shifts = torch.arange(rows, device="cuda") % 3 - 1
    column_shifts = torch.arange(128, device="cuda") % 5 - 2
    input_scales = (
        (torch.tensor([113, 127], device="cuda") + row_shifts[:, None])
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    weight_scales = (
        (torch.tensor([126, 128], device="cuda") + column_shifts[:, None])
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    factors = torch.exp2((row_shifts[:, None] + column_shifts[None, :]).float())
    return (
        encoded,
        input_scales,
        raw.view(torch.float4_e2m1fn_x2),
        weight_scales,
        factors,
    )


def grouped_literal(counts=(1, 0, 33, 2), max_rows=33, dtype=torch.bfloat16):
    rows = sum(counts)
    encoded, scales, weight, weight_scales, factors = fp4_literal(rows)
    weights = [
        (
            weight.clone(),
            (weight_scales.view(torch.uint8) + shift).view(weight_scales.dtype),
        )
        for shift in (-1, 0, 1, 2)
    ]
    packed = [kernel._packed_weight(w, s, 4) for w, s in weights]
    pointers = kernel._grouped_pointer_arrays(packed, encoded.device)
    module = kernel._projection_module()
    return SimpleNamespace(
        encoded=encoded.view(torch.uint8),
        scales=scales.view(torch.uint8),
        pointers=pointers,
        counts=torch.tensor(counts, dtype=torch.int32, device="cuda"),
        output=torch.full((rows, 128), -7, dtype=dtype, device="cuda"),
        scratch=torch.empty(
            module.grouped_workspace_size(rows, 4, max_rows, 128, 64),
            dtype=torch.uint8,
            device="cuda",
        ),
        factors=factors,
        max_rows=max_rows,
        weights=weights,
        packed=packed,
    )


def run_grouped(module, case, *, output=None):
    module.grouped_project(
        case.encoded,
        case.scales,
        *case.pointers,
        case.counts,
        case.output if output is None else output,
        case.scratch,
        case.max_rows,
    )


class GroupedProjectionPrecision(unittest.TestCase):
    def test_channel_gather_keeps_rank_order_for_strided_and_empty_rows(self):
        values = torch.tensor([[0, -1, 1, -1], [2, -1, 3, -1]])[:, ::2]
        expected = torch.tensor(
            [[0, 1, 10, 11, 20, 21, 30, 31], [2, 3, 12, 13, 22, 23, 32, 33]]
        )

        def gather(output, value):
            output.copy_(torch.cat([value + offset for offset in (0, 10, 20, 30)]))

        with (
            patch.object(model, "world_size", 4),
            patch.object(model.dist, "all_gather_into_tensor", side_effect=gather),
        ):
            for value in (values, values[:0], values):
                actual = model._gather_channels(value)
                torch.testing.assert_close(
                    actual, expected[: value.size(0)], rtol=0, atol=0
                )
                self.assertTrue(actual.is_contiguous())
        with patch.object(model, "world_size", 1):
            torch.testing.assert_close(
                model._gather_channels(values), values, rtol=0, atol=0
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
            with self.subTest(device=device), patch.object(model, "world_size", 4):
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
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.zeros((1, 128), dtype=torch.bfloat16, device=device)
                values[:, 0] = 1

                def gather(output, value):
                    output.copy_(value.repeat(4, 1))

                with patch.object(
                    model.dist, "all_gather_into_tensor", side_effect=gather
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
            with self.subTest(device=device), patch.object(model, "world_size", 4):
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
                    model.dist, "all_gather_into_tensor", side_effect=gather
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
            with self.subTest(device=device), patch.object(model, "world_size", 4):
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
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.ones((3, 128), dtype=torch.bfloat16, device=device)
                weights = torch.ones((3, 2), device=device)
                indices = torch.tensor([[0, 1], [0, 3], [0, 1]], device=device)

                def gather(output, value):
                    output.copy_(torch.cat([value * factor for factor in (1, 2, 3, 4)]))

                with patch.object(
                    model.dist, "all_gather_into_tensor", side_effect=gather
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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionPrecisionCuda(unittest.TestCase):
    def test_fp4_projection_decodes_every_packed_four_value_word(self):
        encoded = torch.zeros((4, 32), dtype=torch.float32, device="cuda")
        encoded[:, :4] = torch.eye(4, device="cuda")
        encoded = encoded.to(torch.float8_e4m3fn)
        input_scales = torch.ones((4, 1), device="cuda").to(torch.float8_e8m0fnu)
        levels = torch.tensor(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6),
            device="cuda",
        )
        weight_scales = torch.ones((32768, 1), device="cuda").to(torch.float8_e8m0fnu)
        for first in (0, 32768):
            words = torch.arange(first, first + 32768, device="cuda")
            raw = torch.zeros((32768, 16), dtype=torch.uint8, device="cuda")
            raw[:, 0] = (words & 255).to(torch.uint8)
            raw[:, 1] = (words >> 8).to(torch.uint8)
            expected = levels[
                (words[None, :] >> (4 * torch.arange(4, device="cuda")[:, None])) & 15
            ]
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(first=first, dtype=dtype):
                    actual = kernel.fp4_gemm(
                        encoded,
                        input_scales,
                        raw.view(torch.float4_e2m1fn_x2),
                        weight_scales,
                        act_block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        actual, expected.to(dtype), rtol=0, atol=0
                    )

    def test_fp4_scales_each_k32_dot_before_accumulation(self):
        # Released FP32 and BF16 results are 646 and 648. Power-of-two scaling
        # preserves these rounded values without overflow or underflow.
        for rows in (1, 16, 17, 31, 32, 33, 63, 64, 65, 125, 126):
            encoded, scales, weight, weight_scales, factors = fp4_literal(rows)
            for dtype, value in ((torch.float32, 646), (torch.bfloat16, 648)):
                with self.subTest(rows=rows, dtype=dtype):
                    output = kernel.fp4_gemm(
                        encoded,
                        scales,
                        weight,
                        weight_scales,
                        torch.float8_e8m0fnu,
                        act_block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        output, (factors * value).to(dtype), rtol=0, atol=0
                    )

    def test_complete_down_k_keeps_small_terms_in_source_order(self):
        values = torch.zeros((3, 2304), dtype=torch.bfloat16, device="cuda")
        values[:, (0, 576, 1152)] = 1
        raw = torch.zeros((1280, 1152), dtype=torch.uint8, device="cuda")
        raw[:, 0] = raw[:, 288] = 2
        raw[:, 576] = 10
        weight = raw.view(torch.float4_e2m1fn_x2)
        scales = torch.ones((1280, 72), device="cuda")
        scales[:, 18] = 2**-25
        scales = scales.to(torch.float8_e8m0fnu)
        # FP32 source order is (1 + 2**-25) - 1 = 0, not 2**-25.
        actual = kernel.grouped_fp4_gemm(
            values,
            torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda"),
            [(weight, scales)] * 3,
        )
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
        encoded, input_scales = kernel.act_quant(
            values, 32, "ue8m0", torch.float8_e8m0fnu
        )
        actual = kernel.fp4_gemm(
            encoded, input_scales, weight, scales, act_block_size=32
        )
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)

    def test_grouped_fp4_rounding_replays_with_changed_counts(self):
        for dtype, value in ((torch.bfloat16, 648), (torch.float32, 646)):
            with self.subTest(dtype=dtype):
                self.check_grouped_replay(dtype, value)

    def check_grouped_replay(self, dtype, value):
        module = kernel._projection_module()
        case = grouped_literal(dtype=dtype)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_grouped(module, case)
            eager = case.output.clone()
        stream.synchronize()
        shifts = torch.tensor([-1] + [1] * 33 + [2] * 2, device="cuda")
        expected = (case.factors * torch.exp2(shifts[:, None].float()) * value).to(
            dtype
        )
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                run_grouped(module, case)
            for counts in ((1, 0, 33, 2), (16, 1, 0, 19)):
                with self.subTest(counts=counts), torch.cuda.stream(stream):
                    case.counts.copy_(
                        torch.tensor(counts, device="cuda", dtype=torch.int32)
                    )
                    case.output.fill_(-7)
                    graph.replay()
                    actual = case.output.clone()
                    finished = torch.cuda.Event()
                    finished.record()
                finished.synchronize()
                group_shifts = torch.repeat_interleave(
                    torch.tensor([-1, 0, 1, 2], device="cuda"),
                    torch.tensor(counts, device="cuda"),
                )
                expected = (
                    case.factors * torch.exp2(group_shifts[:, None].float()) * value
                ).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            stream.synchronize()
            graph.reset()

    def test_grouped_output_cannot_alias_encoded_input(self):
        module = kernel._projection_module()
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                case = grouped_literal((1, 0, 2, 0), max_rows=2, dtype=dtype)
                storage = torch.empty(
                    3 * 128 * dtype.itemsize, dtype=torch.uint8, device="cuda"
                )
                case.encoded = storage[: 3 * 64].view(3, 64)
                alias = storage.view(dtype).view(3, 128)
                before = storage.clone()
                with self.assertRaisesRegex(Exception, "operator tensors overlap"):
                    run_grouped(module, case, output=alias)
                torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_grouped_rejects_invalid_output_metadata_before_writes(self):
        module = kernel._projection_module()
        case = grouped_literal((1, 0, 2, 0), max_rows=2, dtype=torch.float32)
        for output in (
            torch.full((3, 128), -7, dtype=torch.float16, device="cuda"),
            torch.full((3, 128), -7, dtype=torch.float64, device="cuda"),
            torch.full((3, 128), -7, dtype=torch.int32, device="cuda"),
            torch.full((3, 256), -7, dtype=torch.float32, device="cuda")[:, ::2],
            torch.full((2, 128), -7, dtype=torch.float32, device="cuda"),
            torch.full((3, 128), -7, dtype=torch.float32),
        ):
            with self.subTest(
                dtype=output.dtype, shape=output.shape, device=output.device
            ):
                with self.assertRaisesRegex(Exception, "Check failed"):
                    run_grouped(module, case, output=output)
                torch.testing.assert_close(
                    output, torch.full_like(output, -7), rtol=0, atol=0
                )
        for dtype in (torch.bfloat16, torch.float32):
            empty = grouped_literal((0, 0, 0, 0), max_rows=2, dtype=dtype)
            run_grouped(module, empty)

    def test_grouped_invalid_device_counts_produce_nan(self):
        module = kernel._projection_module()
        for dtype in (torch.bfloat16, torch.float32):
            case = grouped_literal((1, 0, 2, 0), max_rows=2, dtype=dtype)
            for counts in ((0, 0, 0, 0), (-1, 2, 2, 0), (3, 0, 0, 0)):
                with self.subTest(dtype=dtype, counts=counts):
                    case.counts.copy_(
                        torch.tensor(counts, device="cuda", dtype=torch.int32)
                    )
                    case.output.fill_(-7)
                    run_grouped(module, case)
                    self.assertTrue(bool(case.output.isnan().all()))


if __name__ == "__main__":
    unittest.main()
