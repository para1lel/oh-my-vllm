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

    def test_moe_tp4_keeps_fp32_through_routed_sum_until_rank_reduce(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            up_raw = torch.zeros((64, 16), dtype=torch.uint8, device=device)
            up_raw[:, 0] = 2
            down_raw = torch.zeros((32, 32), dtype=torch.uint8, device=device)
            down_raw[:, 0] = down_raw[:, 16] = 2
            up_scale = torch.ones((64, 1), device=device).to(torch.float8_e8m0fnu)
            down_scale = torch.tensor([1, 1 / 256], device=device).expand(32, 2)
            down_scale = down_scale.contiguous().to(torch.float8_e8m0fnu)
            up = SimpleNamespace(
                weight=up_raw.view(torch.float4_e2m1fn_x2), scale=up_scale
            )
            down = SimpleNamespace(
                weight=down_raw.view(torch.float4_e2m1fn_x2), scale=down_scale
            )
            expert = SimpleNamespace(w1=up, w3=up, w2=down, swiglu_limit=0)
            values = torch.zeros((1, 32), dtype=torch.bfloat16, device=device)
            values[:, 0] = 1
            for tp, partial in ((True, 1.505859375), (False, 1.5078125)):
                with self.subTest(device=device, tensor_parallel=tp):
                    moe = SimpleNamespace(
                        experts=[expert, expert],
                        experts_start_idx=0,
                        experts_end_idx=2,
                        n_routed_experts=2,
                        tensor_parallel=tp,
                        expert_parallel=not tp,
                        gate=lambda x, mask, device=device: (
                            torch.ones((1, 2), device=device),
                            torch.tensor([[0, 1]], device=device),
                        ),
                        shared_experts=torch.zeros_like,
                    )
                    moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)

                    def reduce_routed(output, partial=partial):
                        torch.testing.assert_close(
                            output, torch.full_like(output, partial), rtol=0, atol=0
                        )
                        self.assertEqual(output.dtype, torch.float32)
                        output.sub_(1.5)

                    with (
                        patch.object(model, "world_size", 4),
                        patch.object(
                            model.dist, "all_reduce", side_effect=reduce_routed
                        ) as reduce,
                    ):
                        actual = model.MoE._forward_rows(moe, values)
                    reduce.assert_called_once()
                    torch.testing.assert_close(
                        actual, torch.full_like(values, partial - 1.5), rtol=0, atol=0
                    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionPrecisionCuda(unittest.TestCase):
    def test_fp4_scales_each_k32_dot_before_accumulation(self):
        # Released FP32 and BF16 results are 646 and 648. Power-of-two scaling
        # preserves these rounded values without overflow or underflow.
        for rows in (1, 33, 125, 126):
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
