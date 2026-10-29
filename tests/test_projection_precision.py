import unittest
from types import SimpleNamespace

import torch
from oh_my_vllm.models.deepseek_v41 import kernel


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


def grouped_literal(counts=(1, 0, 33, 2), max_rows=33):
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
        output=torch.full((rows, 128), -7, dtype=torch.bfloat16, device="cuda"),
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
        module = kernel._projection_module()
        case = grouped_literal()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_grouped(module, case)
        stream.synchronize()
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
                    case.factors * torch.exp2(group_shifts[:, None].float()) * 648
                ).to(torch.bfloat16)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            stream.synchronize()
            graph.reset()

    def test_grouped_output_cannot_alias_encoded_input(self):
        module = kernel._projection_module()
        case = grouped_literal((1, 0, 2, 0), max_rows=2)
        storage = torch.empty(3 * 128 * 2, dtype=torch.uint8, device="cuda")
        encoded = storage[: 3 * 64].view(3, 64)
        encoded.copy_(case.encoded)
        case.encoded = encoded
        alias = storage.view(torch.bfloat16).view(3, 128)
        before = storage.clone()
        with self.assertRaisesRegex(Exception, "operator tensors overlap"):
            run_grouped(module, case, output=alias)
            torch.cuda.current_stream().synchronize()
        torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_grouped_invalid_device_counts_produce_nan(self):
        module = kernel._projection_module()
        case = grouped_literal((1, 0, 2, 0), max_rows=2)
        case.counts.zero_()
        run_grouped(module, case)
        self.assertTrue(bool(case.output.isnan().all()))


if __name__ == "__main__":
    unittest.main()
