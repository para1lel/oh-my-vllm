import unittest
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, kernel_weights

from tests.deepseek_operators_fixtures import projection_reference, projection_weight


class DeepSeekOperatorContractsGrouped(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_reuses_values_and_matches_quantized_reference(self):
        width = 128
        weights = [projection_weight(4, 64, width, seed=seed) for seed in (7, 11, 13)]
        counts = torch.tensor((2, 0, 3), device="cuda", dtype=torch.int32)
        values = torch.linspace(-2, 2, 5 * width, device="cuda").view(5, width)
        encoded, input_scales = kernel.act_quant(
            values, 32, "ue8m0", torch.float8_e8m0fnu
        )
        snapshots = [
            (w.view(torch.uint8).clone(), s.view(torch.uint8).clone())
            for w, s in weights
        ]
        with patch.dict(kernel_weights._PACKED_WEIGHTS, clear=True):
            actual = kernel.grouped_fp4_gemm(values, counts, weights)
            expected = []
            first = 0
            for count, (weight, scale), (raw_before, scale_before) in zip(
                (2, 0, 3), weights, snapshots, strict=True
            ):
                if count:
                    packed = kernel_weights._packed_weight(weight, scale, 4)
                    self.assertEqual(packed[0].data_ptr(), weight.data_ptr())
                    expected.append(
                        projection_reference(
                            encoded[first : first + count],
                            input_scales[first : first + count],
                            weight,
                            scale,
                            4,
                        )
                    )
                torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                torch.testing.assert_close(scale.view(torch.uint8), scale_before)
                first += count
            torch.testing.assert_close(actual, torch.cat(expected), atol=0.1, rtol=0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_matches_individual_projections(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(41)
        values = torch.randn(
            (5, 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        weights = []
        for seed in (7, 11, 13):
            weight_generator = torch.Generator(device=device).manual_seed(seed)
            raw = torch.randint(
                0,
                256,
                (32, 16),
                generator=weight_generator,
                device=device,
                dtype=torch.uint8,
            )
            weights.append(
                (
                    raw.view(torch.float4_e2m1fn_x2),
                    torch.ones((32, 1), device=device, dtype=torch.float8_e8m0fnu),
                )
            )
        counts = torch.tensor((2, 0, 3), device=device, dtype=torch.int32)

        actual = kernel.grouped_fp4_gemm(values, counts, weights)

        expected_parts = []
        first = 0
        for count, (weight, scale) in zip(counts.tolist(), weights, strict=True):
            if count:
                encoded, input_scale = kernel.act_quant(
                    values[first : first + count],
                    32,
                    "ue8m0",
                    torch.float8_e8m0fnu,
                )
                expected_parts.append(
                    kernel.fp4_gemm(
                        encoded,
                        input_scale,
                        weight,
                        scale,
                        torch.float8_e8m0fnu,
                        32,
                    )
                )
            first += count
        expected = torch.cat(expected_parts)
        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_device_counts_match_host_counts(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(53)
        values = torch.randn(
            (6, 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        weights = []
        for seed in (29, 31, 37):
            weight_generator = torch.Generator(device=device).manual_seed(seed)
            raw = torch.randint(
                0,
                256,
                (32, 16),
                generator=weight_generator,
                device=device,
                dtype=torch.uint8,
            )
            weights.append(
                (
                    raw.view(torch.float4_e2m1fn_x2),
                    torch.ones((32, 1), device=device, dtype=torch.float8_e8m0fnu),
                )
            )
        counts = torch.tensor((1, 0, 5), device=device, dtype=torch.int32)

        expected = kernel.grouped_fp4_gemm(
            values, counts, weights, counts_list=(1, 0, 5)
        )
        actual = kernel.grouped_fp4_gemm(values, counts, weights)

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_rejects_device_counts_that_miss_a_row(self):
        device = torch.device("cuda")
        raw = torch.zeros((32, 16), device=device, dtype=torch.uint8)
        weights = [
            (
                raw.view(torch.float4_e2m1fn_x2),
                torch.ones((32, 1), device=device, dtype=torch.float8_e8m0fnu),
            )
            for _ in range(4)
        ]
        for rows, counts in ((6, (1, 0, 2, 0)), (2, (1, 1, 5, 0))):
            # The second case drops a group that exceeds the row bound, so the
            # compacted counts alone would look valid.
            values = torch.zeros((rows, 32), device=device, dtype=torch.bfloat16)
            actual = kernel.grouped_fp4_gemm(
                values,
                torch.tensor(counts, device=device, dtype=torch.int32),
                weights,
            )
            self.assertFalse(torch.isfinite(actual).all())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_capture_replays_the_metadata_upload(self):
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(43)
        values = torch.randn(
            (5, 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        weights = []
        for seed in (7, 11, 13):
            weight_generator = torch.Generator(device=device).manual_seed(seed)
            raw = torch.randint(
                0,
                256,
                (32, 16),
                generator=weight_generator,
                device=device,
                dtype=torch.uint8,
            )
            weights.append(
                (
                    raw.view(torch.float4_e2m1fn_x2),
                    torch.ones((32, 1), device=device, dtype=torch.float8_e8m0fnu),
                )
            )
        counts = torch.tensor((2, 0, 3), device=device, dtype=torch.int32)
        expected = kernel.grouped_fp4_gemm(
            values, counts, weights, counts_list=(2, 0, 3)
        )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                actual = kernel.grouped_fp4_gemm(
                    values, counts, weights, counts_list=(2, 0, 3)
                )
            actual.fill_(0)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            graph.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_device_counts_cover_rows_past_one_tile(self):
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(59)
        values = torch.randn(
            (37, 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        weights = []
        for seed in (61, 67, 71):
            weight_generator = torch.Generator(device=device).manual_seed(seed)
            raw = torch.randint(
                0,
                256,
                (32, 16),
                generator=weight_generator,
                device=device,
                dtype=torch.uint8,
            )
            weights.append(
                (
                    raw.view(torch.float4_e2m1fn_x2),
                    torch.ones((32, 1), device=device, dtype=torch.float8_e8m0fnu),
                )
            )
        counts = torch.tensor((19, 0, 18), device=device, dtype=torch.int32)
        expected = kernel.grouped_fp4_gemm(
            values, counts, weights, counts_list=(19, 0, 18)
        )
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                actual = kernel.grouped_fp4_gemm(values, counts, weights)
            actual.fill_(0)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            graph.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_gemm_uses_the_caller_default_device(self):
        device = torch.device("cuda:0")
        generator = torch.Generator(device=device).manual_seed(47)
        values = torch.randn(
            (5, 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        weights = []
        for seed in (17, 19, 23):
            weight_generator = torch.Generator(device=device).manual_seed(seed)
            raw = torch.randint(
                0,
                256,
                (32, 16),
                generator=weight_generator,
                device=device,
                dtype=torch.uint8,
            )
            weights.append(
                (
                    raw.view(torch.float4_e2m1fn_x2),
                    torch.ones((32, 1), device=device, dtype=torch.float8_e8m0fnu),
                )
            )
        counts = torch.tensor((2, 0, 3), device=device, dtype=torch.int32)
        previous = torch.get_default_device()
        torch.set_default_device(device)
        try:
            actual = kernel.grouped_fp4_gemm(
                values, counts, weights, counts_list=(2, 0, 3)
            )
            torch.cuda.synchronize()
        finally:
            torch.set_default_device(previous)
        expected = kernel.grouped_fp4_gemm(
            values, counts, weights, counts_list=(2, 0, 3)
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_swiglu_accepts_an_unaligned_route_slice(self):
        gate = torch.zeros((8, 576), device="cuda", dtype=torch.bfloat16)
        up = torch.zeros_like(gate)
        route = torch.ones((9,), device="cuda", dtype=torch.float32)[1:]

        actual = kernel.swiglu(gate, up, route, 10)

        torch.testing.assert_close(actual, torch.zeros_like(gate))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_swiglu_accepts_six_route_rows_per_prefill_token(self):
        rows = 8192 * 6
        gate = torch.zeros((rows, 576), device="cuda", dtype=torch.bfloat16)
        up = torch.zeros_like(gate)
        route = torch.ones((rows,), device="cuda", dtype=torch.float32)

        actual = kernel.swiglu(gate, up, route, 10)

        self.assertEqual(tuple(actual.shape), tuple(gate.shape))
        torch.testing.assert_close(actual, torch.zeros_like(gate))
