import os
import subprocess
import sys
import unittest
from itertools import pairwise
from unittest.mock import patch

import torch
import torch.nn.functional as F
from oh_my_vllm.models.deepseek_v41 import kernel, model


def _assert_released_sequential_projection(module, output, values, weight, block=8192):
    """Check the released single-chunk order through the HC sequential kernel.

    One 20480-wide chunk is a plain sequence of fused multiply-adds. Zero
    padding after width 5120 keeps that sequence for the FP32 shapes. Rows and
    outputs are compared in blocks to bound the temporary storage. The block
    stays in the 5187..32768 row range, where the HC chunk is 20480.
    """
    rows, outputs = values.size(0), weight.size(0)
    edges = [*list(range(0, rows, block)), rows]
    if len(edges) > 2 and edges[-1] - edges[-2] == 1:
        edges.pop(-2)
    workspace = torch.empty(
        (module.workspace_size(block),), dtype=torch.uint8, device=values.device
    )
    for first, stop in pairwise(edges):
        padded = torch.zeros((stop - first, 20480), device=values.device)
        padded[:, :5120] = values[first:stop]
        for start in range(0, outputs, 24):
            count = min(24, outputs - start)
            padded_weight = torch.zeros((24, 20480), device=values.device)
            padded_weight[:count, :5120] = weight[start : start + count]
            part = torch.empty((stop - first, 24), device=values.device)
            module.hc_chunked_project(padded, padded_weight, part, workspace)
            assert torch.equal(
                output[first:stop, start : start + count], part[:, :count]
            )


class LogicalRowsCpu(unittest.TestCase):
    def test_moe_keeps_logical_gate_rows_across_dispatch_chunks(self):
        class RoutedExpert(torch.nn.Module):
            def __init__(self, coefficient):
                super().__init__()
                self.coefficient = coefficient

            def forward(self, values, weights):
                return (weights * self.coefficient).expand_as(values)

        class ZeroExpert(torch.nn.Module):
            def forward(self, values):
                return torch.zeros_like(values)

        args = model.ModelArgs(
            dim=2, moe_inter_dim=2, n_routed_experts=4, n_activated_experts=2
        )
        moe = model.MoE(0, args)
        moe.experts = torch.nn.ModuleList(RoutedExpert(i + 1) for i in range(4))
        moe.shared_experts = ZeroExpert()
        with torch.no_grad():
            moe.gate.weight.copy_(torch.tensor(((1, 0), (0, 1), (-1, 0), (0, -1))))
            moe.gate.bias.zero_()
        batch, steps = 3, 343
        values = torch.arange(batch * steps * 2, dtype=torch.float32).view(
            batch, steps, 2
        )
        values = values.remainder(19) / 8 + 0.125
        logits = torch.stack(
            [
                F.linear(values[:, step].contiguous(), moe.gate.weight.float())
                for step in range(steps)
            ],
            dim=1,
        ).flatten(0, 1)
        scores = F.softplus(logits).sqrt()
        indices = scores.topk(2, dim=-1).indices
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(-1, keepdim=True) + 1e-20) * args.route_scale
        expected = (weights * (indices + 1)).sum(-1, keepdim=True).expand(-1, 2)
        calls = []
        handle = moe.gate.register_forward_pre_hook(
            lambda module, inputs, kwargs: calls.append(
                (inputs[0].size(0), kwargs["logical_rows"])
            ),
            with_kwargs=True,
        )
        try:
            actual = moe(values, logical_rows=batch)
        finally:
            handle.remove()
        torch.testing.assert_close(actual, expected.view_as(values), rtol=0, atol=0)
        self.assertEqual(calls, [(batch * steps, batch)])

    def test_bf16_linear_falls_back_for_three_dimensional_values(self):
        values = torch.randn(2, 3, 8).bfloat16()
        weight = torch.randn(5, 8).bfloat16()
        expected = (
            F.linear(values.reshape(6, 8).float(), weight.float())
            .bfloat16()
            .reshape(2, 3, 5)
        )
        actual = kernel.bf16_linear(values, weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_native_sparse_attention_is_default_with_explicit_opt_out(self):
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
        code = (
            "from oh_my_vllm.models.deepseek_v41 import kernel; "
            "print(int(kernel._SPARSE_NATIVE_DISABLED))"
        )
        for setting, disabled in ((None, "0"), ("0", "1"), ("1", "0")):
            with self.subTest(setting=setting):
                environment.pop("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", None)
                if setting is not None:
                    environment["OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE"] = setting
                result = subprocess.check_output(
                    [sys.executable, "-c", code], env=environment, text=True
                )
                self.assertEqual(result.strip(), disabled)

    def test_engram_reference_keeps_mask_and_logical_row_shape(self):
        residual = (
            torch.arange(6, dtype=torch.bfloat16).view(2, 3, 1, 1).expand(2, 3, 4, 2)
        )
        kv = torch.zeros((2, 3, 10), dtype=torch.bfloat16)
        kv[..., 8:] = 2
        weight = torch.ones((4, 2), dtype=torch.bfloat16)
        mask = torch.tensor(((True, False, True), (False, True, False)))
        expected = residual + mask[..., None, None].to(torch.bfloat16)
        for logical_rows in (0, 1, 2, 3, 6):
            actual = kernel.engram_mix(
                residual, kv, weight, weight, mask, logical_rows=logical_rows
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "logical row"):
            kernel.engram_mix(residual, kv, weight, weight, mask, logical_rows=4)

    def test_fp32_projection_reference_keeps_row_order_and_validates_groups(self):
        values = torch.arange(6 * 5120, dtype=torch.float32).view(6, 5120)
        weight = torch.zeros((2, 5120))
        weight[0, 0], weight[0, 1] = 1.0, 2.0
        weight[1, 0], weight[1, 1] = -1.0, 1.0
        expected = torch.tensor(
            ((2, 1), (15362, 1), (30722, 1), (46082, 1), (61442, 1), (76802, 1))
        ).float()
        for logical_rows in (0, 1, 2, 3, 6):
            actual = kernel.fp32_project(values, weight, logical_rows=logical_rows)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for invalid in (-1, 4, 7, True, 1.0):
            with self.assertRaisesRegex(ValueError, "logical row"):
                kernel.fp32_project(values, weight, logical_rows=invalid)
        with self.assertRaisesRegex(ValueError, "width must be 5120"):
            kernel.fp32_project(torch.zeros((2, 4)), torch.zeros((1, 4)))

    def test_hc_reference_rejects_invalid_logical_rows(self):
        residual = torch.ones((2, 3, 4, 2), dtype=torch.bfloat16)
        incoming = torch.ones((2, 3, 4))
        weight = torch.zeros((24, 8))
        scale = torch.ones(3)
        base = torch.zeros(24)
        for invalid in (-1, 4, 7, True, 1.0):
            with self.assertRaisesRegex(ValueError, "logical row"):
                kernel.hc_prepare(
                    residual, incoming, weight, scale, base, logical_rows=invalid
                )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LogicalRowsCuda(unittest.TestCase):
    def test_index_rms_uses_native_logical_reduction(self):
        values = (
            torch.linspace(-1, 1, 3 * 6 * 128, device="cuda").bfloat16().view(3, 6, 128)
        )
        weight = torch.ones(128, dtype=torch.bfloat16, device="cuda")
        reference = values.float()
        expected = (
            reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + 1e-20)
        ).bfloat16()
        with patch.object(
            kernel, "_source_steps", side_effect=AssertionError("native RMS expected")
        ):
            actual = kernel.rms(values, weight, 1e-20, logical_rows=3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hc_grouped_projection_keeps_source_step_rows(self):
        generator = torch.Generator(device="cuda").manual_seed(73)
        module = kernel._hyperconnection_module()
        for batch, steps in ((1, 6), (2, 3), (3, 8), (4, 2)):
            with self.subTest(batch=batch, steps=steps):
                rows = batch * steps
                residual = torch.randn(
                    (batch, steps, 4, 5120),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                incoming = torch.rand((rows, 4), generator=generator, device="cuda")
                weight = torch.randn((24, 20480), generator=generator, device="cuda")
                scale = torch.ones(3, device="cuda")
                base = torch.zeros(24, device="cuda")
                storage = kernel.make_hc_workspace(residual.device, max_rows=rows)
                arguments = (
                    residual.view(rows, 4, 5120),
                    incoming,
                    weight,
                    scale,
                    base,
                    storage["mixed"],
                    storage["post"],
                    storage["comb"],
                    storage["collapsed"],
                    storage["next_pre_a"],
                    storage["flat"],
                    storage["statistics"],
                    storage["blas"],
                    batch,
                )
                expected = []
                for step in range(steps):
                    values = residual[:, step].flatten(1).float().contiguous()
                    inverse = torch.rsqrt(
                        values.square().mean(-1, keepdim=True) + 1e-20
                    )
                    expected.append(F.linear(values, weight) * inverse)
                expected = torch.stack(expected, dim=1).view(rows, 24)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    module.hc_prepare_grouped(*arguments)
                    eager = storage["mixed"].clone()
                stream.synchronize()
                # The released reduction order differs from the installed cuBLAS.
                # Exact order is covered by the frozen candidate check.
                torch.testing.assert_close(eager, expected, rtol=1e-5, atol=5e-4)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        module.hc_prepare_grouped(*arguments)
                    with torch.cuda.stream(stream):
                        storage["mixed"].fill_(123)
                        graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(
                        storage["mixed"], expected, rtol=1e-5, atol=5e-4
                    )
                    before = storage["mixed"].clone()
                    for invalid in (-1, 0, rows + 1, rows - 1):
                        with self.assertRaisesRegex(Exception, "logical rows"):
                            module.hc_prepare_grouped(*arguments[:-1], invalid)
                        torch.testing.assert_close(
                            storage["mixed"], before, rtol=0, atol=0
                        )
                    alias_arguments = list(arguments)
                    alias_arguments[5] = (
                        storage["flat"].flatten()[: rows * 24].view(rows, 24)
                    )
                    with self.assertRaisesRegex(Exception, "operator tensors overlap"):
                        module.hc_prepare_grouped(*alias_arguments)
                finally:
                    stream.synchronize()
                    graph.reset()
        module.hc_prepare_grouped(
            residual.view(rows, 4, 5120)[:0],
            incoming[:0],
            weight,
            scale,
            base,
            storage["mixed"][:0],
            storage["post"][:0],
            storage["comb"][:0],
            storage["collapsed"][:0],
            storage["next_pre_a"][:0],
            storage["flat"][:0],
            storage["statistics"][:0],
            storage["blas"][:0],
            0,
        )

    def test_fp32_projection_keeps_batch_major_positions(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        module = kernel._hyperconnection_module()
        for batch, steps, outputs in (
            (1, 6, 384),
            (2, 3, 512),
            (3, 8, 384),
            (4, 2, 128),
        ):
            with self.subTest(batch=batch, steps=steps, outputs=outputs):
                values = torch.randn(
                    (batch, steps, 5120),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                ).float()
                weight = torch.randn(
                    (outputs, 5120),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                ).float()
                expected = torch.stack(
                    [
                        F.linear(values[:, step].contiguous(), weight)
                        for step in range(steps)
                    ],
                    dim=1,
                ).view(-1, outputs)
                flat = values.view(-1, 5120)
                output = torch.empty_like(expected)
                workspace = torch.empty(
                    module.workspace_size(flat.size(0)),
                    dtype=torch.uint8,
                    device="cuda",
                )
                module.fp32_project(flat, weight, output, workspace, batch)
                # Reordered FP32 sums differ from the installed cuBLAS by a few ulp.
                torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-4)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        module.fp32_project(flat, weight, output, workspace, batch)
                    with torch.cuda.stream(stream):
                        flat.mul_(2)
                        output.fill_(123)
                        graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(
                        output, expected * 2, rtol=1e-5, atol=1e-4
                    )
                finally:
                    stream.synchronize()
                    graph.reset()
                before = output.clone()
                for invalid in (-1, flat.size(0) + 1, flat.size(0) - 1):
                    with self.assertRaisesRegex(Exception, "logical rows"):
                        module.fp32_project(flat, weight, output, workspace, invalid)
                    torch.testing.assert_close(output, before, rtol=0, atol=0)
                alias = flat.flatten()[: output.numel()].view_as(output)
                with self.assertRaisesRegex(Exception, "operator tensors overlap"):
                    module.fp32_project(flat, weight, alias, workspace, batch)
                invalid_output = torch.full_like(output, 123, dtype=torch.float16)
                with self.assertRaisesRegex(Exception, "Check failed"):
                    module.fp32_project(flat, weight, invalid_output, workspace, batch)
                torch.testing.assert_close(
                    invalid_output, torch.full_like(invalid_output, 123), rtol=0, atol=0
                )
        module.fp32_project(flat[:0], weight, output[:0], workspace[:0], 0)

    def test_hc_chunk_table_samples_the_released_layout(self):
        # Measured from the released CUDA 13.0 cuBLAS image in torch 2.13.0+cu130
        # with classify_chunks.cu. The generator and library identity are external
        # artifacts; hc_projection.cuh holds the generated table.
        module = kernel._hyperconnection_module()
        expected = {
            1: 0,
            5: 192,
            6: 224,
            20: 160,
            125: 160,
            256: 288,
            512: 576,
            1024: 224,
            1500: 352,
            2048: 448,
            2600: 2944,
            3000: 3424,
            4000: 896,
            4900: 1152,
            5200: 20480,
            6144: 20480,
            7000: 20480,
            8192: 20480,
            8193: 20480,
            32768: 20480,
        }
        for rows, chunk in expected.items():
            with self.subTest(rows=rows):
                self.assertEqual(int(module.hc_chunk(rows)), chunk)
        with self.assertRaisesRegex(Exception, "operator limit"):
            module.hc_chunk(32769)

    def test_hc_chunk_table_has_no_hole(self):
        module = kernel._hyperconnection_module()
        self.assertEqual(int(module.hc_chunk(1)), 0)
        chunk = int(module.hc_chunk(2))
        self.assertGreater(chunk, 0, "HC table has no layout for 2 rows")
        runs, start, previous = [], 2, chunk
        for rows in range(3, 32769):
            chunk = int(module.hc_chunk(rows))
            self.assertGreater(chunk, 0, f"HC table has no layout for {rows} rows")
            if chunk != previous:
                runs.append((start, rows - 1, previous))
                start, previous = rows, chunk
        runs.append((start, 32768, previous))
        self.assertEqual(runs[0][0], 2)
        self.assertEqual(runs[-1][1], 32768)
        self.assertEqual(runs[-1][2], int(module.hc_chunk(32768)))
        for (_, last, _), (first, _, _) in pairwise(runs):
            self.assertEqual(first, last + 1)

    def test_fp32_project_chunk_samples_the_released_layout(self):
        # Measured from the released CUDA 13.0 cuBLAS image in torch 2.13.0+cu130
        # with classify_shapes.cu. The generator and library identity are external
        # artifacts; hc_projection.cuh holds the generated tables.
        module = kernel._hyperconnection_module()
        router = {
            2: 160,
            6: 96,
            8: 128,
            17: 160,
            33: 128,
            65: 160,
            128: 96,
            129: 128,
            257: 160,
            385: 224,
            462: 288,
            513: 384,
            573: 288,
            640: 384,
            641: 448,
            727: 320,
            769: 384,
            897: 448,
            993: 224,
            1025: 5120,
            8192: 5120,
            8193: 5120,
            32768: 5120,
        }
        dspark = {
            2: 128,
            6: 96,
            8: 128,
            17: 64,
            29: 96,
            33: 64,
            54: 96,
            65: 160,
            82: 96,
            129: 128,
            193: 160,
            257: 128,
            385: 160,
            513: 192,
            641: 224,
            769: 256,
            897: 288,
            1025: 320,
            1153: 384,
            1281: 224,
            1537: 256,
            1793: 288,
            2049: 320,
            2305: 352,
            2433: 384,
            2689: 416,
            2817: 448,
            3073: 480,
            3169: 5120,
            32768: 5120,
        }
        compressor = {
            2: 128,
            6: 96,
            8: 128,
            17: 160,
            33: 64,
            65: 160,
            257: 224,
            385: 288,
            513: 384,
            641: 448,
            727: 224,
            769: 256,
            828: 512,
            897: 5120,
            1249: 736,
            1281: 864,
            1409: 5120,
            1474: 864,
            1537: 5120,
            32768: 5120,
        }
        for outputs, table in ((384, router), (128, dspark), (512, compressor)):
            for rows, chunk in table.items():
                with self.subTest(outputs=outputs, rows=rows):
                    self.assertEqual(int(module.project_chunk(rows, outputs)), chunk)
        self.assertEqual(int(module.project_chunk(2, 256)), 0)
        with self.assertRaisesRegex(Exception, "operator limit"):
            module.project_chunk(32769, 384)

    def test_fp32_projection_keeps_the_extended_row_order(self):
        # Rows above 8192 must keep the measured single-chunk arithmetic, and
        # the HC table must have no hole. The reference runs the same fused
        # multiply-add sequence through the HC kernel with zero padding. The
        # measured chunk values are locked by the sampled-expectation test, not
        # by an independent golden.
        module = kernel._hyperconnection_module()
        generator = torch.Generator(device="cuda").manual_seed(107)
        for rows in (8193, 32768):
            values = torch.randn((rows, 5120), generator=generator, device="cuda")
            for outputs in (384, 128, 512):
                with self.subTest(rows=rows, outputs=outputs):
                    weight = (
                        torch.randn((outputs, 5120), generator=generator, device="cuda")
                        * 0.01
                    )
                    output = torch.empty((rows, outputs), device="cuda")
                    workspace = torch.empty(
                        (module.workspace_size(rows),),
                        dtype=torch.uint8,
                        device="cuda",
                    )
                    module.fp32_project(values, weight, output, workspace, 0)
                    _assert_released_sequential_projection(
                        module, output, values, weight
                    )

    def test_fp32_projection_rejects_rows_above_the_measured_limit(self):
        module = kernel._hyperconnection_module()
        rows = 32769
        values = torch.zeros((rows, 5120), device="cuda")
        weight = torch.zeros((384, 5120), device="cuda")
        output = torch.empty((rows, 384), device="cuda")
        with self.assertRaisesRegex(Exception, "32768"):
            module.fp32_project(
                values,
                weight,
                output,
                torch.empty(0, dtype=torch.uint8, device="cuda"),
                0,
            )

    def test_hc_chunked_projection_keeps_the_released_order_for_every_band(self):
        module = kernel._hyperconnection_module()
        runs, start, previous = [], 2, int(module.hc_chunk(2))
        for rows in range(3, 32769):
            chunk = int(module.hc_chunk(rows))
            if chunk != previous:
                runs.append((start, rows - 1))
                start, previous = rows, chunk
        runs.append((start, 32768))
        samples = sorted({row for first, last in runs for row in (first, last)})
        generator = torch.Generator(device="cuda").manual_seed(83)
        weight = torch.randn((24, 20480), generator=generator, device="cuda") * 0.01
        for rows in samples:
            with self.subTest(rows=rows):
                values = torch.randn((rows, 20480), generator=generator, device="cuda")
                fast = torch.empty((rows, 24), device="cuda")
                reference = torch.empty((rows, 24), device="cuda")
                workspace = torch.empty(
                    (module.workspace_size(rows),), dtype=torch.uint8, device="cuda"
                )
                module.hc_chunked_project(values, weight, fast, workspace)
                module.hc_chunked_project(values, weight, reference, workspace[:0])
                self.assertTrue(torch.equal(fast, reference))

    def test_hc_grouped_projection_matches_the_released_order_across_steps(self):
        module = kernel._hyperconnection_module()
        generator = torch.Generator(device="cuda").manual_seed(101)
        for batch, steps in ((3, 8), (4, 2)):
            with self.subTest(batch=batch, steps=steps):
                rows = batch * steps
                source = torch.randn(
                    (batch, steps, 4, 5120),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                incoming = torch.rand((rows, 4), generator=generator, device="cuda")
                weight = (
                    torch.randn((24, 20480), generator=generator, device="cuda") * 0.01
                )
                scale = torch.ones(3, device="cuda")
                base = torch.zeros(24, device="cuda")
                storage = kernel.make_hc_workspace(source.device, max_rows=rows)
                residual = source.view(rows, 4, 5120)
                module.hc_prepare_grouped(
                    residual,
                    incoming,
                    weight,
                    scale,
                    base,
                    storage["mixed"],
                    storage["post"],
                    storage["comb"],
                    storage["collapsed"],
                    storage["next_pre_a"],
                    storage["flat"],
                    storage["statistics"],
                    storage["blas"],
                    batch,
                )
                expected = torch.empty_like(storage["mixed"])
                reference = torch.empty((batch, 24), device="cuda")
                for step in range(steps):
                    module.hc_chunked_project(
                        source[:, step].reshape(batch, 20480).float().contiguous(),
                        weight,
                        reference,
                        storage["blas"][:0],
                    )
                    expected[step::steps] = reference
                expected *= storage["statistics"][:, 1:2]
                self.assertTrue(torch.equal(storage["mixed"], expected))

    def test_hc_chunked_projection_is_capture_safe(self):
        module = kernel._hyperconnection_module()
        generator = torch.Generator(device="cuda").manual_seed(97)
        weight = torch.randn((24, 20480), generator=generator, device="cuda") * 0.01
        for rows in (5, 64, 320):
            with self.subTest(rows=rows):
                values = torch.randn((rows, 20480), generator=generator, device="cuda")
                output = torch.empty((rows, 24), device="cuda")
                workspace = torch.empty(
                    (module.workspace_size(rows),), dtype=torch.uint8, device="cuda"
                )
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    module.hc_chunked_project(values, weight, output, workspace)
                stream.synchronize()
                expected = output.clone()
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        module.hc_chunked_project(values, weight, output, workspace)
                    with torch.cuda.stream(stream):
                        output.fill_(7)
                        graph.replay()
                    stream.synchronize()
                    self.assertTrue(torch.equal(output, expected))
                finally:
                    stream.synchronize()
                    graph.reset()

    def test_markov_linear_projects_every_row(self):
        module = kernel._hyperconnection_module()
        generator = torch.Generator(device="cuda").manual_seed(93)
        values = torch.randn((3, 256), generator=generator, device="cuda")
        weight = torch.randn((128, 256), generator=generator, device="cuda")
        output = torch.full((3, 128), float("nan"), device="cuda")
        module.markov_linear(values, weight, output)
        for row in range(3):
            single = torch.empty((1, 128), device="cuda")
            module.markov_linear(values[row : row + 1].contiguous(), weight, single)
            self.assertTrue(torch.equal(output[row], single[0]))

    def test_markov_linear_rejects_rows_above_the_grid_limit(self):
        module = kernel._hyperconnection_module()
        module.markov_linear(
            torch.zeros((65535, 4), device="cuda"),
            torch.zeros((8, 4), device="cuda"),
            torch.empty((65535, 8), device="cuda"),
        )
        with self.assertRaisesRegex(Exception, "grid.y limit"):
            module.markov_linear(
                torch.zeros((65536, 4), device="cuda"),
                torch.zeros((8, 4), device="cuda"),
                torch.empty((65536, 8), device="cuda"),
            )

    def test_markov_linear_handles_a_partial_output_block(self):
        module = kernel._hyperconnection_module()
        generator = torch.Generator(device="cuda").manual_seed(91)
        values = torch.randn((1, 256), generator=generator, device="cuda")
        weight = torch.randn((64, 256), generator=generator, device="cuda")
        padded = torch.empty((1, 64), device="cuda")
        module.markov_linear(values, weight, padded)
        for outputs in (17, 33):
            with self.subTest(outputs=outputs):
                narrow = torch.empty((1, outputs), device="cuda")
                module.markov_linear(values, weight[:outputs].contiguous(), narrow)
                self.assertTrue(torch.equal(narrow, padded[:, :outputs]))

    def test_confidence_head_uses_the_native_split_dot(self):
        generator = torch.Generator(device="cuda").manual_seed(95)
        head = model.DSparkConfidenceHead(40).cuda()
        with torch.no_grad():
            head.proj.weight.copy_(
                torch.randn((1, 40), generator=generator, device="cuda")
            )
        hidden = torch.randn((2, 3, 32), generator=generator, device="cuda").bfloat16()
        markov = torch.randn((2, 3, 8), generator=generator, device="cuda").bfloat16()
        with patch.object(model, "dot_linear", wraps=kernel.dot_linear) as dot:
            actual = head(hidden, markov)
        self.assertTrue(dot.called)
        self.assertEqual(dot.call_args.args[0].dtype, torch.float32)
        expected = kernel.dot_linear(
            torch.cat([hidden, markov], dim=-1).float(), head.proj.weight
        ).squeeze(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
