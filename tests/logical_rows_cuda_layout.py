import unittest
from itertools import pairwise
from unittest.mock import patch

import torch
import torch.nn.functional as F
from oh_my_vllm.models.deepseek_v41 import kernel, kernel_native, kernel_transform


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
            kernel_transform,
            "_source_steps",
            side_effect=AssertionError("native RMS expected"),
        ):
            actual = kernel.rms(values, weight, 1e-20, logical_rows=3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hc_grouped_projection_keeps_source_step_rows(self):
        generator = torch.Generator(device="cuda").manual_seed(73)
        module = kernel_native._hyperconnection_module()
        for batch, steps in ((1, 6), (2, 3), (3, 8), (4, 2)):
            with self.subTest(batch=batch, steps=steps):
                rows, residual, incoming, weight, scale, base, storage, arguments = (
                    self._grouped_hc_case(batch, steps, generator)
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
        module = kernel_native._hyperconnection_module()
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
                for invalid_workspace in (
                    workspace[:1].expand(workspace.numel()),
                    workspace.cpu(),
                ):
                    with self.assertRaisesRegex(Exception, "Check failed"):
                        module.fp32_project(
                            flat, weight, output, invalid_workspace, batch
                        )
                    self.assertTrue(torch.equal(output, before))
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
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
        router = self._router_chunks()
        dspark = self._dspark_chunks()
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

    def _grouped_hc_case(self, batch, steps, generator):
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
        return rows, residual, incoming, weight, scale, base, storage, arguments

    def _dspark_chunks(self):
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
        return dspark

    def _router_chunks(self):
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
        return router
