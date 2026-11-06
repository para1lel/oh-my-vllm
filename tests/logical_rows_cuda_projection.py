import unittest
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, kernel_native, model, model_dspark

from tests.logical_rows_fixtures import _assert_released_sequential_projection


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LogicalRowsCudaCudaProjection(unittest.TestCase):
    def test_fp32_tiled_projection_keeps_each_chunk_band(self):
        module = kernel_native._hyperconnection_module()
        generator = torch.Generator(device="cuda").manual_seed(217)
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        for outputs in (128, 384, 512):
            boundaries = {1, 2, 7, 8, 9, 31, 32, 33, 32768}
            previous = int(module.project_chunk(2, outputs))
            for rows in range(3, 32769):
                chunk = int(module.project_chunk(rows, outputs))
                if chunk != previous:
                    boundaries.update((rows - 1, rows))
                previous = chunk
            values = torch.randn((32768, 5120), generator=generator, device="cuda")
            values[0].zero_()
            values[1, ::2] = -0.0
            weight = torch.randn((outputs, 5120), generator=generator, device="cuda")
            cases = [(rows, 0) for rows in sorted(boundaries)]
            cases.extend(((24, 3), (32768, 2)))
            for rows, logical_rows in cases:
                with self.subTest(
                    outputs=outputs, rows=rows, logical_rows=logical_rows
                ):
                    actual = torch.empty((rows, outputs), device="cuda")
                    expected = torch.empty_like(actual)
                    module.fp32_project(
                        values[:rows], weight, actual, workspace, logical_rows
                    )
                    module.fp32_project(
                        values[:rows], weight, expected, workspace[:0], logical_rows
                    )
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.int32), expected.view(torch.int32)
                        )
                    )

    def test_fp32_projection_keeps_the_extended_row_order(self):
        # Rows above 8192 must keep the measured single-chunk arithmetic, and
        # the HC table must have no hole. The reference runs the same fused
        # multiply-add sequence through the HC kernel with zero padding. The
        # measured chunk values are locked by the sampled-expectation test, not
        # by an independent golden.
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
        generator = torch.Generator(device="cuda").manual_seed(101)
        for batch, steps in ((3, 8), (4, 2), (4, 817)):
            with self.subTest(batch=batch, steps=steps):
                _rows, source, incoming, weight, scale, base, storage, residual = (
                    self._grouped_hc_storage(batch, steps, generator)
                )
                arguments = (
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
                module.hc_prepare_grouped(*arguments)
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
                self.assertTrue(
                    torch.equal(
                        storage["mixed"].view(torch.int32), expected.view(torch.int32)
                    )
                )
                if steps == 817:
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        module.hc_prepare_grouped(*arguments)
                    torch.cuda.current_stream().wait_stream(stream)
                    storage["mixed"].fill_(123)
                    graph.replay()
                    self.assertTrue(
                        torch.equal(
                            storage["mixed"].view(torch.int32),
                            expected.view(torch.int32),
                        )
                    )
                    torch.cuda.current_stream().synchronize()
                    graph.reset()

    def test_hc_chunked_projection_is_capture_safe(self):
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
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
        module = kernel_native._hyperconnection_module()
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
        with patch.object(model_dspark, "dot_linear", wraps=kernel.dot_linear) as dot:
            actual = head(hidden, markov)
        self.assertTrue(dot.called)
        self.assertEqual(dot.call_args.args[0].dtype, torch.float32)
        expected = kernel.dot_linear(
            torch.cat([hidden, markov], dim=-1).float(), head.proj.weight
        ).squeeze(-1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def _grouped_hc_storage(self, batch, steps, generator):
        rows = batch * steps
        source = torch.randn(
            (batch, steps, 4, 5120),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        incoming = torch.rand((rows, 4), generator=generator, device="cuda")
        weight = torch.randn((24, 20480), generator=generator, device="cuda") * 0.01
        scale = torch.ones(3, device="cuda")
        base = torch.zeros(24, device="cuda")
        storage = kernel.make_hc_workspace(
            source.device, max_rows=min(rows, kernel.NATIVE_ROW_LIMIT)
        )
        for name in (
            "mixed",
            "post",
            "comb",
            "collapsed",
            "next_pre_a",
            "flat",
            "statistics",
        ):
            value = storage[name]
            if value.size(0) < rows:
                storage[name] = torch.empty(
                    (rows, *value.shape[1:]),
                    dtype=value.dtype,
                    device=source.device,
                )
        residual = source.view(rows, 4, 5120)
        return rows, source, incoming, weight, scale, base, storage, residual
