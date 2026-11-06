import unittest

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, kernel_native

from tests.projection_precision_fixtures import (
    grouped_literal,
    run_grouped,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionPrecisionCudaCudaPadded(unittest.TestCase):
    def test_padded_grouped_projection_replays_device_live_rows_and_zeroes_tail(self):
        module = kernel_native._projection_module()
        for dtype in (torch.bfloat16, torch.float32):
            for capacity in (36, 260):
                case = grouped_literal(
                    (1, 0, capacity - 3, 2),
                    max_rows=capacity - 3,
                    dtype=dtype,
                    width=128 if capacity > 64 else 64,
                )
                status = torch.zeros((1,), dtype=torch.int32, device="cuda")
                encoded, scales = case.encoded.clone(), case.scales.clone()

                def run(case=case, status=status, dtype=dtype):
                    return kernel.grouped_fp4_gemm_padded(
                        case.encoded,
                        case.scales,
                        case.counts,
                        case.weights,
                        pointers=case.pointers,
                        status=status,
                        max_rows=case.max_rows,
                        output_dtype=dtype,
                        output=case.output,
                        scratch=case.scratch,
                    )

                run()
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        run()
                    self._check_padded_live_rows(
                        capacity,
                        dtype,
                        stream,
                        case,
                        encoded,
                        scales,
                        status,
                        graph,
                        run,
                        module,
                    )
                    self._check_padded_invalid_counts(
                        capacity, dtype, case, status, stream, graph, run
                    )
                    self._check_padded_binding_errors(
                        capacity, dtype, case, status, run
                    )
                finally:
                    stream.synchronize()
                    graph.reset()

    def test_padded_grouped_zero_capacity_still_rejects_live_counts(self):
        case = grouped_literal((0, 0, 0, 0), max_rows=1)
        status = torch.zeros((1,), dtype=torch.int32, device="cuda")
        module = kernel_native._projection_module()
        module.grouped_project_padded(
            case.encoded,
            case.scales,
            *case.pointers,
            case.counts,
            case.output,
            case.scratch,
            case.max_rows,
            status,
        )
        kernel.require_padded_status(status)
        case.counts[0] = 1
        module.grouped_project_padded(
            case.encoded,
            case.scales,
            *case.pointers,
            case.counts,
            case.output,
            case.scratch,
            case.max_rows,
            status,
        )
        with self.assertRaisesRegex(RuntimeError, "device metadata"):
            kernel.require_padded_status(status)

    def test_grouped_output_cannot_alias_encoded_input(self):
        module = kernel_native._projection_module()
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
        module = kernel_native._projection_module()
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
        module = kernel_native._projection_module()
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

    def _check_padded_binding_errors(self, capacity, dtype, case, status, run):
        case.counts.copy_(torch.tensor((1, 0, 0, 0), device="cuda"))
        status.fill_(8)
        run()
        self.assertEqual(status.item(), 8)
        self.assertEqual(case.output.count_nonzero().item(), 0)
        with self.assertRaisesRegex(ValueError, "shape and dtype"):
            kernel.grouped_fp4_gemm_padded(
                case.encoded,
                case.scales,
                case.counts,
                case.weights,
                pointers=case.pointers,
                status=status,
                output=torch.empty((capacity, 256), dtype=dtype, device="cuda"),
                output_dtype=dtype,
            )
        for bad_weight, bad_scale in (
            (case.weights[-1][0][:32], case.weights[-1][1][:32]),
            (case.weights[-1][0][:, :16], case.weights[-1][1]),
            (case.weights[-1][0], case.weights[-1][1][:, :1]),
        ):
            with (
                self.subTest(bad_weight=bad_weight.shape),
                self.assertRaisesRegex(ValueError, "shapes must match"),
            ):
                kernel.grouped_fp4_gemm_padded(
                    case.encoded,
                    case.scales,
                    case.counts,
                    [*case.weights[:-1], (bad_weight, bad_scale)],
                    pointers=case.pointers,
                    status=status,
                    output_dtype=dtype,
                )

    def _check_padded_invalid_counts(
        self, capacity, dtype, case, status, stream, graph, run
    ):
        invalid = (
            (-1, 2, 2, 0),
            (capacity // 2, capacity - capacity // 2 + 1, 0, 0),
            (capacity - 2, 0, 0, 0),
            (2**31 - 1, 2**31 - 1, 0, 0),
        )
        for counts in invalid:
            for execution in ("eager", "graph"):
                with self.subTest(
                    dtype=dtype,
                    capacity=capacity,
                    invalid=counts,
                    execution=execution,
                ):
                    with torch.cuda.stream(stream):
                        case.counts.copy_(
                            torch.tensor(counts, dtype=torch.int32, device="cuda")
                        )
                        status.zero_()
                        graph.replay() if execution == "graph" else run()
                    stream.synchronize()
                    with self.assertRaisesRegex(RuntimeError, "device metadata"):
                        kernel.require_padded_status(status)
                    self.assertEqual(case.output.count_nonzero().item(), 0)

    def _check_padded_live_rows(
        self, capacity, dtype, stream, case, encoded, scales, status, graph, run, module
    ):
        valid = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 3, 2), (16, 1, 0, 19)]
        if capacity > 64:
            valid += [(32, 32, 0, 0), (32, 33, 0, 0), (128, 1, 0, 131)]
        valid += [(1, 0, capacity - 3, 2)]
        for counts in valid:
            live = sum(counts)
            for execution in ("eager", "graph"):
                with self.subTest(
                    dtype=dtype,
                    capacity=capacity,
                    counts=counts,
                    execution=execution,
                ):
                    with torch.cuda.stream(stream):
                        case.counts.copy_(
                            torch.tensor(counts, dtype=torch.int32, device="cuda")
                        )
                        case.encoded.copy_(encoded)
                        case.scales.copy_(scales)
                        case.encoded[live:].fill_(127)
                        case.scales[live:].fill_(255)
                        case.output.fill_(-7)
                        status.zero_()
                        graph.replay() if execution == "graph" else run()
                        actual = case.output.clone()
                    stream.synchronize()
                    kernel.require_padded_status(status)
                    reference = torch.empty((live, 128), dtype=dtype, device="cuda")
                    module.grouped_project(
                        case.encoded[:live],
                        case.scales[:live],
                        *case.pointers,
                        case.counts,
                        reference,
                        case.scratch,
                        case.max_rows,
                    )
                    expected = torch.zeros_like(actual)
                    expected[:live] = reference
                    self.assertTrue(
                        torch.equal(
                            actual.view(torch.uint8),
                            expected.view(torch.uint8),
                        )
                    )
