import unittest

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, kernel_native

from tests.deepseek_operators_fixtures import (
    projection_reference as projection_reference,
)
from tests.deepseek_operators_fixtures import projection_weight as projection_weight
from tests.deepseek_operators_fixtures import rms_rounding_case as rms_rounding_case
from tests.deepseek_operators_grouped import (
    DeepSeekOperatorContractsGrouped as DeepSeekOperatorContractsGrouped,
)
from tests.deepseek_operators_hyperconnection import (
    DeepSeekHyperconnectionContracts as DeepSeekHyperconnectionContracts,
)
from tests.deepseek_operators_loader import (
    DeepSeekOperatorContractsLoader as DeepSeekOperatorContractsLoader,
)
from tests.deepseek_operators_projection_storage import (
    DeepSeekProjectionStorageContracts as DeepSeekProjectionStorageContracts,
)
from tests.deepseek_operators_references import (
    DeepSeekOperatorContractsReferences as DeepSeekOperatorContractsReferences,
)
from tests.deepseek_operators_rounding import (
    DeepSeekOperatorContractsRounding as DeepSeekOperatorContractsRounding,
)
from tests.deepseek_operators_sparse import (
    DeepSeekOperatorContractsSparse as DeepSeekOperatorContractsSparse,
)


class DeepSeekOperatorContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_engram_staging_rejects_metadata_before_buffer_writes(self):
        module = kernel_native._transform_module()
        weight = torch.arange(256, dtype=torch.uint8).reshape(8, 32)
        scales = torch.full((8, 1), 127, dtype=torch.uint8)
        ids = torch.tensor([1, 3], dtype=torch.int64)
        packed = torch.full((2, 48), 173, dtype=torch.uint8)
        strided_storage = torch.full((2, 96), 173, dtype=torch.uint8)
        for name, bad_weight, bad_scales, bad_ids, bad_packed, start in (
            ("weight dtype", weight.long(), scales, ids, packed, 0),
            ("scales shape", weight, scales[:7], ids, packed, 0),
            ("ids dtype", weight, scales, ids.int(), packed, 0),
            ("packed shape", weight, scales, ids, packed[:, :47], 0),
            ("packed stride", weight, scales, ids, strided_storage[:, ::2], 0),
            ("negative start", weight, scales, ids, packed, -1),
            ("overflow start", weight, scales, ids, packed, 2**63 - 1),
            (
                "aliased weight",
                weight,
                scales,
                ids,
                weight.flatten()[:96].view(2, 48),
                0,
            ),
        ):
            with self.subTest(name=name):
                buffers = (weight, scales, ids, packed, strided_storage)
                before = [value.clone() for value in buffers]
                with self.assertRaises(RuntimeError):
                    module.engram_lookup_stage(
                        bad_weight, bad_scales, bad_ids, bad_packed, start
                    )
                for actual, expected in zip(buffers, before, strict=True):
                    self.assertTrue(torch.equal(actual, expected))
        output = torch.full((2, 32), 3, dtype=torch.bfloat16, device="cuda")
        device_packed = packed.cuda()
        device_strided_storage = strided_storage.cuda()
        for bad_packed, bad_output in (
            (device_packed[:, :47], output),
            (device_strided_storage[:, ::2], output),
            (device_packed, output.float()),
            (device_packed, output[:1]),
        ):
            with self.subTest(shape=tuple(bad_output.shape), dtype=bad_output.dtype):
                saved_packed = device_packed.clone()
                saved_strided = device_strided_storage.clone()
                saved_output = bad_output.clone()
                with self.assertRaises(RuntimeError):
                    module.engram_lookup_decode(bad_packed, bad_output)
                self.assertTrue(torch.equal(device_packed, saved_packed))
                self.assertTrue(torch.equal(device_strided_storage, saved_strided))
                self.assertTrue(
                    torch.equal(
                        bad_output.view(torch.uint8), saved_output.view(torch.uint8)
                    )
                )

    def test_expert_sum_keeps_official_expert_order(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for rows, positions in (
                (4, (23, 19, 21)),
                (127, (18, 22, 20)),
                (128, (23, 19, 21)),
                (129, (18, 22, 20)),
            ):
                with self.subTest(device=device, rows=rows, positions=positions):
                    values = torch.zeros((3, 1283), dtype=torch.bfloat16, device=device)
                    values[:, 1121] = torch.tensor(
                        (-0.0244140625, 0.0145263671875, 2.514570951461792e-8),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    routes = torch.tensor(positions, dtype=torch.int64, device=device)
                    output = torch.empty(
                        (rows, 1283), dtype=torch.float32, device=device
                    )
                    kernel.ordered_expert_sum(values, routes, output, 6)
                    expected = torch.zeros_like(output)
                    expected[3, 1121] = -0.009887670166790485
                    self.assertTrue(
                        torch.equal(
                            output.view(torch.int32), expected.view(torch.int32)
                        )
                    )
                    kernel.ordered_expert_sum(values[:0], routes[:0], output, 6)
                    self.assertEqual(output.count_nonzero().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_expert_sum_graph_replays_changed_routes(self):
        values = torch.tensor(
            (
                (-0.0244140625,) * 1283,
                (0.0145263671875,) * 1283,
                (2.514570951461792e-8,) * 1283,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        routes = torch.tensor((5, 1, 3), dtype=torch.int64, device="cuda")
        output = torch.empty((128, 1283), dtype=torch.float32, device="cuda")
        kernel.ordered_expert_sum(values, routes, output, 6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel.ordered_expert_sum(values, routes, output, 6)
        for positions, token in (((5, 1, 3), 0), ((764, 766, 762), 127)):
            routes.copy_(torch.tensor(positions, dtype=torch.int64, device="cuda"))
            graph.replay()
            expected = torch.zeros_like(output)
            expected[token] = -0.009887670166790485
            self.assertTrue(
                torch.equal(output.view(torch.int32), expected.view(torch.int32))
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_padded_expert_sum_replays_live_prefix_and_token_capacity(self):
        capacity, tokens, width, topk = 128, 128, 1283, 6
        levels = torch.tensor(
            (-0.0244140625, 0.0145263671875, 2.514570951461792e-8), dtype=torch.bfloat16
        )
        finite_cpu = (
            levels[(torch.arange(capacity) // 16) % 3, None]
            .expand(-1, width)
            .contiguous()
        )
        finite = finite_cpu.to("cuda")
        values = finite.clone()
        route_prefix = (
            (torch.arange(96, device="cuda") % 16) * topk
            + 5
            - torch.arange(96, device="cuda") // 16
        )
        routes = torch.full((capacity,), 2**63 - 1, dtype=torch.int64, device="cuda")
        routes[:96].copy_(route_prefix)
        counts = torch.full((6,), 16, dtype=torch.int32, device="cuda")
        output = torch.empty((tokens, width), dtype=torch.float32, device="cuda")
        route_rows = torch.empty((tokens, topk), dtype=torch.int32, device="cuda")
        status = torch.zeros((1,), dtype=torch.int32, device="cuda")

        def run():
            kernel.ordered_expert_sum_padded(
                values, routes, counts, output, topk, status, route_rows=route_rows
            )

        run()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.graph(graph, stream=stream):
                run()
            self._check_padded_expert_live_rows(
                stream,
                values,
                finite,
                routes,
                route_prefix,
                counts,
                output,
                status,
                graph,
                tokens,
                width,
                finite_cpu,
            )
            self._check_padded_expert_invalid_routes(
                capacity, tokens, topk, stream, counts, routes, status, graph, output
            )
            counts.zero_()
            counts[0] = 1
            routes[0] = 0
            values.fill_(float("nan"))
            status.fill_(8)
            run()
            self.assertEqual(status.item(), 8)
            self.assertEqual(output.count_nonzero().item(), 0)
        finally:
            stream.synchronize()
            graph.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_padded_expert_sum_zero_capacity_and_unused_token_rows(self):
        for tokens in (0, 32769):
            with self.subTest(tokens=tokens):
                values = torch.empty((0, 32), dtype=torch.bfloat16, device="cuda")
                routes = torch.empty((0,), dtype=torch.int64, device="cuda")
                counts = torch.zeros((1,), dtype=torch.int32, device="cuda")
                output = torch.full(
                    (tokens, 32), float("nan"), dtype=torch.float32, device="cuda"
                )
                status = torch.zeros((1,), dtype=torch.int32, device="cuda")
                kernel.ordered_expert_sum_padded(
                    values, routes, counts, output, 6, status
                )
                kernel.require_padded_status(status)
                self.assertEqual(output.count_nonzero().item(), 0)
                counts[0] = 1
                kernel.ordered_expert_sum_padded(
                    values, routes, counts, output, 6, status
                )
                with self.assertRaisesRegex(RuntimeError, "device metadata"):
                    kernel.require_padded_status(status)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_routing_keeps_source_softplus_and_selected_sum_order(self):
        for rows in (1, 33):
            for logit in (-0.25, 0.25):
                with self.subTest(rows=rows, logit=logit):
                    logits = torch.full(
                        (rows, 384), -100.0, dtype=torch.float32, device="cuda"
                    )
                    logits[:, :6] = logit
                    bias = torch.full(
                        (384,), -100.0, dtype=torch.float32, device="cuda"
                    )
                    bias[:6] = torch.tensor(
                        [12, 10, 8, 6, 4, 2], dtype=torch.float32, device="cuda"
                    )
                    coefficients, ids = kernel.route(logits, bias, bias, None)
                    torch.testing.assert_close(
                        ids,
                        torch.arange(6, dtype=torch.int32, device="cuda").expand(
                            rows, 6
                        ),
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        coefficients,
                        torch.full_like(coefficients, 0.25),
                        atol=0,
                        rtol=0,
                    )

    def _check_padded_expert_invalid_routes(
        self, capacity, tokens, topk, stream, counts, routes, status, graph, output
    ):
        for invalid_counts, first_routes in (
            ((-1, 1, 0, 0, 0, 0), (5,)),
            ((capacity + 1, 0, 0, 0, 0, 0), (5,)),
            ((1, 0, 0, 0, 0, 0), (-1,)),
            ((1, 0, 0, 0, 0, 0), (tokens * topk,)),
            ((2, 0, 0, 0, 0, 0), (5, 5)),
        ):
            with self.subTest(invalid_counts=invalid_counts, first_routes=first_routes):
                with torch.cuda.stream(stream):
                    counts.copy_(
                        torch.tensor(invalid_counts, dtype=torch.int32, device="cuda")
                    )
                    routes[: len(first_routes)].copy_(
                        torch.tensor(first_routes, dtype=torch.int64, device="cuda")
                    )
                    status.zero_()
                    graph.replay()
                stream.synchronize()
                with self.assertRaisesRegex(RuntimeError, "device metadata"):
                    kernel.require_padded_status(status)
                self.assertEqual(output.count_nonzero().item(), 0)

    def _check_padded_expert_live_rows(
        self,
        stream,
        values,
        finite,
        routes,
        route_prefix,
        counts,
        output,
        status,
        graph,
        tokens,
        width,
        finite_cpu,
    ):
        for live in (96, 3, 65, 0, 1, 33, 96):
            with self.subTest(live=live):
                live_counts = [min(16, max(0, live - group * 16)) for group in range(6)]
                with torch.cuda.stream(stream):
                    values.fill_(float("nan"))
                    values[:live].copy_(finite[:live])
                    routes.fill_(2**63 - 1)
                    routes[:live].copy_(route_prefix[:live])
                    counts.copy_(
                        torch.tensor(live_counts, dtype=torch.int32, device="cuda")
                    )
                    output.fill_(float("nan"))
                    status.zero_()
                    graph.replay()
                    actual = output.cpu()
                stream.synchronize()
                kernel.require_padded_status(status)
                expected = torch.zeros((tokens, width), dtype=torch.float32)
                for row in range(live):
                    expected[row % 16] += finite_cpu[row].float()
                self.assertTrue(
                    torch.equal(actual.view(torch.int32), expected.view(torch.int32))
                )


if __name__ == "__main__":
    unittest.main()
