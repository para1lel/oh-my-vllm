import unittest
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from oh_my_vllm.models.deepseek_v41 import (
    kernel,
    kernel_hc,
    kernel_native,
    kernel_transform,
)


class DeepSeekOperatorContractsSparse(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_sparse_attention_native_path_matches_bf16_reference(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(19)
        query = torch.randn(
            (2, 3, 16, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        window = torch.randn(
            (2, 7, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        compressed = torch.randn(
            (2, 4, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        sink = torch.randn((16,), generator=generator, device=device)
        indices = torch.full((2, 3, 9), -1, device=device, dtype=torch.int32)
        indices[:, :, :7] = torch.arange(7, device=device, dtype=torch.int32)
        indices[:, :, 7:] = torch.tensor((7, 8), device=device, dtype=torch.int32)

        native = kernel._sparse_attn_native(query, sink, indices, window, compressed)
        reference = kernel.sparse_attn(
            query,
            torch.cat((window, compressed), dim=1),
            sink,
            indices,
            512**-0.5,
        )

        torch.testing.assert_close(native, reference, atol=0.06, rtol=0.06)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_sparse_decode_split_boundaries_and_graph(self):
        from oh_my_vllm.operators.sparse_attention.plan import scratch_bytes

        generator = torch.Generator(device="cuda").manual_seed(813)
        module = kernel._sparse_attention_module()
        for rows, heads, slots in (
            (0, 16, 640),
            (1, 16, 0),
            (1, 16, 1),
            (1, 64, 64),
            (2, 16, 640),
            (4, 16, 65),
            (1, 16, 640),
            (5, 16, 129),
        ):
            with self.subTest(rows=rows, heads=heads, slots=slots):
                query, table, sink, indices, output, status, arguments = (
                    self._sparse_decode_inputs(
                        rows, heads, slots, generator, scratch_bytes
                    )
                )

                def run(arguments=arguments):
                    module.attend(*arguments)

                run()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                graph.replay()
                self.assertFalse(bool(status.any()))
                if rows:
                    torch.testing.assert_close(
                        output[0], torch.zeros_like(output[0]), atol=0, rtol=0
                    )
                if rows > 1:
                    reference = kernel.sparse_attn(
                        query[None], table[None], sink, indices[None], 512**-0.5
                    )[0]
                    torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)
                if rows and slots:
                    indices[-1, -1] = 7
                    graph.replay()
                    self.assertTrue(bool(status[-1].all()))
                    torch.testing.assert_close(
                        output[-1], torch.zeros_like(output[-1]), atol=0, rtol=0
                    )
                    indices[-1, -1] = 0
                    query[-1, 0, 0] = float("nan")
                    graph.replay()
                    self.assertTrue(bool(status[-1, 0]))
                    torch.testing.assert_close(
                        output[-1, :16],
                        torch.zeros_like(output[-1, :16]),
                        atol=0,
                        rtol=0,
                    )
                    query[-1].zero_()
                    sink[0] = float("inf")
                    graph.replay()
                    self.assertTrue(bool(status[:, 0].all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_sparse_decode_keeps_cross_block_mma_order(self):
        query = torch.zeros((1, 1, 16, 512), device="cuda", dtype=torch.bfloat16)
        table = torch.zeros((1, 4, 512), device="cuda", dtype=torch.bfloat16)
        table[0, :, 0] = torch.tensor((2**26, -(2**26), 1, 0), device="cuda")
        indices = torch.full((1, 1, 130), -1, device="cuda", dtype=torch.int32)
        indices[0, 0, [0, 64, 80, 129]] = torch.arange(
            4, device="cuda", dtype=torch.int32
        )
        sink = torch.zeros((16,), device="cuda", dtype=torch.float32)

        actual = kernel._sparse_attn_native(query, sink, indices, table, table[:, :0])

        expected = torch.zeros_like(actual)
        expected[..., 0] = 0.2
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_sparse_attention_status_failure_blocks_the_step(self):
        device = torch.device("cuda")
        query = torch.zeros((1, 1, 16, 512), device=device, dtype=torch.bfloat16)
        window = torch.zeros((1, 1, 512), device=device, dtype=torch.bfloat16)
        compressed = torch.empty((1, 0, 512), device=device, dtype=torch.bfloat16)
        sink = torch.zeros((16,), device=device)
        indices = torch.zeros((1, 1, 1), device=device, dtype=torch.int32)

        class FailedAttention:
            @staticmethod
            def attend(*arguments):
                arguments[-2].zero_()
                arguments[-1].fill_(1)

        with (
            patch.object(kernel, "_SPARSE_NATIVE_DISABLED", False),
            patch.object(kernel, "_SPARSE_NATIVE_FAILED", False),
            patch.object(
                kernel, "_sparse_attention_module", return_value=FailedAttention()
            ),
        ):
            kernel.reset_sparse_status()
            output = kernel._sparse_attn_native(
                query, sink, indices, window, compressed
            )
            self.assertEqual(tuple(output.shape), (1, 1, 16, 512))
            with self.assertRaises(RuntimeError):
                kernel.require_sparse_status()
        kernel.reset_sparse_status()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_large_row_native_paths_accept_rows_above_operator_limit(self):
        device = torch.device("cuda")
        rows = kernel.NATIVE_ROW_LIMIT + 1
        generator = torch.Generator(device=device).manual_seed(23)

        residual = torch.randn(
            (rows, 4, 5120), generator=generator, device=device, dtype=torch.bfloat16
        )
        incoming = torch.rand((rows, 4), generator=generator, device=device)
        projection = (
            torch.randn((24, 4 * 5120), generator=generator, device=device) * 0.01
        )
        scale = torch.randn((3,), generator=generator, device=device)
        base = torch.randn((24,), generator=generator, device=device)
        branch = torch.randn(
            (rows, 5120), generator=generator, device=device, dtype=torch.bfloat16
        )

        hyperconnection = Mock(wraps=kernel_native._hyperconnection_module())
        with patch.object(
            kernel_hc, "_hyperconnection_module", return_value=hyperconnection
        ):
            collapsed, next_pre, post, comb = kernel.hc_prepare(
                residual, incoming, projection, scale, base
            )
            expanded = kernel.hc_expand(residual, branch, post, comb)

        flat = residual.float().flatten(1)
        mixes = F.linear(flat, projection) * torch.rsqrt(
            flat.square().mean(-1, keepdim=True) + 1e-20
        )
        expected_pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
        expected_post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
        expected_comb = torch.softmax(
            (mixes[..., 8:] * scale[2] + base[8:]).reshape(-1, 4, 4),
            dim=-1,
        )
        expected_comb = expected_comb + 1e-6
        expected_comb = expected_comb / (expected_comb.sum(dim=-2, keepdim=True) + 1e-6)
        for _ in range(19):
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-1, keepdim=True) + 1e-6
            )
            expected_comb = expected_comb / (
                expected_comb.sum(dim=-2, keepdim=True) + 1e-6
            )
        expected_collapsed = torch.sum(
            incoming.unsqueeze(-1) * residual.float(), dim=1
        ).to(torch.bfloat16)
        expected_expanded = expected_post.unsqueeze(-1) * branch.float().unsqueeze(-2)
        expected_expanded = expected_expanded + torch.sum(
            expected_comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=1
        )

        torch.testing.assert_close(collapsed, expected_collapsed, atol=0.01, rtol=0.01)
        torch.testing.assert_close(next_pre, expected_pre, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(post, expected_post, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(comb, expected_comb, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            expanded, expected_expanded.to(torch.bfloat16), atol=0.02, rtol=0.01
        )
        self.assertEqual(hyperconnection.hc_prepare.call_count, 1)
        self.assertEqual(hyperconnection.hc_expand.call_count, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_large_row_sparse_attention_accepts_rows_above_operator_limit(self):
        device = torch.device("cuda")
        rows = kernel.NATIVE_ROW_LIMIT + 1
        generator = torch.Generator(device=device).manual_seed(29)
        query = torch.randn(
            (1, rows, 16, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        window = torch.randn(
            (1, 7, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        compressed = torch.randn(
            (1, 4, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        sink = torch.randn((16,), generator=generator, device=device)
        indices = torch.full((1, rows, 9), -1, device=device, dtype=torch.int32)
        indices[:, :, :7] = torch.arange(7, device=device, dtype=torch.int32)
        indices[:, :, 7:] = torch.tensor((7, 8), device=device, dtype=torch.int32)

        sparse_attention = Mock(wraps=kernel._sparse_attention_module())
        with patch.object(
            kernel, "_sparse_attention_module", return_value=sparse_attention
        ):
            native = kernel._sparse_attn_native(
                query, sink, indices, window, compressed
            )
            with (
                patch.object(kernel, "_SPARSE_NATIVE_DISABLED", True),
                patch.object(kernel, "_SPARSE_NATIVE_FAILED", False),
            ):
                public = kernel.sparse_attn(
                    query,
                    torch.cat((window, compressed), dim=1),
                    sink,
                    indices,
                    512**-0.5,
                    native_window=window,
                    native_compressed=compressed,
                )

        reference = kernel.sparse_attn(
            query,
            torch.cat((window, compressed), dim=1),
            sink,
            indices,
            512**-0.5,
        )

        self.assertIsNotNone(native)
        torch.testing.assert_close(native, reference, atol=0.06, rtol=0.06)
        torch.testing.assert_close(public, native, atol=0, rtol=0)
        self.assertEqual(sparse_attention.attend.call_count, 2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_large_row_rms_accepts_rows_above_operator_limit(self):
        device = torch.device("cuda")
        rows = kernel.NATIVE_ROW_LIMIT + 1
        generator = torch.Generator(device=device).manual_seed(31)
        values = torch.randn(
            (rows, 512), generator=generator, device=device, dtype=torch.bfloat16
        )
        weight = torch.randn(
            (512,), generator=generator, device=device, dtype=torch.bfloat16
        )
        transform = Mock(wraps=kernel_native._transform_module())
        with patch.object(
            kernel_transform, "_transform_module", return_value=transform
        ):
            actual = kernel.rms(values, weight, 1e-6)
        expected = (
            values.float()
            * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
        self.assertEqual(transform.rms.call_count, 1)

    def _sparse_decode_inputs(self, rows, heads, slots, generator, scratch_bytes):
        query = torch.randn(
            (rows, heads, 512),
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        table = torch.randn(
            (7, 512), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        sink = torch.zeros((heads,), device="cuda", dtype=torch.float32)
        indices = torch.arange(slots, device="cuda", dtype=torch.int32) % 7
        indices = indices.expand(rows, slots).contiguous()
        if rows:
            indices[0].fill_(-1)
        output = torch.empty_like(query)
        status = torch.empty((rows, heads // 16), device="cuda", dtype=torch.uint8)
        scratch = torch.empty(
            (scratch_bytes(rows, heads, slots),),
            device="cuda",
            dtype=torch.uint8,
        )
        scales = torch.empty((7, 0), device="cuda", dtype=torch.uint8)
        compressed = torch.empty((0, 512), device="cuda", dtype=torch.bfloat16)
        compressed_scales = torch.empty((0, 0), device="cuda", dtype=torch.uint8)

        arguments = (
            query,
            sink,
            indices,
            table,
            scales,
            compressed,
            compressed_scales,
            scratch,
            output,
            status,
        )
        return query, table, sink, indices, output, status, arguments
