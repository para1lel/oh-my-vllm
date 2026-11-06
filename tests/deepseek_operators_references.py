import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from oh_my_vllm.models.deepseek_v41 import kernel
from oh_my_vllm.models.deepseek_v41.loader import source_index


class DeepSeekOperatorContractsReferences(unittest.TestCase):
    def test_source_index_returns_the_digest_for_the_loaded_index_bytes(self):
        raw_index = (
            b'{"weight_map":{"model.embed.weight":"model-00001-of-00048.safetensors"}}'
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors.index.json").write_bytes(raw_index)

            weight_map, digest = source_index(root)

        self.assertEqual(
            weight_map,
            {"model.embed.weight": "model-00001-of-00048.safetensors"},
        )
        self.assertEqual(digest, hashlib.sha256(raw_index).hexdigest())

    def test_rms_reference_path_matches_the_model_equation(self):
        generator = torch.Generator().manual_seed(11)
        values = torch.randn((3, 512), generator=generator, dtype=torch.bfloat16)
        weight = torch.randn((512,), generator=generator, dtype=torch.bfloat16)
        epsilon = 1e-20

        actual = kernel.rms(values, weight, epsilon)
        expected = (
            values.float()
            * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + epsilon)
            * weight.float()
        ).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_sparse_attention_reference_path_matches_the_model_equation(self):
        generator = torch.Generator().manual_seed(13)
        query = torch.randn((1, 2, 4, 32), generator=generator, dtype=torch.bfloat16)
        kv = torch.randn((1, 5, 32), generator=generator, dtype=torch.bfloat16)
        sink = torch.randn((4,), generator=generator)
        indices = torch.tensor([[[0, 1, 3], [0, 2, -1]]], dtype=torch.int32)
        actual = kernel.sparse_attn(query, kv, sink, indices, 32**-0.5)
        gathered = (
            kv[:, None]
            .expand(-1, query.size(1), -1, -1)
            .gather(2, indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, -1, 32))
        )
        valid = indices >= 0
        scores = (
            torch.einsum("bshd,bskd->bshk", query.float(), gathered.float()) * 32**-0.5
        )
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
        maximum = scores.amax(-1, keepdim=True)
        probabilities = (scores - maximum).exp()
        denominator = (
            probabilities.sum(-1, keepdim=True)
            + (sink.view(1, 1, -1, 1) - maximum).exp()
        )
        expected = (
            torch.einsum(
                "bshk,bskd->bshd",
                probabilities.to(torch.bfloat16).float(),
                gathered.float(),
            )
            / denominator
        ).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_hyperconnection_reference_path_matches_the_operator_contract(self):
        generator = torch.Generator().manual_seed(7)
        residual = torch.randn((2, 4, 5120), generator=generator, dtype=torch.bfloat16)
        incoming = torch.rand((2, 4), generator=generator, dtype=torch.float32)
        projection = torch.randn((24, 4 * 5120), generator=generator) * 0.01
        scale = torch.randn((3,), generator=generator)
        base = torch.randn((24,), generator=generator)
        branch = torch.randn((2, 5120), generator=generator, dtype=torch.bfloat16)

        collapsed, next_pre, post, comb = kernel.hc_prepare(
            residual, incoming, projection, scale, base
        )
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

        torch.testing.assert_close(collapsed, expected_collapsed, atol=0, rtol=0)
        torch.testing.assert_close(next_pre, expected_pre, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(post, expected_post, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(comb, expected_comb, atol=1e-6, rtol=1e-6)

        expanded = kernel.hc_expand(residual, branch, post, comb)
        expected_expanded = expected_post.unsqueeze(-1) * branch.float().unsqueeze(-2)
        expected_expanded = expected_expanded + torch.sum(
            expected_comb.unsqueeze(-1) * residual.float().unsqueeze(-2), dim=1
        )
        torch.testing.assert_close(
            expanded, expected_expanded.to(torch.bfloat16), atol=0, rtol=0
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_hyperconnection_native_path_matches_reference(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(17)
        residual = torch.randn(
            (1, 4, 5120), generator=generator, device=device, dtype=torch.bfloat16
        )
        incoming = torch.rand((1, 4), generator=generator, device=device)
        projection = (
            torch.randn((24, 4 * 5120), generator=generator, device=device) * 0.01
        )
        scale = torch.randn((3,), generator=generator, device=device)
        base = torch.randn((24,), generator=generator, device=device)
        branch = torch.randn(
            (1, 5120), generator=generator, device=device, dtype=torch.bfloat16
        )

        collapsed, next_pre, post, comb = kernel.hc_prepare(
            residual, incoming, projection, scale, base
        )
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

        torch.testing.assert_close(collapsed, expected_collapsed, atol=0, rtol=0)
        torch.testing.assert_close(next_pre, expected_pre, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(post, expected_post, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(comb, expected_comb, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(
            kernel.hc_expand(residual, branch, post, comb),
            expected_expanded.to(torch.bfloat16),
            atol=0.002,
            rtol=0.002,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_long_routing_and_swiglu_paths_accept_prefill_rows(self):
        device = torch.device("cuda")
        torch.manual_seed(13)
        rows = 512
        logits = torch.randn((rows, 384), device=device)
        bias = torch.randn((384,), device=device)

        coefficients, expert_ids = kernel.route(logits, bias, None, None)
        self.assertEqual(tuple(coefficients.shape), (rows, 6))
        self.assertEqual(tuple(expert_ids.shape), (rows, 6))
        self.assertTrue(torch.isfinite(coefficients).all())

        gate = torch.randn((rows, 2304), device=device, dtype=torch.bfloat16)
        up = torch.randn_like(gate)
        output = kernel.swiglu(
            gate,
            up,
            torch.ones((rows, 1), device=device),
            limit=10,
        )
        self.assertEqual(tuple(output.shape), tuple(gate.shape))
        self.assertTrue(torch.isfinite(output).all())

    def test_grouped_fp4_reference_path_handles_variable_expert_batches(self):
        values = torch.tensor(
            (
                (1.0,) * 32,
                (-1.0,) * 32,
                (1.0,) * 32,
                (-1.0,) * 32,
                (1.0,) * 32,
            ),
            dtype=torch.bfloat16,
        )
        counts = torch.tensor((2, 0, 3), dtype=torch.int32)
        weights = (
            (
                torch.ones((4, 32), dtype=torch.bfloat16),
                torch.ones((4, 1), dtype=torch.float32),
            ),
            (
                torch.full((4, 32), 3, dtype=torch.bfloat16),
                torch.ones((4, 1), dtype=torch.float32),
            ),
            (
                torch.full((4, 32), 2, dtype=torch.bfloat16),
                torch.ones((4, 1), dtype=torch.float32),
            ),
        )

        actual = kernel.grouped_fp4_gemm(values, counts, weights)

        expected = torch.cat(
            (
                torch.nn.functional.linear(values[:2].float(), weights[0][0].float()),
                torch.nn.functional.linear(values[2:].float(), weights[2][0].float()),
            )
        ).to(actual.dtype)
        torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
