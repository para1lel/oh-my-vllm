import unittest
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import model_context
from oh_my_vllm.models.deepseek_v41.kernel import act_quant, fp8_gemm
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    MoE,
)

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelMoeCuda(DeepSeekModelFixture, unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_moe_grouped_fp4_path_matches_expert_reference(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            moe = self._grouped_fp4_moe()

            class FixedGate(torch.nn.Module):
                def forward(self, value, image_mask=None):
                    weights = torch.tensor(
                        (
                            (0.25, 0.75),
                            (0.5, 0.5),
                            (0.75, 0.25),
                            (0.6, 0.4),
                        ),
                        device=value.device,
                        dtype=torch.float32,
                    )
                    indices = torch.tensor(
                        ((0, 2), (1, 3), (2, 0), (3, 1)),
                        device=value.device,
                        dtype=torch.long,
                    )
                    return weights[: value.size(0)], indices[: value.size(0)]

            class ZeroExpert(torch.nn.Module):
                def forward(self, value):
                    return torch.zeros_like(value)

            moe.gate = FixedGate()
            moe.shared_experts = ZeroExpert()
            value = torch.randn((4, 1, 32), device="cuda", dtype=torch.bfloat16)
            actual = moe(value)

            rows = value.view(-1, 32)
            weights, indices = moe.gate(rows)
            flat_indices = indices.flatten()
            order = torch.argsort(flat_indices)
            sorted_tokens = (
                torch.arange(rows.size(0), device="cuda")
                .view(-1, 1)
                .expand_as(indices)
                .flatten()[order]
            )
            sorted_weights = weights.flatten()[order]
            sorted_rows = rows.index_select(0, sorted_tokens)
            counts = torch.bincount(flat_indices, minlength=4).tolist()
            expected_rows = torch.zeros_like(rows, dtype=torch.float32)
            first = 0
            for index, count in enumerate(counts):
                last = first + count
                if count:
                    expected_rows.index_add_(
                        0,
                        sorted_tokens[first:last],
                        moe.experts[index](
                            sorted_rows[first:last],
                            sorted_weights[first:last].unsqueeze(-1),
                        ).float(),
                    )
                first = last
            expected = expected_rows.to(value.dtype).view_as(value)
            torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_moe_device_counts_avoid_the_host_read_for_many_rows(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        world_size_patch = patch.object(model_context, "world_size", 1)
        world_size_patch.start()
        try:
            moe = self._device_count_moe()

            rows_count = 10  # two activated experts give 20 routed rows
            positions = torch.arange(rows_count, device="cuda")

            class FixedGate(torch.nn.Module):
                def forward(self, value, image_mask=None):
                    weights = torch.linspace(
                        0.1, 0.9, rows_count * 2, device=value.device
                    ).view(rows_count, 2)
                    indices = torch.stack(
                        (positions % 4, (positions + 1) % 4), dim=1
                    ).long()
                    return weights, indices

            class ZeroExpert(torch.nn.Module):
                def forward(self, value):
                    return torch.zeros_like(value)

            moe.gate = FixedGate()
            moe.shared_experts = ZeroExpert()
            value = torch.randn(
                (rows_count, 1, 32), device="cuda", dtype=torch.bfloat16
            )

            rows = value.view(-1, 32)
            weights, indices = moe.gate(rows)
            flat_indices = indices.flatten()
            order = torch.argsort(flat_indices)
            sorted_tokens = (
                torch.arange(rows.size(0), device="cuda")
                .view(-1, 1)
                .expand_as(indices)
                .flatten()[order]
            )
            sorted_weights = weights.flatten()[order]
            sorted_rows = rows.index_select(0, sorted_tokens)
            counts = torch.bincount(flat_indices, minlength=4).tolist()
            expected_rows = torch.zeros_like(rows, dtype=torch.float32)
            first = 0
            for index, count in enumerate(counts):
                last = first + count
                if count:
                    expected_rows.index_add_(
                        0,
                        sorted_tokens[first:last],
                        moe.experts[index](
                            sorted_rows[first:last],
                            sorted_weights[first:last].unsqueeze(-1),
                        ).float(),
                    )
                first = last
            expected = expected_rows.to(value.dtype).view_as(value)

            original = torch.Tensor.tolist

            def guarded(tensor):
                if tensor.is_cuda:
                    raise AssertionError("host read of a CUDA routed count")
                return original(tensor)

            with patch.object(torch.Tensor, "tolist", guarded):
                actual = moe(value)
            torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)
        finally:
            world_size_patch.stop()
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fp8_projection_accepts_long_prefill_rows(self):
        previous_device = torch.get_default_device()
        previous_dtype = torch.get_default_dtype()
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        try:
            torch.manual_seed(11)
            values = torch.randn(512, 128, device="cuda", dtype=torch.bfloat16)
            weight = torch.randn(32, 128, device="cuda", dtype=torch.float32).to(
                torch.float8_e4m3fn
            )
            weight_scale = torch.ones((1, 4), device="cuda", dtype=torch.float8_e8m0fnu)
            encoded, scales = act_quant(
                values,
                block_size=32,
                scale_fmt="ue8m0",
                scale_dtype=torch.float8_e8m0fnu,
            )

            result = fp8_gemm(
                encoded,
                scales,
                weight,
                weight_scale,
                block_size=32,
            )

            self.assertEqual(tuple(result.shape), (512, 32))
            self.assertTrue(torch.isfinite(result).all())
        finally:
            torch.set_default_device(previous_device)
            torch.set_default_dtype(previous_dtype)

    def _grouped_fp4_moe(self):
        torch.manual_seed(0)
        args = ModelArgs(
            dim=32,
            moe_inter_dim=32,
            n_routed_experts=4,
            n_activated_experts=2,
            n_shared_experts=1,
            dtype="bf16",
            expert_dtype="fp4",
            parallel_strategy="TP4",
            swiglu_limit=10,
        )
        moe = MoE(0, args).eval()
        for expert in moe.experts:
            for linear in (expert.w1, expert.w2, expert.w3):
                raw = torch.randint(
                    0, 256, linear.weight.shape, device="cuda", dtype=torch.uint8
                )
                linear.weight.data.copy_(raw.view(torch.float4_e2m1fn_x2))
                linear.scale.data.fill_(1)
        return moe

    def _device_count_moe(self):
        torch.manual_seed(1)
        args = ModelArgs(
            dim=32,
            moe_inter_dim=32,
            n_routed_experts=4,
            n_activated_experts=2,
            n_shared_experts=1,
            dtype="bf16",
            expert_dtype="fp4",
            parallel_strategy="TP4",
            swiglu_limit=10,
        )
        moe = MoE(0, args).eval()
        for expert in moe.experts:
            for linear in (expert.w1, expert.w2, expert.w3):
                raw = torch.randint(
                    0, 256, linear.weight.shape, device="cuda", dtype=torch.uint8
                )
                linear.weight.data.copy_(raw.view(torch.float4_e2m1fn_x2))
                linear.scale.data.fill_(1)
        return moe
