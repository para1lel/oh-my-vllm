import os
import subprocess
import sys
import unittest

import torch
import torch.nn.functional as F
from oh_my_vllm.models.deepseek_v41 import kernel, model

from tests.logical_rows_cuda_layout import LogicalRowsCuda as LogicalRowsCuda
from tests.logical_rows_cuda_projection import (
    LogicalRowsCudaCudaProjection as LogicalRowsCudaCudaProjection,
)
from tests.logical_rows_fixtures import (
    _assert_released_sequential_projection as _assert_released_sequential_projection,
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


if __name__ == "__main__":
    unittest.main()
