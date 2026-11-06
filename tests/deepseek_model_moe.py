import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import model, model_expert, model_linear
from oh_my_vllm.models.deepseek_v41.model import (
    ModelArgs,
    MoE,
)

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelMoe(DeepSeekModelFixture, unittest.TestCase):
    def test_moe_merges_routed_experts_and_shared_output(self):
        class FixedGate(torch.nn.Module):
            def forward(self, value, image_mask=None):
                weights = torch.tensor(
                    (
                        (0.25, 0.75),
                        (0.5, 0.5),
                        (0.75, 0.25),
                        (0.6, 0.4),
                    ),
                    dtype=value.dtype,
                )
                indices = torch.tensor(
                    (
                        (0, 2),
                        (1, 3),
                        (2, 0),
                        (3, 1),
                    ),
                    dtype=torch.long,
                )
                return weights[: value.size(0)], indices[: value.size(0)]

        class FixedExpert(torch.nn.Module):
            def __init__(self, offset):
                super().__init__()
                self.offset = offset

            def forward(self, value, weights):
                return (value + self.offset) * weights

        class ZeroExpert(torch.nn.Module):
            def forward(self, value):
                return torch.zeros_like(value)

        args = ModelArgs(
            dim=2,
            moe_inter_dim=4,
            n_routed_experts=4,
            n_activated_experts=2,
            n_shared_experts=1,
            dtype="bf16",
            expert_dtype=None,
            parallel_strategy="TP4",
        )
        moe = MoE(0, args)
        moe.gate = FixedGate()
        moe.experts = torch.nn.ModuleList(FixedExpert(index + 1) for index in range(4))
        moe.shared_experts = ZeroExpert()
        value = torch.tensor(
            (
                ((1.0, 2.0),),
                ((3.0, 4.0),),
                ((5.0, 6.0),),
                ((7.0, 8.0),),
            )
        )

        result = moe(value)

        expected = torch.tensor(
            (
                ((3.5, 4.5),),
                ((6.0, 7.0),),
                ((7.5, 8.5),),
                ((10.2, 11.2),),
            )
        )
        torch.testing.assert_close(result, expected)

    def test_fp8_expert_quantizes_shared_up_input_once(self):
        expert = model.Expert(32, 32, dtype=torch.float8_e4m3fn)
        with torch.no_grad():
            for projection, value in (
                (expert.w1, 1.0),
                (expert.w3, 0.5),
                (expert.w2, 0.25),
            ):
                projection.weight.copy_(
                    torch.full(projection.weight.shape, value).to(torch.float8_e4m3fn)
                )
                projection.scale.view(torch.uint8).fill_(127)
        values = torch.ones((4, 32), dtype=torch.bfloat16)
        with (
            patch.object(
                model_expert, "act_quant", wraps=model_expert.act_quant
            ) as quantize,
            patch.object(model_linear, "act_quant", quantize),
        ):
            actual = expert(values)
        torch.testing.assert_close(
            actual, torch.full_like(actual, 4096), rtol=0, atol=0
        )
        self.assertEqual(quantize.call_count, 2)

    def test_expert_keeps_replaced_projection_calls(self):
        values = torch.ones((4, 32), dtype=torch.bfloat16)
        expert = model.Expert(32, 32, dtype=torch.bfloat16)
        expert.w1, expert.w3, expert.w2 = (torch.nn.Identity() for _ in range(3))
        torch.testing.assert_close(
            expert(values), torch.full_like(values, 0.73046875), rtol=0, atol=0
        )

        class ZeroLinear(model.Linear):
            def forward(self, values):
                return torch.zeros_like(values)

        for mode in ("subclass", "instance", "foreign_instance", "hook", "global_hook"):
            with self.subTest(mode=mode), ExitStack() as stack:
                expert = model.Expert(32, 32, dtype=torch.float8_e4m3fn)
                if mode == "subclass":
                    expert.w1 = ZeroLinear(32, 32, dtype=torch.float8_e4m3fn)
                with torch.no_grad():
                    for projection in (expert.w1, expert.w3, expert.w2):
                        projection.weight.copy_(
                            torch.ones(projection.weight.shape).to(torch.float8_e4m3fn)
                        )
                        projection.scale.view(torch.uint8).fill_(127)

                def zero_gate(module, inputs, output, gate=expert.w1):
                    return torch.zeros_like(output) if module is gate else None

                if mode == "instance":
                    stack.enter_context(
                        patch.object(
                            expert.w1, "forward", return_value=torch.zeros_like(values)
                        )
                    )
                elif mode == "foreign_instance":
                    replacement = model.Linear(32, 32, dtype=torch.float8_e4m3fn)
                    with torch.no_grad():
                        replacement.weight.view(torch.uint8).zero_()
                        replacement.scale.view(torch.uint8).fill_(127)
                    stack.enter_context(
                        patch.object(expert.w1, "forward", replacement.forward)
                    )
                elif mode == "hook":
                    stack.enter_context(expert.w1.register_forward_hook(zero_gate))
                elif mode == "global_hook":
                    stack.enter_context(
                        torch.nn.modules.module.register_module_forward_hook(zero_gate)
                    )
                actual = expert(values)
                torch.testing.assert_close(
                    actual, torch.zeros_like(actual), rtol=0, atol=0
                )

    def test_moe_bounds_routed_rows_without_changing_outputs(self):
        class FixedGate(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, value, image_mask=None):
                self.calls += 1
                weights = torch.tensor(
                    ((0.25, 0.75),),
                    dtype=value.dtype,
                    device=value.device,
                ).expand(value.size(0), -1)
                indices = torch.tensor(
                    ((0, 1),),
                    dtype=torch.long,
                    device=value.device,
                ).expand(value.size(0), -1)
                return weights, indices

        class FixedExpert(torch.nn.Module):
            def __init__(self, output):
                super().__init__()
                self.output = output

            def forward(self, value, weights):
                return torch.full_like(value, self.output)

        class SharedExpert(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.rows = []

            def forward(self, value):
                self.rows.append(value.size(0))
                output = value.clone()
                output[:, 0] = -1
                return output

        args = ModelArgs(
            dim=2,
            moe_inter_dim=4,
            n_routed_experts=2,
            n_activated_experts=2,
            n_shared_experts=1,
            dtype="bf16",
            expert_dtype=None,
            parallel_strategy="TP4",
        )
        moe = MoE(0, args)
        gate = FixedGate()
        moe.gate = gate
        moe.experts = torch.nn.ModuleList(
            FixedExpert(output) for output in (1, 1 / 256)
        )
        moe.shared_experts = SharedExpert()
        value = torch.zeros((1025, 1, 2), dtype=torch.bfloat16)
        value[:, 0, 1] = torch.arange(1025) % 3

        result = moe(value)

        expected = torch.full_like(value, 1 / 256)
        expected[:, 0, 1] = value[:, 0, 1] + 1
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertGreater(gate.calls, 1)
        self.assertEqual(moe.shared_experts.rows, [1025])
