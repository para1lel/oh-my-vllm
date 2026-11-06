import unittest
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import model, model_context
from oh_my_vllm.models.deepseek_v41.quantization import fp4_values

from tests.projection_precision_fixtures import (
    prepared_moe_literal,
)


class GroupedProjectionPrecisionBindings(unittest.TestCase):
    def test_prepared_experts_refresh_after_rebinding_and_state_assignment(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with (
                self.subTest(device=device),
                patch.object(model_context, "world_size", 1),
            ):
                moe, values, routing = prepared_moe_literal(device)

                def check(expected, moe=moe, values=values, routing=routing):
                    actual = moe._forward_rows(values, routing=routing)
                    want = torch.zeros_like(values)
                    want[:, 0] = expected
                    torch.testing.assert_close(actual, want, rtol=0, atol=0)

                check(1.5)
                for expert in moe.experts:
                    expert.w2 = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2).to(
                        device
                    )
                    with torch.no_grad():
                        expert.w2.weight.view(torch.uint8).zero_()
                        expert.w2.weight.view(torch.uint8)[0, 0] = 4
                        expert.w2.scale.fill_(1)
                self.assertIsNone(moe._routed_state.weights)
                check(3)

                for expert in moe.experts:
                    weight = expert.w2.weight.detach().clone()
                    weight.view(torch.uint8).zero_()
                    weight.view(torch.uint8)[0, 0] = 6
                    expert.w2.weight = torch.nn.Parameter(weight, requires_grad=False)
                self.assertIsNone(moe._routed_state.weights)
                check(6)
                self.assertIs(
                    moe._routed_state.weights.down[0][0], moe.experts[0].w2.weight
                )

                for expert in moe.experts:
                    scale = expert.w2.scale.detach().clone()
                    expert.w2.scale = torch.nn.Parameter(scale, requires_grad=False)
                self.assertIsNone(moe._routed_state.weights)
                check(6)
                self.assertIs(
                    moe._routed_state.weights.down[0][1], moe.experts[0].w2.scale
                )

                state = {
                    name: value.view(torch.uint8).clone().view(value.dtype)
                    for name, value in moe.state_dict().items()
                }
                for name, value in state.items():
                    if name.endswith("w2.weight"):
                        value.view(torch.uint8)[0, 0] = 6
                moe.load_state_dict(state, assign=True)
                check(6)
                moe._apply(
                    lambda value: (
                        fp4_values(value).to(torch.bfloat16)
                        if value.dtype == torch.float4_e2m1fn_x2
                        else value
                    )
                )
                check(5.84375)

    def test_non_fp4_prepare_recovers_grouped_after_child_replacement(self):
        with patch.object(model_context, "world_size", 1):
            args = model.ModelArgs(
                dim=32,
                moe_inter_dim=64,
                n_routed_experts=4,
                n_activated_experts=2,
                expert_dtype="bf16",
                parallel_strategy="TEP4",
            )
            moe = model.MoE(0, args).to("cpu")

            class ZeroShared(torch.nn.Module):
                def forward(self, value):
                    return torch.zeros_like(value)

            moe.shared_experts = ZeroShared()
            moe.prepare_routed_experts()
            self.assertIs(moe._routed_state.weights, False)
            self.assertTrue(moe._routed_state.registered)
            for expert in moe.experts:
                for name, in_features, out_features in (
                    ("w1", 32, 64),
                    ("w3", 32, 64),
                    ("w2", 64, 32),
                ):
                    layer = model.Linear(
                        in_features,
                        out_features,
                        dtype=torch.float4_e2m1fn_x2,
                    )
                    with torch.no_grad():
                        layer.weight.view(torch.uint8).zero_()
                        layer.weight.view(torch.uint8)[0, 0] = 2
                        layer.scale.fill_(1)
                    setattr(expert, name, layer)
            self.assertIsNone(moe._routed_state.weights)
            values = torch.zeros((1, 32), dtype=torch.bfloat16)
            values[:, 0] = 1
            routing = (
                torch.ones((1, 2)),
                torch.tensor([[0, 1]]),
            )
            actual = moe._forward_rows(values, routing=routing)
            want = torch.zeros_like(values)
            want[:, 0] = 1.5
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
            self.assertIsNotNone(moe._routed_state.weights)

    def test_child_binding_changes_and_live_clamp_refresh_routed_execution(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for change, expected in (
                ("load", 3),
                ("convert", 1.4609375),
                ("weight", 3),
                ("scale", 3),
                ("clamp", 0.3125),
            ):
                with (
                    self.subTest(device=device, change=change),
                    patch.object(model_context, "world_size", 1),
                ):
                    moe, values, routing = prepared_moe_literal(device)
                    moe._forward_rows(values, routing=routing)
                    for expert in moe.experts:
                        if change == "load":
                            state = {
                                name: value.view(torch.uint8).clone().view(value.dtype)
                                for name, value in expert.w2.state_dict().items()
                            }
                            state["weight"].view(torch.uint8)[0, 0] = 4
                            expert.w2.load_state_dict(state, assign=True)
                        elif change == "convert":
                            expert.w2._apply(
                                fn=lambda value: (
                                    fp4_values(value).to(torch.bfloat16)
                                    if value.dtype == torch.float4_e2m1fn_x2
                                    else value
                                ),
                                recurse=True,
                            )
                        elif change == "weight":
                            weight = expert.w2.weight.detach().clone()
                            weight.view(torch.uint8).zero_()
                            weight.view(torch.uint8)[0, 0] = 4
                            expert.w2.weight = torch.nn.Parameter(
                                weight, requires_grad=False
                            )
                        elif change == "scale":
                            scale = expert.w2.scale.detach().clone().fill_(2)
                            expert.w2.scale = torch.nn.Parameter(
                                scale, requires_grad=False
                            )
                        else:
                            expert.swiglu_limit = 0.5
                    if change != "clamp":
                        self.assertIsNone(moe._routed_state.weights)
                    actual = moe._forward_rows(values, routing=routing)
                    want = torch.zeros_like(values)
                    want[:, 0] = expected
                    torch.testing.assert_close(actual, want, rtol=0, atol=0)
                    for expert in moe.experts:
                        expert.swiglu_limit = 0
                        expert.w2 = model.Linear(
                            64, 32, dtype=torch.float4_e2m1fn_x2
                        ).to(device)
                        with torch.no_grad():
                            expert.w2.weight.view(torch.uint8).zero_()
                            expert.w2.weight.view(torch.uint8)[0, 0] = 4
                            expert.w2.scale.fill_(1)
                    moe.prepare_routed_experts()
                    grouped = moe._grouped_routed(
                        values.repeat(2, 1),
                        torch.zeros(2, dtype=torch.long, device=device),
                        torch.ones(2, device=device),
                        [1, 1, 0, 0],
                        torch.tensor([1, 1, 0, 0], device=device),
                    )
                    self.assertIsNotNone(grouped)
                    want = values.repeat(2, 1).zero_()
                    want[:, 0] = 1.5
                    torch.testing.assert_close(grouped[1], want, rtol=0, atol=0)

    def test_prepared_expert_parallel_ranges_keep_zero_row_steps(self):
        devices = [("cpu", "cpu")]
        if torch.cuda.is_available():
            devices.extend((("cuda", "cpu"), ("cuda", "cuda")))
        for device, default_device in devices:
            for strategy in ("TEP4", "DEP4"):
                for rank in (0, 1):
                    with (
                        self.subTest(
                            device=device,
                            default_device=default_device,
                            strategy=strategy,
                            rank=rank,
                        ),
                        torch.device(default_device),
                        patch.object(
                            model_context, "world_size", 4 if strategy == "TEP4" else 1
                        ),
                        patch.object(model_context, "expert_world_size", 4),
                        patch.object(model_context, "rank", rank),
                    ):
                        moe, values, pointers, first_value = self._zero_row_moe_fixture(
                            device, strategy, rank
                        )
                        for counts, rows, expected in (
                            ([1, 0, 1, 0, 0, 0, 0, 0], [rank], first_value),
                            (
                                [0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 1] if rank else [],
                                first_value,
                            ),
                            ([0, 1, 0, 1, 0, 0, 0, 0], [rank], 1.5),
                            (
                                [2, 0, 0, 0, 0, 0, 0, 0],
                                [] if rank else [0, 1],
                                first_value,
                            ),
                            ([1, 0, 1, 0, 0, 0, 0, 0], [rank], first_value),
                        ):
                            pointers = self._check_expert_parallel_step(
                                moe, values, device, counts, rows, expected, pointers
                            )
                        expert = moe.experts[moe.experts_start_idx]
                        with torch.no_grad():
                            weight = expert.w2.weight.detach().clone()
                            weight.view(torch.uint8)[0, 0] = 2 if rank else 6
                        expert.w2.weight = torch.nn.Parameter(
                            weight, requires_grad=False
                        )
                        self.assertIsNone(moe._routed_state.pointers)
                        grouped = moe._grouped_routed(
                            values.repeat(2, 1),
                            torch.arange(2, device=device),
                            torch.ones(2, device=device),
                            counts,
                            torch.tensor(counts, device=device),
                        )
                        want = torch.zeros((1, 32), dtype=torch.bfloat16, device=device)
                        want[:, 0] = 0.75 if rank else 3
                        torch.testing.assert_close(grouped[1], want, rtol=0, atol=0)
                        if device == "cuda":
                            self.assertIsNotNone(moe._routed_state.pointers)
                            self.assertIsNot(moe._routed_state.pointers, pointers)

    def test_rebinding_one_owner_does_not_invalidate_another_shared_owner(self):
        with patch.object(model_context, "world_size", 1):
            first, values, routing = prepared_moe_literal("cpu")
            second, _, _ = prepared_moe_literal("cpu")
            shared = first.experts[0].w2
            second.experts[0].w2 = shared
            first.prepare_routed_experts()
            second.prepare_routed_experts()
            self.assertIsNotNone(first._routed_state.weights)
            self.assertIsNotNone(second._routed_state.weights)

            replacement = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
            with torch.no_grad():
                replacement.weight.view(torch.uint8).zero_()
                replacement.weight.view(torch.uint8)[0, 0] = 4
                replacement.scale.fill_(1)
            first.experts[0].w2 = replacement
            self.assertIsNone(first._routed_state.weights)
            first._forward_rows(values, routing=routing)
            self.assertIsNotNone(first._routed_state.weights)
            self.assertIsNotNone(second._routed_state.weights)

            shared._apply(
                lambda value: (
                    fp4_values(value).to(torch.bfloat16)
                    if value.dtype == torch.float4_e2m1fn_x2
                    else value
                )
            )
            self.assertIsNotNone(first._routed_state.weights)
            self.assertIsNone(second._routed_state.weights)
            first._forward_rows(values, routing=routing)
            second._forward_rows(values, routing=routing)

    def test_shared_projection_invalidates_all_live_owners(self):
        with patch.object(model_context, "world_size", 1):
            first, _, _ = prepared_moe_literal("cpu")
            second, _, _ = prepared_moe_literal("cpu")
            shared = first.experts[0].w2
            second.experts[0].w2 = shared
            first.prepare_routed_experts()
            second.prepare_routed_experts()
            shared._apply(
                lambda value: (
                    fp4_values(value).to(torch.bfloat16)
                    if value.dtype == torch.float4_e2m1fn_x2
                    else value
                )
            )
            self.assertIsNone(first._routed_state.weights)
            self.assertIsNone(second._routed_state.weights)

    def _check_expert_parallel_step(
        self, moe, values, device, counts, rows, expected, pointers
    ):
        grouped = moe._grouped_routed(
            values.repeat(2, 1),
            torch.arange(2, device=device),
            torch.ones(2, device=device),
            counts,
            torch.tensor(counts, device=device),
        )
        if not rows:
            self.assertIsNone(grouped)
        else:
            torch.testing.assert_close(grouped[0], torch.tensor(rows, device=device))
            want = torch.zeros((len(rows), 32), dtype=torch.bfloat16, device=device)
            want[:, 0] = expected
            torch.testing.assert_close(grouped[1], want, rtol=0, atol=0)
        if device == "cuda":
            self.assertIsNotNone(moe._routed_state.pointers)
            if pointers is not None:
                self.assertIs(moe._routed_state.pointers, pointers)
            pointers = moe._routed_state.pointers
            for (
                weight_ptrs,
                scale_ptrs,
                weight_host,
                scale_host,
            ) in pointers:
                self.assertEqual(weight_ptrs.device, values.device)
                self.assertEqual(scale_ptrs.device, values.device)
                for host in (weight_host, scale_host):
                    self.assertEqual(host.device.type, "cpu")
                    self.assertTrue(host.is_pinned())
        return pointers

    def _zero_row_moe_fixture(self, device, strategy, rank):
        moe, values, _ = prepared_moe_literal(device, strategy, 8)
        if rank:
            with torch.no_grad():
                for expert in moe.experts[2:4]:
                    expert.w2.weight.view(torch.uint8)[0, 0] = 6
        with torch.no_grad():
            second = moe.experts[moe.experts_start_idx + 1]
            second.w2.weight.view(torch.uint8)[0, 0] = 4
        empty = moe._grouped_routed(
            values[:0],
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, device=device),
            [0] * 8,
            torch.zeros(8, dtype=torch.int32, device=device),
        )
        self.assertIsNone(empty)
        self.assertIsNone(moe._routed_state.pointers)
        pointers = None
        first_value = 3 if rank else 0.75
        return moe, values, pointers, first_value
