import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel_native, model_context
from oh_my_vllm.models.deepseek_v41.model import ModelArgs, MoE
from oh_my_vllm.models.deepseek_v41.model_moe import _route_counts

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelPrefillWaves(DeepSeekModelFixture, unittest.TestCase):
    def test_moe_prefill_keeps_collective_waves_and_reuses_storage(self):
        from oh_my_vllm.models.deepseek_v41 import kernel
        from oh_my_vllm.models.deepseek_v41.moe_prefill import PrefillWorkspace

        Native = self._prefill_native_type()

        def gate(value, image_mask=None):
            ids = (value[:, 1].long()[:, None] + torch.tensor([0, 3, 6])) % 8
            weights = torch.tensor([0.25, 0.5, 0.75]).expand(value.size(0), -1)
            return weights, ids

        for strategy in ("TEP4", "TP4"):
            for wave in (4096, 8192):
                with self.subTest(strategy=strategy, wave=wave), ExitStack() as stack:
                    native, moe, dim, workspace, addresses, gate_mock = (
                        self._prefill_wave_fixture(
                            stack,
                            strategy,
                            wave,
                            kernel,
                            Native,
                            PrefillWorkspace,
                            gate,
                        )
                    )
                    value, shared = self._check_prefill_waves(
                        native,
                        gate_mock,
                        wave,
                        dim,
                        moe,
                        gate,
                        strategy,
                        workspace,
                        addresses,
                    )
                    if strategy == "TEP4":
                        native.projections.clear()
                        native.collectives.clear()
                        gate_mock.reset_mock()
                        indices = torch.tensor([[0, 4, 6]]).expand(1025, -1)
                        weights = torch.ones((1025, 3))
                        routing = workspace.routing(
                            moe, value[0], None, (weights, indices)
                        )
                        actual = moe._forward_rows(
                            value[0],
                            routing=routing,
                            shared_output=shared,
                            workspace=workspace,
                        )
                        self.assertTrue(
                            torch.equal(actual, torch.full_like(actual, 1.5))
                        )
                        self.assertEqual(native.projections, [])
                        self.assertEqual(
                            native.collectives,
                            [("reduce", (1024, dim)), ("reduce", (1, dim))],
                        )
                        gate_mock.assert_not_called()

    def test_route_counts_match_bincount_without_a_host_read(self):
        indices = torch.tensor([0, 3, 3, 5, 383, 0], dtype=torch.int32)
        expected = torch.bincount(indices.to(torch.int64), minlength=384).to(
            torch.int32
        )

        actual = _route_counts(indices, 384)

        self.assertEqual(actual.dtype, torch.int32)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            _route_counts(torch.empty(0, dtype=torch.int32), 4),
            torch.zeros(4, dtype=torch.int32),
        )

    def _check_prefill_waves(
        self, native, gate_mock, wave, dim, moe, gate, strategy, workspace, addresses
    ):
        for length in (wave + 1025, 1025):
            native.projections.clear()
            native.collectives.clear()
            gate_mock.reset_mock()
            value = torch.zeros((1, length, dim), dtype=torch.bfloat16)
            value[0, :, 0] = torch.arange(length) % 3
            value[0, :, 1] = torch.arange(length) % 8
            shared = torch.full((length, dim), -0.5, dtype=torch.bfloat16)
            with patch.object(moe.shared_experts, "forward", return_value=shared):
                actual = moe(value, prefill_eager=True)
            ids = self._check_prefill_routed_output(
                gate, value, length, dim, moe, strategy, shared, actual
            )
            collective_rows = [
                min(1024, length - first) for first in range(0, length, 1024)
            ]
            expected_collectives = []
            for rows in collective_rows:
                if strategy == "TEP4":
                    expected_collectives.append(("reduce", (rows, dim)))
                else:
                    expected_collectives.extend(
                        (("product", (rows * 3, 32)), ("sum", (rows, 32)))
                    )
            self.assertEqual(native.collectives, expected_collectives)
            self.assertEqual(
                [call.args[0].size(0) for call in gate_mock.call_args_list],
                collective_rows,
            )
            self.assertEqual(
                max(count for role, count in native.projections if role == "gate"),
                min(wave, length) * 3
                if strategy == "TP4"
                else int(
                    (
                        (ids[: min(wave, length)] >= 2) & (ids[: min(wave, length)] < 4)
                    ).sum()
                ),
            )
            workspace.reserve(moe, torch.device("cpu"), 4)
            self.assertEqual(
                addresses,
                {name: value.data_ptr() for name, value in workspace.buffers.items()},
            )
        return value, shared

    def _check_prefill_routed_output(
        self, gate, value, length, dim, moe, strategy, shared, actual
    ):
        weights, ids = gate(value[0])
        expected = torch.zeros((length, dim), dtype=torch.float32)
        for expert in range(moe.experts_start_idx, moe.experts_end_idx):
            token, slot = (ids == expert).nonzero(as_tuple=True)
            contribution = (
                value[0, token, 0].float() * weights[token, slot] + expert / 8
            ).to(torch.bfloat16)
            expected[token] += contribution[:, None].float()
        if strategy == "TEP4":
            expected += 2
        else:
            expected += torch.arange(4).repeat_interleave(32)[None, :] / 8
        expected = (expected + shared).to(torch.bfloat16).view_as(actual)
        self.assertTrue(
            torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        )
        return ids

    def _prefill_wave_fixture(
        self, stack, strategy, wave, kernel, Native, PrefillWorkspace, gate
    ):
        stack.enter_context(patch.object(model_context, "world_size", 4))
        stack.enter_context(patch.object(model_context, "rank", 1))
        native = Native()
        for module in (
            "_projection_module",
            "_transform_module",
            "_activation_module",
            "_routing_module",
        ):
            stack.enter_context(
                patch.object(kernel_native, module, return_value=native)
            )
        stack.enter_context(
            patch.object(torch.distributed, "all_reduce", native.all_reduce)
        )
        stack.enter_context(
            patch.object(
                torch.distributed,
                "all_gather_into_tensor",
                native.all_gather_into_tensor,
            )
        )
        dim = 128 if strategy == "TP4" else 32
        moe = MoE(
            0,
            ModelArgs(
                dim=dim,
                moe_inter_dim=128,
                n_routed_experts=8,
                n_activated_experts=3,
                dtype="bf16",
                expert_dtype=None,
                parallel_strategy=strategy,
            ),
        )
        groups = moe.n_local_experts
        projection = tuple((torch.empty((32, dim // 2)), None) for _ in range(groups))
        down = tuple(
            (torch.empty((32, 64 if strategy == "TP4" else 16)), None)
            for _ in range(groups)
        )
        moe._routed_state.weights = SimpleNamespace(
            gate=projection, up=projection, down=down
        )
        moe._routed_state.pointers = tuple(
            (role, moe.experts_start_idx) for role in ("gate", "up", "down")
        )
        workspace = PrefillWorkspace(wave)
        workspace.reserve(moe, torch.device("cpu"), 4)
        addresses = {
            name: value.data_ptr() for name, value in workspace.buffers.items()
        }
        stack.enter_context(
            patch.object(moe, "_prepare_prefill_workspace", return_value=workspace)
        )
        gate_mock = stack.enter_context(
            patch.object(moe.gate, "forward", side_effect=gate)
        )
        return native, moe, dim, workspace, addresses, gate_mock

    def _prefill_native_type(self):
        class Native:
            def __init__(self):
                self.projections = []
                self.collectives = []

            def grouped_workspace_size(self, *_args):
                return 64

            def quantize(self, values, encoded, scales, *_args):
                encoded.copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))
                scales.fill_(127)

            def grouped_project(
                self,
                encoded,
                scales,
                role,
                first_expert,
                counts,
                output,
                scratch,
                max_rows,
            ):
                self.projections.append((role, encoded.size(0)))
                values = encoded.view(torch.float8_e4m3fn).float()
                output.copy_(values[:, :1].expand_as(output))
                if role == "down":
                    first = 0
                    for expert, count in enumerate(counts.tolist(), first_expert):
                        output[first : first + count].add_(expert / 8)
                        first += count

            def swiglu(self, gate, up, weights, output, *_args):
                output.copy_(gate.float() * weights[:, None])

            def expert_sum(self, values, route_rows, output):
                output.zero_()
                for row, routes in enumerate(route_rows.tolist()):
                    for route in sorted(route for route in routes if route >= 0):
                        output[row].add_(values[route].float())

            def all_reduce(self, values):
                self.collectives.append(("reduce", tuple(values.shape)))
                values.add_(2)

            def all_gather_into_tensor(self, output, values):
                self.collectives.append(
                    (
                        "product" if values.dtype == torch.bfloat16 else "sum",
                        tuple(values.shape),
                    )
                )
                for rank, shard in enumerate(output.view(4, *values.shape)):
                    shard.copy_(values)
                    if values.dtype == torch.float32:
                        shard.add_(rank / 8)

        return Native
