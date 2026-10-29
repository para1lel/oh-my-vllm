import os
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F
from oh_my_vllm.models.deepseek_v41 import kernel, model


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
        values = torch.arange(12, dtype=torch.float32).view(6, 2)
        weight = torch.tensor(((1.0, 2.0), (-1.0, 1.0)))
        expected = torch.tensor(
            ((2, 1), (8, 1), (14, 1), (20, 1), (26, 1), (32, 1))
        ).float()
        for logical_rows in (0, 1, 2, 3, 6):
            actual = kernel.fp32_project(values, weight, logical_rows=logical_rows)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for invalid in (-1, 4, 7, True, 1.0):
            with self.assertRaisesRegex(ValueError, "logical row"):
                kernel.fp32_project(values, weight, logical_rows=invalid)

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


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LogicalRowsCuda(unittest.TestCase):
    def test_index_rms_uses_native_logical_reduction(self):
        values = (
            torch.linspace(-1, 1, 3 * 6 * 128, device="cuda").bfloat16().view(3, 6, 128)
        )
        weight = torch.ones(128, dtype=torch.bfloat16, device="cuda")
        reference = values.float()
        expected = (
            reference * torch.rsqrt(reference.square().mean(-1, keepdim=True) + 1e-20)
        ).bfloat16()
        with patch.object(
            kernel, "_source_steps", side_effect=AssertionError("native RMS expected")
        ):
            actual = kernel.rms(values, weight, 1e-20, logical_rows=3)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hc_grouped_projection_keeps_source_step_rows(self):
        generator = torch.Generator(device="cuda").manual_seed(73)
        module = kernel._hyperconnection_module()
        for batch, steps in ((1, 6), (2, 3), (3, 8), (4, 2)):
            with self.subTest(batch=batch, steps=steps):
                rows = batch * steps
                residual = torch.randn(
                    (batch, steps, 4, 5120),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                incoming = torch.rand((rows, 4), generator=generator, device="cuda")
                weight = torch.randn((24, 20480), generator=generator, device="cuda")
                scale = torch.ones(3, device="cuda")
                base = torch.zeros(24, device="cuda")
                storage = kernel.make_hc_workspace(residual.device, max_rows=rows)
                arguments = (
                    residual.view(rows, 4, 5120),
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
                expected = []
                for step in range(steps):
                    values = residual[:, step].flatten(1).float().contiguous()
                    inverse = torch.rsqrt(
                        values.square().mean(-1, keepdim=True) + 1e-20
                    )
                    expected.append(F.linear(values, weight) * inverse)
                expected = torch.stack(expected, dim=1).view(rows, 24)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    module.hc_prepare_grouped(*arguments)
                    eager = storage["mixed"].clone()
                stream.synchronize()
                torch.testing.assert_close(eager, expected, rtol=0, atol=0)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        module.hc_prepare_grouped(*arguments)
                    with torch.cuda.stream(stream):
                        storage["mixed"].fill_(123)
                        graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(
                        storage["mixed"], expected, rtol=0, atol=0
                    )
                    before = storage["mixed"].clone()
                    for invalid in (-1, 0, rows + 1, rows - 1):
                        with self.assertRaisesRegex(Exception, "logical rows"):
                            module.hc_prepare_grouped(*arguments[:-1], invalid)
                        torch.testing.assert_close(
                            storage["mixed"], before, rtol=0, atol=0
                        )
                    alias_arguments = list(arguments)
                    alias_arguments[5] = (
                        storage["flat"].flatten()[: rows * 24].view(rows, 24)
                    )
                    with self.assertRaisesRegex(Exception, "operator tensors overlap"):
                        module.hc_prepare_grouped(*alias_arguments)
                finally:
                    stream.synchronize()
                    graph.reset()
        module.hc_prepare_grouped(
            residual.view(rows, 4, 5120)[:0],
            incoming[:0],
            weight,
            scale,
            base,
            storage["mixed"][:0],
            storage["post"][:0],
            storage["comb"][:0],
            storage["collapsed"][:0],
            storage["next_pre_a"][:0],
            storage["flat"][:0],
            storage["statistics"][:0],
            storage["blas"][:0],
            0,
        )

    def test_fp32_projection_keeps_batch_major_positions(self):
        generator = torch.Generator(device="cuda").manual_seed(42)
        module = kernel._hyperconnection_module()
        for batch, steps, outputs in (
            (1, 6, 384),
            (2, 3, 128),
            (3, 8, 384),
            (4, 2, 128),
        ):
            with self.subTest(batch=batch, steps=steps, outputs=outputs):
                values = torch.randn(
                    (batch, steps, 5120),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                ).float()
                weight = torch.randn(
                    (outputs, 5120),
                    generator=generator,
                    device="cuda",
                    dtype=torch.bfloat16,
                ).float()
                expected = torch.stack(
                    [
                        F.linear(values[:, step].contiguous(), weight)
                        for step in range(steps)
                    ],
                    dim=1,
                ).view(-1, outputs)
                flat = values.view(-1, 5120)
                output = torch.empty_like(expected)
                workspace = torch.empty(
                    module.workspace_size(flat.size(0)),
                    dtype=torch.uint8,
                    device="cuda",
                )
                module.fp32_project(flat, weight, output, workspace, batch)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        module.fp32_project(flat, weight, output, workspace, batch)
                    with torch.cuda.stream(stream):
                        flat.mul_(2)
                        output.fill_(123)
                        graph.replay()
                    stream.synchronize()
                    torch.testing.assert_close(output, expected * 2, rtol=0, atol=0)
                finally:
                    stream.synchronize()
                    graph.reset()
                before = output.clone()
                for invalid in (-1, flat.size(0) + 1, flat.size(0) - 1):
                    with self.assertRaisesRegex(Exception, "logical rows"):
                        module.fp32_project(flat, weight, output, workspace, invalid)
                    torch.testing.assert_close(output, before, rtol=0, atol=0)
                alias = flat.flatten()[: output.numel()].view_as(output)
                with self.assertRaisesRegex(Exception, "operator tensors overlap"):
                    module.fp32_project(flat, weight, alias, workspace, batch)
                invalid_output = torch.full_like(output, 123, dtype=torch.float16)
                with self.assertRaisesRegex(Exception, "Check failed"):
                    module.fp32_project(flat, weight, invalid_output, workspace, batch)
                torch.testing.assert_close(
                    invalid_output, torch.full_like(invalid_output, 123), rtol=0, atol=0
                )
        module.fp32_project(flat[:0], weight, output[:0], workspace[:0], 0)


if __name__ == "__main__":
    unittest.main()
