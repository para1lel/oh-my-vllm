import hashlib
import json
import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41.loader import RankLoader
from oh_my_vllm.models.deepseek_v41.model import (
    Gate,
    ModelArgs,
    MoE,
    RowParallelLinear,
)
from safetensors.torch import save_file

from tests.deepseek_model_fixtures import DeepSeekModelFixture


class DeepSeekModelRankWeights(DeepSeekModelFixture, unittest.TestCase):
    def test_rank_loader_keeps_full_down_k_and_matching_output_row_scales(self):
        config = json.loads(
            (Path(__file__).parent / "fixtures/deepseek_v41/config.json").read_text()
        )
        layout, source, expected = {}, {}, {}
        for prefix in ("layers.0", "mtp.2"):
            name = f"{prefix}.ffn.experts.0.w2"
            weight = (torch.arange(128)[None, :] + torch.arange(128)[:, None] * 3).to(
                torch.int8
            )
            scales = (torch.arange(128) % 15 + 120).to(torch.uint8)
            scales = scales[:, None].expand(128, 8).contiguous()
            for suffix, value, dtype in (
                ("weight", weight, "I8"),
                ("scale", scales.view(torch.float8_e8m0fnu), "F8_E8M0"),
            ):
                key = f"{name}.{suffix}"
                layout[key] = {"dtype": dtype, "shape": list(value.shape)}
                source[key] = value
                expected[key] = value.view(torch.uint8)
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_file(source, root / "weights.safetensors")
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(source, "weights.safetensors")})
            )
            for rank in range(4):
                with patch(
                    "oh_my_vllm.models.deepseek_v41.loader.tensor_layout",
                    return_value=layout,
                ):
                    loader = RankLoader(root, config, rank, strategy="TP4")
                try:
                    for name, value in expected.items():
                        for device in devices:
                            with self.subTest(rank=rank, name=name, device=device):
                                destination = torch.empty(
                                    (32, value.size(1)),
                                    dtype=(
                                        torch.float4_e2m1fn_x2
                                        if name.endswith("weight")
                                        else torch.float8_e8m0fnu
                                    ),
                                    device=device,
                                )
                                loader._copy_parameter(name, destination)
                                self.assertTrue(
                                    torch.equal(
                                        destination.cpu().view(torch.uint8),
                                        value[rank * 32 : (rank + 1) * 32],
                                    )
                                )
                finally:
                    loader.close()

    def test_shared_expert_forward_uses_full_weights_after_routed_exchange(self):
        args = ModelArgs(
            dim=4,
            moe_inter_dim=8,
            n_layers=1,
            n_routed_experts=4,
            n_activated_experts=1,
            dspark_n_routed_experts=8,
            expert_dtype=None,
        )
        values = torch.ones((1, 1, 4), dtype=torch.bfloat16)
        gate_weight = torch.tensor([[0.5, 0.5, 0, 0]] * 8, dtype=torch.bfloat16)
        up_weight = torch.tensor([[1.0, 0.0, 0, 0]] * 8, dtype=torch.bfloat16)
        down_weight = torch.tensor([[0.125] * 8] * 4, dtype=torch.bfloat16)
        product = torch.nn.functional.silu(torch.ones(1, 8)).to(torch.bfloat16)
        shared_output = torch.nn.functional.linear(product, down_weight)
        expected = (shared_output.float() + 3).to(torch.bfloat16).view_as(values)
        for strategy in ("TP4", "TEP4", "DEP4"):
            for rank in range(4):
                for layer in (0, 1):
                    with (
                        self.subTest(strategy=strategy, rank=rank, layer=layer),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model_context.world_size", 4
                        ),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model_context.rank", rank
                        ),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model_context.default_dtype",
                            torch.bfloat16,
                        ),
                    ):
                        moe = MoE(layer, replace(args, parallel_strategy=strategy))
                        if strategy == "DEP4":
                            moe.dep4_exchange = lambda _layer, rows, _routing: (
                                torch.full_like(rows, 3, dtype=torch.float32)
                            )
                        shared = moe.shared_experts
                        self.assertEqual(tuple(shared.w1.weight.shape), (8, 4))
                        self.assertEqual(tuple(shared.w2.weight.shape), (4, 8))
                        self.assertEqual(tuple(shared.w3.weight.shape), (8, 4))
                        with torch.no_grad():
                            for parameter in moe.parameters():
                                parameter.zero_()
                            shared.w1.weight.copy_(gate_weight)
                            shared.w2.weight.copy_(down_weight)
                            shared.w3.weight.copy_(up_weight)

                        def reduce_routed(value):
                            self.assertTrue(torch.equal(value, torch.zeros_like(value)))
                            value.fill_(3)

                        def gather_routed(output, value):
                            self.assertTrue(torch.equal(value, torch.zeros_like(value)))
                            output.fill_(3 if value.dtype == torch.float32 else 0)

                        with (
                            patch(
                                "torch.distributed.all_reduce",
                                side_effect=reduce_routed,
                            ),
                            patch(
                                "torch.distributed.all_gather_into_tensor",
                                side_effect=gather_routed,
                            ),
                        ):
                            actual = moe(values)
                        self.assertTrue(torch.equal(actual, expected))

    def test_rank_loader_preserves_full_shared_fp8_values_and_scales(self):
        config, layout, source, expected = self._shared_fp8_checkpoint()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = root / "model-00001-of-00048.safetensors"
            save_file(source, shard)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(source, shard.name)})
            )
            before = hashlib.sha256(shard.read_bytes()).digest()
            for strategy in ("TP4", "TEP4", "DEP4"):
                for rank in range(4):
                    with (
                        self.subTest(strategy=strategy, rank=rank),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.loader.tensor_layout",
                            return_value=layout,
                        ),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model_context.world_size", 4
                        ),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model_context.rank", rank
                        ),
                        patch(
                            "oh_my_vllm.models.deepseek_v41.model_context.default_dtype",
                            torch.float8_e4m3fn,
                        ),
                    ):
                        model = torch.nn.Module()
                        for namespace in ("layers", "mtp"):
                            block = torch.nn.Module()
                            block.ffn = torch.nn.Module()
                            block.ffn.shared_experts = MoE(
                                0,
                                ModelArgs(
                                    dim=128,
                                    moe_inter_dim=128,
                                    n_routed_experts=4,
                                    parallel_strategy=strategy,
                                ),
                            ).shared_experts
                            model.add_module(namespace, torch.nn.ModuleList([block]))
                        loader = RankLoader(
                            root, config, rank, strategy=strategy, tile_rows=17
                        )
                        try:
                            for name, value in expected.items():
                                self.assertEqual(
                                    loader.specs[name]["shape"], tuple(value.shape)
                                )
                            info = loader.load_into(model)
                            self.assertEqual(info["parameter_count"], 12)
                            for name, value in model.named_parameters():
                                self.assertTrue(
                                    torch.equal(
                                        value.view(torch.uint8), expected[name]
                                    ),
                                    name,
                                )
                            for blocks in (model.layers, model.mtp):
                                for projection in ("w1", "w2", "w3"):
                                    linear = getattr(
                                        blocks[0].ffn.shared_experts, projection
                                    )
                                    self.assertIs(linear.weight.scale, linear.scale)
                        finally:
                            loader.close()
            self.assertEqual(hashlib.sha256(shard.read_bytes()).digest(), before)

    def test_dense_owner_keeps_small_rank_term_before_bf16_rounding(self):
        from oh_my_vllm.models.deepseek_v41.local_tp import LocalRowProjection

        # Captured TEP4 values at layer 21, query 158, channel 348.
        partials = torch.tensor(
            [-0.04150390625, -2.2351741790771484e-08, -0.462890625, 0.11669921875],
            dtype=torch.bfloat16,
        )
        calls = []

        def project(values, weight):
            rank = int(weight[0, 0])
            calls.append((rank, tuple(values.shape), values.dtype))
            return partials[rank].expand(values.size(0), 1).clone()

        with torch.device("cpu"):
            inputs = torch.ones((2, 4), dtype=torch.bfloat16)
            weights = torch.arange(4, dtype=torch.bfloat16).reshape(1, 4)
            actual = LocalRowProjection(4)(inputs, weights, None, project)
        self.assertEqual(actual.dtype, torch.float32)
        self.assertTrue(
            torch.equal(
                actual, torch.full((2, 1), -0.3876953423023224, dtype=torch.float32)
            )
        )
        self.assertTrue(
            torch.equal(
                actual.to(torch.bfloat16),
                torch.full((2, 1), -0.388671875, dtype=torch.bfloat16),
            )
        )
        self.assertEqual(calls, [(rank, (2, 1), torch.bfloat16) for rank in range(4)])

    def test_attention_output_rounds_before_tensor_parallel_reduction(self):
        with patch("oh_my_vllm.models.deepseek_v41.model_context.world_size", 4):
            attention = RowParallelLinear(8, 2, dtype=torch.bfloat16)
            with torch.no_grad():
                attention.weight.copy_(torch.tensor([[1, 1 / 256]] * 2))

            def reduce(value):
                self.assertEqual(value.dtype, torch.float32)
                torch.testing.assert_close(value, torch.ones((1, 2)), rtol=0, atol=0)
                value.sub_(1)

            with patch(
                "torch.distributed.all_reduce",
                side_effect=reduce,
            ):
                actual = attention(torch.ones((1, 2), dtype=torch.bfloat16))
            torch.testing.assert_close(
                actual, torch.zeros((1, 2), dtype=torch.bfloat16), rtol=0, atol=0
            )

    def test_router_keeps_float32_projection_before_expert_selection(self):
        args = ModelArgs(
            dim=2,
            n_routed_experts=6,
            n_activated_experts=3,
            route_scale=1.5,
            score_func="sqrtsoftplus",
            vision_n_layers=0,
        )
        gate = Gate(0, args)
        gate.weight = torch.nn.Parameter(
            torch.tensor(
                [[1, 1 / 256], [1, 0], [2, 0], [3, 0], [-1, 0], [-2, 0]],
                dtype=torch.bfloat16,
            )
        )
        with torch.no_grad():
            gate.bias.zero_()
        weights, indices = gate(torch.ones((1, 2), dtype=torch.bfloat16))
        scores = [math.sqrt(math.log1p(math.exp(value))) for value in (3, 2, 257 / 256)]
        expected = torch.tensor(
            [[value / sum(scores) * 1.5 for value in scores]], dtype=torch.float32
        )
        self.assertTrue(torch.equal(indices, torch.tensor([[3, 2, 0]])))
        torch.testing.assert_close(weights, expected, rtol=2e-7, atol=2e-7)

    def _shared_fp8_checkpoint(self):
        config = json.loads(
            (Path(__file__).parent / "fixtures/deepseek_v41/config.json").read_text()
        )
        shapes = {
            "w1.weight": ((128, 128), "F8_E4M3", torch.float8_e4m3fn),
            "w1.scale": ((4, 4), "F8_E8M0", torch.float8_e8m0fnu),
            "w2.weight": ((128, 128), "F8_E4M3", torch.float8_e4m3fn),
            "w2.scale": ((4, 4), "F8_E8M0", torch.float8_e8m0fnu),
            "w3.weight": ((128, 128), "F8_E4M3", torch.float8_e4m3fn),
            "w3.scale": ((4, 4), "F8_E8M0", torch.float8_e8m0fnu),
        }
        layout, source, expected = {}, {}, {}
        for namespace in ("layers", "mtp"):
            for index, (suffix, (shape, dtype, torch_dtype)) in enumerate(
                shapes.items()
            ):
                name = f"{namespace}.0.ffn.shared_experts.{suffix}"
                layout[name] = {"shape": list(shape), "dtype": dtype}
                raw = (torch.arange(math.prod(shape)) % 16 + 112 + index).to(
                    torch.uint8
                )
                expected[name] = raw.reshape(shape)
                source_name = name.replace("ffn", "mlp").replace(
                    ".scale", ".weight_scale_inv"
                )
                source[f"model.{source_name}"] = raw.view(torch_dtype).reshape(shape)
        return config, layout, source, expected
