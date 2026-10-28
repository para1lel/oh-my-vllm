import hashlib
import json
import struct
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F
from oh_my_vllm.models.deepseek_v41 import kernel
from oh_my_vllm.models.deepseek_v41.loader import RankLoader, source_index
from oh_my_vllm.operators.activation.plan import make_plan as make_activation_plan
from oh_my_vllm.operators.hyperconnection.plan import (
    make_plan as make_hyperconnection_plan,
)
from oh_my_vllm.operators.projection.plan import make_plan as make_projection_plan
from oh_my_vllm.operators.routing.plan import make_plan as make_routing_plan
from oh_my_vllm.operators.sparse_attention.plan import (
    make_plan as make_sparse_attention_plan,
)
from oh_my_vllm.operators.transform.plan import make_plan as make_transform_plan


class DeepSeekOperatorContracts(unittest.TestCase):
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
        scores = torch.cat([scores, sink.view(1, 1, -1, 1).expand(1, 2, -1, 1)], dim=-1)
        weights = torch.softmax(scores, dim=-1)[..., :-1]
        expected = torch.einsum("bshk,bskd->bshd", weights, gathered.float()).to(
            torch.bfloat16
        )
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
    def test_cuda_sparse_attention_native_path_matches_quantized_reference(self):
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
        window_codes, window_scales = kernel._pack_fp8_kv(window)
        window_values = window_codes.view(torch.float8_e4m3fn).float()
        window_scale_values = window_scales.view(torch.float8_e8m0fnu).float()
        window_scale_values = window_scale_values.unsqueeze(-1).expand(
            *window_scale_values.shape, 32
        )
        window_values = window_values * window_scale_values.reshape_as(window_values)
        compressed_codes, compressed_scales = kernel._pack_fp4_kv(compressed)
        low = (compressed_codes & 15).long()
        high = (compressed_codes >> 4).long()
        levels = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
            device=device,
        )
        compressed_values = torch.stack(
            (
                levels[low & 7] * torch.where((low & 8).bool(), -1.0, 1.0),
                levels[high & 7] * torch.where((high & 8).bool(), -1.0, 1.0),
            ),
            dim=-1,
        ).flatten(-2, -1)
        compressed_scale_values = compressed_scales.view(torch.float8_e4m3fn).float()
        compressed_scale_values = compressed_scale_values.unsqueeze(-1).expand(
            *compressed_scale_values.shape, 16
        )
        compressed_values = compressed_values * compressed_scale_values.reshape_as(
            compressed_values
        )
        reference = kernel.sparse_attn(
            query,
            torch.cat((window_values, compressed_values), dim=1),
            sink,
            indices,
            512**-0.5,
        )

        torch.testing.assert_close(native, reference, atol=0.06, rtol=0.06)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_sparse_attention_status_failure_falls_back(self):
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

        with patch.object(
            kernel, "_sparse_attention_module", return_value=FailedAttention()
        ):
            self.assertIsNone(
                kernel._sparse_attn_native(query, sink, indices, window, compressed)
            )

    def test_rank_loader_maps_a_host_engram_slice_from_safetensors(self):
        name = "layers.1.engram.embed.weight"
        values = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32)
        raw = values.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
        header = json.dumps(
            {
                name: {
                    "dtype": "F8_E4M3",
                    "shape": [8, 32],
                    "data_offsets": [0, len(raw)],
                }
            },
            separators=(",", ":"),
        ).encode()
        header += b" " * (-len(header) % 8)

        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + raw)
            loader = RankLoader.__new__(RankLoader)
            loader.root = Path(directory)
            loader.index = {name: shard.name}
            loader.sources = {name: name}
            loader.layout = {name: {"shape": [8, 32]}}
            loader.specs = {
                name: {
                    "shape": (4, 32),
                    "dtype": "F8_E4M3",
                    "stored_dtype": "F8_E4M3",
                    "axis": 0,
                    "start": 4,
                }
            }
            loader.resources = ExitStack()
            loader.headers = {}
            loader.mappings = {}
            destination = torch.empty(4, 32, dtype=torch.float8_e4m3fn)

            loader._copy_parameter(name, destination)
            expected = destination.float().clone()
            del destination
            loader.close()

        self.assertTrue(
            torch.equal(expected, values.to(torch.float8_e4m3fn)[4:].float())
        )

    def test_rank_loader_reads_sharded_one_dimensional_parameters_from_the_rank_slice(
        self,
    ):
        loader = RankLoader.__new__(RankLoader)
        loader.specs = {
            "layers.0.attn.attn_sink": {
                "shape": (16,),
                "dtype": "F32",
                "stored_dtype": "F32",
                "axis": 0,
                "start": 16,
            }
        }
        loader._read = Mock(return_value=torch.arange(16, dtype=torch.float32))
        destination = torch.empty(16)

        loader._copy_parameter("layers.0.attn.attn_sink", destination)

        self.assertEqual(
            loader._read.call_args.args[1],
            (slice(16, 32),),
        )
        self.assertTrue(torch.equal(destination, torch.arange(16, dtype=torch.float32)))

    def test_rank_loader_advances_rows_for_axis_one_tiles(self):
        name = "layers.0.attn.wo_b.weight"
        loader = RankLoader.__new__(RankLoader)
        loader.layout = {name: {"shape": [4096, 64]}}
        loader.specs = {
            name: {
                "shape": (4096, 16),
                "dtype": "F32",
                "stored_dtype": "F32",
                "axis": 1,
                "start": 16,
            }
        }
        loader.tile_rows = 2048

        def read(_, selection):
            first, stop = selection[0].start, selection[0].stop
            rows = torch.arange(first, stop, dtype=torch.float32).unsqueeze(1)
            return rows.expand(-1, 16)

        loader._read = Mock(side_effect=read)
        destination = torch.empty(4096, 16)

        loader._copy_parameter(name, destination)

        self.assertEqual(loader._read.call_count, 2)
        self.assertEqual(loader._read.call_args_list[1].args[1][0], slice(2048, 4096))
        self.assertEqual(destination[0, 0], 0)
        self.assertEqual(destination[2048, 0], 2048)

    def test_native_operator_plans_cover_deepseek_shapes(self):
        self.assertEqual(
            make_activation_plan(rows=32, width=2304, limit=10, routed=True).width,
            2304,
        )
        self.assertEqual(make_routing_plan(rows=64, experts=384).topk, 6)
        self.assertEqual(
            make_sparse_attention_plan(rows=64, heads=64, slots=640).slots,
            640,
        )
        self.assertEqual(
            make_projection_plan(rows=64, outputs=5120, width=5120, bits=8).bits,
            8,
        )
        self.assertEqual(make_hyperconnection_plan(rows=64).rows, 64)
        self.assertEqual(
            make_transform_plan(kind="rms", rows=64, width=5120, epsilon=1e-20).kind,
            "rms",
        )
        self.assertEqual(
            make_transform_plan(kind="rope", rows=64, width=512, heads=64).heads,
            64,
        )

    def test_native_operator_plans_reject_unrelated_shapes(self):
        with self.assertRaises(ValueError):
            make_activation_plan(rows=32, width=5120, limit=10, routed=True)
        with self.assertRaises(ValueError):
            make_routing_plan(rows=64, experts=256)
        with self.assertRaises(ValueError):
            make_sparse_attention_plan(rows=64, heads=32, slots=641)
        with self.assertRaises(ValueError):
            make_projection_plan(rows=64, outputs=5120, width=5120, bits=3)


if __name__ == "__main__":
    unittest.main()
