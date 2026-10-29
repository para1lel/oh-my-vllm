import gc
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


def projection_weight(bits, outputs, width, *, offset=0, seed=43):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    columns = width * bits // 8
    storage = torch.empty(outputs * columns + offset, dtype=torch.uint8, device="cuda")
    raw = storage[offset:].view(outputs, columns)
    if bits == 4:
        raw.copy_(
            torch.randint(
                0, 256, raw.shape, generator=generator, device="cuda", dtype=torch.uint8
            )
        )
        weight = raw.view(torch.float4_e2m1fn_x2)
    else:
        values = torch.randn(raw.shape, generator=generator, device="cuda")
        raw.copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))
        weight = raw.view(torch.float8_e4m3fn)
    shape = (outputs if bits == 4 else outputs // 32, width // 32)
    scales = torch.randint(
        125, 129, shape, generator=generator, device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    return weight, scales


def projection_reference(encoded, input_scales, weight, scales, bits):
    if bits == 4:
        raw = weight.view(torch.uint8)
        levels = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=weight.device
        )
        low, high = raw & 15, raw >> 4
        values = torch.stack(
            (
                levels[(low & 7).long()] * torch.where((low & 8).bool(), -1.0, 1.0),
                levels[(high & 7).long()] * torch.where((high & 8).bool(), -1.0, 1.0),
            ),
            dim=-1,
        ).flatten(-2)
        weight_scales = scales.float()
    else:
        values = weight.float()
        weight_scales = scales.float().repeat_interleave(32, dim=0)
    values = values * weight_scales.repeat_interleave(32, dim=-1)
    inputs = encoded.float() * input_scales.float().repeat_interleave(32, dim=-1)
    return F.linear(inputs, values).to(torch.bfloat16)


class DeepSeekOperatorContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fp8_projection_scales_each_k32_dot_before_accumulation(self):
        # Raw K32 dots are -1 and 323. Their scaled FP32 sum rounds to 646,
        # then the BF16 midpoint rounds to 648, not 644.
        for rows in (1, 33, 125, 126):
            inputs = torch.zeros((rows, 64), device="cuda")
            weight = torch.zeros((160, 64), device="cuda")
            for column, left, right in (
                (8, -1, 1),
                (32, 16, 16),
                (48, 8, 8),
                (63, 3, 1),
            ):
                inputs[:, column] = left
                weight[:, column] = right
            inputs = inputs.to(torch.float8_e4m3fn)
            weight = weight.to(torch.float8_e4m3fn)
            row_shifts = torch.arange(rows, device="cuda") % 3 - 1
            group_shifts = torch.arange(5, device="cuda") - 2
            input_scales = (
                (torch.tensor([113, 127], device="cuda") + row_shifts[:, None])
                .to(torch.uint8)
                .view(torch.float8_e8m0fnu)
            )
            weight_scales = (
                (torch.tensor([126, 128], device="cuda") + group_shifts[:, None])
                .to(torch.uint8)
                .view(torch.float8_e8m0fnu)
            )
            factors = torch.exp2(
                row_shifts[:, None].float()
                + group_shifts.repeat_interleave(32)[None, :].float()
            )
            for dtype, value in ((torch.float32, 646), (torch.bfloat16, 648)):
                with self.subTest(rows=rows, dtype=dtype):
                    actual = kernel.fp8_gemm(
                        inputs,
                        input_scales,
                        weight,
                        weight_scales,
                        torch.float8_e8m0fnu,
                        block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        actual, (factors * value).to(dtype), rtol=0, atol=0
                    )

    def test_sparse_reference_rounds_unnormalized_probabilities_before_values(self):
        query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16)
        query[..., 0] = 32
        kv = torch.zeros((1, 2, 512), dtype=torch.bfloat16)
        kv[0, :, 0] = torch.tensor([0.0, -0.5])
        kv[0, :, 1] = torch.tensor([1.0, -1.0])
        sink = torch.full((16,), -2.0)
        indices = torch.tensor([[[0, 1]]], dtype=torch.int32)
        actual = kernel.sparse_attn(query, kv, sink, indices, 512**-0.5)
        torch.testing.assert_close(
            actual[..., 1],
            torch.full((1, 1, 16), 0.3125, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_sparse_sink_does_not_change_probability_rounding(self):
        query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16, device="cuda")
        window = torch.ones((1, 1, 512), dtype=torch.bfloat16, device="cuda")
        sink = torch.ones(16, dtype=torch.float32, device="cuda")
        indices = torch.zeros((1, 1, 1), dtype=torch.int32, device="cuda")
        actual = kernel._sparse_attn_native(query, sink, indices, window, window[:, :0])
        self.assertIsNotNone(actual)
        torch.testing.assert_close(
            actual, torch.full_like(query, 0.26953125), rtol=0, atol=0
        )

    def test_sparse_attention_preserves_ordered_blocks_and_masked_slots(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        # Released reference: one block and a later maximum have different rounding.
        cases = (
            ([1, 0], -2.0, 0.3125),
            ([1] + [-1] * 63 + [0], -2.0, 0.310546875),
            ([0] * 65, -2.0, 0.99609375),
            ([-1] * 65, -2.0, 0.0),
            ([-1] * 65, -1e31, 0.0),
            ([], -1e31, 0.0),
        )
        for device in devices:
            query = torch.zeros((1, 1, 16, 512), dtype=torch.bfloat16, device=device)
            query[..., 0] = 32
            kv = torch.zeros((1, 2, 512), dtype=torch.bfloat16, device=device)
            kv[0, 1, 0] = -0.5
            kv[0, 0, 1], kv[0, 1, 1] = 1, -1
            for selected, sink_value, expected in cases:
                with self.subTest(device=device, indices=selected, sink=sink_value):
                    sink = torch.full((16,), sink_value, device=device)
                    indices = torch.tensor(
                        [[selected]], dtype=torch.int32, device=device
                    )
                    actual = kernel.sparse_attn(query, kv, sink, indices, 512**-0.5)
                    torch.testing.assert_close(
                        actual[..., 1],
                        torch.full_like(actual[..., 1], expected),
                        atol=0,
                        rtol=0,
                    )
                    if expected == 0:
                        torch.testing.assert_close(actual, torch.zeros_like(actual))
                    if device == "cuda":
                        native = kernel._sparse_attn_native(
                            query, sink, indices, kv, kv[:, :0]
                        )
                        self.assertIsNotNone(native)
                        torch.testing.assert_close(native, actual, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_fp32_projection_replays_on_caller_stream(self):
        for width, outputs in ((128, 32), (1024, 512)):
            encoded = torch.ones((2, width), device="cuda").to(torch.float8_e4m3fn)
            scales = torch.ones((2, width // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            weight = torch.ones((outputs, width), device="cuda").to(torch.float8_e4m3fn)
            weight_scales = torch.ones((outputs // 32, width // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            module = kernel._projection_module()
            proxy = Mock(wraps=module)
            with patch.object(kernel, "_projection_module", return_value=proxy):
                output = kernel.fp8_gemm(
                    encoded,
                    scales,
                    weight,
                    weight_scales,
                    torch.float8_e8m0fnu,
                    block_size=32,
                    output_dtype=torch.float32,
                )
            arguments = proxy.project.call_args.args
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph):
                    module.project(*arguments)
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    encoded.view(torch.uint8).fill_(64)
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    output, torch.full_like(output, 2 * width), rtol=0, atol=0
                )
            finally:
                torch.cuda.synchronize()
                graph.reset()

    def test_hyperconnection_sinkhorn_preserves_zero_probability_support(self):
        mixes = torch.zeros((1, 1, 24), dtype=torch.float32)
        scale = torch.ones(3, dtype=torch.float32)
        base = torch.tensor(
            [0.0] * 8 + [0.0, -1000.0, -1000.0, -1000.0] * 4,
            dtype=torch.float32,
        )
        expected = torch.full((1, 1, 4, 4), 0.24999975, dtype=torch.float32)
        actual = kernel.hc_split_sinkhorn(mixes, scale, base)[2]
        with self.subTest(backend="reference"):
            torch.testing.assert_close(actual, expected, rtol=0, atol=2e-7)
        if torch.cuda.is_available():
            actual = kernel.hc_prepare(
                torch.ones((1, 1, 4, 5120), dtype=torch.bfloat16, device="cuda"),
                torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32, device="cuda"
                ),
                torch.zeros((24, 20480), dtype=torch.float32, device="cuda"),
                scale.cuda(),
                base.cuda(),
            )[3]
            with self.subTest(backend="cuda"):
                torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=2e-7)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_retains_fp32_partials_before_reduction(self):
        for rows, width in ((1, 32), (33, 128), (33, 160)):
            encoded = torch.zeros((rows, width), dtype=torch.float32, device="cuda")
            encoded[:, :2] = torch.tensor([1, 1 / 256], device="cuda")
            encoded = encoded.to(torch.float8_e4m3fn)
            input_scales = torch.ones((rows, width // 32), device="cuda").to(
                torch.float8_e8m0fnu
            )
            for bits in (4, 8):
                with self.subTest(bits=bits, rows=rows, width=width):
                    if bits == 4:
                        weight = torch.full(
                            (32, width // 2), 0x22, dtype=torch.uint8, device="cuda"
                        ).view(torch.float4_e2m1fn_x2)
                        scales = torch.ones((32, width // 32), device="cuda").to(
                            torch.float8_e8m0fnu
                        )
                        gemm = kernel.fp4_gemm
                        kwargs = {"act_block_size": 32}
                    else:
                        weight = torch.ones((32, width), device="cuda").to(
                            torch.float8_e4m3fn
                        )
                        scales = torch.ones((1, width // 32), device="cuda").to(
                            torch.float8_e8m0fnu
                        )
                        gemm = kernel.fp8_gemm
                        kwargs = {"block_size": 32}
                    inputs = (
                        encoded,
                        input_scales,
                        weight,
                        scales,
                        torch.float8_e8m0fnu,
                    )
                    actual = gemm(*inputs, output_dtype=torch.float32, **kwargs)
                    torch.testing.assert_close(
                        actual,
                        torch.full(
                            (rows, 32), 257 / 256, dtype=torch.float32, device="cuda"
                        ),
                        rtol=0,
                        atol=0,
                    )
                    self.assertEqual(actual.dtype, torch.float32)
                    rounded = gemm(*inputs, **kwargs)
                    torch.testing.assert_close(
                        rounded,
                        torch.ones_like(actual, dtype=torch.bfloat16),
                        rtol=0,
                        atol=0,
                    )
                    empty = gemm(
                        encoded[:0],
                        input_scales[:0],
                        *inputs[2:],
                        output_dtype=torch.float32,
                        **kwargs,
                    )
                    self.assertEqual(empty.shape, (0, 32))
                    self.assertEqual(empty.dtype, torch.float32)
                    with self.assertRaisesRegex(ValueError, "output dtype"):
                        gemm(*inputs, output_dtype=torch.float16, **kwargs)

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_packing_reuses_compatible_encoded_storage(self):
        for bits, outputs, width, offset in (
            (4, 32, 128, 64),
            (4, 576, 128, 64),
            (8, 64, 128, 16),
            (8, 64, 576, 16),
        ):
            with self.subTest(bits=bits, outputs=outputs, width=width):
                weight, scales = projection_weight(bits, outputs, width, offset=offset)
                raw_before = weight.view(torch.uint8).clone()
                scales_before = scales.view(torch.uint8).clone()
                with patch.dict(kernel._PACKED_WEIGHTS, clear=True):
                    packed = kernel._packed_weight(weight, scales, bits)
                    self.assertIs(kernel._packed_weight(weight, scales, bits), packed)
                    self.assertEqual(packed[0].data_ptr(), weight.data_ptr())
                    self.assertEqual(
                        packed[0].untyped_storage().data_ptr(),
                        weight.untyped_storage().data_ptr(),
                    )
                    self.assertNotEqual(packed[1].data_ptr(), scales.data_ptr())
                    torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                    torch.testing.assert_close(scales.view(torch.uint8), scales_before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_alias_retains_source_storage(self):
        for bits in (4, 8):
            with (
                self.subTest(bits=bits),
                patch.dict(kernel._PACKED_WEIGHTS, clear=True),
            ):
                weight, scales = projection_weight(bits, 32, 128, offset=64)
                pointer = weight.data_ptr()
                storage_pointer = weight.untyped_storage().data_ptr()
                expected = weight.view(torch.uint8).clone()
                key = (pointer, scales.data_ptr(), bits, weight.device.index)
                kernel._packed_weight(weight, scales, bits)
                del weight, scales
                gc.collect()
                retained = kernel._PACKED_WEIGHTS[key][0]
                self.assertEqual(retained.data_ptr(), pointer)
                self.assertEqual(retained.untyped_storage().data_ptr(), storage_pointer)
                torch.testing.assert_close(retained, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_packing_keeps_incompatible_values_separate(self):
        for width, offset, overlap in (
            (576, 0, False),
            (128, 16, False),
            (128, 0, True),
        ):
            with self.subTest(width=width, offset=offset, overlap=overlap):
                weight, scales = projection_weight(4, 32, width, offset=offset)
                if overlap:
                    scales = (
                        weight.view(torch.uint8)
                        .flatten()[: 32 * (width // 32)]
                        .view(32, width // 32)
                        .view(torch.float8_e8m0fnu)
                    )
                raw_before = weight.view(torch.uint8).clone()
                scales_before = scales.view(torch.uint8).clone()
                with patch.dict(kernel._PACKED_WEIGHTS, clear=True):
                    packed = kernel._packed_weight(weight, scales, 4)
                    self.assertNotEqual(packed[0].data_ptr(), weight.data_ptr())
                    padded_width = (width + 127) // 128 * 128
                    self.assertEqual(tuple(packed[0].shape), (32, padded_width // 2))
                    torch.testing.assert_close(packed[0][:, : width // 2], raw_before)
                    torch.testing.assert_close(
                        packed[0][:, width // 2 :],
                        torch.zeros_like(packed[0][:, width // 2 :]),
                    )
                    torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                    torch.testing.assert_close(scales.view(torch.uint8), scales_before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_packing_rejects_invalid_source_views(self):
        weight, scales = projection_weight(4, 32, 128)
        unaligned, _ = projection_weight(4, 32, 128, offset=1)
        for invalid in (unaligned, weight.transpose(0, 1)):
            with (
                self.subTest(stride=invalid.stride()),
                patch.dict(kernel._PACKED_WEIGHTS, clear=True),
                self.assertRaises(RuntimeError),
            ):
                kernel._packed_weight(invalid, scales, 4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_projection_alias_matches_copy_and_quantized_reference(self):
        for bits, width in ((4, 128), (4, 576), (8, 128), (8, 576)):
            with self.subTest(bits=bits, width=width):
                weight, scales = projection_weight(
                    bits, 64, width, offset=64 if bits == 4 else 16
                )
                values = torch.linspace(-2, 2, 3 * width, device="cuda").view(3, width)
                encoded, input_scales = kernel.act_quant(
                    values, 32, "ue8m0", torch.float8_e8m0fnu
                )
                raw_before = weight.view(torch.uint8).clone()
                scales_before = scales.view(torch.uint8).clone()
                with patch.dict(kernel._PACKED_WEIGHTS, clear=True):
                    actual = kernel._projection_gemm(
                        encoded, input_scales, weight, scales, bits
                    )
                    packed = kernel._packed_weight(weight, scales, bits)
                    copy = torch.empty_like(packed[0])
                    copied_scales = torch.empty_like(packed[1])
                    kernel._projection_module().pack_weights(
                        weight.view(torch.uint8),
                        scales.view(torch.uint8),
                        copy,
                        copied_scales,
                        bits,
                    )
                    with patch.object(
                        kernel,
                        "_packed_weight",
                        return_value=(copy, copied_scales, width),
                    ):
                        copied_output = kernel._projection_gemm(
                            encoded, input_scales, weight, scales, bits
                        )
                    torch.testing.assert_close(actual, copied_output, atol=0, rtol=0)
                    expected = projection_reference(
                        encoded, input_scales, weight, scales, bits
                    )
                    torch.testing.assert_close(actual, expected, atol=0.1, rtol=0.02)
                    torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                    torch.testing.assert_close(scales.view(torch.uint8), scales_before)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_reuses_values_and_matches_quantized_reference(self):
        width = 128
        weights = [projection_weight(4, 64, width, seed=seed) for seed in (7, 11, 13)]
        counts = torch.tensor((2, 0, 3), device="cuda", dtype=torch.int32)
        values = torch.linspace(-2, 2, 5 * width, device="cuda").view(5, width)
        encoded, input_scales = kernel.act_quant(
            values, 32, "ue8m0", torch.float8_e8m0fnu
        )
        snapshots = [
            (w.view(torch.uint8).clone(), s.view(torch.uint8).clone())
            for w, s in weights
        ]
        with (
            patch.dict(kernel._PACKED_WEIGHTS, clear=True),
            patch.dict(kernel._GROUPED_POINTERS, clear=True),
        ):
            actual = kernel.grouped_fp4_gemm(values, counts, weights)
            expected = []
            first = 0
            for count, (weight, scale), (raw_before, scale_before) in zip(
                (2, 0, 3), weights, snapshots, strict=True
            ):
                if count:
                    packed = kernel._packed_weight(weight, scale, 4)
                    self.assertEqual(packed[0].data_ptr(), weight.data_ptr())
                    expected.append(
                        projection_reference(
                            encoded[first : first + count],
                            input_scales[first : first + count],
                            weight,
                            scale,
                            4,
                        )
                    )
                torch.testing.assert_close(weight.view(torch.uint8), raw_before)
                torch.testing.assert_close(scale.view(torch.uint8), scale_before)
                first += count
            torch.testing.assert_close(actual, torch.cat(expected), atol=0.1, rtol=0.02)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_grouped_fp4_matches_individual_projections(self):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(41)
        values = torch.randn(
            (5, 32), generator=generator, device=device, dtype=torch.bfloat16
        )
        weights = []
        for seed in (7, 11, 13):
            weight_generator = torch.Generator(device=device).manual_seed(seed)
            raw = torch.randint(
                0,
                256,
                (32, 16),
                generator=weight_generator,
                device=device,
                dtype=torch.uint8,
            )
            weights.append(
                (
                    raw.view(torch.float4_e2m1fn_x2),
                    torch.ones((32, 1), device=device, dtype=torch.float8_e8m0fnu),
                )
            )
        counts = torch.tensor((2, 0, 3), device=device, dtype=torch.int32)

        actual = kernel.grouped_fp4_gemm(values, counts, weights)

        expected_parts = []
        first = 0
        for count, (weight, scale) in zip(counts.tolist(), weights, strict=True):
            if count:
                encoded, input_scale = kernel.act_quant(
                    values[first : first + count],
                    32,
                    "ue8m0",
                    torch.float8_e8m0fnu,
                )
                expected_parts.append(
                    kernel.fp4_gemm(
                        encoded,
                        input_scale,
                        weight,
                        scale,
                        torch.float8_e8m0fnu,
                        32,
                    )
                )
            first += count
        expected = torch.cat(expected_parts)
        torch.testing.assert_close(actual, expected, atol=0.08, rtol=0.08)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_swiglu_accepts_an_unaligned_route_slice(self):
        gate = torch.zeros((8, 576), device="cuda", dtype=torch.bfloat16)
        up = torch.zeros_like(gate)
        route = torch.ones((9,), device="cuda", dtype=torch.float32)[1:]

        actual = kernel.swiglu(gate, up, route, 10)

        torch.testing.assert_close(actual, torch.zeros_like(gate))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_swiglu_accepts_six_route_rows_per_prefill_token(self):
        rows = 8192 * 6
        gate = torch.zeros((rows, 576), device="cuda", dtype=torch.bfloat16)
        up = torch.zeros_like(gate)
        route = torch.ones((rows,), device="cuda", dtype=torch.float32)

        actual = kernel.swiglu(gate, up, route, 10)

        self.assertEqual(tuple(actual.shape), tuple(gate.shape))
        torch.testing.assert_close(actual, torch.zeros_like(gate))

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

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_sparse_attention_failure_disables_long_query_retry(self):
        device = torch.device("cuda")
        rows = kernel.NATIVE_ROW_LIMIT + 1
        query = torch.zeros((1, rows, 16, 512), device=device, dtype=torch.bfloat16)
        window = torch.zeros((1, 1, 512), device=device, dtype=torch.bfloat16)
        compressed = torch.empty((1, 0, 512), device=device, dtype=torch.bfloat16)
        sink = torch.zeros((16,), device=device)
        indices = torch.zeros((1, rows, 1), device=device, dtype=torch.int32)
        kv = window
        native = Mock(return_value=None)

        with (
            patch.object(kernel, "_SPARSE_NATIVE_DISABLED", True),
            patch.object(kernel, "_SPARSE_NATIVE_FAILED", False, create=True),
            patch.object(kernel, "_sparse_attn_native", native),
        ):
            kernel.sparse_attn(
                query,
                kv,
                sink,
                indices,
                512**-0.5,
                native_window=window,
                native_compressed=compressed,
            )
            kernel.sparse_attn(
                query,
                kv,
                sink,
                indices,
                512**-0.5,
                native_window=window,
                native_compressed=compressed,
            )

        self.assertEqual(native.call_count, 1)

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

        hyperconnection = Mock(wraps=kernel._hyperconnection_module())
        with patch.object(
            kernel, "_hyperconnection_module", return_value=hyperconnection
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
            with patch.object(kernel, "_SPARSE_NATIVE_DISABLED", True):
                public = kernel.sparse_attn(
                    query,
                    torch.cat((window, compressed), dim=1),
                    sink,
                    indices,
                    512**-0.5,
                    native_window=window,
                    native_compressed=compressed,
                )

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
        transform = Mock(wraps=kernel._transform_module())
        with patch.object(kernel, "_transform_module", return_value=transform):
            actual = kernel.rms(values, weight, 1e-6)
        expected = (
            values.float()
            * torch.rsqrt(values.float().square().mean(-1, keepdim=True) + 1e-6)
            * weight.float()
        ).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, atol=0.002, rtol=0.002)
        self.assertEqual(transform.rms.call_count, 1)

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
        loader.rank = 0
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

    def test_rank_loader_loads_multiple_parameter_dtypes_through_the_public_boundary(
        self,
    ):
        loader = RankLoader.__new__(RankLoader)
        loader.rank = 0
        loader.specs = {
            "first": {"shape": (2, 3), "dtype": "BF16"},
            "second": {"shape": (4,), "dtype": "F32"},
        }
        values = {
            "first": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            "second": torch.arange(4, dtype=torch.float32),
        }
        loader._read_parameter = Mock(side_effect=values.__getitem__)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Parameter(torch.empty(2, 3, dtype=torch.bfloat16))
                self.second = torch.nn.Parameter(torch.empty(4, dtype=torch.float32))

        model = TinyModel()
        info = loader.load_into(model)

        self.assertEqual(info["parameter_count"], 2)
        torch.testing.assert_close(model.first, values["first"])
        torch.testing.assert_close(model.second, values["second"])
        self.assertEqual(model.first.data_ptr() % 16, 0)
        self.assertEqual(model.second.data_ptr() % 16, 0)

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
