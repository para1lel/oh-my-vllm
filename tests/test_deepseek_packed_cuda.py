import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel
from oh_my_vllm.models.deepseek_v41.indexing import score_chunks
from oh_my_vllm.models.deepseek_v41.packed_cache import (
    IndexWorkspace,
    PackedHistory,
    PageTable,
)
from oh_my_vllm.models.deepseek_v41.quantization import fp4_act_quant

MIXED_CASES = (
    (1, 1, 16, 127),
    (1, 4, 64, 128),
    (1, 5, 16, 129),
    (1, 6, 64, 131),
    (4, 1, 16, 127),
    (4, 1, 64, 128),
    (4, 4, 16, 129),
    (4, 5, 64, 131),
    (4, 6, 16, 127),
    (1, 4, 16, 131),
    (1, 1, 64, 129),
    (4, 6, 64, 128),
)
MIXED_SCALE_FAILURES = (("zero", 0), ("negative", 128), ("producer_overflow", 127))


def mixed_inputs(batch, rows, heads, width):
    levels = torch.tensor(
        (
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        )
    )
    scale_codes = torch.tensor((1, 7, 8, 56, 119, 126), dtype=torch.uint8)
    scales = (
        scale_codes[torch.arange(batch * width * 32).remainder(6)]
        .view(torch.float8_e4m3fn)
        .float()
        .view(batch, width, 32, 1)
    )
    request_levels = torch.stack([levels.roll(request) for request in range(batch)])[
        :, None, None, :
    ]
    source = (request_levels.float() * scales).flatten(-2).bfloat16()
    slots = torch.tensor([0] if batch == 1 else [3, 0, 2, 1], dtype=torch.int64)
    window = torch.randn(batch, 128, 512)
    query = torch.randn(batch, rows, heads, 512) / 128
    indices = (
        torch.arange(320, dtype=torch.int32)
        .remainder(128 + width + 1)
        .sub(1)
        .view(1, 1, 320)
        .expand(batch, rows, 320)
        .contiguous()
    )
    sink = torch.randn(heads, dtype=torch.float32)
    return source, slots, window, query, indices, sink


class _PackedHistoryFixture:
    def history(self, batch, length, dim, group, scale_dtype, *, device="cuda"):
        args = SimpleNamespace(
            max_batch_size=batch, max_seq_len=length, lazy_cache=False
        )
        history = PackedHistory(args, dim, 1, group, scale_dtype)
        table = PageTable(
            batch,
            length,
            128,
            1 + batch * ((length + 127) // 128),
            torch.device(device),
        )
        self.addCleanup(table.close)
        history.bind(table)
        self.addCleanup(history.close)
        blocks = table.values.size(1)
        for slot in range(batch):
            physical_slot = (slot + 1) % batch
            ids = list(
                range(1 + physical_slot * blocks, 1 + (physical_slot + 1) * blocks)
            )
            if slot % 2:
                ids.reverse()
            for block in ids:
                history.reserve_page(block)
            table.bind(slot, ids)
        return history


class DeepSeekPackedRecipes(_PackedHistoryFixture, unittest.TestCase):
    def test_mixed_inputs_write_and_read_the_exact_physical_tail(self):
        previous = torch.get_default_dtype()
        self.addCleanup(torch.set_default_dtype, previous)
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cpu"):
            for batch, rows, heads, width in MIXED_CASES:
                with self.subTest(batch=batch, rows=rows, heads=heads, width=width):
                    source, slots, _window, query, indices, _sink = mixed_inputs(
                        batch, rows, heads, width
                    )
                    history = self.history(
                        batch, 256, 512, 16, torch.float8_e4m3fn, device="cpu"
                    )
                    codes, scales = fp4_act_quant(
                        source, 16, scale_dtype=torch.float8_e4m3fn
                    )
                    history.write(slots, 0, codes, scales)
                    view = history.view(slots, width)
                    self.assertEqual(source.size(1), width)
                    self.assertEqual(codes.size(1), width)
                    self.assertEqual(view.size(1), width)
                    self.assertEqual(query.size(0) * query.size(1), batch * rows)
                    self.assertEqual(int(indices.max()), 128 + width - 1)
                    expected = fp4_act_quant(
                        source.clone(),
                        16,
                        inplace=True,
                        scale_dtype=torch.float8_e4m3fn,
                    )
                    torch.testing.assert_close(
                        view.materialize().view(torch.int16),
                        expected.view(torch.int16),
                        atol=0,
                        rtol=0,
                    )
                    for slot in slots.tolist():
                        block = int(history.table.values[slot, width // 128])
                        tail = history.page(block)
                        self.assertEqual(tail.codes[width % 128].count_nonzero(), 0)
                        self.assertTrue((tail.scales[width % 128] == 1).all())


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeepSeekPackedCUDA(_PackedHistoryFixture, unittest.TestCase):
    def setUp(self):
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        self.addCleanup(torch.set_default_dtype, previous)
        device = torch.device("cuda")
        device.__enter__()
        self.addCleanup(device.__exit__, None, None, None)
        torch.manual_seed(29)

    def test_paged_index_gather_preserves_complete_shape_scores_and_graph_scratch(self):
        history = self.history(4, 256, 128, 32, torch.float8_e8m0fnu)
        workspace = IndexWorkspace(4, 256, 128, torch.device("cuda"), False)
        history.workspace = workspace
        slots = torch.tensor([3, 0, 2, 1], dtype=torch.int64)
        source = torch.randn(4, 129, 128)
        codes, scales = fp4_act_quant(source, 32)
        history.write(slots, 0, codes, scales)
        old = source.clone()
        fp4_act_quant(old, 32, inplace=True)
        expected_keys = torch.zeros(4, 192, 128)
        expected_keys[:, :129] = old
        view = history.view(slots, 192)
        actual_keys = view.materialize()
        torch.testing.assert_close(
            actual_keys.view(torch.int16),
            expected_keys.view(torch.int16),
            atol=0,
            rtol=0,
        )
        query = torch.randn(4, 6, 8, 128)
        weights = torch.randn(4, 6, 8)
        expected = [
            value.clone() for _, _, value in score_chunks(query, expected_keys, weights)
        ]
        actual = [value for _, _, value in score_chunks(query, actual_keys, weights)]
        for left, right in zip(actual, expected, strict=True):
            torch.testing.assert_close(
                left.view(torch.int16), right.view(torch.int16), atol=0, rtol=0
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            view.materialize()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        self.addCleanup(graph.reset)
        with torch.cuda.graph(graph, stream=stream):
            result = view.materialize()
        self.assertEqual(result.data_ptr(), workspace.values.data_ptr())
        source.mul_(0.5)
        codes, scales = fp4_act_quant(source, 32)
        history.write(slots, 0, codes, scales)
        expected_keys[:, :129] = fp4_act_quant(source.clone(), 32, inplace=True)
        graph.replay()
        torch.testing.assert_close(
            result.view(torch.int16), expected_keys.view(torch.int16), atol=0, rtol=0
        )
        kernel.require_sparse_status()
        graph.reset()

    def test_mixed_attention_matches_bf16_tables_on_both_native_paths(self):
        for batch, rows, heads, width in MIXED_CASES:
            with (
                self.subTest(batch=batch, rows=rows, heads=heads, width=width),
                patch.object(kernel, "_SPARSE_STATUS", {}),
            ):
                history = self.history(batch, 256, 512, 16, torch.float8_e4m3fn)
                source, slots, window, query, indices, sink = mixed_inputs(
                    batch, rows, heads, width
                )
                codes, stored_scales = fp4_act_quant(
                    source, 16, scale_dtype=torch.float8_e4m3fn
                )
                history.write(slots, 0, codes, stored_scales)
                old = source.clone()
                fp4_act_quant(old, 16, inplace=True, scale_dtype=torch.float8_e4m3fn)
                view = history.view(slots, width)
                self.assertEqual(codes.size(1), width)
                self.assertEqual(view.size(1), width)
                expected = kernel._sparse_attn_native(query, sink, indices, window, old)
                actual = kernel._sparse_attn_native(query, sink, indices, window, view)
                kernel.require_sparse_status()
                torch.testing.assert_close(
                    actual.view(torch.int16), expected.view(torch.int16), atol=0, rtol=0
                )

    def test_native_gather_decodes_all_codes_scale_boundaries_and_signed_zero(self):
        levels = torch.tensor(
            (
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ),
            dtype=torch.float64,
            device="cpu",
        )
        packed = torch.tensor(
            (0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE) * 2,
            dtype=torch.uint8,
            device="cpu",
        )
        for dtype, group, scale_bytes in (
            (torch.float8_e4m3fn, 16, (1, 7, 8, 56, 119, 126)),
            (torch.float8_e8m0fnu, 32, (0, 1, 126, 127, 128, 252, 253, 254)),
        ):
            with self.subTest(dtype=dtype), patch.object(kernel, "_SPARSE_STATUS", {}):
                history = self.history(1, 128, 32, group, dtype)
                raw_scales = (
                    torch.tensor(scale_bytes, dtype=torch.uint8, device="cpu")
                    .view(-1, 1)
                    .expand(-1, 32 // group)
                    .contiguous()
                )
                raw_codes = packed.expand(len(scale_bytes), -1).clone()
                expected_values = levels.repeat(2).expand(len(scale_bytes), -1).clone()
                if dtype == torch.float8_e8m0fnu:
                    raw_codes[-2:] = torch.tensor(
                        (0x80, 0x91, 0xA2, 0xB3) * 4, dtype=torch.uint8, device="cpu"
                    )
                    expected_values[-2:] = torch.tensor(
                        (0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5) * 4,
                        dtype=torch.float64,
                        device="cpu",
                    )
                expected = (
                    (
                        expected_values.unflatten(-1, (-1, group))
                        * raw_scales.view(dtype).double().unsqueeze(-1)
                    )
                    .flatten(-2)
                    .bfloat16()
                )
                slots = torch.tensor([0], dtype=torch.int64)
                history.write(
                    slots,
                    0,
                    raw_codes.cuda().unsqueeze(0),
                    raw_scales.cuda().view(dtype).unsqueeze(0),
                )
                actual = history.view(slots, len(scale_bytes)).materialize()[0]
                torch.testing.assert_close(
                    actual.cpu().view(torch.int16),
                    expected.view(torch.int16),
                    atol=0,
                    rtol=0,
                )
                kernel.require_sparse_status()

    def test_invalid_native_gather_scales_and_bf16_overflow_reject_the_step(self):
        for dtype, group, bad_scales in (
            (torch.float8_e4m3fn, 16, (0, 127, 128, 255)),
            (torch.float8_e8m0fnu, 32, (255, 254)),
        ):
            history = self.history(1, 128, 32, group, dtype)
            slots = torch.tensor([0], dtype=torch.int64)
            for scale in bad_scales:
                with (
                    self.subTest(dtype=dtype, scale=scale),
                    patch.object(kernel, "_SPARSE_STATUS", {}),
                    patch.object(kernel, "_SPARSE_NATIVE_DISABLED", False),
                    patch.object(kernel, "_SPARSE_NATIVE_FAILED", False),
                ):
                    history.write(
                        slots,
                        0,
                        torch.full((1, 1, 16), 0x77, dtype=torch.uint8),
                        torch.full((1, 1, 32 // group), scale, dtype=torch.uint8).view(
                            dtype
                        ),
                    )
                    output = history.view(slots, 1).materialize()
                    self.assertFalse(torch.isfinite(output).all())
                    with self.assertRaisesRegex(RuntimeError, "no state is committed"):
                        kernel.require_sparse_status()

    def test_invalid_packed_scales_reject_fast_and_ordinary_attention(self):
        for failure, scale_code in MIXED_SCALE_FAILURES:
            history = self.history(1, 256, 512, 16, torch.float8_e4m3fn)
            slots = torch.tensor([0], dtype=torch.int64)
            source = torch.ones(1, 129, 512)
            if failure == "producer_overflow":
                source[:, -1].fill_(2800)
            codes, scales = fp4_act_quant(source, 16, scale_dtype=torch.float8_e4m3fn)
            if failure == "producer_overflow":
                self.assertTrue((scales.view(torch.uint8)[:, -1] == scale_code).all())
            history.write(slots, 0, codes, scales)
            if failure != "producer_overflow":
                history.page(2).scales[0, 0] = scale_code
            view = history.view(slots, 129)
            for rows in (1, 5):
                with (
                    self.subTest(failure=failure, rows=rows),
                    patch.object(kernel, "_SPARSE_STATUS", {}),
                    patch.object(kernel, "_SPARSE_NATIVE_DISABLED", False),
                    patch.object(kernel, "_SPARSE_NATIVE_FAILED", False),
                ):
                    query = torch.zeros(1, rows, 16, 512)
                    indices = torch.full((1, rows, 1), 129, dtype=torch.int32)
                    result = kernel._sparse_attn_native(
                        query,
                        torch.zeros(16, dtype=torch.float32),
                        indices,
                        torch.zeros(1, 1, 512),
                        view,
                    )
                    torch.testing.assert_close(
                        result, torch.zeros_like(result), atol=0, rtol=0
                    )
                    with self.assertRaisesRegex(RuntimeError, "no state is committed"):
                        kernel.require_sparse_status()
