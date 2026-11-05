import hashlib
import os
import unittest
from pathlib import Path

import torch
from oh_my_vllm.operators.runtime.native import load
from oh_my_vllm.operators.transform.api import FUNCTIONS


def assert_bytes(test, actual, expected):
    test.assertEqual(actual.shape, expected.shape)
    test.assertEqual(actual.dtype, expected.dtype)
    test.assertTrue(
        torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )
    )


def engram_inputs(rows):
    return [
        torch.zeros((rows, 4, 5120), dtype=torch.bfloat16, device="cuda"),
        torch.zeros((rows, 25600), dtype=torch.bfloat16, device="cuda"),
        torch.ones((4, 5120), dtype=torch.bfloat16, device="cuda"),
        torch.ones((4, 5120), dtype=torch.bfloat16, device="cuda"),
        torch.empty(0, dtype=torch.uint8, device="cuda"),
        torch.full((rows, 4, 5120), -7, dtype=torch.bfloat16, device="cuda"),
    ]


class LogicalTransformInterface(unittest.TestCase):
    def test_native_exports_keep_existing_operations_and_add_logical_selection(self):
        self.assertTrue(
            {"rms", "rope", "quantize", "rms_grouped", "engram"}.issubset(FUNCTIONS)
        )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class LogicalTransformContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load("transform", FUNCTIONS)[0]

    def test_weighted_heads_keeps_bfloat16_rounding_and_graph_stream(self):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            for rows, width in (
                (0, 8),
                (1, 1),
                (3, 7),
                (3, 24),
                (53, 1152),
                (107, 128),
                (7, 129),
            ):
                values = torch.randn((rows, 8, width), device="cuda").bfloat16()
                weights = torch.randn((rows, 8), device="cuda").bfloat16()
                output = torch.empty((rows, width), dtype=torch.float32, device="cuda")
                self.module.weighted_heads(values, weights, output)
                expected = (values.relu() * weights.unsqueeze(-1)).sum(dim=1).float()
                assert_bytes(self, output, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                self.module.weighted_heads(values, weights, output)
            weights.neg_()
            graph.replay()
            expected = (values.relu() * weights.unsqueeze(-1)).sum(dim=1).float()
            assert_bytes(self, output, expected)
        stream.synchronize()

    def test_weighted_heads_rejects_metadata_and_aliases_before_writes(self):
        values = torch.ones((3, 8, 16), dtype=torch.bfloat16, device="cuda")
        weights = torch.ones((3, 8), dtype=torch.bfloat16, device="cuda")
        output = torch.full((3, 16), -7, dtype=torch.float32, device="cuda")
        misaligned = torch.ones((25,), dtype=weights.dtype, device="cuda")[1:].view(
            3, 8
        )
        strided = torch.ones((3, 8, 32), dtype=values.dtype, device="cuda")[:, :, ::2]
        invalid = (
            (values.float(), weights, output),
            (values[:, :4], weights, output),
            (values[:, :, :15], weights, output[:, :15]),
            (values, weights.cpu(), output),
            (values, misaligned, output),
            (strided, weights, output),
            (values, weights, output.bfloat16()),
            (values, weights, values.view(torch.float32).flatten()[:48].view(3, 16)),
            (values, output.view(torch.bfloat16).flatten()[:24].view(3, 8), output),
        )
        for arguments in invalid:
            with self.subTest(shapes=[tuple(value.shape) for value in arguments]):
                before = [value.clone() for value in arguments]
                with self.assertRaisesRegex(Exception, "Check failed"):
                    self.module.weighted_heads(*arguments)
                for actual, expected in zip(arguments, before, strict=True):
                    assert_bytes(self, actual, expected)

    def test_topk_bf16_physical_tail_and_graph_on_caller_stream(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for rows, width in (
                (3, 512),
                (
                    2 * torch.cuda.get_device_properties(0).multi_processor_count + 1,
                    513,
                ),
                (3, 16401),
            ):
                physical = (width + 511) // 512 * 512
                scores = torch.full(
                    (rows, physical), torch.nan, dtype=torch.bfloat16, device="cuda"
                )
                scores[:, :width].normal_()
                scores[0, :width] = -torch.inf
                scores[1, 0] = torch.inf
                output = torch.empty((rows, 512), dtype=torch.int64, device="cuda")
                self.module.topk_bf16(scores, output, width)
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        self.module.topk_bf16(scores, output, width)
                    for replay in (False, True):
                        if replay:
                            scores[:, :width].neg_()
                            graph.replay()
                        self.assertTrue(bool(((output >= 0) & (output < width)).all()))
                        self.assertTrue(bool((output.diff(dim=-1) > 0).all()))
                        logical = scores[:, :width].cpu().float()
                        expected = (
                            logical.topk(512, sorted=False).values.sort(-1).values
                        )
                        actual = logical.gather(-1, output.cpu()).sort(-1).values
                        self.assertTrue(torch.equal(actual, expected))
                finally:
                    stream.synchronize()
                    graph.reset()
        stream.synchronize()

    def test_topk_bf16_accepts_16_byte_output_alignment_in_eager_and_graph(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for width in (512, 1025):
                physical = (width + 511) // 512 * 512
                scores = torch.zeros(
                    (physical,), dtype=torch.bfloat16, device="cuda"
                ).as_strided((1, physical), (1, 1))
                self.assertTrue(scores.is_contiguous())
                self.assertEqual(scores.stride(), (1, 1))
                scores[:, :512] = 1
                storage = torch.full((516,), -7, dtype=torch.int64, device="cuda")
                output = storage[2:-2].view(1, 512)
                self.assertEqual(output.data_ptr() % 32, 16)
                self.module.topk_bf16(scores, output, width)
                expected = torch.arange(512, dtype=torch.int64, device="cuda").view(
                    1, 512
                )
                self.assertTrue(torch.equal(output, expected))
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        self.module.topk_bf16(scores, output, width)
                    scores.zero_()
                    scores[:, width - 512 : width] = 2
                    output.fill_(-7)
                    graph.replay()
                    expected.add_(width - 512)
                    self.assertTrue(torch.equal(output, expected))
                    self.assertTrue(bool((storage[:2] == -7).all()))
                    self.assertTrue(bool((storage[-2:] == -7).all()))
                finally:
                    stream.synchronize()
                    graph.reset()
        stream.synchronize()

    def test_topk_bf16_rejects_incomplete_physical_views_and_aliases(self):
        width, physical = 16385, 16896
        storage = torch.full(
            (physical + 2048,), -7, dtype=torch.bfloat16, device="cuda"
        )
        scores = storage[:physical].view(1, physical)
        output = torch.full((1, 512), -7, dtype=torch.int64, device="cuda")
        alias = storage[16400 : 16400 + 2048].view(torch.int64).view(1, 512)
        invalid = (
            (scores[:, :width], output, width),
            (scores[:, ::2], output, width),
            (scores.float(), output, width),
            (scores.cpu(), output, width),
            (scores, output.int(), width),
            (scores, output, 511),
            (scores, output, physical + 1),
            (scores, output, 2**23),
            (scores, alias, width),
            (storage[1 : physical + 1].view_as(scores), output, width),
        )
        for arguments in invalid:
            with self.subTest(width=arguments[-1]):
                with self.assertRaisesRegex(Exception, "Check failed"):
                    self.module.topk_bf16(*arguments)
                assert_bytes(self, storage, torch.full_like(storage, -7))
                assert_bytes(self, output, torch.full_like(output, -7))
        self.module.topk_bf16(scores[:0], output[:0], width)

    def test_grouped_rms_uses_logical_rows_without_changing_physical_rms(self):
        generator = torch.Generator(device="cuda").manual_seed(113)
        for width in (128, 512, 1024, 1280, 5120):
            values = torch.randn(
                (12, width), generator=generator, device="cuda"
            ).bfloat16()
            weight = torch.randn(
                (width,), generator=generator, device="cuda"
            ).bfloat16()
            for epsilon in (1e-20, 1e-6):
                for logical in (1, 2, 3, 4, 6, 12):
                    with self.subTest(width=width, epsilon=epsilon, logical=logical):
                        expected = torch.cat(
                            [
                                (
                                    (
                                        part.float()
                                        * torch.rsqrt(
                                            part.float().square().mean(-1, keepdim=True)
                                            + epsilon
                                        )
                                    )
                                    * weight.float()
                                ).bfloat16()
                                for part in values.split(logical)
                            ]
                        )
                        output = torch.empty_like(values)
                        self.module.rms_grouped(
                            values, weight, output, epsilon, logical
                        )
                        assert_bytes(self, output, expected)
                output = torch.empty_like(values)
                grouped = torch.empty_like(values)
                self.module.rms(values, weight, output, epsilon)
                self.module.rms_grouped(values, weight, grouped, epsilon, 12)
                assert_bytes(self, output, grouped)

    def test_grouped_rms_rejects_bad_logical_metadata_before_writes(self):
        values = torch.ones((6, 5120), dtype=torch.bfloat16, device="cuda")
        weight = values[0].clone()
        output = torch.full_like(values, -7)
        for logical in (-1, 0, 4, 7, 1.5):
            with self.subTest(logical=logical):
                with self.assertRaisesRegex(Exception, "Check failed|[Ii]nt"):
                    self.module.rms_grouped(values, weight, output, 1e-20, logical)
                assert_bytes(self, output, torch.full_like(output, -7))
        for arguments in (
            (values, weight, values, 1e-20, 1),
            (values, weight[:-1], output, 1e-20, 1),
            (values, weight, output, 1e-5, 1),
        ):
            with self.assertRaisesRegex(Exception, "Check failed"):
                self.module.rms_grouped(*arguments)
            assert_bytes(self, values, torch.ones_like(values))
            assert_bytes(self, output, torch.full_like(output, -7))
        self.module.rms_grouped(values[:0], weight, output[:0], 1e-20, 0)
        with self.assertRaisesRegex(Exception, "Check failed|[Ii]nt"):
            self.module.rms_grouped(values[:0], weight, output[:0], 1e-20, 1)

    def test_engram_mask_keeps_source_signed_zero_arithmetic(self):
        arguments = engram_inputs(2)
        residual, kv, _, _, _, output = arguments
        residual[:, :, ::2] = -0.0
        kv[:, 20480 + 1 :: 2] = -0.0
        arguments[4] = torch.zeros(2, dtype=torch.uint8, device="cuda")
        self.module.engram(*arguments, 1)
        expected = (
            residual.float() + 0.0 * kv[:, 20480:].float().unsqueeze(1)
        ).bfloat16()
        assert_bytes(self, output, expected)

    def test_engram_rejects_metadata_and_all_output_aliases_before_writes(self):
        arguments = engram_inputs(6)
        residual, kv, q_weight, k_weight, mask, output = arguments
        strided = torch.full((6, 4, 10240), -7, dtype=torch.bfloat16, device="cuda")[
            :, :, ::2
        ]
        misaligned = torch.full(
            (output.numel() + 1,), -7, dtype=torch.bfloat16, device="cuda"
        )[1:].view_as(output)
        invalid = [
            [residual.float(), kv, q_weight, k_weight, mask, output],
            [residual, kv[:, :-1], q_weight, k_weight, mask, output],
            [residual, kv, q_weight.float(), k_weight, mask, output],
            [residual, kv, q_weight, k_weight.cpu(), mask, output],
            [
                residual,
                kv,
                q_weight,
                k_weight,
                torch.zeros(6, dtype=torch.bool, device="cuda"),
                output,
            ],
            [
                residual,
                kv,
                q_weight,
                k_weight,
                torch.zeros(5, dtype=torch.uint8, device="cuda"),
                output,
            ],
            [
                residual,
                kv,
                q_weight,
                k_weight,
                torch.zeros((6, 1), dtype=torch.uint8, device="cuda"),
                output,
            ],
            [
                residual,
                kv,
                q_weight,
                k_weight,
                torch.zeros(12, dtype=torch.uint8, device="cuda")[::2],
                output,
            ],
            [residual, kv, q_weight, k_weight, mask, strided],
            [residual, kv, q_weight, k_weight, mask, misaligned],
            [residual, kv, q_weight, k_weight, mask, output.float()],
            [residual.view(6, -1), kv, q_weight, k_weight, mask, output],
            [strided, kv, q_weight, k_weight, mask, output],
            [misaligned, kv, q_weight, k_weight, mask, output],
            [residual, kv, q_weight, k_weight, mask, residual],
            [
                residual,
                kv,
                q_weight,
                k_weight,
                mask,
                kv.view(-1)[: output.numel()].view_as(output),
            ],
        ]
        for index in (2, 3, 4):
            alias = arguments.copy()
            if index == 4:
                alias[index] = output.view(torch.uint8).view(-1)[:6]
            else:
                alias[index] = output[0]
            invalid.append(alias)
        storage = torch.zeros(output.numel() + 8, dtype=torch.bfloat16, device="cuda")
        invalid.append(
            [
                storage[:-8].view_as(residual),
                kv,
                q_weight,
                k_weight,
                mask,
                storage[8:].view_as(output),
            ]
        )
        for candidate in invalid:
            with self.subTest(shapes=[tuple(v.shape) for v in candidate]):
                before = [value.clone() for value in candidate]
                with self.assertRaisesRegex(Exception, "Check failed|[Ii]nt"):
                    self.module.engram(*candidate, 1)
                for value, original in zip(candidate, before, strict=True):
                    assert_bytes(self, value, original)
        for logical in (-1, 4, 7, 1.5):
            with self.subTest(logical=logical):
                with self.assertRaisesRegex(Exception, "Check failed|[Ii]nt"):
                    self.module.engram(*arguments, logical)
                assert_bytes(self, output, torch.full_like(output, -7))

    def test_engram_rejects_rows_above_bound_before_writes(self):
        arguments = engram_inputs(1)
        arguments[0] = torch.empty(
            (32769, 4, 5120), dtype=torch.bfloat16, device="cuda"
        )
        with self.assertRaisesRegex(Exception, "32768"):
            self.module.engram(*arguments, 0)
        assert_bytes(self, arguments[5], torch.full_like(arguments[5], -7))

    def test_engram_empty_validates_metadata_without_launch(self):
        arguments = engram_inputs(0)
        self.module.engram(*arguments, 0)
        arguments[5] = arguments[2].view(-1)[8:8].view(0, 4, 5120)
        self.module.engram(*arguments, 0)
        for logical in (-1, 1):
            with self.assertRaisesRegex(Exception, "Check failed|[Ii]nt"):
                self.module.engram(*arguments, logical)
        arguments[2] = arguments[2].float()
        with self.assertRaisesRegex(Exception, "Check failed|[Ii]nt"):
            self.module.engram(*arguments, 0)

    def test_engram_and_grouped_rms_graph_replay_on_caller_stream(self):
        arguments = engram_inputs(6)
        residual, kv, q_weight, _k_weight, _, output = arguments
        arguments[4] = torch.zeros(6, dtype=torch.uint8, device="cuda")
        norm = torch.empty((24, 5120), dtype=torch.bfloat16, device="cuda")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                self.module.engram(*arguments, 2)
                self.module.rms_grouped(
                    output.view(24, 5120), q_weight[0], norm, 1e-20, 8
                )
            for value in (1, 2):
                with torch.cuda.stream(stream):
                    residual.fill_(value)
                    kv.fill_(value)
                    arguments[4].fill_(value - 1)
                    output.fill_(-7)
                    graph.replay()
                    observed = output.clone()
                    normalized = norm.clone()
                    output.fill_(-7)
                    norm.fill_(-7)
                    self.module.engram(*arguments, 2)
                    self.module.rms_grouped(
                        output.view(24, 5120), q_weight[0], norm, 1e-20, 8
                    )
                    eager_output = output.clone()
                    eager_norm = norm.clone()
                    finished = torch.cuda.Event()
                    finished.record()
                finished.synchronize()
                gate = (
                    torch.sigmoid(
                        torch.tensor(5120**0.25, dtype=torch.float32, device="cuda")
                    )
                    if value == 2
                    else 0.0
                )
                expected = (
                    residual.float() + gate * kv[:, 20480:].float().unsqueeze(1)
                ).bfloat16()
                assert_bytes(self, observed, expected)
                assert_bytes(self, normalized, torch.ones_like(normalized))
                assert_bytes(self, eager_output, expected)
                assert_bytes(self, eager_norm, torch.ones_like(eager_norm))
        finally:
            stream.synchronize()
            graph.reset()


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("OH_MY_VLLM_TRANSFORM_REFERENCE"),
    "CUDA and independent source fixtures are required",
)
class LogicalTransformSourceFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from safetensors.torch import load_file

        root = Path(os.environ["OH_MY_VLLM_TRANSFORM_REFERENCE"])
        cls.module = load("transform", FUNCTIONS)[0]
        cls.fixtures = {}
        paths = (
            (
                "engram",
                "target-engram/arithmetic-reference.safetensors",
                "efd4f32c39e44460a5675828bdbdd3b85b1d383288afa64f6b820071a60f5679",
            ),
            (
                "rms",
                "target-verification/rms-reference.safetensors",
                "f57626ec73d2f025da82200ee5dbabbec0fafa1a3ba9146bdc933af3f0e212f9",
            ),
            (
                "synthetic",
                "target-native-transform/synthetic-reference.safetensors",
                "435c6f78fbf0c223df7a5ddbb427ce88e79f4410d948ead17a5aba8aa0d60802",
            ),
        )
        for label, relative, expected in paths:
            path = root / relative
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("independent source fixture identity differs")
            cls.fixtures[label] = load_file(str(path), device="cuda")

    def test_actual_engram_logical_and_physical_topologies_are_byte_exact(self):
        frames = self.fixtures["engram"]
        for physical in (1, 2, 3, 4, 6, 8, 16, 33):
            residual = frames["input"].view(1, 4, 5120).repeat(physical, 1, 1)
            kv = frames["kv"].view(1, 25600).repeat(physical, 1)
            output = torch.empty_like(residual)
            mask = residual.view(torch.uint8).view(-1)[:0]
            for logical in (0, 1, 2, 3, 4, 6, 8, 16, 33):
                if logical and physical % logical:
                    continue
                with self.subTest(physical=physical, logical=logical):
                    self.module.engram(
                        residual,
                        kv,
                        frames["qweight"],
                        frames["kweight"],
                        mask,
                        output,
                        logical,
                    )
                    source = frames[f"m{logical or physical}.output"].view(-1, 4, 5120)[
                        :1
                    ]
                    assert_bytes(self, output, source.repeat(physical, 1, 1))

    def test_distinct_token_rows_use_independent_source_arithmetic(self):
        frames = self.fixtures["synthetic"]
        for batch in (1, 2, 3, 4):
            for steps in (2, 3, 6, 8):
                prefix = f"b{batch}s{steps}."
                residual = frames[prefix + "residual"].view(-1, 4, 5120)
                kv = frames[prefix + "kv"].view(-1, 25600)
                output = torch.empty_like(residual)
                q_weight = frames[prefix + "q_weight"]
                k_weight = frames[prefix + "k_weight"]
                for logical, mask, field in (
                    (0, torch.empty(0, dtype=torch.uint8, device="cuda"), "physical"),
                    (
                        batch,
                        torch.empty(0, dtype=torch.uint8, device="cuda"),
                        "logical",
                    ),
                    (batch, frames[prefix + "token_mask"].view(-1), "masked"),
                ):
                    with self.subTest(batch=batch, steps=steps, field=field):
                        self.module.engram(
                            residual, kv, q_weight, k_weight, mask, output, logical
                        )
                        assert_bytes(
                            self, output, frames[prefix + field].view_as(output)
                        )

    def test_distinct_source_arithmetic_graph_replays_masks_on_caller_stream(self):
        frames = self.fixtures["synthetic"]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for batch in (1, 2, 3, 4):
            prefix = f"b{batch}s6."
            residual = frames[prefix + "residual"].view(-1, 4, 5120)
            kv = frames[prefix + "kv"].view(-1, 25600)
            q_weight, k_weight = (
                frames[prefix + name] for name in ("q_weight", "k_weight")
            )
            mask = frames[prefix + "token_mask"].view(-1).clone()
            output = torch.empty_like(residual)
            stream.wait_stream(torch.cuda.current_stream())
            graph = torch.cuda.CUDAGraph()
            try:
                with torch.cuda.graph(graph, stream=stream):
                    self.module.engram(
                        residual, kv, q_weight, k_weight, mask, output, batch
                    )
                for enabled in (True, False, True):
                    with torch.cuda.stream(stream):
                        if enabled:
                            mask.copy_(frames[prefix + "token_mask"].view(-1))
                        else:
                            mask.zero_()
                        output.fill_(-7)
                        graph.replay()
                        observed = output.clone()
                        finished = torch.cuda.Event()
                        finished.record()
                    finished.synchronize()
                    expected = (
                        frames[prefix + "masked"].view_as(output)
                        if enabled
                        else residual
                    )
                    assert_bytes(self, observed, expected)
            finally:
                stream.synchronize()
                graph.reset()

    def test_actual_rms_norms_select_logical_topology_and_keep_default(self):
        frames = self.fixtures["rms"]
        for name in sorted(
            key.removesuffix(".input") for key in frames if key.endswith(".input")
        ):
            weight = frames[name + ".weight"]
            values = frames[name + ".input"].view(1, -1).repeat(48, 1)
            output = torch.empty_like(values)
            for logical in (1, 2, 3, 4, 6, 8, 16):
                with self.subTest(name=name, logical=logical):
                    self.module.rms_grouped(values, weight, output, 1e-20, logical)
                    expected = (
                        frames[name + f".m{logical}"]
                        .view(logical, -1)[:1]
                        .repeat(48, 1)
                    )
                    assert_bytes(self, output, expected)
                    self.module.rms(values[:logical], weight, output[:logical], 1e-20)
                    assert_bytes(
                        self,
                        output[:logical],
                        frames[name + f".m{logical}"].view(logical, -1),
                    )
            values = values[:33]
            output = output[:33]
            self.module.rms(values, weight, output, 1e-20)
            assert_bytes(self, output, frames[name + ".m33"].view_as(output))


if __name__ == "__main__":
    unittest.main()
