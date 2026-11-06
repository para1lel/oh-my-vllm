import os
import unittest

import torch

from tests.logical_transform_fixtures import (
    LogicalTransformSourceFixturesFixture,
    assert_bytes,
)


@unittest.skipUnless(
    torch.cuda.is_available() and os.environ.get("OH_MY_VLLM_TRANSFORM_REFERENCE"),
    "CUDA and independent source fixtures are required",
)
class LogicalTransformSourceFixtures(
    LogicalTransformSourceFixturesFixture, unittest.TestCase
):
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
