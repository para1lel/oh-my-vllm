import hashlib
import json
import math
import os
import statistics
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from development.serving import deepseek_accounting as source
from development.serving import deepseek_accounting_math as acct

TINY = acct.ModelSpec(
    vocab_size=16,
    hidden_size=32,
    num_layers=1,
    num_heads=4,
    head_dim=8,
    q_lora_rank=8,
    o_lora_rank=8,
    o_groups=4,
    moe_intermediate=32,
    routed_experts=2,
    experts_per_token=1,
    shared_experts=1,
    window_size=2,
    compress_ratios=(0,),
    kv_source_layers=(),
    index_source_layers=(),
    index_heads=4,
    index_head_dim=8,
    index_topk=1,
    candidate_source_layer=-1,
    candidate_positions=8,
    hc_mult=1,
    hc_sinkhorn_iters=1,
    engram_layers=(),
    engram_columns=0,
    engram_head_dim=8,
    engram_scale_bytes=1,
    draft_tokens=5,
)


class ResourcePricing(unittest.TestCase):
    def test_group_seconds_use_tensor_sum_and_scalar_max(self):
        self.assertEqual(
            acct.group_seconds(acct.Group("x", acct.Work(hbm_bytes=7.7e12))), 1
        )
        self.assertEqual(
            acct.group_seconds(acct.Group("x", acct.Work(fp8_flops=4.5e15))), 1.875
        )
        self.assertEqual(
            acct.group_seconds(
                acct.Group("x", acct.Work(fp8_flops=4.5e15, bf16_flops=2.25e15))
            ),
            2,
        )
        self.assertEqual(
            acct.group_seconds(acct.Group("x", acct.Work(fp32_flops=75e12))), 1
        )
        self.assertEqual(
            acct.group_seconds(acct.Group("x", acct.Work(nvlink_bytes=0.9e12))), 1
        )
        self.assertEqual(
            acct.group_seconds(acct.Group("x", acct.Work(host_bytes=63.02e9))), 1
        )
        self.assertEqual(
            acct.group_seconds(
                acct.Group("x", acct.Work(hbm_bytes=7.7e12), repetitions=2)
            ),
            2,
        )

    def test_critical_path_takes_max_per_group_then_sums(self):
        one = acct.Work(hbm_bytes=7.7e12)
        zero = acct.Work()
        rank0 = [acct.Group("a", one), acct.Group("b", zero)]
        rank1 = [acct.Group("a", zero), acct.Group("b", one)]
        self.assertAlmostEqual(acct.critical_seconds([rank0, rank1]).seconds, 2.0)
        with self.assertRaises(ValueError):
            acct.critical_seconds(
                [
                    [acct.Group("a", acct.Work())],
                    [acct.Group("b", acct.Work())],
                ]
            )
        with self.assertRaises(ValueError):
            acct.critical_seconds([])

    def test_add_and_scale_work_vectors(self):
        work = acct.Work(hbm_bytes=1, fp8_flops=2)
        self.assertEqual(acct.add_work(work, work), acct.Work(hbm_bytes=2, fp8_flops=4))
        self.assertEqual(acct.scale_work(work, 3), acct.Work(hbm_bytes=3, fp8_flops=6))
        with self.assertRaises(ValueError):
            acct.scale_work(work, 0)

    def test_serial_seconds_add_empty_groups_as_zero(self):
        groups = [acct.Group("launch", acct.Work(), repetitions=1000) for _ in range(3)]
        self.assertEqual(acct.serial_seconds(groups), 0)
        self.assertTrue(math.isfinite(acct.serial_seconds(groups)))


class ProjectionWork(unittest.TestCase):
    def test_fp8_fp4_bf16_and_activation_bytes(self):
        fp8 = acct.projection_work(2, 8, 8, storage="fp8")
        self.assertEqual(fp8.hbm_bytes, 65 + 16 + 32)
        self.assertEqual(fp8.fp8_flops, 256)

        fp4 = acct.projection_work(1, 2304, 5120, storage="fp4")
        self.assertEqual(fp4.hbm_bytes, 6_266_880 + 5_120 + 4_608)
        self.assertEqual(fp4.fp8_flops, 23_592_960)

        promoted = acct.projection_work(1, 1280, 5120, storage="bf16", compute="fp32")
        self.assertEqual(promoted.hbm_bytes, 13_107_200 + 20_480 + 5_120)
        self.assertEqual(promoted.fp32_flops, 13_107_200)

        sharded = acct.projection_work(
            1, 32768, 1280, storage="fp8", shard_axis=0, shards=4
        )
        self.assertEqual(sharded.hbm_bytes, 10_496_000 + 1_280 + 16_384)
        self.assertEqual(sharded.fp8_flops, 20_971_520)

    def test_projection_rejects_invalid_shape_and_shards(self):
        for rows, out_features, in_features, kwargs in (
            (0, 1280, 5120, {"storage": "fp8"}),
            (1, 0, 5120, {"storage": "fp8"}),
            (1, 1280, 5120, {"storage": "fp8", "shards": 0}),
            (1, 1280, 5120, {"storage": "fp8", "shard_axis": 0}),
            (1, 101, 5120, {"storage": "fp8", "shard_axis": 0, "shards": 4}),
            (1, 2304, 100, {"storage": "fp4"}),
            (1, 1280, 5120, {"storage": "int8"}),
            (1, 1280, 5120, {"storage": "fp8", "compute": "tf32"}),
        ):
            with (
                self.subTest(rows=rows, out=out_features, in_=in_features, **kwargs),
                self.assertRaises(ValueError),
            ):
                acct.projection_work(rows, out_features, in_features, **kwargs)


class PairAccounting(unittest.TestCase):
    def test_window_and_compressed_pairs_match_bruteforce(self):
        for start in range(0, 13):
            for rows in range(1, 9):
                for window in range(1, 7):
                    expected = sum(
                        min(window, p + 1) for p in range(start, start + rows)
                    )
                    self.assertEqual(acct.window_pairs(start, rows, window), expected)
                for ratio in range(1, 4):
                    for topk in range(1, 7):
                        expected = sum(
                            min(topk, (p + 1) // ratio)
                            for p in range(start, start + rows)
                        )
                        self.assertEqual(
                            acct.compressed_pairs(start, rows, ratio, topk),
                            expected,
                        )

    def test_indexer_pairs_are_uncapped_or_candidate_capped(self):
        self.assertEqual(
            acct.index_score_pairs(0, 8, 2),
            sum((p + 1) // 2 for p in range(8)),
        )
        self.assertEqual(
            acct.index_score_pairs(0, 8, 1, candidate_limit=3),
            sum(min(3, p + 1) for p in range(8)),
        )
        self.assertEqual(
            acct.index_score_pairs(0, 8, 2, candidate_limit=2),
            sum(min(2, (p + 1) // 2) for p in range(8)),
        )


class Collectives(unittest.TestCase):
    def test_ring_formulas(self):
        self.assertEqual(acct.all_reduce_wire_bytes(4, 400), 600)
        self.assertEqual(acct.all_gather_wire_bytes(4, 400), 300)
        self.assertEqual(acct.reduce_scatter_wire_bytes(4, 100), 300)
        self.assertEqual(acct.all_reduce_wire_bytes(1, 400), 0)

    def test_route_pressure_uses_max_of_independent_directions(self):
        routes = [
            [0, 1, 2, 3],
            [4, 0, 5, 6],
            [7, 8, 0, 9],
            [1, 2, 3, 0],
        ]
        self.assertEqual(acct.route_wire_pressure_bytes(routes, 0, 10), 120)
        self.assertEqual(acct.route_wire_pressure_bytes(routes, 1, 10), 150)
        self.assertEqual(acct.route_wire_pressure_bytes([[0]], 0, 10), 0)

        hotspot = [
            [0, 0, 0, 0],
            [5, 0, 0, 0],
            [7, 0, 0, 0],
            [3, 0, 0, 0],
        ]
        self.assertEqual(acct.route_wire_pressure_bytes(hotspot, 0, 10), 150)
        self.assertEqual(acct.route_wire_pressure_bytes(hotspot, 1, 10), 50)

    def test_collectives_reject_bad_counts(self):
        with self.assertRaises(ValueError):
            acct.all_reduce_wire_bytes(0, 10)
        with self.assertRaises(ValueError):
            acct.all_reduce_wire_bytes(4, -1)
        with self.assertRaises(ValueError):
            acct.route_wire_pressure_bytes([[0, 1], [1]], 0, 10)
        with self.assertRaises(ValueError):
            acct.route_wire_pressure_bytes([[0, 1], [1, 0]], 0, 0)


class CollectiveProfile(unittest.TestCase):
    def test_profile_matches_all_raw_points_and_rank_medians(self):
        raw_path = os.environ.get("DEEPSEEK_NCCL_PROFILE_RAW")
        if not raw_path:
            self.skipTest("set DEEPSEEK_NCCL_PROFILE_RAW for external raw validation")
        raw = Path(raw_path).read_bytes()
        self.assertEqual(
            hashlib.sha256(raw).hexdigest(), acct.COLLECTIVE_PROFILE_RAW_SHA256
        )
        measured = {}
        ranks = json.loads(raw)
        self.assertEqual(len(ranks), 4)
        for records in ranks:
            for row in records:
                kind = (
                    "all_reduce"
                    if row["name"].startswith("all_reduce")
                    else row["name"]
                )
                dtype = "bf16" if row["dtype"] == "torch.bfloat16" else "fp32"
                key = (kind, dtype, row["bytes_per_rank"])
                measured.setdefault(key, []).append(row["median_us"])
        expected = {}
        for (kind, dtype, size), latencies in sorted(measured.items()):
            self.assertEqual(len(latencies), 4)
            expected.setdefault((kind, dtype), []).append(
                (size, statistics.median(latencies))
            )
        self.assertEqual(
            acct.COLLECTIVE_LATENCY_PROFILE_US,
            {key: tuple(points) for key, points in expected.items()},
        )
        self.assertNotIn(("all_to_all", "fp32"), expected)
        self.assertNotIn(
            2048, dict(acct.COLLECTIVE_LATENCY_PROFILE_US[("all_reduce", "fp32")])
        )

    def test_profile_digest_changes_with_table_and_invalidates_trace_binding(self):
        original_binding = source.ideal_trace(
            dict(
                mode="ordinary",
                strategy="tp4",
                batch=1,
                input_tokens=20,
                output_tokens=1,
            )
        )["binding"]
        digest = acct.collective_profile_digest()
        key = ("all_reduce", "fp32")
        points = acct.COLLECTIVE_LATENCY_PROFILE_US[key]
        replacement = ((points[0][0], points[0][1] + 0.25), *points[1:])
        with patch.dict(acct.COLLECTIVE_LATENCY_PROFILE_US, {key: replacement}):
            self.assertNotEqual(acct.collective_profile_digest(), digest)
            self.assertNotEqual(
                source.rate_profile_digest(), original_binding["rate_profile"]
            )
            with self.assertRaisesRegex(source.BindingError, "rate profile differs"):
                source.validate_binding(original_binding)

    def test_raw_identity_changes_the_profile_and_rate_digests(self):
        profile = acct.collective_profile_digest()
        rate = acct.rate_profile_digest()
        with patch.object(acct, "COLLECTIVE_PROFILE_RAW_SHA256", "0" * 64):
            self.assertNotEqual(acct.collective_profile_digest(), profile)
            self.assertNotEqual(acct.rate_profile_digest(), rate)

    def test_interpolation_is_monotonic_and_clamps_without_extrapolation(self):
        key = ("all_reduce", "fp32")
        with patch.dict(
            acct.COLLECTIVE_LATENCY_PROFILE_US,
            {key: ((16, 10.0), (32, 8.0), (64, 20.0))},
        ):
            samples = [
                acct.collective_latency_seconds(*key, size)
                for size in (0, 1, 16, 24, 32, 48, 64, 128)
            ]
        self.assertEqual(samples, sorted(samples))
        self.assertEqual(samples[0], 0)
        self.assertAlmostEqual(samples[5], 15e-6)
        self.assertEqual(samples[-1], samples[-2])
        self.assertEqual(samples[1], samples[2])


class ModelTable(unittest.TestCase):
    def test_layer_table_matches_the_checkpoint(self):
        table = acct.layer_table()
        self.assertEqual(len(table), 40)
        self.assertEqual(
            [layer.compress_ratio for layer in table], [0, 0] + [2] * 18 + [1] * 20
        )
        self.assertTrue(table[2].owns_compressed_kv)
        self.assertFalse(table[3].owns_compressed_kv)
        self.assertTrue(table[20].owns_index_keys)
        self.assertFalse(table[24].owns_compressed_kv)
        self.assertTrue(table[24].uses_candidate_mask)
        self.assertTrue(table[1].has_engram)
        self.assertTrue(table[14].has_engram)
        self.assertEqual(acct.ENGRAM_BYTES_PER_ROW, 264)
        self.assertEqual(acct.RATES["host_bytes"], 63.02e9)
        self.assertEqual(len(acct.rate_profile_digest()), 64)


class PerLayerAndFormat(unittest.TestCase):
    def test_compressor_above_ratio_one_uses_fp32_storage(self):
        model = replace(
            TINY,
            compress_ratios=(2,),
            kv_source_layers=(0,),
            index_source_layers=(),
        )
        work = acct.compressor_work(1, model)
        # Two FP32 projections: 4 * 8 * 32 weight and 4 * (32 + 8) activation.
        self.assertEqual(work.hbm_bytes, 2 * (4 * 8 * 32 + 4 * (32 + 8)))
        self.assertEqual(work.fp32_flops, 2 * 2 * 1 * 8 * 32)


if __name__ == "__main__":
    unittest.main()
