import hashlib
import json
import math
import os
import statistics
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from development.serving import deepseek_accounting_v2 as acct

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


def binding(**overrides):
    values = dict(
        source="test-source",
        native_modules=["device"],
        checkpoint_config="config",
        checkpoint_index="index",
        tokenizer="tokenizer",
        gpu_uuids=["gpu0", "gpu1", "gpu2", "gpu3"],
        strategy="tp4",
        effective_strategy="tp4",
        owner_plan="owner",
        sampling="greedy-block-5",
        case_id="case",
        accepted=0,
        proposed=0,
    )
    values.update(overrides)
    return acct.build_binding(**values)


def decode_trace(accepted, proposed, *, strategy="tp4"):
    rows = accepted + 1
    calls = []
    for rank in range(acct.WORLD_SIZE):
        calls.append(
            {
                "rank": rank,
                "call": 0,
                "phase": "decode",
                "requests": 1,
                "rows": rows,
                "global_rows": rows,
                "logit_rows": rows,
                "context_end": 100,
                "accepted": accepted,
                "proposed": proposed,
                "sparse_pairs": 0,
                "index_pairs": 0,
                "engram_unique_rows": 0,
                "touched_experts": 1,
                "layer_ids": list(range(acct.DEEPSEEK_V41_MODEL.num_layers)),
                "route_matrix": [[0, 0, 0, 0] for _ in range(acct.WORLD_SIZE)],
            }
        )
    return {
        "binding": binding(
            strategy=strategy,
            effective_strategy=strategy,
            accepted=accepted,
            proposed=proposed,
        ),
        "calls": calls,
    }


class WorkValidation(unittest.TestCase):
    def test_work_rejects_nonfinite_negative_and_non_numbers(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                acct.Work(hbm_bytes=value)
        with self.assertRaises(TypeError):
            acct.Work(hbm_bytes=True)
        with self.assertRaises(TypeError):
            acct.Work(host_bytes="1")

    def test_group_rejects_bad_repetitions_and_work(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            acct.Group("group", acct.Work(), repetitions=0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            acct.Group("group", acct.Work(), repetitions=True)
        with self.assertRaises(TypeError):
            acct.Group("group", {})

    def test_group_seconds_use_tensor_sum_and_scalar_max(self):
        self.assertEqual(
            acct.group_seconds(acct.Group("x", acct.Work(hbm_bytes=7.7e12))), 1
        )
        self.assertEqual(
            acct.group_seconds(acct.Group("x", acct.Work(fp8_flops=4.5e15))), 1
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

    def test_sparse_pairs_add_window_and_compressed_without_padding(self):
        self.assertEqual(acct.window_pairs(0, 1, 7), 1)
        self.assertEqual(acct.window_pairs(0, 8, 7), 35)
        self.assertEqual(
            acct.sparse_pairs(0, 8, 1, window=7, topk=5),
            acct.window_pairs(0, 8, 7) + acct.compressed_pairs(0, 8, 1, 5),
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

    def test_profile_digest_changes_with_table_and_invalidates_v2_binding(self):
        original_binding = binding()
        digest = acct.collective_profile_digest()
        legacy_digest = acct.rate_profile_digest(acct.LEGACY_VERSION)
        key = ("all_reduce", "fp32")
        points = acct.COLLECTIVE_LATENCY_PROFILE_US[key]
        replacement = ((points[0][0], points[0][1] + 0.25), *points[1:])
        with patch.dict(acct.COLLECTIVE_LATENCY_PROFILE_US, {key: replacement}):
            self.assertNotEqual(acct.collective_profile_digest(), digest)
            self.assertNotEqual(
                acct.rate_profile_digest(), original_binding["rate_profile"]
            )
            self.assertEqual(
                acct.rate_profile_digest(acct.LEGACY_VERSION), legacy_digest
            )
            with self.assertRaisesRegex(acct.BindingError, "rate profile differs"):
                acct.validate_binding(original_binding)

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
        self.assertIn("route matrix", " ".join(acct.MISSING_RUNTIME_COUNTERS))


class Communication(unittest.TestCase):
    def plan(self, **overrides):
        values = dict(
            rank=0,
            phase="prefill",
            requests=1,
            rows=8,
            global_rows=8,
            logit_rows=1,
            context_end=128,
            accepted=0,
            proposed=0,
            sparse_pairs=0,
            index_pairs=0,
            engram_unique_rows=0,
            touched_experts=1,
            route_matrix=((0, 0, 0, 0),) * 4,
        )
        values.update(overrides)
        return acct.CallPlan(**values)

    def wire(self, groups, name):
        return next(group.work.nvlink_bytes for group in groups if group.name == name)

    def test_idle_latency_model_is_positive_and_interpolated(self):
        first = acct.collective_latency_seconds("all_reduce", "fp32", 1024)
        second = acct.collective_latency_seconds("all_reduce", "fp32", 20480)
        self.assertGreater(first, 0)
        self.assertGreater(second, first)
        with self.assertRaises(ValueError):
            acct.collective_latency_seconds("broadcast", "fp32", 1024)

    def test_strategy_collective_bytes(self):
        model = acct.DEEPSEEK_V41_MODEL
        tp4 = acct.communication_groups(self.plan(), model, "tp4", "wire")
        rows, hidden, layers, vocab = 8, 5120, 40, 129280
        self.assertEqual(
            self.wire(tp4, "communication.dense"),
            1.5 * rows * hidden * 4 * layers
            + 1.5 * rows * hidden * 2
            + 0.75 * 1 * vocab * 4,
        )
        assignments = rows * layers * 6
        self.assertEqual(
            self.wire(tp4, "communication.routed"),
            assignments * 0.75 * 2304 * 2 + rows * layers * 0.75 * hidden * 4,
        )

        tep4 = acct.communication_groups(self.plan(), model, "tep4", "wire")
        self.assertEqual(
            self.wire(tep4, "communication.routed"),
            rows * layers * 1.5 * hidden * 4,
        )

        route = ((3, 3, 3, 3),) * 4
        dep4 = acct.communication_groups(
            self.plan(route_matrix=route), model, "dep4", "wire"
        )
        self.assertEqual(
            self.wire(dep4, "communication.dispatch"), 9 * hidden * 2 * layers
        )
        self.assertEqual(
            self.wire(dep4, "communication.combine"), 9 * hidden * 4 * layers
        )

    def test_lower_bound_uses_distinct_input_and_output_bytes(self):
        original = acct.collective_latency_seconds
        with patch.object(acct, "collective_latency_seconds", wraps=original) as lookup:
            groups = acct.communication_groups(
                self.plan(rows=1, global_rows=1, context_end=256),
                acct.DEEPSEEK_V41_MODEL,
                "tp4",
            )
        calls = Counter(call.args for call in lookup.call_args_list)
        self.assertEqual(calls[("all_gather", "bf16", 6912)], 1)
        self.assertEqual(calls[("all_gather", "fp32", 5120)], 1)
        self.assertEqual(calls[("all_gather", "fp32", 129280)], 1)
        self.assertEqual(calls[("all_reduce", "fp32", 20480)], 1)
        self.assertEqual(
            self.wire(groups, "communication.dense"),
            40 * 30720 + 15360 + 387840,
        )
        self.assertEqual(
            self.wire(groups, "communication.routed"),
            40 * (20736 + 15360) + 1.5 * (3 * 512 + 5 * 1028),
        )
        for dtype, input_bytes, output_bytes, expected_wire in (
            ("bf16", 6912, 27648, 20736),
            ("fp32", 5120, 20480, 15360),
            ("fp32", 129280, 517120, 387840),
        ):
            work = acct._collective_work(
                "all_gather", dtype, input_bytes, output_bytes, 1, "lower_bound"
            )
            self.assertEqual(work.nvlink_bytes, expected_wire)
        legacy = acct.communication_groups(
            self.plan(rows=1, global_rows=1, context_end=256),
            acct.DEEPSEEK_V41_MODEL,
            "tp4",
            "wire",
        )
        self.assertEqual(
            self.wire(legacy, "communication.routed") - 40 * 15360,
            40 * 20736,
        )

    def test_index_score_uses_default_layer_ratios_and_effective_width(self):
        model = acct.DEEPSEEK_V41_MODEL
        plan = self.plan(rows=1, global_rows=1, context_end=256)
        self.assertEqual(model.index_source_layers, (2, 8, 14, 20, 24, 28, 32, 36))
        self.assertEqual(
            Counter(acct._index_score_collectives(plan, model)),
            Counter({(512, 1): 3, (1028, 1): 5}),
        )
        layers = [layer for layer in acct.layer_table() if layer.owns_index_keys]
        self.assertEqual(
            [acct._index_score_layer_width(plan, layer) for layer in layers],
            [128] * 3 + [257] * 5,
        )
        large = self.plan(rows=384, global_rows=384, context_end=800000)
        self.assertEqual(len(acct._index_score_collectives(large, model)), 8)
        self.assertEqual(
            sum(
                size * count
                for size, count in acct._index_score_collectives(large, model)
            ),
            384 * 4 * (3 * 400000 + 5 * 800001),
        )
        doubled = replace(plan, rows=2, global_rows=2)
        self.assertEqual(
            acct._index_score_collectives(doubled, model),
            tuple(
                (size * 2, count)
                for size, count in acct._index_score_collectives(plan, model)
            ),
        )

    def test_tep4_prices_dense_routed_and_index_reductions_separately(self):
        model = acct.DEEPSEEK_V41_MODEL
        plan = self.plan(rows=1, global_rows=1, context_end=256)
        original = acct._collective_work
        with patch.object(acct, "_collective_work", wraps=original) as collect:
            groups = acct.communication_groups(plan, model, "tep4")
        fp32 = [
            call.args
            for call in collect.call_args_list
            if call.args[:2] == ("all_reduce", "fp32")
        ]
        self.assertEqual(
            Counter((args[2], args[3], args[4]) for args in fp32),
            Counter({(20480, 20480, 40): 2, (512, 512, 1): 3, (1028, 1028, 1): 5}),
        )
        routed = next(group for group in groups if group.name == "communication.routed")
        self.assertEqual(
            routed.work.nvlink_bytes, 40 * 30720 + 1.5 * (3 * 512 + 5 * 1028)
        )
        expected = 40 * acct.collective_latency_seconds("all_reduce", "fp32", 20480)
        expected += 3 * acct.collective_latency_seconds("all_reduce", "fp32", 512)
        expected += 5 * acct.collective_latency_seconds("all_reduce", "fp32", 1028)
        self.assertAlmostEqual(routed.work.collective_seconds, expected)

    def test_dep4_default_uses_same_byte_proxy_and_labels_model(self):
        with patch.object(
            acct, "collective_latency_seconds", return_value=20e-6
        ) as lookup:
            acct._dep4_all_to_all_seconds(46080, "fp32")
        lookup.assert_called_once_with("all_to_all", "bf16", 61440)
        with self.assertRaisesRegex(ValueError, "no measured latency profile"):
            acct.collective_latency_seconds("all_to_all", "fp32", 61440)
        case = dict(
            mode="dspark5", strategy="dep4", batch=4, input_tokens=2, output_tokens=4
        )
        result = acct.reference_case(case)
        self.assertEqual(result.version, acct.VERSION)
        self.assertEqual(result.communication_model, "lower_bound")
        self.assertTrue(result.communication_is_modeled)
        self.assertFalse(result.communication_is_measured)
        self.assertEqual(
            result.communication_rule_version, acct.DEP4_PROXY_RULE_VERSION
        )
        self.assertEqual(result.profile_digest, acct.collective_profile_digest())

    def test_collective_cost_uses_max_without_double_pricing(self):
        with patch.object(acct, "collective_latency_seconds", return_value=20e-6):
            small = acct._collective_work(
                "all_reduce", "fp32", 20480, 20480, 2, "lower_bound"
            )
            large = acct._collective_work(
                "all_reduce", "fp32", 100000000, 100000000, 2, "lower_bound"
            )
        self.assertEqual(small.collective_seconds, 40e-6)
        self.assertEqual(large.collective_seconds, 2 * 150000000 / 0.9e12)

    def test_dep4_label_requires_dep4_execution(self):
        for requested in acct.STRATEGIES:
            acct.validate_execution_strategy(requested, requested)
        with self.assertRaisesRegex(ValueError, "real DEP4"):
            acct.validate_execution_strategy("dep4", "tep4")
        with self.assertRaises(ValueError):
            acct.validate_execution_strategy("tp4", "tep4")


class PerLayerAndFormat(unittest.TestCase):
    def plan(self, **overrides):
        values = dict(
            rank=0,
            phase="decode",
            requests=1,
            rows=4,
            global_rows=4,
            logit_rows=4,
            context_end=10,
            accepted=0,
            proposed=0,
            sparse_pairs=0,
            index_pairs=0,
            engram_unique_rows=0,
            touched_experts=0,
            route_matrix=((0, 0, 0, 0),) * 4,
        )
        values.update(overrides)
        return acct.CallPlan(**values)

    def test_expert_weights_are_counted_for_each_layer(self):
        two_layers = replace(TINY, num_layers=2, compress_ratios=(0, 0))
        self.assertEqual(acct._touched_experts(self.plan(), TINY, "tp4"), 2)
        self.assertEqual(acct._touched_experts(self.plan(), two_layers, "tp4"), 4)

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

    def test_engram_h2d_uses_the_supplied_model_format(self):
        model = replace(
            TINY,
            engram_layers=(0,),
            engram_columns=2,
            engram_head_dim=8,
            engram_scale_bytes=1,
        )
        groups = acct.call_groups(
            self.plan(engram_unique_rows=3),
            model,
            "tp4",
            include_weights=False,
        )
        host = next(
            group.work.host_bytes for group in groups if group.name == "engram_h2d"
        )
        self.assertEqual(host, 3 * 9)


class TinyModelGolden(unittest.TestCase):
    def test_hand_computed_prefill_seconds(self):
        case = {
            "mode": "ordinary",
            "strategy": "tp4",
            "batch": 4,
            "input_tokens": 1,
            "output_tokens": 1,
        }
        result = acct.reference_case(
            case, strategy="tp4", model=TINY, communication_model="wire"
        )
        self.assertEqual(result.calls, 1)
        self.assertEqual(result.acceptance, "ideal")
        # Hand sum: embedding 320, routed 1392, shared 4227, input 1059,
        # output 801, sparse 128 BF16, router 672, HC 1920, head 1088 HBM;
        # dense wire 1344 and routed wire 576 at 0.9e12.
        self.assertAlmostEqual(
            result.seconds,
            3.624169443001443e-09,
            delta=1e-18,
        )


class DecodeCredit(unittest.TestCase):
    def test_proposed_drafts_do_not_add_work(self):
        one = acct.reference_trace(decode_trace(1, 1), strategy="tp4")
        five = acct.reference_trace(decode_trace(1, 5), strategy="tp4")
        self.assertEqual(one.seconds, five.seconds)

    def test_credit_follows_emitted_tokens(self):
        one = acct.reference_trace(decode_trace(1, 1), strategy="tp4")
        five = acct.reference_trace(decode_trace(5, 5), strategy="tp4")
        self.assertGreater(five.seconds, one.seconds)


class TraceValidation(unittest.TestCase):
    def ideal(self, strategy="tp4"):
        case = {
            "mode": "dspark5",
            "strategy": strategy,
            "batch": 4,
            "input_tokens": 1000,
            "output_tokens": 64,
        }
        return acct.ideal_trace(case, strategy=strategy)

    def test_ideal_trace_matches_the_direct_case(self):
        for strategy in acct.STRATEGIES:
            case = {
                "mode": "dspark5",
                "strategy": strategy,
                "batch": 4,
                "input_tokens": 1000,
                "output_tokens": 64,
            }
            direct = acct.reference_case(case, strategy=strategy)
            trace = acct.reference_trace(self.ideal(strategy), strategy=strategy)
            self.assertAlmostEqual(direct.seconds, trace.seconds, places=12)

    def test_trace_version_and_communication_model_must_match(self):
        trace = self.ideal()
        with self.assertRaisesRegex(acct.BindingError, "communication model"):
            acct.reference_trace(trace, strategy="tp4", communication_model="wire")
        trace["binding"]["version"] = acct.LEGACY_VERSION
        trace["binding"]["rate_profile"] = acct.rate_profile_digest(acct.LEGACY_VERSION)
        result = acct.reference_trace(trace, strategy="tp4")
        self.assertEqual(result.version, acct.LEGACY_VERSION)
        self.assertEqual(result.communication_model, "wire")
        with self.assertRaisesRegex(acct.BindingError, "communication model"):
            acct.reference_trace(
                trace, strategy="tp4", communication_model="lower_bound"
            )

    def test_bad_trace_keeps_contract_error_types(self):
        for bad_trace in (None, [], {}, {"binding": None, "calls": []}):
            expected = (
                acct.BindingError
                if isinstance(bad_trace, dict) and "binding" in bad_trace
                else acct.TraceError
            )
            with self.subTest(trace=bad_trace), self.assertRaises(expected):
                acct.reference_trace(bad_trace, strategy="tp4")

    def test_missing_rank_record_is_rejected(self):
        trace = self.ideal()
        trace["calls"] = [call for call in trace["calls"] if call["rank"] != 2]
        with self.assertRaisesRegex(acct.TraceError, "lacks a rank"):
            acct.reference_trace(trace, strategy="tp4")

    def test_duplicate_call_is_rejected(self):
        trace = self.ideal()
        trace["calls"].append(dict(trace["calls"][0]))
        with self.assertRaisesRegex(acct.TraceError, "repeats a rank call"):
            acct.reference_trace(trace, strategy="tp4")

    def test_incomplete_layer_set_is_rejected(self):
        trace = self.ideal()
        trace["calls"][0]["layer_ids"] = [0, 1]
        with self.assertRaisesRegex(acct.TraceError, "layer set"):
            acct.reference_trace(trace, strategy="tp4")

    def test_counter_above_the_causal_limit_is_rejected(self):
        trace = self.ideal()
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["proposed"] = 10_000
        with self.assertRaisesRegex(acct.TraceError, "causal draft limit"):
            acct.reference_trace(trace, strategy="tp4")

    def test_rows_must_match_accepted_tokens_plus_requests(self):
        trace = self.ideal()
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["rows"] += 1
        record["global_rows"] += 1
        with self.assertRaisesRegex(acct.TraceError, "credited rows"):
            acct.reference_trace(trace, strategy="tp4")

    def test_index_pairs_above_the_causal_limit_are_rejected(self):
        trace = self.ideal()
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["index_pairs"] = 10**9
        with self.assertRaisesRegex(acct.TraceError, "index pair count"):
            acct.reference_trace(trace, strategy="tp4")

    def test_replicated_rank_metadata_must_match(self):
        trace = self.ideal()
        record = next(
            call
            for call in trace["calls"]
            if call["phase"] == "decode" and call["rank"] == 1
        )
        record["context_end"] += 1
        with self.assertRaisesRegex(acct.TraceError, "replicated ranks"):
            acct.reference_trace(trace, strategy="tp4")

    def test_rank_work_counters_may_differ(self):
        trace = self.ideal()
        records = [
            call
            for call in trace["calls"]
            if call["phase"] == "decode" and call["rank"] in (0, 1)
        ]
        records[0]["engram_unique_rows"] -= 1
        records[1]["engram_unique_rows"] += 1
        records[0]["sparse_pairs"] -= 1
        records[1]["sparse_pairs"] += 1
        records[0]["index_pairs"] -= 1
        records[1]["index_pairs"] += 1
        acct.reference_trace(trace, strategy="tp4")

    def test_prefill_zero_rows_with_logits_is_rejected(self):
        trace = self.ideal()
        record = next(call for call in trace["calls"] if call["phase"] == "prefill")
        record["rows"] = 0
        record["touched_experts"] = 0
        record["sparse_pairs"] = 0
        record["index_pairs"] = 0
        record["engram_unique_rows"] = 0
        with self.assertRaisesRegex(acct.TraceError, "zero-row"):
            acct.reference_trace(trace, strategy="tp4")

    def test_context_end_must_fit_the_model_context(self):
        trace = self.ideal()
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["context_end"] = acct.DEEPSEEK_V41_MODEL.max_context_tokens
        with self.assertRaisesRegex(acct.TraceError, "context end"):
            acct.reference_trace(trace, strategy="tp4")

    def test_short_prompts_satisfy_the_index_causal_bound(self):
        for input_tokens in (2, 3, 4):
            case = {
                "mode": "dspark5",
                "strategy": "dep4",
                "batch": 4,
                "input_tokens": input_tokens,
                "output_tokens": 4,
            }
            trace = acct.ideal_trace(case, strategy="dep4")
            result = acct.reference_trace(trace, strategy="dep4")
            self.assertGreater(result.seconds, 0)

    def test_layer_ids_must_not_repeat(self):
        trace = self.ideal()
        trace["calls"][0]["layer_ids"] = [*list(range(40)), 0]
        with self.assertRaisesRegex(acct.TraceError, "layer set"):
            acct.reference_trace(trace, strategy="tp4")

    def test_route_matrix_must_be_square(self):
        trace = self.ideal()
        trace["calls"][0]["route_matrix"] = [[0], [0], [0], [0]]
        with self.assertRaisesRegex(acct.TraceError, "four by four"):
            acct.reference_trace(trace, strategy="tp4")

    def test_dep4_requires_matching_route_rows(self):
        trace = self.ideal("dep4")
        for record in trace["calls"]:
            record["route_matrix"] = [[0] * 4 for _ in range(4)]
        with self.assertRaisesRegex(acct.TraceError, "route row sum"):
            acct.reference_trace(trace, strategy="dep4")

    def test_route_cells_are_bounded_by_the_source_assignments(self):
        trace = self.ideal("dep4")
        records = [record for record in trace["calls"] if record["call"] == 1]
        for record in records:
            record["route_matrix"] = [[10**9] * 4 for _ in range(4)]
        with self.assertRaisesRegex(acct.TraceError, "route row sum"):
            acct.reference_trace(trace, strategy="dep4")

    def test_touched_experts_are_bounded_by_the_assignments(self):
        trace = self.ideal("tp4")
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["touched_experts"] = 40 * 384
        with self.assertRaisesRegex(acct.TraceError, "touched expert"):
            acct.reference_trace(trace, strategy="tp4")

    def test_tep4_accepts_data_dependent_expert_touches(self):
        calls = []
        for rank in range(acct.WORLD_SIZE):
            calls.append(
                {
                    "rank": rank,
                    "call": 0,
                    "phase": "decode",
                    "requests": 4,
                    "rows": 24,
                    "global_rows": 24,
                    "logit_rows": 24,
                    "context_end": 100,
                    "accepted": 20,
                    "proposed": 20,
                    "sparse_pairs": 0,
                    "index_pairs": 0,
                    "engram_unique_rows": 0,
                    "touched_experts": 40 * 37,
                    "layer_ids": list(range(acct.DEEPSEEK_V41_MODEL.num_layers)),
                    "route_matrix": [[0] * 4 for _ in range(4)],
                }
            )
        trace = {
            "binding": binding(
                strategy="tep4",
                effective_strategy="tep4",
                accepted=20,
                proposed=20,
            ),
            "calls": calls,
        }
        result = acct.reference_trace(trace, strategy="tep4")
        self.assertGreater(result.seconds, 0)

    def test_sparse_pairs_are_bounded_by_the_rank_share(self):
        trace = self.ideal("tp4")
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["sparse_pairs"] = record["rows"] * 40 * 640
        with self.assertRaisesRegex(acct.TraceError, "sparse pair"):
            acct.reference_trace(trace, strategy="tp4")

    def test_sparse_pairs_above_the_head_share_are_rejected(self):
        trace = self.ideal("tp4")
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        bound = record["rows"] * 40 * 640 // 4
        record["sparse_pairs"] = bound + 1
        with self.assertRaisesRegex(acct.TraceError, "sparse pair"):
            acct.reference_trace(trace, strategy="tp4")

    def test_index_pairs_above_the_query_share_are_rejected(self):
        trace = self.ideal("tp4")
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        bound = record["rows"] * 8 * (record["context_end"] + 1) // 4
        record["index_pairs"] = bound + 1
        with self.assertRaisesRegex(acct.TraceError, "index pair"):
            acct.reference_trace(trace, strategy="tp4")

    def test_index_pairs_are_bounded_by_the_rank_share(self):
        trace = self.ideal("tp4")
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["index_pairs"] = record["rows"] * 8 * (record["context_end"] + 1)
        with self.assertRaisesRegex(acct.TraceError, "index pair"):
            acct.reference_trace(trace, strategy="tp4")

    def test_engram_rows_are_bounded_by_the_causal_lookup_count(self):
        trace = self.ideal("tp4")
        record = next(call for call in trace["calls"] if call["phase"] == "decode")
        record["engram_unique_rows"] = record["rows"] * 2 * 24 + 1
        with self.assertRaisesRegex(acct.TraceError, "Engram"):
            acct.reference_trace(trace, strategy="tp4")

    def test_effective_strategy_must_match_the_requested_strategy(self):
        trace = self.ideal("dep4")
        trace["binding"]["effective_strategy"] = "tep4"
        with self.assertRaisesRegex(acct.BindingError, "real DEP4"):
            acct.reference_trace(trace, strategy="dep4")

    def test_zero_row_first_prefill_keeps_the_weight_credit(self):
        trace = self.ideal("tp4")
        original = acct.reference_trace(trace, strategy="tp4").seconds
        for record in trace["calls"]:
            record["call"] += 1
        prefill = [record for record in trace["calls"] if record["call"] == 1]
        for record in prefill:
            trace["calls"].append(
                {
                    **record,
                    "call": 0,
                    "requests": 0,
                    "rows": 0,
                    "global_rows": 0,
                    "logit_rows": 0,
                    "context_end": 0,
                    "sparse_pairs": 0,
                    "index_pairs": 0,
                    "engram_unique_rows": 0,
                    "touched_experts": 0,
                    "route_matrix": [[0] * 4 for _ in range(4)],
                }
            )
        with_zero = acct.reference_trace(trace, strategy="tp4").seconds
        self.assertAlmostEqual(original, with_zero, places=12)

    def test_trace_strategy_must_match_the_binding(self):
        trace = self.ideal("tp4")
        with self.assertRaisesRegex(acct.TraceError, "strategy"):
            acct.reference_trace(trace, strategy="tep4")

    def test_decode_counters_are_partitioned_one_time(self):
        trace = self.ideal("tp4")
        records = [call for call in trace["calls"] if call["call"] == 1]
        self.assertEqual(len(records), 4)
        self.assertEqual(len({call["index_pairs"] for call in records}), 1)
        self.assertEqual(len({call["engram_unique_rows"] for call in records}), 1)
        self.assertGreater(records[0]["index_pairs"], 0)

    def test_zero_row_dep4_rank_adds_no_work(self):
        trace = acct.ideal_trace(
            {
                "mode": "dspark5",
                "strategy": "dep4",
                "batch": 2,
                "input_tokens": 100,
                "output_tokens": 8,
            },
            strategy="dep4",
        )
        zero = next(
            call
            for call in trace["calls"]
            if call["rank"] == 3 and call["phase"] == "decode"
        )
        self.assertEqual(zero["rows"], 0)
        groups = acct.call_groups(
            acct.CallPlan(
                rank=3,
                phase="decode",
                requests=0,
                rows=0,
                global_rows=8,
                logit_rows=0,
                context_end=107,
                accepted=0,
                proposed=0,
                sparse_pairs=0,
                index_pairs=0,
                engram_unique_rows=0,
                touched_experts=0,
                route_matrix=((0, 0, 0, 0),) * 4,
            ),
            strategy="dep4",
        )
        nonzero = acct.call_groups(
            next(
                acct._decode_plans(
                    {
                        "mode": "dspark5",
                        "batch": 2,
                        "input_tokens": 100,
                        "output_tokens": 8,
                    },
                    acct.DEEPSEEK_V41_MODEL,
                    "dep4",
                )
            )[0],
            strategy="dep4",
        )
        self.assertEqual(
            [group.name for group in groups],
            [group.name for group in nonzero],
        )
        self.assertEqual(
            sum(
                group.work.hbm_bytes
                + group.work.fp8_flops
                + group.work.bf16_flops
                + group.work.fp32_flops
                + group.work.nvlink_bytes
                + group.work.host_bytes
                for group in groups
            ),
            0,
        )
        result = acct.reference_trace(trace, strategy="dep4")
        self.assertGreater(result.seconds, 0)


class Binding(unittest.TestCase):
    def test_build_and_validate_binding(self):
        value = binding()
        self.assertEqual(value["version"], acct.VERSION)
        self.assertEqual(value["rate_profile"], acct.rate_profile_digest())
        self.assertIs(acct.validate_binding(value), value)

    def test_incomplete_or_changed_identity_is_rejected(self):
        value = binding()
        for field in ("source", "gpu_uuids", "accepted"):
            changed = dict(value)
            changed.pop(field)
            with self.assertRaises(acct.BindingError):
                acct.validate_binding(changed)
        changed = dict(value, rate_profile="other")
        with self.assertRaises(acct.BindingError):
            acct.validate_binding(changed)
        with self.assertRaises(acct.BindingError):
            acct.validate_binding(value, expected=dict(value, source="other"))
        with self.assertRaises(acct.BindingError):
            acct.build_binding()
        with self.assertRaises(acct.BindingError):
            acct.build_binding(
                **{
                    **{name: value[name] for name in acct.BINDING_FIELDS},
                    "gpu_uuids": ["same", "same", "same", "same"],
                }
            )


class ReportedCases(unittest.TestCase):
    def test_reported_8192_reference_values(self):
        case = {
            "mode": "dspark5",
            "batch": 4,
            "input_tokens": 400000,
            "output_tokens": 8192,
        }
        for strategy, expected in (
            ("tp4", 21.230),
            ("tep4", 20.831),
            ("dep4", 14.427),
        ):
            with self.subTest(strategy=strategy):
                result = acct.reference_case(
                    {**case, "strategy": strategy},
                    strategy=strategy,
                    communication_model="wire",
                )
                self.assertAlmostEqual(result.seconds, expected, delta=0.002)


if __name__ == "__main__":
    unittest.main()
