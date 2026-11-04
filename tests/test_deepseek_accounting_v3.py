import copy
import unittest
from dataclasses import replace
from itertools import pairwise
from unittest.mock import patch

from development.serving import deepseek_accounting_v2 as v2
from development.serving import deepseek_accounting_v3 as acct

MODEL = replace(
    acct.DEEPSEEK_V41_MODEL,
    num_layers=2,
    compress_ratios=(1, 1),
    kv_source_layers=(0,),
    index_source_layers=(0, 1),
    candidate_source_layer=0,
    candidate_positions=8,
    index_topk=4,
    engram_layers=(0,),
)


def case(strategy="tp4", **values):
    return dict(
        mode="dspark5",
        strategy=strategy,
        batch=4,
        input_tokens=20,
        output_tokens=8,
        **values,
    )


def trace(strategy="tp4", **values):
    value = case(strategy)
    value.update(values)
    return acct.ideal_trace(value, model=MODEL)


class ValidRowsAndPlacement(unittest.TestCase):
    def test_score_bytes_are_causal_and_candidate_limited(self):
        layers = v2.layer_table(MODEL)
        rows = [{"start": 0, "count": 20}]
        self.assertEqual(acct.valid_index_pairs(rows, layers[0], MODEL), 210)
        self.assertEqual(acct.valid_index_pairs(rows, layers[1], MODEL), 132)
        records = trace()["calls"][:4]
        heads, _ = acct.index_alternatives(records, layers[1], MODEL, "tp4", [True] * 4)
        # Four requests; 132 useful FP32 scores each. Ring all-reduce sends 3/2.
        self.assertEqual(heads[0][-1].work.nvlink_bytes, 4 * 132 * 4 * 1.5)

    def test_query_placement_keeps_gather_and_full_head_compute(self):
        records = trace(input_tokens=400000)["calls"][4:8]
        layer = v2.layer_table(MODEL)[0]
        heads, queries = acct.index_alternatives(
            records, layer, MODEL, "tp4", [True] * 4
        )
        self.assertLess(
            v2.critical_seconds(queries).seconds, v2.critical_seconds(heads).seconds
        )
        self.assertGreater(queries[0][1].work.collective_seconds, 0)
        self.assertGreater(queries[0][-1].work.collective_seconds, 0)
        self.assertAlmostEqual(
            sum(groups[2].work.bf16_flops for groups in queries),
            sum(groups[1].work.bf16_flops for groups in heads),
        )
        self.assertGreater(queries[0][2].work.hbm_bytes, 0)

    def test_engram_compares_complete_host_read_with_sharded_reduction(self):
        records = trace()["calls"][4:8]
        full, sharded = acct.engram_alternatives(
            records, v2.layer_table(MODEL)[0], MODEL
        )
        self.assertEqual(full[0][0].work.host_bytes, 24 * 264)
        self.assertEqual(sharded[0][0].work.host_bytes, 6 * 264)
        self.assertGreater(sharded[0][1].work.collective_seconds, 0)
        self.assertLess(
            v2.critical_seconds(full).seconds, v2.critical_seconds(sharded).seconds
        )

    def test_history_reads_reuse_across_queries_and_heads(self):
        layer = v2.layer_table(MODEL)[0]
        one = acct.historical_reads([dict(start=100, count=1)], layer, MODEL)
        six = acct.historical_reads([dict(start=100, count=6)], layer, MODEL)
        # One historical window union, four selected compressed rows, and 100 keys.
        self.assertEqual(one, ((100 + 3) * 512 * 2, 100 * 128 * 2))
        self.assertEqual(six, (100 * 512 * 2, 100 * 128 * 2))
        self.assertLess(six[0] + six[1], 6 * (one[0] + one[1]))
        candidate = v2.layer_table(MODEL)[1]
        self.assertEqual(
            acct.historical_reads([dict(start=100, count=1)], candidate, MODEL)[1],
            7 * 128 * 2,
        )


class ReviewRegressions(unittest.TestCase):
    def test_query_floor_balances_single_row_requests(self):
        records = trace(input_tokens=800000)["calls"][4:8]
        for record in records:
            record["request_rows"] = [
                dict(id=i, start=800000, count=1) for i in range(64)
            ]
        layer = v2.layer_table(MODEL)[0]
        _, query = acct.index_alternatives(records, layer, MODEL, "tp4", [False] * 4)
        expected = 2 * 64 * 800001 * 32 * 128 / 4
        self.assertEqual(
            [groups[2].work.bf16_flops for groups in query], [expected] * 4
        )

    def test_compressed_key_projection_uses_each_request_position(self):
        model = replace(MODEL, compress_ratios=(2, 1))
        layer = v2.layer_table(model)[0]
        records = trace()["calls"][4:8]
        for record in records:
            record["request_rows"] = [
                dict(id=i, start=400000, count=1) for i in range(4)
            ]
        heads, _ = acct.index_alternatives(records, layer, model, "tp4", [False] * 4)
        one = acct._one_layer(model, layer)
        self.assertEqual(
            heads[0][0].work, acct._index_projection(4, 0, one, "tp4", False)
        )
        for record in records:
            record["request_rows"] = [dict(id=1, start=400001, count=1)]
        heads, _ = acct.index_alternatives(records, layer, model, "tp4", [False] * 4)
        self.assertEqual(
            heads[0][0].work, acct._index_projection(1, 1, one, "tp4", False)
        )
        delta = (
            acct._index_projection(1, 1, one, "tp4", False).bf16_flops
            - acct._index_projection(1, 0, one, "tp4", False).bf16_flops
        )
        self.assertEqual(delta, 2 * 128 * 512)

    def test_dispatch_constrains_touched_experts(self):
        value = trace("dep4", output_tokens=2)
        for record in value["calls"][4:8]:
            rank = record["rank"]
            record["expert_assignments"][0] = [6 * int(i == rank) for i in range(4)]
            record["dispatch_rows"][0] = [int(i == rank) for i in range(4)]
            record["touched_experts"][0] = 2
        with self.assertRaisesRegex(acct.TraceError, "touched expert"):
            acct.validate_trace(value, strategy="dep4", model=MODEL)

    def test_index_choices_minimize_final_overlap_price(self):
        records = trace()["calls"][4:8]
        options = [
            acct.index_alternatives(records, layer, MODEL, "tp4", [True] * 4)
            for layer in v2.layer_table(MODEL)
        ]
        common = [[acct.Group("common", acct.Work(fp32_flops=1e10))] for _ in range(4)]
        chosen = acct.choose_index_groups(
            copy.deepcopy(common), options, records, MODEL, "tp4"
        )
        prices = []
        for first in range(2):
            for second in range(2):
                groups = [
                    common[rank] + options[0][first][rank] + options[1][second][rank]
                    for rank in range(4)
                ]
                if first == 1 and second == 0:
                    for rank in groups:
                        rank.extend(acct.candidate_handoff(records, MODEL))
                prices.append(acct.overlap_seconds(groups, 24))
        self.assertAlmostEqual(acct.overlap_seconds(chosen, 24), min(prices))
        self.assertGreater(
            acct.candidate_handoff(records, MODEL)[0].work.collective_seconds, 0
        )


class Routing(unittest.TestCase):
    def test_dep4_dispatch_deduplicates_and_combine_uses_fp32(self):
        records = trace("dep4")["calls"][4:8]
        for record in records:
            record["dispatch_rows"] = [[2, 2, 2, 2], [2, 2, 2, 2]]
        dispatch = acct._route_work(records, 0, 0, MODEL, 2)
        combine = acct._route_work(records, 0, 0, MODEL, 4)
        self.assertEqual(dispatch.nvlink_bytes, 6 * 5120 * 2)
        self.assertEqual(combine.nvlink_bytes, 6 * 5120 * 4)
        self.assertGreater(combine.collective_seconds, 0)

    def test_per_layer_hotspot_and_zero_request_expert_rank(self):
        value = trace("dep4", batch=3)
        records = value["calls"][4:8]
        for record in records:
            count = sum(row["count"] for row in record["request_rows"])
            record["expert_assignments"][0] = [0, 0, 0, count * 6]
            record["dispatch_rows"][0] = [0, 0, 0, count]
            record["touched_experts"][0] = 6 if record["rank"] == 3 else 0
        acct.validate_trace(value, strategy="dep4", model=MODEL)
        self.assertEqual(records[3]["request_rows"], [])
        self.assertEqual(acct.received_assignments(records, 3, 0, "dep4"), 108)
        groups = acct.call_groups(records, MODEL, "dep4")
        work = next(
            group.work for group in groups[3] if group.name == "0.routed_experts"
        )
        self.assertEqual(work.fp8_flops, 108 * 6 * 5120 * 2304)
        self.assertGreater(work.hbm_bytes, 0)
        self.assertNotEqual(acct.received_assignments(records, 3, 1, "dep4"), 108)

    def test_tep4_uses_received_column_and_tp4_uses_all_columns(self):
        records = trace("tep4")["calls"][4:8]
        for record in records:
            record["expert_assignments"][0] = [144, 0, 0, 0]
        self.assertEqual(acct.received_assignments(records, 0, 0, "tep4"), 144)
        self.assertEqual(acct.received_assignments(records, 1, 0, "tep4"), 0)
        self.assertEqual(acct.received_assignments(records, 1, 0, "tp4"), 144)

    def test_dispatch_cannot_exceed_distinct_tokens(self):
        value = trace("dep4")
        value["calls"][4]["dispatch_rows"][0] = [9, 9, 9, 9]
        with self.assertRaisesRegex(acct.TraceError, "dispatch rows"):
            acct.validate_trace(value, strategy="dep4", model=MODEL)


class ManifestValidation(unittest.TestCase):
    def test_renumbered_duplicate_decode_fails(self):
        value = trace(mode="ordinary", output_tokens=2)
        duplicate = copy.deepcopy(value["calls"][4:8])
        for record in duplicate:
            record["call"] = 2
        value["calls"].extend(duplicate)
        with self.assertRaisesRegex(acct.TraceError, "recomputed"):
            acct.reference_trace(value, strategy="tp4", model=MODEL)

    def test_manifest_and_http_usage_are_bound(self):
        value = trace()
        for request in value["manifest"]["requests"]:
            request["response_id"] = f"chatcmpl_123_abc_{request['id']}"
        value["binding"]["manifest_sha256"] = acct.digest(value["manifest"])
        acct.validate_trace(value, strategy="tp4", model=MODEL)
        value["manifest"]["requests"][0]["output_tokens"] += 1
        with self.assertRaisesRegex(acct.BindingError, "manifest digest"):
            acct.validate_trace(value, strategy="tp4", model=MODEL)
        value["binding"]["manifest_sha256"] = acct.digest(value["manifest"])
        with self.assertRaisesRegex(acct.TraceError, "output differs"):
            acct.validate_trace(value, strategy="tp4", model=MODEL)

    def test_missing_prefill_and_noncontiguous_decode_fail(self):
        for operation in ("prefill", "position"):
            value = trace()
            if operation == "prefill":
                value["calls"] = value["calls"][4:]
            else:
                for record in value["calls"][4:8]:
                    record["request_rows"][0]["start"] += 1
            with self.assertRaises(acct.TraceError):
                acct.validate_trace(value, strategy="tp4", model=MODEL)

    def test_chunked_prefill_reads_output_head_at_first_use(self):
        value = trace(output_tokens=1)
        last = copy.deepcopy(value["calls"])
        for record in value["calls"]:
            for row in record["request_rows"]:
                row["count"], row["logit_rows"] = 10, 0
            record["expert_assignments"] = [[60, 60, 60, 60]] * MODEL.num_layers
            record["touched_experts"] = [120] * MODEL.num_layers
        for record in last:
            record["call"] = 1
            for row in record["request_rows"]:
                row["start"], row["count"] = 10, 10
            record["expert_assignments"] = [[60, 60, 60, 60]] * MODEL.num_layers
            record["touched_experts"] = [120] * MODEL.num_layers
        value["calls"].extend(last)
        original = acct.call_groups
        with patch.object(acct, "call_groups", wraps=original) as groups:
            result = acct.reference_trace(value, strategy="tp4", model=MODEL)
        self.assertGreater(result.seconds, 0)
        self.assertEqual(groups.call_args_list[0].kwargs["weights"], [True] * 4)
        self.assertEqual(groups.call_args_list[1].kwargs["weights"], [False] * 4)
        self.assertEqual(groups.call_args_list[1].kwargs["head_weights"], [True] * 4)
        head = next(
            g
            for g in original(
                last, MODEL, "tp4", weights=[False] * 4, head_weights=[True] * 4
            )[0]
            if g.name == "output_head"
        )
        self.assertEqual(
            head.work.hbm_bytes, 129280 * 5120 + 4 * (5120 + 129280 // 4) * 4
        )
        # Duplicate a complete chunk with a new call ID: it still cannot pass.
        extra = copy.deepcopy(last)
        for record in extra:
            record["call"] = 2
        value["calls"].extend(extra)
        with self.assertRaises(acct.TraceError):
            acct.validate_trace(value, strategy="tp4", model=MODEL)


class HistoryAndCausalPartition(unittest.TestCase):
    def test_shared_kv_is_read_once_when_selection_fits_cache(self):
        records = trace(output_tokens=2)["calls"][4:8]
        groups = acct.call_groups(records, MODEL, "tp4")
        first = next(
            g.work.hbm_bytes for g in groups[0] if g.name == "0.sparse_attention"
        )
        next_layer = next(
            g.work.hbm_bytes for g in groups[0] if g.name == "1.sparse_attention"
        )
        self.assertEqual(first, 4 * (20 + 3) * 512 * 2)
        self.assertEqual(next_layer, 4 * 20 * 512 * 2)

    def test_reuse_above_cache_capacity_still_reads_hbm(self):
        capacity = 128 * 1024 * 1024
        group = acct.HistoryGroup(
            "score", acct.Work(hbm_bytes=2 * capacity), reusable_hbm_bytes=2 * capacity
        )
        first, seen = acct._credit_index((group,), 0)
        second, _ = acct._credit_index((group,), seen)
        self.assertEqual(first[0].work.hbm_bytes, 2 * capacity)
        self.assertEqual(second[0].work.hbm_bytes, capacity)

    def test_query_floor_is_below_integer_causal_partitions(self):
        records = trace(output_tokens=1)["calls"][:4]
        for record in records:
            record["request_rows"] = [dict(id=1, start=0, count=20)]
        layer = v2.layer_table(MODEL)[0]
        _, query = acct.index_alternatives(records, layer, MODEL, "tp4", [False] * 4)
        floor = query[0][2].work.bf16_flops / (2 * 32 * 128)
        self.assertEqual(floor, 210 / 4)
        for cuts in ((0, 5, 10, 15, 20), (0, 10, 14, 17, 20)):
            counts = [sum(range(a + 1, b + 1)) for a, b in pairwise(cuts)]
            self.assertLessEqual(floor, max(counts))


class KeyWeightFirstUse(unittest.TestCase):
    def test_short_first_chunk_keeps_later_key_weight_credit(self):
        model = replace(MODEL, compress_ratios=(2, 1))
        value = acct.ideal_trace(case(), model=model)
        first = copy.deepcopy(value["calls"][:4])
        for record in first:
            for row in record["request_rows"]:
                row["count"], row["logit_rows"] = 1, 0
            record["expert_assignments"] = [[6] * 4] * model.num_layers
            record["touched_experts"] = [24] * model.num_layers
        for record in value["calls"]:
            record["call"] += 1
            if record["phase"] == "prefill":
                for row in record["request_rows"]:
                    row["start"], row["count"] = 1, 19
                record["expert_assignments"] = [[114] * 4] * model.num_layers
        value["calls"] = first + value["calls"]
        with patch.object(acct, "call_groups", wraps=acct.call_groups) as calls:
            acct.reference_trace(value, strategy="tp4", model=model)
        self.assertEqual(
            calls.call_args_list[0].kwargs["key_weights"], [[False, False]] * 4
        )
        self.assertEqual(
            calls.call_args_list[1].kwargs["key_weights"], [[True, False]] * 4
        )
        self.assertEqual(calls.call_args_list[1].kwargs["weights"], [False] * 4)
        first_credit = calls.call_args_list[0].kwargs["touched_weights"]
        later_credit = calls.call_args_list[1].kwargs["touched_weights"]
        for rank in range(4):
            for layer in range(model.num_layers):
                self.assertLessEqual(
                    first_credit[rank][layer], first[rank]["touched_experts"][layer]
                )
                self.assertEqual(
                    first_credit[rank][layer] + later_credit[rank][layer],
                    max(
                        first[rank]["touched_experts"][layer],
                        value["calls"][4 + rank]["touched_experts"][layer],
                    ),
                )

        one = acct._one_layer(model, v2.layer_table(model)[0])
        work = acct._index_projection(76, 40, one, "tp4", False, True)
        without = acct._index_projection(76, 40, one, "tp4", False, False)
        self.assertEqual(work.hbm_bytes - without.hbm_bytes, 128 * 512 * 2)


class OverlapAndVersion(unittest.TestCase):
    def test_free_query_ownership_does_not_add_average_work_to_hot_rank(self):
        ranks = []
        for rank in range(4):
            ranks.append(
                [
                    acct.Group(
                        "fixed", acct.Work(fp8_flops=4.5e15 if rank == 0 else 0)
                    ),
                    acct.PooledHistoryGroup("query", acct.Work(bf16_flops=1.125e15)),
                ]
            )
        self.assertEqual(acct.overlap_seconds(ranks, 2), 1)

    def test_two_microbatch_bound_single_row_and_nonzero_communication(self):
        groups = [
            [
                acct.Group("compute", acct.Work(fp32_flops=75e12)),
                acct.Group("communication", acct.Work(collective_seconds=1)),
            ]
        ] * 4
        serial = v2.critical_seconds(groups).seconds
        self.assertEqual(serial, 2)
        self.assertEqual(acct.overlap_seconds(groups, 1), serial)
        bound = acct.overlap_seconds(groups, 2)
        self.assertEqual(bound, 1)
        self.assertLessEqual(bound, serial)
        groups[0][1] = acct.Group("communication", acct.Work(collective_seconds=3))
        self.assertGreaterEqual(acct.overlap_seconds(groups, 2), 3)

    def test_profile_remains_an_estimate_with_wire_floor(self):
        with patch.object(v2, "collective_latency_seconds", return_value=20e-6):
            acct._collect.cache_clear()
            small = acct._collect("all_reduce", "fp32", 20480)
            large = acct._collect("all_reduce", "fp32", 100000000)
        acct._collect.cache_clear()
        self.assertEqual(small.collective_seconds, 20e-6)
        self.assertEqual(large.collective_seconds, 150000000 / 900e9)

    def test_new_identity_and_legacy_reproducibility(self):
        value = case()
        old = acct.reference_case(value, model=MODEL, version=v2.VERSION)
        self.assertEqual(old, v2.reference_case(value, model=MODEL))
        new = acct.reference_case(value, model=MODEL)
        self.assertEqual(new.version, "deepseek-http-reference-v3")
        self.assertNotEqual(new.seconds, old.seconds)
        self.assertEqual(
            acct.reference_trace(trace(), strategy="tp4", model=MODEL).seconds,
            new.seconds,
        )


if __name__ == "__main__":
    unittest.main()
