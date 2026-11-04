import copy
import unittest
from dataclasses import replace
from unittest.mock import patch

from development.serving import deepseek_accounting as v31
from development.serving import deepseek_accounting_v2 as v2
from development.serving import deepseek_accounting_v4 as acct

MODEL = replace(
    v2.DEEPSEEK_V41_MODEL,
    num_layers=1,
    compress_ratios=(0,),
    kv_source_layers=(),
    index_source_layers=(),
    engram_layers=(),
)


def routed_trace(strategy, *, outputs=2, batch=4, model=MODEL):
    case = dict(
        mode="dspark5",
        strategy=strategy,
        batch=batch,
        input_tokens=20,
        output_tokens=outputs,
    )
    trace = v31.ideal_trace(case, model=model)
    for records in _calls(trace):
        if records[0]["phase"] == "prefill":
            continue
        _literal_routes(records, strategy, model)
    return trace


def _calls(trace):
    calls = {}
    for record in trace["calls"]:
        calls.setdefault(record["call"], []).append(record)
    return [calls[key] for key in sorted(calls)]


def _literal_routes(records, strategy, model):
    total = sum(
        row["count"]
        for record in (records if strategy == "dep4" else records[:1])
        for row in record["request_rows"]
    )
    for record in records:
        count = sum(row["count"] for row in record["request_rows"])
        record["expert_assignments"] = [
            [count * 6, 0, 0, 0] for _ in range(model.num_layers)
        ]
        record["dispatch_rows"] = [
            [count, 0, 0, 0] if strategy == "dep4" else [0, 0, 0, 0]
            for _ in range(model.num_layers)
        ]
        touched = 6 if total and (strategy == "tp4" or record["rank"] == 0) else 0
        record["touched_experts"] = [touched] * model.num_layers


def split_decode(trace, strategy, model=MODEL):
    result = copy.deepcopy(trace)
    result["calls"] = []
    next_call = 0
    for records in _calls(trace):
        if records[0]["phase"] == "prefill":
            for record in copy.deepcopy(records):
                record["call"] = next_call
                result["calls"].append(record)
            next_call += 1
            continue
        rows = [
            row
            for record in (records if strategy == "dep4" else records[:1])
            for row in record["request_rows"]
        ]
        for row in rows:
            for offset in range(row["count"]):
                single = dict(
                    row,
                    start=row["start"] + offset,
                    count=1,
                    accepted=0,
                    proposed=0,
                    logit_rows=1,
                )
                pieces = []
                owner = next(
                    request["owner_rank"]
                    for request in result["manifest"]["requests"]
                    if request["id"] == row["id"]
                )
                for rank in range(4):
                    record = copy.deepcopy(records[rank])
                    record["call"] = next_call
                    record["request_rows"] = (
                        [copy.deepcopy(single)]
                        if strategy != "dep4" or rank == owner
                        else []
                    )
                    pieces.append(record)
                _literal_routes(pieces, strategy, model)
                result["calls"].extend(pieces)
                next_call += 1
    result["binding"]["accepted"] = 0
    result["binding"]["proposed"] = 0
    return result


class WholeHttpFloor(unittest.TestCase):
    def test_call_splitting_reordering_and_renumbering_cannot_change_resources(self):
        for strategy in ("tp4", "tep4", "dep4"):
            with self.subTest(strategy=strategy):
                trace = routed_trace(strategy, outputs=7)
                split = split_decode(trace, strategy)
                for record in split["calls"]:
                    record["call"] = record["call"] * 11 + 17
                split["calls"].reverse()
                first = acct.reference_trace(trace, strategy=strategy, model=MODEL)
                second = acct.reference_trace(split, strategy=strategy, model=MODEL)
                self.assertEqual(first.seconds, second.seconds)
                self.assertEqual(first.ranks, second.ranks)
                self.assertEqual(first.expert_union_lower, second.expert_union_lower)
                self.assertEqual(
                    first.pooled_wire_sent_bytes, second.pooled_wire_sent_bytes
                )
                self.assertNotEqual(first.calls, second.calls)
                self.assertNotEqual(first.accepted, second.accepted)

    def test_fp8_k32_and_replicated_matrices_match_literal_work(self):
        trace = routed_trace("tp4", outputs=2, batch=1)
        result = acct.reference_trace(trace, strategy="tp4", model=MODEL)
        rows, logits = 21, 2
        fp8 = (
            2 * rows * 1280 * 5120
            + 2 * rows * (64 * 512 // 4) * 1280
            + 2 * rows * 512 * 5120
            + 2 * rows * 5120 * (8 * 1024 // 4)
            + 6 * rows * 2304 * 5120
            + 6 * (6 * rows) * 5120 * 2304 // 4
        )
        fp32 = (
            4 * rows * 24 * 20480
            + 2 * rows * 384 * 5120
            + 2 * logits * (129280 // 4) * 5120
            + fp8 / 32
        )
        bf16 = 2 * rows * (8192 // 4) * 4096 + 4 * (21 * 22 // 2) * (64 // 4) * 512
        for rank in result.ranks:
            self.assertEqual(rank.fp8_flops, fp8)
            self.assertEqual(rank.fp32_flops, fp32)
            self.assertEqual(rank.bf16_flops, bf16)
            self.assertGreater(rank.weight_union_lower_bytes, 0)
            self.assertEqual(rank.hbm_bytes, 0)

    def test_direct_dep4_diagnostic_accumulates_directions_before_maximum(self):
        dispatch = [[[0] * 4 for _ in range(4)] for _ in range(2)]
        dispatch[0][0][1] = 10
        dispatch[1][1][0] = 10
        dispatch[0][2][2] = 100000
        sent, received = acct._dep4_wire(dispatch, MODEL)
        self.assertEqual(sent, [60 * 5120, 60 * 5120, 0, 0])
        self.assertEqual(received, sent)
        self.assertLess(max(max(sent), max(received)), 80 * 5120)
        result = acct.reference_trace(
            routed_trace("dep4"), strategy="dep4", model=MODEL
        )
        self.assertTrue(any(row.direct_protocol_sent_bytes for row in result.ranks))
        self.assertTrue(all(row.wire_sent_bytes == 0 for row in result.ranks))
        self.assertTrue(all(row.wire_received_bytes == 0 for row in result.ranks))

    def test_pooled_moe_wire_uses_final_bf16_and_excludes_unproved_collectives(self):
        self.assertEqual(acct._pooled_wire(1, MODEL, "tep4"), 3 * 5120 * 2)
        self.assertEqual(
            acct._pooled_wire(1, MODEL, "tp4"),
            3 * 5120 * 2 + 3 * 6 * 2304 * 2,
        )
        self.assertEqual(acct._pooled_wire(1, MODEL, "dep4"), 0)

    def test_empty_dep4_owner_still_computes_received_experts(self):
        case = dict(
            mode="dspark5",
            strategy="dep4",
            batch=3,
            input_tokens=20,
            output_tokens=7,
        )
        trace = v31.ideal_trace(case, model=MODEL)
        result = acct.reference_trace(trace, strategy="dep4", model=MODEL)
        self.assertGreater(result.ranks[3].fp8_flops, 0)
        self.assertGreater(result.expert_union_lower[3][0], 0)

    def test_missing_or_duplicated_accepted_rows_and_ranks_are_rejected(self):
        trace = routed_trace("tp4")
        missing = copy.deepcopy(trace)
        missing["calls"].pop()
        with self.assertRaises(v31.TraceError):
            acct.reference_trace(missing, strategy="tp4", model=MODEL)
        repeated = copy.deepcopy(trace)
        for record in copy.deepcopy(trace["calls"][4:]):
            record["call"] += 100
            repeated["calls"].append(record)
        with self.assertRaises(v31.TraceError):
            acct.reference_trace(repeated, strategy="tp4", model=MODEL)

    def test_empirical_profile_changes_do_not_change_strict_value_or_identity(self):
        trace = routed_trace("tp4")
        first = acct.reference_trace(trace, strategy="tp4", model=MODEL)
        changed = {
            key: tuple((size, latency * 100) for size, latency in points)
            for key, points in v2.COLLECTIVE_LATENCY_PROFILE_US.items()
        }
        with patch.dict(v2.COLLECTIVE_LATENCY_PROFILE_US, changed, clear=True):
            source = copy.deepcopy(trace)
            source["binding"]["rate_profile"] = v31.rate_profile_digest()
            second = acct.reference_trace(source, strategy="tp4", model=MODEL)
        self.assertEqual(first.seconds, second.seconds)
        self.assertEqual(first.rate_profile, second.rate_profile)
        self.assertEqual(first.assumptions_sha256, second.assumptions_sha256)
        with patch.object(v2, "collective_latency_seconds", side_effect=AssertionError):
            self.assertEqual(
                acct.reference_trace(trace, strategy="tp4", model=MODEL).seconds,
                first.seconds,
            )

    def test_rate_and_assumption_identity_changes_are_visible(self):
        original = acct.rate_profile_digest(MODEL)
        with patch.object(acct, "RATES", {**acct.RATES, "fp32_flops": 1}):
            self.assertNotEqual(acct.rate_profile_digest(MODEL), original)
        with patch.object(
            acct, "ASSUMPTIONS", {**acct.ASSUMPTIONS, "host_credit": "changed"}
        ):
            self.assertNotEqual(acct.rate_profile_digest(MODEL), original)

    def test_legacy_reference_remains_group_dependent_and_unchanged(self):
        trace = routed_trace("tp4", outputs=2)
        trace["manifest"]["mode"] = "ordinary"
        trace["binding"]["manifest_sha256"] = v31.digest(trace["manifest"])
        split = split_decode(trace, "tp4")
        first = v31.reference_trace(trace, strategy="tp4", model=MODEL)
        second = v31.reference_trace(split, strategy="tp4", model=MODEL)
        self.assertEqual(first.version, "deepseek-http-reference-v3.1")
        self.assertGreater(second.seconds, first.seconds)
        self.assertEqual(
            acct.reference_trace(trace, strategy="tp4", model=MODEL).seconds,
            acct.reference_trace(split, strategy="tp4", model=MODEL).seconds,
        )

    def test_index_queries_without_a_reachable_key_receive_no_projection_credit(self):
        model = replace(
            MODEL,
            num_layers=2,
            compress_ratios=(0, 2),
            kv_source_layers=(1,),
            index_source_layers=(1,),
        )
        layer = v2.layer_table(model)[1]
        empty, weights, score = acct._index([1], layer, model, "tp4")
        self.assertEqual(
            (empty.fp8_flops, empty.bf16_flops, weights, score), (0, 0, 0, 0)
        )
        work, weights, score = acct._index([2], layer, model, "tp4")
        self.assertEqual(work.fp8_flops, 2 * (32 * 128 // 4) * 1280)
        self.assertEqual(work.bf16_flops, 2 * (32 // 4) * 5120 + 2 * 128 * 512)
        self.assertEqual(score, 2 * 32 * 128)
        self.assertGreater(weights, 0)

    def test_candidate_latest_block_counts_only_its_valid_positions(self):
        counts = [min(index, 16) for index in range(17)] + list(range(9, 17)) + [9]
        total = 0
        self.assertEqual(acct.candidate_pairs(0, 1, 16), 0)
        for length, expected in enumerate(counts[1:], start=1):
            actual = acct.candidate_pairs(length, 1, 16)
            self.assertEqual(actual - total, expected)
            total = actual
        boundary = acct.candidate_pairs(16385, 1, 16384)
        previous = acct.candidate_pairs(16384, 1, 16384)
        self.assertEqual(boundary - previous, 16377)
        self.assertEqual((16384 - 16377) * 2 * 32 * 128, 57344)
        self.assertEqual(
            acct.candidate_pairs(34, 2, 16) - acct.candidate_pairs(33, 2, 16),
            9,
        )
        self.assertEqual(
            acct.candidate_pairs(35, 2, 16) - acct.candidate_pairs(34, 2, 16),
            9,
        )

    def test_full_model_v31_literal_counterexample_numbers_remain_reproducible(self):
        model = v2.DEEPSEEK_V41_MODEL
        trace = routed_trace("tp4", outputs=2, model=model)
        trace["manifest"]["mode"] = "ordinary"
        trace["binding"]["manifest_sha256"] = v31.digest(trace["manifest"])
        split = split_decode(trace, "tp4", model)
        self.assertAlmostEqual(
            v31.reference_trace(trace, strategy="tp4").seconds,
            0.009802319330475422,
            places=15,
        )
        self.assertAlmostEqual(
            v31.reference_trace(split, strategy="tp4").seconds,
            0.01924492533442352,
            places=15,
        )

    def test_case_scale_is_separate_from_runtime_evidence(self):
        case = dict(
            mode="dspark5",
            strategy="tep4",
            batch=4,
            input_tokens=20,
            output_tokens=7,
        )
        result = acct.reference_case(case, model=MODEL)
        self.assertEqual(result.version, acct.VERSION)
        self.assertEqual(result.acceptance, "ideal-routing-resource-floor")


if __name__ == "__main__":
    unittest.main()
