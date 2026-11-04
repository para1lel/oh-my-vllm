import unittest
from dataclasses import replace

from development.serving import deepseek_accounting as acct
from development.serving import deepseek_accounting_v2 as v2
from development.serving import deepseek_accounting_v3 as v3
from tests.test_deepseek_accounting_v3 import MODEL


class ScaleAccumulation(unittest.TestCase):
    def test_fp8_and_fp4_count_one_fma_per_output_and_k32_group(self):
        for storage in ("fp8", "fp4"):
            for rows in (1, 7, 1024):
                with self.subTest(storage=storage, rows=rows):
                    work = v2.projection_work(rows, 128, 96, storage=storage)
                    priced = acct.scale_accumulation_work(work)
                    self.assertEqual(priced.fp32_flops, 2 * rows * 128 * 3)
                    self.assertEqual(priced.fp8_flops, work.fp8_flops)
                    self.assertEqual(priced.hbm_bytes, work.hbm_bytes)

    def test_output_and_input_shards_use_local_output_and_k(self):
        for axis in (0, 1):
            work = v2.projection_work(
                7, 512, 512, storage="fp8", shard_axis=axis, shards=4
            )
            priced = acct.scale_accumulation_work(work)
            outputs, k_groups = (128, 16) if axis == 0 else (512, 4)
            self.assertEqual(priced.fp32_flops, 2 * 7 * outputs * k_groups)

    def test_bf16_and_existing_fp32_work_are_not_scaled(self):
        for storage in ("bf16", "fp32"):
            work = v2.projection_work(7, 128, 96, storage=storage)
            self.assertIs(acct.scale_accumulation_work(work), work)
        work = acct.Work(fp8_flops=4.5e15, fp32_flops=75e12)
        group = acct.Group("projection", work)
        self.assertEqual(v2.group_seconds(group), 1)
        self.assertEqual(acct.group_seconds(group), 2.875)
        self.assertEqual(acct.critical_seconds([[group]] * 4).seconds, 2.875)
        self.assertEqual(acct._resource_times([group])[2], 2.875)
        self.assertEqual(acct.overlap_seconds([[group]] * 4, 2), 2.875)

    def test_routed_work_uses_received_assignments_and_local_columns(self):
        for strategy in acct.STRATEGIES:
            for phase in ("prefill", "decode"):
                case = dict(
                    mode="dspark5",
                    strategy=strategy,
                    batch=3,
                    input_tokens=20,
                    output_tokens=8,
                )
                trace = acct.ideal_trace(case, model=MODEL)
                records = [
                    r
                    for r in trace["calls"]
                    if r["call"] == (0 if phase == "prefill" else 1)
                ]
                groups = acct.call_groups(records, MODEL, strategy)
                for rank in range(4):
                    assignments = acct.received_assignments(records, rank, 0, strategy)
                    routed = next(
                        g for g in groups[rank] if g.name == "0.routed_experts"
                    )
                    output_shards = 4 if strategy == "tp4" else 1
                    expected = 6 * assignments * 5120 * 2304 / (32 * output_shards)
                    self.assertEqual(
                        acct.scale_accumulation_work(routed.work).fp32_flops, expected
                    )
                if strategy == "dep4":
                    self.assertEqual(records[3]["request_rows"], [])
                    self.assertGreater(
                        acct.received_assignments(records, 3, 0, strategy), 0
                    )

    def test_dense_and_index_projections_enter_the_fp32_resource(self):
        for strategy in acct.STRATEGIES:
            case = dict(
                mode="dspark5",
                strategy=strategy,
                batch=4,
                input_tokens=20,
                output_tokens=8,
            )
            records = acct.ideal_trace(case, model=MODEL)["calls"][:4]
            ranks = acct.call_groups(records, MODEL, strategy)
            names = (
                "0.attention_input",
                "0.attention_output",
                "0.shared_expert",
                "0.engram_projection",
                "0.index.projection",
            )
            for name in names:
                group = next(g for g in ranks[0] if g.name == name)
                self.assertGreater(
                    acct.scale_accumulation_work(group.work).fp32_flops,
                    group.work.fp32_flops,
                )

    def test_v31_identity_and_old_v3_reproduction(self):
        case = dict(
            mode="dspark5", strategy="tp4", batch=4, input_tokens=20, output_tokens=8
        )
        old = v3.reference_case(case, model=MODEL)
        self.assertEqual(
            acct.reference_case(case, model=MODEL, version=v3.VERSION), old
        )
        old_trace = v3.ideal_trace(case, model=MODEL)
        self.assertEqual(
            acct.reference_trace(old_trace, strategy="tp4", model=MODEL).seconds,
            old.seconds,
        )
        new_trace = acct.ideal_trace(case, model=MODEL)
        self.assertEqual(
            new_trace["binding"]["version"], "deepseek-http-reference-v3.1"
        )
        self.assertNotEqual(
            new_trace["binding"]["rate_profile"], old_trace["binding"]["rate_profile"]
        )
        large = replace(MODEL, vocab_size=16)
        case.update(batch=64, input_tokens=400000, output_tokens=1)
        self.assertGreater(
            acct.reference_case(case, model=large).seconds,
            v3.reference_case(case, model=large).seconds,
        )


if __name__ == "__main__":
    unittest.main()
