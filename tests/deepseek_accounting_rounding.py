import unittest

from development.serving import deepseek_accounting as acct
from development.serving import deepseek_accounting_math as arithmetic
from tests.deepseek_accounting_fixtures import MODEL, binding, decode_trace


class ScaleAccumulation(unittest.TestCase):
    def test_fp8_and_fp4_count_one_fma_per_output_and_k32_group(self):
        for storage in ("fp8", "fp4"):
            for rows in (1, 7, 1024):
                with self.subTest(storage=storage, rows=rows):
                    work = arithmetic.projection_work(rows, 128, 96, storage=storage)
                    priced = arithmetic.scale_accumulation_work(work)
                    self.assertEqual(priced.fp32_flops, 2 * rows * 128 * 3)
                    self.assertEqual(priced.fp8_flops, work.fp8_flops)
                    self.assertEqual(priced.hbm_bytes, work.hbm_bytes)

    def test_output_and_input_shards_use_local_output_and_k(self):
        for axis in (0, 1):
            work = arithmetic.projection_work(
                7, 512, 512, storage="fp8", shard_axis=axis, shards=4
            )
            priced = arithmetic.scale_accumulation_work(work)
            outputs, k_groups = (128, 16) if axis == 0 else (512, 4)
            self.assertEqual(priced.fp32_flops, 2 * 7 * outputs * k_groups)

    def test_bf16_and_existing_fp32_work_are_not_scaled(self):
        for storage in ("bf16", "fp32"):
            work = arithmetic.projection_work(7, 128, 96, storage=storage)
            self.assertIs(arithmetic.scale_accumulation_work(work), work)
        work = arithmetic.Work(fp8_flops=4.5e15, fp32_flops=75e12)
        group = arithmetic.Group("projection", work)
        self.assertEqual(arithmetic.group_seconds(group), 2.875)
        self.assertEqual(arithmetic.critical_seconds([[group]] * 4).seconds, 2.875)
        self.assertEqual(acct.overlap_seconds([[group]] * 4, 2), 2.875)

    def test_routed_work_uses_received_assignments_and_local_columns(self):
        for strategy in arithmetic.STRATEGIES:
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
                        arithmetic.scale_accumulation_work(routed.work).fp32_flops,
                        expected,
                    )
                if strategy == "dep4":
                    self.assertEqual(records[3]["request_rows"], [])
                    self.assertGreater(
                        acct.received_assignments(records, 3, 0, strategy), 0
                    )

    def test_dense_and_index_projections_enter_the_fp32_resource(self):
        for strategy in arithmetic.STRATEGIES:
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
                    arithmetic.scale_accumulation_work(group.work).fp32_flops,
                    group.work.fp32_flops,
                )


class DecodeCredit(unittest.TestCase):
    def test_proposed_drafts_do_not_add_work(self):
        one = acct.reference_trace(decode_trace(1, 1), strategy="tp4", model=MODEL)
        five = acct.reference_trace(decode_trace(1, 5), strategy="tp4", model=MODEL)
        self.assertEqual(one.seconds, five.seconds)

    def test_credit_follows_emitted_tokens(self):
        one = acct.reference_trace(decode_trace(1, 1), strategy="tp4", model=MODEL)
        five = acct.reference_trace(decode_trace(5, 5), strategy="tp4", model=MODEL)
        self.assertGreater(five.seconds, one.seconds)


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
                    **value,
                    "gpu_uuids": ["same", "same", "same", "same"],
                }
            )
