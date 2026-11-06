import unittest
from copy import deepcopy

from . import accounting, kernels
from .qwen_fp8_roofline import (
    Span,
    checkpoint_writes,
    kv_storage,
    meets_efficiency_target,
    pool_storage,
    resource_floor,
    target_components,
)


def workload(mode="ordinary", prompt=2, output=2, steps=None):
    return dict(
        workload="serving-tools-v1",
        mode=mode,
        requests=[
            dict(
                ordinal=0,
                id=7,
                response_id="chatcmpl_test_7",
                input_tokens=prompt,
                output_tokens=output,
                cached_prefix_tokens=0,
                accepted_decode_tokens=[1] * (output - 1) if steps is None else steps,
            )
        ],
    )


class QwenFP8Roofline(unittest.TestCase):
    def test_fp8_vectors_pages_and_pool_ratio(self):
        self.assertEqual(
            kv_storage(1, layers=1), dict(codes=2048, scales=32, total=2080)
        )
        self.assertEqual(kv_storage(128, layers=1)["total"], 266240)
        self.assertEqual(kv_storage(128)["total"], 4259840)
        self.assertEqual(kv_storage(128, layers=17)["total"], 4526080)
        self.assertEqual(kv_storage(129, layers=1)["total"], 268320)
        ordinary = pool_storage(1, mode="ordinary")
        mtp = pool_storage(1, mode="mtp4")
        self.assertEqual((ordinary["pages"], ordinary["checkpoints"]), (64, 1))
        self.assertEqual(ordinary["kv"]["total"], 272629760)
        self.assertEqual(mtp["kv"]["total"], 289669120)
        self.assertEqual(ordinary["gdn_bytes"], 153944064)
        self.assertEqual(pool_storage(0, mode="ordinary")["kv"]["total"], 0)

    def test_checkpoint_and_session_tail_are_distinct_from_pages(self):
        cases = (
            (0, 128, 1),
            (0, 8191, 1),
            (0, 8192, 1),
            (0, 8193, 2),
            (8192, 8193, 1),
            (8191, 8192, 1),
            (8191, 16384, 2),
        )
        for start, end, expected in cases:
            with self.subTest(start=start, end=end):
                span = Span(start, end)
                self.assertEqual(checkpoint_writes(span), expected)
                value = target_components(span)
                self.assertEqual(value.gdn_write_recurrent, expected * 150994944)
                self.assertEqual(value.gdn_write_convolution, expected * 2949120)
                self.assertEqual(value.gdn_read_recurrent, 150994944 if start else 0)
                self.assertEqual(value.gdn_read_convolution, 2949120 if start else 0)

    def test_ba_precision_matches_fixed_dispatch_boundaries(self):
        cases = (
            (False, [(0, 1)], "fp32"),
            (False, [(0, 32)], "fp32"),
            (False, [(0, 33)], "bf16"),
            (False, [(0, 16), (0, 17)], "bf16"),
            (True, [(2, 1)], "bf16"),
            (True, [(2, 2)], "fp32"),
            (True, [(2, 32)], "fp32"),
            (True, [(2, 33)], "bf16"),
            (True, [(2, 1), (3, 1)], "bf16"),
            (True, [(2, 16), (3, 16)], "fp32"),
            (True, [(2, 16), (3, 17)], "bf16"),
        )
        for verify, queries, precision in cases:
            with self.subTest(verify=verify, queries=queries):
                groups = kernels.gdn_groups(queries, verify)
                matrix = any(group.name == "gdn.ba" for group in groups)
                self.assertEqual(matrix, precision == "bf16")
                for start, count in queries:
                    span = Span(start, start + count)
                    omitted = target_components(span)
                    actual = target_components(span, ba_precision=precision)
                    ba = count * 47185920
                    self.assertEqual(
                        actual.bf16_flops - omitted.bf16_flops,
                        ba if matrix else 0,
                    )
                    self.assertEqual(
                        actual.fp32_flops - omitted.fp32_flops,
                        0 if matrix else ba,
                    )
        actual = target_components(Span(2, 3), ba_precision="fp32")
        self.assertEqual(actual.bf16_flops, 1179648)
        self.assertEqual(actual.fp32_flops, 50630016)

    def test_step_ba_precision_and_omission_reach_aggregate(self):
        omitted = resource_floor(workload())
        self.assertEqual(omitted["total"]["bf16_flops"], 3932160)
        self.assertEqual(omitted["total"]["fp32_flops"], 13774080)
        for mode, precisions, bf16, fp32 in (
            ("ordinary", ["fp32", "fp32", "fp32"], 3932160, 202517760),
            ("mtp4", ["fp32", "bf16", "bf16"], 412876800, 108145920),
        ):
            data = workload(mode)
            data["requests"][0]["target_ba_precisions"] = precisions
            result = resource_floor(data)
            with self.subTest(mode=mode):
                self.assertEqual(result["total"]["bf16_flops"], bf16)
                self.assertEqual(result["total"]["fp32_flops"], fp32)
                tail = result["components"]["compute_only_exact_end"]
                self.assertEqual(
                    tail["bf16_flops"], 1572864 + (47185920 if mode == "mtp4" else 0)
                )
        for precisions in ([], ["fp32"], ["fp32", "fp32", "fp8"], None):
            data = workload()
            data["requests"][0]["target_ba_precisions"] = precisions
            with self.subTest(precisions=precisions), self.assertRaises(ValueError):
                resource_floor(data)
        with self.assertRaises(ValueError):
            target_components(Span(2, 3), ba_precision="fp8")

    def test_small_accepted_span_has_hand_calculated_resources(self):
        value = target_components(Span(2, 4), ba_precision="bf16")
        self.assertEqual((value.kv_read_codes, value.kv_read_scales), (65536, 1024))
        self.assertEqual((value.kv_write_codes, value.kv_write_scales), (65536, 1024))
        self.assertEqual(value.fp8_flops, 97307852800)
        self.assertEqual(value.bf16_flops, 97124352)
        self.assertEqual(value.fp32_flops, 6889344)
        self.assertEqual(value.necessary_bytes, 308021248)

    def test_http_exact_end_is_computed_once_in_both_modes(self):
        for mode in ("ordinary", "mtp4"):
            result = resource_floor(workload(mode))
            with self.subTest(mode=mode):
                total = result["total"]
                self.assertEqual(
                    total["kv_write_codes"], 131072 + (6144 if mode == "mtp4" else 0)
                )
                self.assertEqual(
                    total["kv_write_scales"], 2048 + (96 if mode == "mtp4" else 0)
                )
                self.assertEqual(total["gdn_write_recurrent"], 452984832)
                tail = result["components"]["compute_only_exact_end"]
                self.assertEqual(tail["kv_write_codes"], 32768)
                self.assertEqual(tail["gdn_write_recurrent"], 150994944)
                self.assertEqual(tail["fp8_flops"], 48653926400)
                self.assertFalse(result["complete_acceptance_calculator"])
                self.assertTrue(result["omissions"])
        boundary = resource_floor(workload(prompt=8191, output=1))
        self.assertEqual(
            boundary["components"]["compute_only_exact_end"]["gdn_write_recurrent"],
            150994944,
        )
        self.assertEqual(boundary["total"]["gdn_write_recurrent"], 301989888)
        self.assertEqual(boundary["total"]["kv_write_codes"], 268435456)

    def test_rejected_replay_padding_offload_do_not_increase_credit(self):
        data = workload("mtp4", output=4, steps=[3])
        before = deepcopy(data)
        expected = resource_floor(data)
        data.update(
            replay_count=10000,
            padded_rows=10000,
            offload_bytes=10**12,
            rejected_tokens=10000,
            nccl_bytes=10**12,
        )
        data["requests"][0].update(replay_count=10000, rejected_tokens=10000)
        self.assertEqual(resource_floor(data), expected)
        self.assertEqual(before["requests"][0]["accepted_decode_tokens"], [3])
        self.assertEqual(expected["total"]["gdn_write_recurrent"], 452984832)
        self.assertEqual(expected["total"]["kv_write_codes"], 206848)

    def test_old_accounting_is_unchanged_by_new_calculation(self):
        rates = dict(accounting.RATES)
        old_attention = accounting.attention([(2, 2)])
        old_groups = kernels.target_groups([dict(start=0, count=8192, end=8192)], False)
        resource_floor(workload("mtp4", prompt=8191, output=1))
        self.assertEqual(accounting.VERSION, "http-reference-v5")
        self.assertEqual(accounting.RATES, rates)
        self.assertEqual(accounting.attention([(2, 2)]), old_attention)
        self.assertEqual(old_attention["hbm_bytes"], 8192)
        self.assertEqual(
            kernels.target_groups([dict(start=0, count=8192, end=8192)], False),
            old_groups,
        )

    def test_missing_invalid_or_unbound_data_fail(self):
        for data in (
            None,
            {},
            dict(workload="other"),
            dict(workload="serving-tools-v1", mode="ordinary", requests=[]),
        ):
            with self.subTest(data=data), self.assertRaises(ValueError):
                resource_floor(data)
        for field, value in (
            ("input_tokens", None),
            ("input_tokens", True),
            ("output_tokens", 0),
            ("cached_prefix_tokens", 1),
            ("cached_prefix_tokens", False),
            ("accepted_decode_tokens", None),
            ("accepted_decode_tokens", [True]),
            ("accepted_decode_tokens", [2]),
            ("accepted_decode_tokens", []),
            ("response_id", "chatcmpl_bad_9"),
            ("ordinal", 1),
            ("id", -1),
        ):
            data = workload()
            data["requests"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                resource_floor(data)
        data = workload("mtp4", output=7, steps=[6])
        with self.assertRaises(ValueError):
            resource_floor(data)
        data = workload()
        data["requests"].append(deepcopy(data["requests"][0]))
        with self.assertRaises(ValueError):
            resource_floor(data)
        for call in (
            lambda: Span(1, 1),
            lambda: Span(False, 1),
            lambda: kv_storage(-1),
            lambda: kv_storage(1, layers=2),
            lambda: pool_storage(1, mode="unknown"),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_efficiency_gate_includes_exactly_forty_percent(self):
        self.assertTrue(meets_efficiency_target(4, 10))
        self.assertFalse(meets_efficiency_target(3.999, 10))
        for invalid in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                meets_efficiency_target(invalid, 10)
