import unittest
from copy import deepcopy

from .accounting import RATES, linear, reference_seconds
from .kernels import (
    draft_groups,
    ffn_groups,
    gdn_groups,
    group,
    normalization,
    pass_groups,
    reference,
    resources,
    sample_groups,
    serial_seconds,
    wy_groups,
)
from .roofline import lower_bound
from .roofline_test_support import _invoke_benchmark as _invoke_benchmark
from .roofline_test_support import _run_http_fixture as _run_http_fixture
from .roofline_test_support import _verification_trace as _verification_trace
from .roofline_test_support import (
    _verify_checkpoint_identity as _verify_checkpoint_identity,
)
from .roofline_test_support import (
    _verify_tokenizer_identity as _verify_tokenizer_identity,
)
from .roofline_test_support import _write_benchmark_inputs as _write_benchmark_inputs
from .roofline_test_support import prefill_trace as prefill_trace
from .roofline_test_support import request_manifest as request_manifest
from .roofline_test_support import tokenizer_fixture as tokenizer_fixture
from .trace import canonical_passes


class Roofline(unittest.TestCase):
    def test_accounting_rejects_inflation_and_unbound_or_fabricated_work(self):
        manifest = request_manifest((9, 2), length=32768)
        complete = prefill_trace((9, 2), length=32768, chunk=32768)
        expected = lower_bound(complete, manifest)
        for order, chunk in (((2, 9), 4096), ((9, 2), 784), ((2, 9), 32768)):
            events = prefill_trace(order, length=32768, chunk=chunk)
            for event in events:
                event.update(tiles=100000, padded_rows=65536, copied_state_bytes=10**12)
            with self.subTest(order=order, chunk=chunk):
                self.assertEqual(lower_bound(events, manifest), expected)
        passes = canonical_passes(complete, manifest)
        self.assertEqual(
            [[row["count"] for row in item["requests"]] for item in passes],
            [[16464, 15680], [16304, 16464], [624]],
        )
        self.assertEqual([row["id"] for row in passes[0]["requests"]], [9, 2])
        for mutate in (
            lambda value: value["requests"].append(value["requests"][0]),
            lambda value: value["requests"].pop(),
            lambda value: value["requests"][0].update(
                id=100, response_id="chatcmpl_100"
            ),
            lambda value: value["requests"][0].update(response_id="chatcmpl_100"),
            lambda value: value["requests"][0].update(output_tokens=2),
            lambda value: value["requests"][0].update(input_tokens=32767),
            lambda value: value.update(mode="mtp4"),
        ):
            changed = deepcopy(manifest)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                lower_bound(complete, changed)
        for events in (
            complete[1:],
            [complete[0], *complete],
            complete[:-1],
            [*complete, dict(operation="unknown")],
        ):
            with self.subTest(events=events[-1]), self.assertRaises(ValueError):
                lower_bound(events, manifest)
        mtp = _verification_trace()
        manifest = request_manifest(length=4, output=4, mode="mtp4")
        result = lower_bound(mtp, manifest)
        self.assertEqual(
            result["acceptance"][1], dict(tokens=4, accepted=2, proposed=2)
        )
        self.assertEqual(
            [
                row["context_count"]
                for item in canonical_passes(mtp, manifest)
                for row in item["requests"]
            ],
            [4, 2],
        )
        for index, field, value in (
            (5, "proposed", 4),
            (6, "count", 5),
            (8, "position", 6),
            (8, "accepted", 3),
            (3, "position", 3),
        ):
            changed = deepcopy(mtp)
            changed[index]["requests"][0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                lower_bound(changed, manifest)
        with self.assertRaises(ValueError):
            lower_bound([*mtp[:2], *mtp[3:5], mtp[2], *mtp[5:]], manifest)

    def test_reference_resources_follow_kernel_dependencies(self):
        self.assertEqual(resources(sample_groups(1))["fp32_flops"], 25602)
        self.assertEqual(resources(draft_groups([(4, 1)], 0))["fp32_flops"], 147286)
        for stage in range(1, 4):
            heads = next(
                item
                for item in draft_groups([(4 + stage, 1)], stage)
                if item.name == "fa.heads"
            )
            self.assertEqual(heads.work["hbm_bytes"], 4 * 512)
        logical = dict(
            mtp=True,
            verify=False,
            requests=[
                dict(
                    start=0,
                    count=32768,
                    end=32768,
                    tokens=1,
                    context_count=32768,
                    proposed=0,
                )
            ],
        )
        final_norm = next(
            item for item in pass_groups(logical) if item.name == "output.norm"
        )
        self.assertEqual(final_norm.work["fp32_flops"], 32768 * (5 * 5120 + 2))
        self.assertEqual(final_norm.work["hbm_bytes"], 4 * 5120)
        logical["mtp"] = False
        ordinary_norm = next(
            item for item in pass_groups(logical) if item.name == "output.norm"
        )
        self.assertEqual(ordinary_norm.work["fp32_flops"], 5 * 5120 + 2)
        initial = wy_groups([(0, 128)], 0)
        continued = wy_groups([(784, 128)], 0)
        self.assertEqual(initial[0].work, continued[0].work)
        for index in (1, 2):
            self.assertEqual(
                continued[index].work["bf16_flops"] - initial[index].work["bf16_flops"],
                2 * 48 * 128 * 128 * 64,
            )
        work = dict(
            hbm_bytes=7.7e12,
            fp8_flops=4.5e15,
            bf16_flops=2.25e15,
            tf32_flops=1.1e15,
            fp32_flops=75e12,
            fp64_flops=37e12,
        )
        self.assertEqual(reference_seconds(work), 3)
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                reference_seconds(dict(work, fp8_flops=value))
        events = prefill_trace(length=32768, chunk=4096)
        manifest = request_manifest(length=32768)
        passes = canonical_passes(events, manifest)
        bound = lower_bound(events, manifest)
        self.assertEqual(bound["seconds"], reference(passes)["seconds"])
        self.assertGreater(bound["seconds"], reference_seconds(bound["work"]))
        # A complete prompt fits the budget without a checkpoint-aligned split.
        self.assertEqual([item["requests"][0]["count"] for item in passes], [32768])
        small = canonical_passes(prefill_trace(length=16), request_manifest())
        self.assertEqual(
            reference(small)["work"]["fp8_flops"],
            2
            * 16
            * (
                48 * (16384 * 5120 + 5120 * 6144)
                + 16 * (14336 * 5120 + 5120 * 6144)
                + 64 * (34816 * 5120 + 5120 * 17408)
            ),
        )

    def test_serial_groups_cache_budget_and_precision_dispatch(self):
        for rows in (64, 65, 624, 784, 1024, 1025):
            with self.subTest(ffn_rows=rows):
                serial = [
                    group("norm", normalization(rows)),
                    group(
                        "upper",
                        linear(rows, 34816, 5120, fp8=True),
                        dict(fp32_flops=rows * 2 * 17408),
                    ),
                    group("down", linear(rows, 5120, 17408, fp8=True)),
                ]
                selected = ffn_groups(rows)
                self.assertEqual(resources(selected), resources(serial))
                if 65 <= rows <= 1024:
                    self.assertEqual(
                        [item.name for item in selected], ["ffn.prefill-pdl"]
                    )
                    self.assertEqual(
                        serial_seconds(selected), reference_seconds(resources(serial))
                    )
                    self.assertLess(serial_seconds(selected), serial_seconds(serial))
                else:
                    self.assertEqual(
                        [item.name for item in selected],
                        ["ffn.norm", "ffn.upper", "ffn.activation", "ffn.down"]
                        if rows == 64
                        else ["ffn.norm", "ffn.upper", "ffn.down"],
                    )
        memory = group("memory", {"hbm_bytes": 8 * RATES["hbm_bytes"]})
        compute = group("compute", {"fp8_flops": 6 * RATES["fp8_flops"]})
        self.assertEqual(serial_seconds([memory, compute]), 14)
        self.assertEqual(reference_seconds(resources([memory, compute])), 8)
        cached = [
            group(
                "first",
                {
                    "hbm_bytes": 10 * RATES["hbm_bytes"],
                    "fp8_flops": 3 * RATES["fp8_flops"],
                },
            ),
            group(
                "second",
                {
                    "hbm_bytes": 4 * RATES["hbm_bytes"],
                    "bf16_flops": 6 * RATES["bf16_flops"],
                },
            ),
        ]
        self.assertEqual(serial_seconds(cached, cache_bytes=5 * RATES["hbm_bytes"]), 11)
        self.assertEqual(
            serial_seconds(cached, cache_bytes=100 * RATES["hbm_bytes"]), 9
        )
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(cache=value), self.assertRaises(ValueError):
                serial_seconds(cached, cache_bytes=value)
        small = gdn_groups([(0, 128)], False)
        large = gdn_groups([(0, 129)], False)
        self.assertIn("gdn.recurrent", [item.name for item in small])
        self.assertNotIn("gdn.recurrent", [item.name for item in large])
        self.assertIn("gdn.wy.prepare", [item.name for item in large])
        scalar = next(
            item
            for item in gdn_groups([(0, 2)], False)
            if item.name == "gdn.convolution"
        )
        self.assertEqual(scalar.work["bf16_flops"], 0)
        self.assertEqual(scalar.work["fp32_flops"], 2 * (7 * 10240 + 2 * 96 * 5120))
        matrix = next(
            item for item in gdn_groups([(1, 1)], True) if item.name == "gdn.ba"
        )
        self.assertEqual(matrix.work["bf16_flops"], 2 * 96 * 5120)
        self.assertEqual(matrix.work["fp32_flops"], 0)
        prepare, carry, output = wy_groups([(783, 1)], 1234)
        self.assertEqual(prepare.work["bf16_flops"], 28672)
        self.assertEqual(prepare.work["fp32_flops"], 6288)
        self.assertEqual(carry.work["hbm_bytes"], 1234)
        self.assertEqual(output.work["fp32_flops"], 12336)
