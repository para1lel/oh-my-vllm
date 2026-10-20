import unittest
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from .cases import load_cases


class PerformanceContracts(unittest.TestCase):
    def test_accounting_ignores_implementation_work(self):
        from .accounting import CACHE_CAPACITY, attention, lower_bound, work

        small = attention(1, 1, 1, 4)
        self.assertEqual(small["fp8_flops"], 0)
        self.assertEqual(small["bf16_flops"], 4 * 4 * 24 * 256)
        self.assertEqual(small["hbm_bytes"], 3 * 4096)

        cases = [case for case in load_cases() if case.purpose == "performance"]
        device = dict(name="NVIDIA B200", l2_bytes=126 * 1024 * 1024)
        for case in cases:
            self.assertGreater(lower_bound(case, device), 0, case.id)
            changed = replace(
                case,
                options={
                    **case.options,
                    "scratch_bytes": 10**15,
                    "kernel_count": 100000,
                    "padded_rows": 10**9,
                },
            )
            self.assertEqual(work(case), work(changed))
        hidden = next(
            case
            for case in cases
            if case.id == "output.hidden.performance.single-decode"
        )
        streaming = next(
            case for case in cases if case.id == "ffn.performance.single-decode"
        )
        self.assertEqual(work(hidden).resources["hbm_bytes"], 0)
        self.assertEqual(
            work(streaming).resources["hbm_bytes"],
            work(streaming).unique_read_bytes - CACHE_CAPACITY,
        )
        with self.assertRaisesRegex(ValueError, "cache"):
            lower_bound(hidden, {**device, "l2_bytes": CACHE_CAPACITY + 1})
        from .performance_inputs import bf16
        from .storage import decode

        np.testing.assert_array_equal(
            decode(bf16(np.array([1.0, 1.00390625, 1.01171875])), "bfloat16"),
            [1.0, 1.0, 1.015625],
        )
        from .fixtures import load_fixture
        from .performance_inputs import make

        for case in cases:
            if case.boundary != "mtp.chain":
                continue
            prototype = next(
                entry
                for entry in load_cases()
                if entry.id == case.id.replace(".performance.", ".numerics.")
            )
            checkpoint = load_fixture(prototype).manifest["checkpoint"]
            fixture = make(case, SimpleNamespace(identity=checkpoint))
            allowed = np.unpackbits(fixture.inputs["mask"], axis=-1)
            self.assertTrue(np.all(allowed.sum(axis=-1) == 16))

    def test_ratio_gate_excludes_reset_and_fails_uncertainty(self):
        from .timing import (
            Sample,
            ShortInterval,
            accept,
            calibrate,
            check,
            collect,
            sample,
        )

        clock = SimpleNamespace(value=0)

        class Prepared:
            device = SimpleNamespace(synchronize=lambda: None)

            def reset(self):
                clock.value += 100

            def invoke(self):
                clock.value += 1

        class Events:
            def begin(self):
                self.started = clock.value

            def record_end(self):
                self.duration = clock.value - self.started

            def elapsed(self):
                return self.duration

        measured = sample(Prepared(), Events(), 3)
        self.assertEqual(measured.gpu_seconds, 1)
        self.assertEqual(clock.value, 103)
        self.assertEqual(check([Sample(1, 500, 1)] * 30, 1, 1).decision, "pass")
        self.assertEqual(check([Sample(2, 0, 1)] * 30, 1, 1).decision, "fail")
        ambiguous = [Sample(value, 0, 1) for value in [0.9, 1.1] * 15]
        self.assertEqual(check(ambiguous, 1, 1).decision, "unresolved")
        history = []

        def short_block(repeats):
            history.append(repeats)
            duration = 0.0005 if len(history) == 6 else 0.001
            return Sample(duration, 0, repeats)

        retries = []
        collection = collect(short_block, 30, True, retries=retries)
        self.assertEqual(len(collection), 30)
        self.assertEqual({value.repeats for value in collection}, {50})
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["discarded_samples"], 30)
        with self.assertRaises(ShortInterval):
            collect(lambda n: Sample(0.0001, 0, n), 30, True, repeats=25)
        calls = []

        def measure(prepared, *, count, repeats=None):
            calls.append(count)
            return [Sample(value, 0, 1) for value in [0.9, 1.1] * (count // 2)]

        samples, result = accept(None, 1, 1, measure_function=measure)
        self.assertEqual(calls, [30, 90])
        self.assertEqual(len(samples), 120)
        self.assertEqual(result.decision, "unresolved")
        self.assertEqual(calibrate([[Sample(1.23451, 0, 1)] * 30] * 5, 1), 1.2346)
        with self.assertRaisesRegex(ValueError, "five"):
            calibrate([[Sample(1, 0, 1)] * 30], 1)
        with self.assertRaises(ValueError):
            check([Sample(1, 0, 1)] * 29, 1, 1)
