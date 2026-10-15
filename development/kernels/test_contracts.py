import math
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory

from oh_my_vllm.kernels.native import source_digest
from oh_my_vllm.kernels.specification import MODEL, OPERATORS, UPSTREAM_COMMIT

from .acceptance import CHECKS, FLAGS, assess
from .provenance import case_id
from .roofline import ExpectedCase, Observation, Work, expected_efficiencies
from .work import work_for
from .workload import PROFILES, SERVING_OPERATORS, modeled_cases, stage_cases


def numerical_evidence(operator, variant="default"):
    metric = {
        "passed": True,
        "nrmse": 0,
        "max_abs": 0,
        "reference_peak": 1,
        "max_tolerance_ratio": 0,
        "tolerance_name": "bf16",
        "required_atol": 0,
    }
    evidence = {key: deepcopy(metric) for key in CHECKS[operator]}
    evidence.update({key: True for key in FLAGS[operator]})
    evidence.update(passed=True, acceptance_state_selection=True)
    for key in {"cache_writes", "state_writes", "convolution_writes"} & evidence.keys():
        evidence[key] = [deepcopy(metric)]
        evidence[key][0]["tolerance_name"] = "state"
    if operator == "fa_attention" and variant == "query_only":
        evidence["cache_writes"] = []
    if operator == "gdn_attention":
        evidence["activation_quantization"].update(codes_exact=True, scales_exact=True)
        for key in ("normalization", "input_projection", "ba_projection"):
            evidence[key]["worst_chunk_nrmse"] = evidence[key].pop("nrmse")
            evidence[key]["rows_checked"] = 1
    if operator == "vision_block":
        evidence["output"]["tolerance_name"] = "vision"
    if operator in {"mtp_context", "mtp_draft"}:
        evidence["attention"] = numerical_evidence(
            "fa_attention", "kv_only" if operator == "mtp_context" else variant
        )
        if operator == "mtp_context":
            evidence["endpoints"] = {"passed": True, "reference": deepcopy(metric)}
        else:
            evidence["sample"] = numerical_evidence("sample")
            evidence["attention"]["output"]["tolerance_name"] = "mtp_attention"
            evidence["ffn"]["tolerance_name"] = "mtp_hidden"
            evidence["ffn_residual"]["tolerance_name"] = "mtp_residual"
            evidence["sample"]["logits"]["tolerance_name"] = "mtp_logits"
    return evidence


class Contracts(unittest.TestCase):
    def test_suite_requires_both_modes_and_complete_numerical_evidence(self):
        reports = []
        for operator in SERVING_OPERATORS:
            for mode in ("eager", "graph"):
                cases = [
                    case for case in modeled_cases() if case.shape.operator == operator
                ]
                samples = {
                    case_id(case.shape): [
                        work_for(case.shape).lower_bound_seconds() / 0.75
                    ]
                    * 5
                    for case in cases
                }
                checks = {
                    case_id(case.shape): numerical_evidence(
                        operator, case.shape.variant
                    )
                    for case in cases
                }
                for evidence in checks.values():
                    evidence["graph"] = deepcopy(evidence)
                reports.append(
                    {
                        "operator": operator,
                        "mode": mode,
                        "source_id": "source",
                        "context": {
                            "mode": mode,
                            "device": "GPU-test",
                            "upstream_commit": UPSTREAM_COMMIT,
                            "checkpoint": {
                                "config": MODEL["config_sha256"],
                                "index": "a" * 64,
                            },
                        },
                        "device": "GPU-test, B200, driver, 1965 MHz, 3996 MHz, 1000 W",
                        "cache_policy": {
                            "method": "ld.global.cg scan before start event",
                            "bytes": 512,
                            "l2_bytes": 128,
                        },
                        "samples": samples,
                        "checks": checks,
                        "work": {
                            case_id(case.shape): asdict(work_for(case.shape))
                            for case in cases
                        },
                        "observations": [
                            asdict(
                                Observation(
                                    operator,
                                    case_id(case.shape),
                                    case.calls,
                                    work_for(case.shape).lower_bound_seconds(),
                                    samples[case_id(case.shape)][0],
                                    True,
                                    "source",
                                )
                            )
                            for case in cases
                        ],
                    }
                )
        self.assertTrue(assess(reports)["accepted"])
        reports[0]["checks"][next(iter(reports[0]["checks"]))][
            "input_projection_upstream_drift"
        ] = {"passed": False, "nrmse": 1.0}
        self.assertTrue(assess(reports)["accepted"])
        self.assertFalse(assess(reports[:-1])["accepted"])
        with self.assertRaises(ValueError):
            assess([*reports, reports[0]])
        key = next(iter(reports[0]["checks"]))
        mutations = (
            lambda report: report["checks"][key].update(passed=False),
            lambda report: report["checks"][key].update(output={"passed": False}),
            lambda report: report["checks"][key].update(
                output={"passed": True, "nrmse": 1, "max_abs": 100, "reference_peak": 1}
            ),
            lambda report: report["checks"].update(
                {key: {"passed": True, "graph": {"passed": True}}}
            ),
            lambda report: report["checks"][key].update(
                output={
                    "passed": True,
                    "nrmse": 0,
                    "max_abs": 0,
                    "reference_peak": 1,
                    "max_tolerance_ratio": 1.01,
                }
            ),
            lambda report: report["work"][key].update(fp64_flops=1),
            lambda report: report["checks"][key]["output"].update(
                tolerance_name="mtp_hidden"
            ),
            lambda report: report["checks"][key]["output"].update(required_atol=0.04),
            lambda report: report["samples"][key].pop(),
            lambda report: report["observations"].pop(),
            lambda report: report["cache_policy"].update(bytes=128),
            lambda report: report["context"].update(checkpoint={"config": "other"}),
            lambda report: report["context"].update(checkpoint={}),
            lambda report: report["context"].update(device=""),
        )
        for mutate in mutations:
            changed = deepcopy(reports)
            mutate(changed[0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                assess(changed)
        changed = deepcopy(reports)
        key = next(iter(changed[1]["checks"]))
        changed[1]["checks"][key]["graph"]["passed"] = False
        with self.assertRaises(ValueError):
            assess(changed)
        for operator, field in (
            ("vision_block", "all_rows_finite"),
            ("mtp_context", "attention"),
            ("mtp_draft", "sample"),
            ("mtp_draft", "ffn_residual"),
            ("mtp_draft", "ffn_residual_store_exact"),
        ):
            changed = deepcopy(reports)
            report = next(
                report for report in changed if report["operator"] == operator
            )
            report["checks"][next(iter(report["checks"]))][field] = {}
            with (
                self.subTest(operator=operator, field=field),
                self.assertRaises(ValueError),
            ):
                assess(changed)
        slower = deepcopy(reports)
        for key, samples in slower[0]["samples"].items():
            slower[0]["samples"][key] = [value * 2 for value in samples]
        for observation in slower[0]["observations"]:
            observation["measured_seconds"] *= 2
        self.assertFalse(assess(slower)["accepted"])

    def test_native_identity_tracks_only_transitive_dependencies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source, header, leaf, unrelated = (
                root / name for name in ("op.cu", "shared.cuh", "leaf.cuh", "other.cuh")
            )
            source.write_text('#include "shared.cuh"\n')
            header.write_text('#include "leaf.cuh"\n')
            leaf.write_text('#include "shared.cuh"\n#define WIDTH 128\n')
            unrelated.write_text("unrelated\n")
            identity = source_digest(source)
            unrelated.write_text("changed\n")
            self.assertEqual(source_digest(source), identity)
            leaf.write_text('#include "shared.cuh"\n#define WIDTH 256\n')
            self.assertNotEqual(source_digest(source), identity)
            leaf.unlink()
            with self.assertRaises(FileNotFoundError):
                source_digest(source)

    def test_serving_distribution_covers_operator_paths(self):
        cases = modeled_cases()
        self.assertAlmostEqual(sum(profile.probability for profile in PROFILES), 1)
        self.assertEqual(
            {case.shape.operator for case in cases},
            set(SERVING_OPERATORS),
        )
        stages = stage_cases()
        self.assertEqual(
            {c.shape.operator for c in stages}, {o.name for o in OPERATORS}
        )
        self.assertTrue(all(case.calls > 0 for case in cases))
        self.assertTrue(
            all(work_for(case.shape).lower_bound_seconds() > 0 for case in cases)
        )
        self.assertTrue(all(case.shape.rows <= MODEL["step_tokens"] for case in cases))
        self.assertTrue(
            any(
                case.shape.operator == "mtp_context" and case.shape.rows > 784
                for case in cases
            )
        )
        self.assertTrue(any(case.shape.variant == "kv_only" for case in stages))
        self.assertTrue(any(case.shape.variant == "query_only" for case in cases))
        self.assertTrue(any(case.shape.variant == "embedding_media" for case in cases))
        self.assertTrue(
            any(
                case.shape.operator == "sample" and case.shape.rows == 20
                for case in cases
            )
        )

    def test_expected_efficiency_uses_fixed_weights_and_complete_coverage(self):
        work = Work(7700)
        lower = work.lower_bound_seconds()
        expected = {"a": ExpectedCase("op", 1, work), "b": ExpectedCase("op", 3, work)}
        observations = [
            Observation("op", "a", 1, lower, lower * 2, True, "source"),
            Observation("op", "b", 3, lower, lower * 5, True, "source"),
        ]
        efficiency = expected_efficiencies(observations, expected, "source")["op"]
        self.assertAlmostEqual(efficiency, 0.275)
        for invalid in (
            observations[:1],
            [*observations, observations[0]],
            [replace(observations[0], calls=0.01), observations[1]],
            [replace(observations[0], operator="other"), observations[1]],
            [replace(observations[0], lower_bound_seconds=lower * 2), observations[1]],
            [replace(observations[0], source_id="other"), observations[1]],
            [replace(observations[0], correct=False), observations[1]],
            [replace(observations[0], measured_seconds=math.nan), observations[1]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                expected_efficiencies(invalid, expected, "source")
        with self.assertRaises(ValueError):
            expected_efficiencies([], {}, "source")

    def test_tensor_core_work_uses_one_shared_resource(self):
        combined = Work(
            0,
            fp8_flops=int(4.5e15),
            bf16_flops=int(2.25e15),
            tf32_flops=int(1.125e15),
        )
        self.assertEqual(combined.lower_bound_seconds(), 3)
        self.assertEqual(Work(0, fp64_flops=int(37e12)).lower_bound_seconds(), 1)
        gdn = next(
            case.shape
            for case in modeled_cases()
            if case.shape.operator == "gdn_attention" and case.shape.rows == 1
        )
        self.assertEqual(work_for(gdn).fp32_flops, 48 * (7 * 128 * 128 + 3 * 128))
        mtp = replace(gdn, operator="mtp_context", rows=4)
        self.assertEqual(work_for(mtp).fp64_flops, 0)
        self.assertGreaterEqual(work_for(mtp).bf16_flops, 4 * 2 * 5120 * 10240)
        vision = next(
            case.shape
            for case in modeled_cases()
            if case.shape.operator == "vision_block"
        )
        attention = 2 * vision.rows * vision.frame_tokens * 1152
        self.assertEqual(work_for(vision).tf32_flops, 0)
        self.assertGreaterEqual(work_for(vision).bf16_flops, 2 * attention)
        with self.assertRaises(ValueError):
            Work(-1).lower_bound_seconds()
        with self.assertRaises(ValueError):
            Work(math.nan).lower_bound_seconds()


if __name__ == "__main__":
    unittest.main()
