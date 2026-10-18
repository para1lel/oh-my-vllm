import hashlib
import json
import math
import struct
import unittest
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from oh_my_vllm.kernels.native import SOURCES, source_digest
from oh_my_vllm.kernels.specification import MODEL, OPERATORS, UPSTREAM_COMMIT
from oh_my_vllm.worker.checkpoint import Checkpoint

from development.peaks import Peaks
from development.peaks import source_id as calibration_source_id

from .acceptance import CHECKS, FLAGS, assess, verify_identity
from .gdn import make_schedule
from .provenance import ROOT, case_id, source_identity, suite_identity
from .providers import MODULES
from .roofline import ExpectedCase, Observation, Work, expected_efficiencies
from .work import work_for
from .workload import (
    NUMERICAL_POLICY,
    PROFILES,
    REPORT_VERSION,
    SERVING_OPERATORS,
    modeled_cases,
    numerical_cases,
    stage_cases,
)

PEAKS = Peaks(
    hbm=7.7e12, fp8=4.5e15, bf16=2.25e15, tf32=1.125e15, fp32=75e12, fp64=37e12
)


def calibration_profile(peaks=PEAKS):
    measurements = {}
    for name, rate in vars(peaks).items():
        size = 1 << 30 if name == "hbm" else 4096
        option = 4 if name == "hbm" else 0 if name in {"fp8", "bf16", "tf32"} else 2
        work = (
            2147483648
            if name == "hbm"
            else 137438953472
            if name in {"fp8", "bf16", "tf32"}
            else 67108864
        )
        measurements[name] = [
            dict(size=size, option=option, work=work, samples_seconds=[work / rate] * 7)
        ]
    return dict(
        version=1,
        uuid="GPU-test",
        peaks=vars(peaks),
        measurements=measurements,
        hardware=dict(sm_count=1, l2_bytes=16),
        device={},
    )


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
    if operator == "sample":
        flags = {
            key: True
            for key in (
                "passed",
                "probabilities_exact",
                "tokens_exact",
                "selected_probability_exact",
                "rng_unchanged",
            )
        }
        evidence["greedy"] = {
            "passed": True,
            "cases": {
                case: {mode: deepcopy(flags) for mode in ("eager", "graph")}
                for case in (
                    "penalties_and_mask",
                    "tied_logits",
                    "penalized_tie",
                    "empty_mask",
                )
            },
        }
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


def acceptance_reports():
    reports = []
    for operator in SERVING_OPERATORS:
        shapes = [shape for shape in numerical_cases() if shape.operator == operator]
        for mode in ("eager", "graph"):
            cases = [
                case for case in modeled_cases() if case.shape.operator == operator
            ]
            samples = {
                case_id(case.shape): [
                    work_for(case.shape).lower_bound_seconds(PEAKS) / 0.75
                ]
                * 5
                for case in cases
            }
            evidence = {}
            for shape in shapes:
                check = (
                    {"passed": True, "empty_rows": True, "unchanged_slots": True}
                    if operator == "gdn_attention" and shape.rows == 0
                    else numerical_evidence(operator, shape.variant)
                )
                evidence[case_id(shape)] = {
                    "shape": asdict(shape),
                    "status": "passed",
                    "eager": check,
                    "graph": deepcopy(check),
                }
            report = {
                "report_version": REPORT_VERSION,
                "operator": operator,
                "mode": mode,
                "source_id": "source",
                "suite_id": "suite",
                "native_id": "native",
                "binary_ids": {"library": "binary"},
                "context": {
                    "mode": mode,
                    "device": "GPU-test",
                    "calibration": calibration_profile(),
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
                "checks": {
                    case_id(case.shape): (
                        {"status": "passed", "numerical_case_id": case_id(case.shape)}
                        if case_id(case.shape) in evidence
                        else {"status": "not_run"}
                    )
                    for case in cases
                },
                "work": {
                    case_id(case.shape): asdict(work_for(case.shape)) for case in cases
                },
                "observations": [
                    asdict(
                        Observation(
                            operator,
                            case_id(case.shape),
                            case.calls,
                            work_for(case.shape).lower_bound_seconds(PEAKS),
                            samples[case_id(case.shape)][0],
                            "source",
                        )
                    )
                    for case in cases
                ],
            }
            report["numerical_evidence"] = {
                "policy": NUMERICAL_POLICY,
                "source_id": "source",
                "suite_id": "suite",
                "native_id": "native",
                "binary_ids": {"library": "binary"},
                "checkpoint": dict(report["context"]["checkpoint"]),
                "cases": evidence,
            }
            reports.append(report)
    return reports


def numerical_result(report, *, mode="eager"):
    return next(iter(report["numerical_evidence"]["cases"].values()))[mode]


def checkpoint_fixture(root):
    config = b'{"fixture":true}'
    (root / "config.json").write_bytes(config)
    weights = {name: f"{name}.safetensors" for name in ("weight", "embedding")}
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    for name, filename in weights.items():
        header = json.dumps(
            {name: {"dtype": "F8_E4M3", "shape": [1], "data_offsets": [0, 1]}}
        ).encode()
        header += b" " * (-len(header) % 8)
        (root / filename).write_bytes(struct.pack("<Q", len(header)) + header + b"\x01")
    return hashlib.sha256(config).hexdigest()


class Contracts(unittest.TestCase):
    def test_suite_requires_both_modes_and_complete_numerical_evidence(self):
        reports = acceptance_reports()
        self.assertTrue(assess(reports)["accepted"])
        numerical_result(reports[0])["input_projection_upstream_drift"] = {
            "passed": False,
            "nrmse": 1.0,
        }
        self.assertTrue(assess(reports)["accepted"])
        self.assertFalse(assess(reports[:-1])["accepted"])
        with self.assertRaises(ValueError):
            assess([*reports, reports[0]])
        key = next(iter(reports[0]["samples"]))
        mutations = (
            lambda report: numerical_result(report).update(passed=False),
            lambda report: numerical_result(report).update(output={"passed": False}),
            lambda report: numerical_result(report).update(
                output={"passed": True, "nrmse": 1, "max_abs": 100, "reference_peak": 1}
            ),
            lambda report: numerical_result(report).clear(),
            lambda report: numerical_result(report).update(
                output={
                    "passed": True,
                    "nrmse": 0,
                    "max_abs": 0,
                    "reference_peak": 1,
                    "max_tolerance_ratio": 1.01,
                }
            ),
            lambda report: report["work"][key].update(fp64_flops=1),
            lambda report: numerical_result(report)["output"].update(
                tolerance_name="mtp_hidden"
            ),
            lambda report: numerical_result(report)["output"].update(
                required_atol=0.04
            ),
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
        numerical_result(changed[1], mode="graph")["passed"] = False
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
            numerical_result(report)[field] = {}
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
        self.assertTrue(assess(slower)["accepted"])
        for greedy in ({"passed": True}, {"passed": True, "cases": {}}):
            changed = deepcopy(reports)
            report = next(
                report for report in changed if report["operator"] == "sample"
            )
            numerical_result(report)["greedy"] = greedy
            with self.subTest(greedy=greedy), self.assertRaises(ValueError):
                assess(changed)
        changed = deepcopy(reports)
        report = next(report for report in changed if report["operator"] == "sample")
        del numerical_result(report)["greedy"]
        with self.assertRaises(ValueError):
            assess(changed)

    def test_audit_rejects_missing_or_mismatched_small_numerical_evidence(self):
        reports = acceptance_reports()
        self.assertTrue(assess(reports)["accepted"])
        for shape in numerical_cases():
            self.assertLessEqual(
                shape.rows, 392 if shape.operator == "vision_block" else 65
            )
            self.assertLessEqual(shape.context, 786)
            if shape.position == 770:
                self.assertEqual(make_schedule(shape).checkpoint_rows, (13, 64))
        for field in ("source_id", "suite_id", "native_id", "binary_ids", "checkpoint"):
            changed = deepcopy(reports)
            changed[0]["numerical_evidence"][field] = "different"
            with self.subTest(binding=field), self.assertRaises(ValueError):
                assess(changed)
        for mutate in (
            lambda report: report.pop("numerical_evidence"),
            lambda report: report.update(numerical_evidence={}),
            lambda report: report.update(report_version=1),
            lambda report: report["numerical_evidence"]["cases"].popitem(),
            lambda report: next(
                iter(report["numerical_evidence"]["cases"].values())
            ).pop("graph"),
        ):
            changed = deepcopy(reports)
            mutate(changed[0])
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                assess(changed)
        changed = deepcopy(reports)
        shifted = next(
            item
            for item in changed[0]["numerical_evidence"]["cases"].values()
            if item["shape"]["position"] == 770
        )
        shifted["shape"]["position"] = 0
        with self.assertRaises(ValueError):
            assess(changed)
        changed = deepcopy(reports)
        large = next(
            key
            for key, value in changed[0]["checks"].items()
            if value["status"] == "not_run"
        )
        changed[0]["checks"][large] = {"status": "passed", "numerical_case_id": large}
        with self.assertRaises(ValueError):
            assess(changed)
        changed = deepcopy(reports)
        item = next(iter(changed[0]["numerical_evidence"]["cases"].values()))
        item["shape"]["rows"] = 32144
        with self.assertRaises(ValueError):
            assess(changed)

    def test_operator_identities_bind_runtime_checkpoint_and_graph_sources(self):
        original_read = Path.read_bytes
        suite = suite_identity()
        source, _ = source_identity("native", (), {}, "ffn")
        for name in ("checkpoint", "device"):
            path = ROOT / "python/oh_my_vllm/worker" / f"{name}.py"

            def changed_runtime_source(value, runtime_path=path):
                data = original_read(value)
                return (
                    data + b"\n# changed runtime dependency\n"
                    if value == runtime_path
                    else data
                )

            with (
                self.subTest(module=name),
                patch.object(Path, "read_bytes", changed_runtime_source),
            ):
                self.assertNotEqual(suite, suite_identity())
                self.assertNotEqual(source, source_identity("native", (), {}, "ffn")[0])

        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = checkpoint_fixture(root)
            with patch.dict(MODEL, {"config_sha256": config}):
                loaded = Checkpoint(root)
                loaded.raw_tensor("weight")
                frozen = loaded.verify_identity()
                for name in (
                    "config.json",
                    "model.safetensors.index.json",
                    "weight.safetensors",
                ):
                    path = root / name
                    original = path.read_bytes()
                    path.write_bytes(original + b" ")
                    self.assertEqual(loaded.identity, frozen)
                    with (
                        self.subTest(checkpoint=name),
                        self.assertRaisesRegex(ValueError, "file bytes changed"),
                    ):
                        loaded.verify_identity(frozen)
                    path.write_bytes(original)
                    self.assertEqual(loaded.verify_identity(frozen), frozen)
                loaded.raw_tensor("embedding")
                with self.assertRaisesRegex(ValueError, "loaded checkpoint set"):
                    loaded.verify_identity(frozen)

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

    def test_native_identity_resolves_device_and_rejects_changed_source(self):
        with TemporaryDirectory() as directory:
            builds = []
            for name in ("ffn", "device"):
                parent = Path(directory) / name
                parent.mkdir()
                binary = parent / "library.so"
                binary.write_bytes(name.encode())
                build = {
                    "functions": list(MODULES[name]),
                    "library": str(binary),
                    "source_id": hashlib.sha256(name.encode()).hexdigest(),
                    "binary_id": hashlib.sha256(binary.read_bytes()).hexdigest(),
                    "source_fingerprint": source_digest(SOURCES / f"{name}.cu"),
                }
                (parent / "manifest.json").write_text(json.dumps(build))
                builds.append(build)
            calibration = calibration_profile()
            calibration.update(
                source_id=calibration_source_id(),
                native=dict(
                    library=builds[0]["library"],
                    binary_id=builds[0]["binary_id"],
                    cublaslt_library=builds[0]["library"],
                    cublaslt=builds[0]["binary_id"],
                ),
            )
            context = {
                "device": "GPU-test",
                "calibration": calibration,
                "upstream_commit": UPSTREAM_COMMIT,
                "device_source": builds[1]["source_id"],
            }
            identity, binaries = source_identity(
                builds[0]["source_id"],
                tuple(Path(build["library"]) for build in builds),
                context,
                "ffn",
            )
            report = {
                "suite_id": suite_identity(),
                "operator": "ffn",
                "builds": builds,
                "context": context,
                "native_id": builds[0]["source_id"],
                "source_id": identity,
                "binary_ids": binaries,
            }
            verify_identity(report)
            builds[1]["source_fingerprint"] = "0" * 64
            manifest = Path(builds[1]["library"]).parent / "manifest.json"
            manifest.write_text(json.dumps(builds[1]))
            with self.assertRaisesRegex(ValueError, "native source has changed"):
                verify_identity(report)

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
            all(work_for(case.shape).lower_bound_seconds(PEAKS) > 0 for case in cases)
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
        lower = work.lower_bound_seconds(PEAKS)
        expected = {"a": ExpectedCase("op", 1, work), "b": ExpectedCase("op", 3, work)}
        observations = [
            Observation("op", "a", 1, lower, lower * 2, "source"),
            Observation("op", "b", 3, lower, lower * 5, "source"),
        ]
        efficiency = expected_efficiencies(observations, expected, "source", PEAKS)[
            "op"
        ]
        self.assertAlmostEqual(efficiency, 0.275)
        self.assertEqual(
            replace(observations[0], measured_seconds=lower / 2).efficiency(), 2
        )
        for invalid in (
            observations[:1],
            [*observations, observations[0]],
            [replace(observations[0], calls=0.01), observations[1]],
            [replace(observations[0], operator="other"), observations[1]],
            [replace(observations[0], lower_bound_seconds=lower * 2), observations[1]],
            [replace(observations[0], source_id="other"), observations[1]],
            [replace(observations[0], measured_seconds=math.nan), observations[1]],
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                expected_efficiencies(invalid, expected, "source", PEAKS)
        with self.assertRaises(ValueError):
            expected_efficiencies([], {}, "source", PEAKS)

    def test_tensor_core_work_uses_one_shared_resource(self):
        combined = Work(
            0,
            fp8_flops=int(4.5e15),
            bf16_flops=int(2.25e15),
            tf32_flops=int(1.125e15),
        )
        self.assertEqual(combined.lower_bound_seconds(PEAKS), 3)
        self.assertEqual(Work(0, fp64_flops=int(37e12)).lower_bound_seconds(PEAKS), 1)
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
            Work(-1).lower_bound_seconds(PEAKS)
        with self.assertRaises(ValueError):
            Work(math.nan).lower_bound_seconds(PEAKS)

    def test_measured_peaks_require_complete_rates_and_matching_uuid(self):
        from development.peaks import Peaks, profile_peaks

        rates = dict(hbm=100, fp8=200, bf16=100, tf32=50, fp32=20, fp64=10)
        peaks = Peaks(**rates)
        self.assertEqual(Work(100, 200, 100, 20, 50, 10).lower_bound_seconds(peaks), 3)
        self.assertEqual(Work(0, fp32_flops=40).lower_bound_seconds(peaks), 2)
        profile = calibration_profile(peaks)
        self.assertEqual(profile_peaks(profile, "GPU-test"), peaks)
        for mutate in (
            lambda value: value.update(uuid="GPU-other"),
            lambda value: value["peaks"].pop("fp64"),
            lambda value: value["peaks"].update(hbm=0),
            lambda value: value["peaks"].update(fp8=math.nan),
            lambda value: value["peaks"].update(bf16=200),
            lambda value: value["measurements"]["hbm"][0].update(work=1),
            lambda value: value["hardware"].update(sm_count=0),
            lambda value: value["measurements"]["tf32"][0].update(
                samples_seconds=[0] * 7
            ),
        ):
            changed = deepcopy(profile)
            mutate(changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                profile_peaks(changed, "GPU-test")


if __name__ == "__main__":
    unittest.main()
