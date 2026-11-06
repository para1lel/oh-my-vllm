import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from . import deepseek_acceptance as gate
from . import deepseek_accounting as source
from . import deepseek_accounting_floor as floor
from . import deepseek_accounting_packed as packed
from . import deepseek_accounting_roofline as roofline
from .deepseek_benchmark import runtime_reference
from .deepseek_runtime_trace import assemble_trace
from .test_deepseek_benchmark import runtime_fixture
from .test_deepseek_closeout import _reject_unreviewed_hardware, measured_rows


class LongHttpGate(unittest.TestCase):
    def assess(self, rows):
        acceptance = (
            rows[0]["reference_acceptance"] if rows else "trace-project-estimate"
        )
        with patch.object(
            gate,
            "runtime_reference",
            return_value=(SimpleNamespace(seconds=33, acceptance=acceptance), {}),
        ):
            return gate.evaluate(rows)

    def test_responses_are_required_and_bound_to_each_trace_request(self):
        rows = measured_rows()
        for mutation in (
            "missing",
            "empty",
            "partial",
            "wrong_prompt",
            "wrong_completion",
            "wrong_id",
            "duplicate_id",
            "cached",
            "wrong_dep_owner",
        ):
            broken = copy.deepcopy(rows)
            result = broken[0]
            if mutation == "missing":
                del result["responses"]
            elif mutation == "empty":
                result["responses"] = []
            elif mutation == "partial":
                result["responses"].pop()
            elif mutation == "wrong_prompt":
                result["responses"][0]["usage"]["prompt_tokens"] = 1
            elif mutation == "wrong_completion":
                result["responses"][0]["usage"]["completion_tokens"] = 1
            elif mutation == "wrong_id":
                result["responses"][0]["id"] = "chatcmpl_other_999999"
            elif mutation == "duplicate_id":
                result["responses"][0]["id"] = result["responses"][1]["id"]
            elif mutation == "cached":
                result["responses"][0]["usage"]["prompt_tokens_details"][
                    "cached_tokens"
                ] = 1
            else:
                result = next(row for row in broken if row["strategy"] == "dep4")
                result["trace"]["manifest"]["requests"][0]["owner_rank"] = 0
            with self.subTest(mutation=mutation):
                report = self.assess(broken)
                self.assertFalse(report["target_passed"])
                self.assertEqual(report["status"], "invalid")

    def test_real_calculator_binds_responses_and_reviewed_b200_hardware(self):
        for strategy, proposal in (
            ("tp4", False),
            ("tp4", True),
            ("tep4", False),
            ("dep4", False),
        ):
            case = dict(
                mode="dspark5",
                strategy=strategy,
                batch=4,
                input_tokens=400000,
                output_tokens=64,
            )
            harness, events, responses = runtime_fixture(
                case, reference_version=roofline.VERSION
            )
            modes = {
                name: {
                    "batch_decode": True,
                    "decode_graph": name == "tp4",
                    "decode_graph_limit": 32,
                    "proposal_graph": proposal if name == "tp4" else False,
                }
                for name in ("tp4", "tep4", "dep4")
            }
            harness.update(
                binary="cpu-fixture-binary",
                harness_numeric_runtime={"python": "cpu-fixture"},
                configuration={"execution_modes": modes},
            )
            for device in harness["topology"]:
                device["name"] = "NVIDIA B200"
            for event_rows in events:
                event_rows[0]["execution"]["runtime"]["execution_modes"] = modes[
                    strategy
                ].copy()
            trace = assemble_trace(events, responses, harness=harness, case=case)
            reference, metadata = runtime_reference(
                trace,
                strategy=strategy,
                expected_binding=trace["binding"],
                version=roofline.VERSION,
            )
            seconds = reference.seconds * 2
            measured = {
                **case,
                **metadata,
                "status": "completed",
                "effective_strategy": strategy,
                "request_count": case["batch"],
                "measurement_identity": harness,
                "seconds": seconds,
                "reference_seconds": reference.seconds,
                "reference_acceptance": reference.acceptance,
                "efficiency_percent": source.efficiency_percent(
                    reference.seconds, seconds
                ),
                "trace": trace,
                "responses": responses,
            }
            gate.validate_measurement(measured)
            _reject_unreviewed_hardware(self, measured, strategy)
            for value in (1, 1.0) if proposal else (0, 0.0):
                events[2][0]["execution"]["runtime"]["execution_modes"][
                    "proposal_graph"
                ] = value
                with self.assertRaises(source.BindingError):
                    assemble_trace(events, responses, harness=harness, case=case)
                trace["binding"]["runtime_modes"] = {
                    **modes[strategy],
                    "proposal_graph": value,
                }
                with self.assertRaises(ValueError):
                    gate.validate_measurement(measured)
                trace["binding"]["runtime_modes"] = modes[strategy]

    def test_complete_exact_threshold_and_one_row_below(self):
        rows = measured_rows()
        report = self.assess(rows)
        self.assertTrue(report["target_passed"])
        self.assertEqual(report["observed_long_cases"], 19)
        rows[0]["seconds"] = 101
        rows[0]["efficiency_percent"] = 100 * 33 / 101
        report = self.assess(rows)
        self.assertFalse(report["target_passed"])
        self.assertEqual(len(report["failed_threshold_cases"]), 1)

    def test_missing_duplicate_failed_and_mixed_identity_cannot_pass(self):
        rows = measured_rows()
        variants = [rows[:-1], [*rows, rows[0]]]
        failed = copy.deepcopy(rows)
        failed[0]["status"] = "failed"
        variants.append(failed)
        mixed = copy.deepcopy(rows)
        mixed[0]["measurement_identity"]["binary"] = "different"
        variants.append(mixed)
        wrong_modes = copy.deepcopy(rows)
        wrong_modes[0]["trace"]["binding"]["runtime_modes"] = {
            "batch_decode": False,
            "decode_graph": False,
        }
        variants.append(wrong_modes)
        for variant in variants:
            with self.subTest(size=len(variant)):
                self.assertFalse(self.assess(variant)["target_passed"])
        for version in (floor.VERSION, packed.VERSION):
            historical = measured_rows(version)
            for row in historical:
                for device in row["measurement_identity"]["topology"]:
                    del device["name"]
            partial = self.assess(historical)
            self.assertEqual(partial["status"], "ineligible")
            self.assertTrue(partial["reference_threshold_met"])
            self.assertFalse(partial["target_passed"])

    def test_qualification_cannot_be_supplied_by_a_result_label(self):
        rows = measured_rows()
        rows[0]["reference_scope"] = "complete physical roofline"
        self.assertFalse(self.assess(rows)["target_passed"])
        rows = measured_rows()
        del rows[0]["measurement_identity"]["configuration"]["execution_modes"]
        self.assertFalse(self.assess(rows)["target_passed"])
        for value in (None, 1):
            rows = measured_rows()
            row = rows[0]
            modes = row["measurement_identity"]["configuration"]["execution_modes"][
                row["strategy"]
            ]
            if value is None:
                del modes["proposal_graph"]
            else:
                modes["proposal_graph"] = value
            self.assertFalse(self.assess(rows)["target_passed"])
        rows = measured_rows()
        rows[0]["rate_profile"] = "old-calculator-digest"
        self.assertFalse(self.assess(rows)["target_passed"])
        rows = measured_rows()
        rows[0]["assumptions_sha256"] = "changed-assumptions"
        self.assertFalse(self.assess(rows)["target_passed"])
