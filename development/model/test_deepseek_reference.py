"""CPU failure contracts for the independent DeepSeek reference entry."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .check_deepseek_reference import CandidateCapture, coverage_matches, main
from .deepseek_reference_contract import (
    acceptance_record,
    compare_tensor,
    endpoint_schema,
    load_reference,
    validate_tensors,
)


class DeepSeekReferenceTests(unittest.TestCase):
    def test_changed_fixture_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text("{}")
            (root / "endpoints.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "frozen reference SHA256"):
                load_reference(root)

    def test_missing_endpoint_and_nonfinite_reference_are_rejected(self):
        schema = {"x.hidden": dict(shape=[1, 2], dtype="torch.float32")}
        with self.assertRaisesRegex(ValueError, "endpoint keys differ"):
            validate_tensors({}, schema)
        with self.assertRaisesRegex(ValueError, "metadata differs"):
            validate_tensors({"x.hidden": torch.zeros(2)}, schema)
        with self.assertRaisesRegex(ValueError, "nonfinite reference"):
            validate_tensors({"x.hidden": torch.tensor([[0.0, float("nan")]])}, schema)
        self.assertEqual(len(endpoint_schema()), 143)

    def test_relative_l2_uses_each_reference_vector(self):
        expected = torch.tensor([[3.0, 4.0], [3000.0, 4000.0]], dtype=torch.float64)
        actual = torch.tensor([[3.0, 4.06], [3000.0, 4000.0]], dtype=torch.float64)
        result = compare_tensor("x.hidden", expected, actual)
        self.assertAlmostEqual(result["value"], 0.012)
        self.assertFalse(result["passed"])
        self.assertEqual(result["limit"], 0.01)
        actual[0, 1] = 4.04
        self.assertTrue(compare_tensor("x.hidden", expected, actual)["passed"])

    def test_small_logit_error_does_not_allow_changed_greedy_id(self):
        expected = torch.tensor([[1.0, 1.001]], dtype=torch.float32)
        actual = torch.tensor([[1.001, 1.0]], dtype=torch.float32)
        result = compare_tensor("x.logits", expected, actual)
        self.assertLess(result["value"], 0.01)
        self.assertFalse(result["passed"])
        self.assertFalse(result["greedy_ids_equal"])
        self.assertFalse(
            compare_tensor("x.ids", torch.tensor([1]), torch.tensor([2]))["passed"]
        )

    def test_confidence_nonfinite_and_zero_reference_failures(self):
        expected = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        actual = torch.tensor([[0.5, 0.5101]], dtype=torch.float64)
        self.assertFalse(compare_tensor("x.confidence", expected, actual)["passed"])
        actual[0, 1] = float("inf")
        self.assertFalse(compare_tensor("x.hidden", expected, actual)["passed"])
        self.assertFalse(
            compare_tensor("x.hidden", torch.zeros(1, 2), torch.ones(1, 2))["passed"]
        )
        self.assertTrue(
            compare_tensor("x.hidden", torch.zeros(1, 2), torch.zeros(1, 2))["passed"]
        )

    def test_capture_keeps_the_actual_worker_prediction(self):
        capture = CandidateCapture.__new__(CandidateCapture)
        capture.worker = SimpleNamespace(
            _target=lambda *_args, **_options: (
                torch.tensor([0]),
                torch.tensor([[0.0, 1.0]]),
                torch.zeros(1, 1, 2),
            )
        )
        saved, checked = {}, {}
        capture.save = lambda label, value, **_options: saved.update(
            {label: value.tolist()}
        )
        capture.check = lambda label, actual, expected: checked.update(
            {label: actual == expected}
        )
        capture.target([123], 1, "short.2")
        self.assertEqual(saved["short.2.argmax"], [0])
        self.assertFalse(checked["short.2.sampled_ids"])

    def test_state_compares_only_the_live_compressor_tail(self):
        kv = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        score = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
        cache = SimpleNamespace(
            _window=[],
            _mtp_window=[],
            _tails=[(kv, score)],
            _engram=[],
            fa_pages={},
            mamba_pages={},
        )
        capture = CandidateCapture.__new__(CandidateCapture)
        capture.worker, capture.torch = SimpleNamespace(cache=cache), torch
        boundary = capture.state(128)
        partial = capture.state(129)
        kv[0, 1].add_(10)
        score[0, 1].add_(10)
        self.assertEqual(capture.state(128), boundary)
        self.assertEqual(capture.state(129), partial)
        kv[0, 0].add_(10)
        self.assertEqual(capture.state(128), boundary)
        self.assertNotEqual(capture.state(129), partial)
        changed_kv = capture.state(129)
        score[0, 0].add_(10)
        self.assertEqual(capture.state(128), boundary)
        self.assertNotEqual(capture.state(129), changed_kv)

    def test_missing_duplicate_or_truthy_comparisons_cannot_pass(self):
        records = [
            dict(label=f"short.{end}.{suffix}", path="driver", passed=True)
            for end in range(1, 5)
            for suffix in ("logits", "hidden", "argmax")
        ]
        self.assertFalse(coverage_matches(("short",), records, [], owns=True))
        records.extend(
            dict(label=f"short.{end}.sampled_ids", passed=True) for end in range(1, 5)
        )
        self.assertTrue(coverage_matches(("short",), records, [], owns=True))
        self.assertFalse(coverage_matches(("short",), records[:-1], [], owns=True))
        self.assertFalse(
            coverage_matches(("short",), [*records, records[0]], [], owns=True)
        )
        for truthy in ("false", float("nan"), 1):
            changed = [dict(record) for record in records]
            changed[0]["passed"] = truthy
            self.assertFalse(coverage_matches(("short",), changed, [], owns=True))
        self.assertFalse(coverage_matches(("authentic",), records[:1], [], owns=True))
        self.assertFalse(
            coverage_matches(("short",), [], [dict(case="short")], owns=False)
        )

    def test_reports_cannot_be_written_inside_the_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = [
                "check",
                "--reference",
                str(root),
                "--output",
                str(root / "candidate"),
                "--validate-only",
            ]
            with patch("sys.argv", arguments), self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse((root / "candidate").exists())

    def test_forced_acceptance_and_correction_positions(self):
        logits = torch.zeros(1, 6, 8)
        for row, predicted in enumerate((2, 3, 4, 5, 6, 7)):
            logits[0, row, predicted] = 1
        result = acceptance_record([1, 2, 3, 0, 5, 6], logits)
        self.assertEqual(
            result,
            dict(
                accepted_drafts=2,
                accepted_endpoint=129,
                correction=4,
                probe_endpoint=130,
                emitted_ids=[2, 3, 4],
            ),
        )
        for chain, count, correction in (
            ([1, 0, 3, 4, 5, 6], 0, 2),
            ([1, 2, 0, 4, 5, 6], 1, 3),
            ([1, 2, 3, 4, 0, 6], 3, 5),
            ([1, 2, 3, 4, 5, 0], 4, 6),
            ([1, 2, 3, 4, 5, 6], 5, 7),
        ):
            result = acceptance_record(chain, logits)
            self.assertEqual(
                (result["accepted_drafts"], result["correction"]), (count, correction)
            )
        with self.assertRaisesRegex(ValueError, "one known token and five drafts"):
            acceptance_record([1], logits)


if __name__ == "__main__":
    unittest.main()
