import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .cases import digest, load_cases, validate_cases
from .fixtures import Fixture, load_fixture


def fixture_files(root, case):
    values = {
        "inputs.value": np.zeros((1, 5120), np.uint16),
        "inputs.residual": np.zeros((1, 5120), np.uint16),
        "expected.value": np.zeros((1, 5120), np.float64),
        "expected.residual": np.zeros((1, 5120), np.float64),
    }
    shard = root / "arrays.npz"
    np.savez(shard, **values)
    manifest = {
        "version": "operator-fixture-v1",
        "case": case.definition(),
        "checkpoint": {"config": "a" * 64},
        "reference": {
            "source": "b" * 64,
            "torch": "fixture-test",
            "precision": "torch.float64",
        },
        "files": {"arrays.npz": hashlib.sha256(shard.read_bytes()).hexdigest()},
        "tensors": {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "tensor_dtype": "bfloat16" if name.startswith("inputs.") else "float64",
                "parts": [
                    {
                        "file": "arrays.npz",
                        "key": name,
                        "offset": 0,
                        "count": value.size,
                    }
                ],
            }
            for name, value in values.items()
        },
        "writes": {},
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return replace(case, fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class Contracts(unittest.TestCase):
    def test_numerics_rejects_wrong_state_and_missing_limits(self):
        from .numerics import compare, float_metrics

        case = next(c for c in load_cases() if c.id == "gdn.numerics.batched-decode")
        expected = {
            "value": np.ones((4, 2)),
            "residual": np.ones((4, 2)),
            "recurrent": np.ones((4, 2, 2)),
            "convolution": np.ones((4, 2, 3)),
            "guards": np.ones(2, dtype=bool),
        }
        fixture = Fixture({"case": case.definition()}, {}, {}, expected)
        budget = {
            "normalized_l2": 0.01,
            "max_abs": 0.01,
            "rms_floor": 1e-6,
            "near_zero_max_abs": 1e-6,
        }
        channel_budget = {
            "kind": "float",
            **budget,
            "normalized_l2": 1.0,
            "max_abs": [0.1, 0.2],
        }
        reference = np.ones((1, 2))
        self.assertTrue(
            float_metrics(np.array([[1.09, 1.19]]), reference, channel_budget)[1]
        )
        self.assertFalse(
            float_metrics(np.array([[1.19, 1.09]]), reference, channel_budget)[1]
        )
        with self.assertRaisesRegex(ValueError, "limits"):
            float_metrics(reference, reference, {**channel_budget, "max_abs": [1.0]})
        limits = {
            "case_sha256": digest(case.definition()),
            "fields": {
                name: {"kind": "exact"}
                if name == "guards"
                else {"kind": "float", **budget}
                for name in expected
            },
        }
        self.assertTrue(compare(expected, fixture, limits).passed)
        for field in ("recurrent", "guards"):
            actual = {name: value.copy() for name, value in expected.items()}
            actual[field][0] = 0
            result = compare(actual, fixture, limits)
            self.assertFalse(result.passed)
            self.assertIn(field, result.failures)
        with self.assertRaisesRegex(ValueError, "limits"):
            compare(expected, fixture, {**limits, "fields": {}})
        with self.assertRaisesRegex(ValueError, "identity"):
            compare(expected, fixture, {**limits, "case_sha256": "0" * 64})
        case = next(c for c in load_cases() if c.id == "output.sample.numerics.batch")
        expected = {
            "probabilities": np.tile([0.5, 0.5, 0.0, 0.0], (4, 1)),
            "tokens": np.zeros(4, dtype=np.int64),
            "selected_probability": np.full(4, 0.5),
        }
        fixture = Fixture(
            {"case": case.definition()},
            {"mask": np.full((4, 1), 3, np.uint8)},
            {},
            expected,
        )
        limits = {
            "case_sha256": case.identity,
            "probability": {"total_variation": 0.01, "softmax_logit_envelope": False},
            "fields": {
                name: {"kind": "exact"}
                if name == "tokens"
                else {"kind": "float", **budget}
                for name in expected
            },
        }
        self.assertTrue(compare(expected, fixture, limits).passed)
        for field, column, value in (
            ("probabilities", 2, 1e-6),
            ("probabilities", 0, 0.5001),
            ("selected_probability", None, 0.5001),
        ):
            actual = {name: array.copy() for name, array in expected.items()}
            if column is None:
                actual[field][0] = value
            else:
                actual[field][0, column] = value
            self.assertFalse(compare(actual, fixture, limits).passed)
        case = next(c for c in load_cases() if c.id == "mtp.chain.numerics.batch-2")
        reference = np.full((4, 2, 2), 1e6)
        reference[0] = 1
        fixture = Fixture({"case": case.definition()}, {}, {}, {"hidden": reference})
        limits = {
            "case_sha256": case.identity,
            "fields": {
                "hidden": {
                    "kind": "float",
                    **budget,
                    "max_abs": 1e6,
                }
            },
        }
        actual = reference.copy()
        actual[0] += 0.5
        self.assertFalse(compare({"hidden": actual}, fixture, limits).passed)

    def test_fixture_loading_is_strict_and_reference_free(self):
        case = next(
            item for item in load_cases() if item.id == "ffn.numerics.single-decode"
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bound = fixture_files(root, case)
            fixture = load_fixture(bound, root)
            self.assertEqual(set(fixture.expected), {"value", "residual"})
            np.testing.assert_array_equal(
                fixture.expected["value"], np.zeros((1, 5120))
            )
            script = (
                "import sys; sys.modules['torch'] = None; "
                "from dataclasses import replace; from pathlib import Path; "
                "from development.operators.cases import load_cases; "
                "from development.operators.fixtures import load_fixture; "
                "case = next(c for c in load_cases() if c.id == sys.argv[1]); "
                "bound = replace(case, fixture_sha256=sys.argv[3]); "
                "load_fixture(bound, Path(sys.argv[2])); "
                "assert sys.modules['torch'] is None"
            )
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    script,
                    case.id,
                    str(root),
                    bound.fixture_sha256,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            shard = root / "arrays.npz"
            original = shard.read_bytes()
            shard.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "identity"):
                load_fixture(bound, root)
            shard.unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                load_fixture(bound, root)
            shard.write_bytes(original)
            path = root / "manifest.json"
            metadata = json.loads(path.read_text())
            metadata["tensors"].pop("expected.residual")
            path.write_text(json.dumps(metadata))
            changed = replace(
                bound, fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            with self.assertRaisesRegex(ValueError, "outputs"):
                load_fixture(changed, root)

    def test_case_limits_reject_shape_sweeps(self):
        cases = load_cases()
        for incomplete in (
            [],
            cases[1:],
            [case for case in cases if case.boundary != "fa"],
        ):
            with self.assertRaisesRegex(ValueError, "coverage"):
                validate_cases(incomplete)
        selected = [
            case
            for case in cases
            if case.boundary == "ffn" and case.purpose == "numerics"
        ]
        with self.assertRaisesRegex(ValueError, "five"):
            validate_cases([*selected, replace(selected[0], id="ffn.numerics.extra")])
        case = selected[0]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bound = fixture_files(root, case)
            changed = replace(bound, shape={"rows": 17})
            with self.assertRaisesRegex(ValueError, "case"):
                load_fixture(changed, root)
