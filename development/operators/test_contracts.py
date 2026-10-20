import hashlib
import json
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .cases import load_cases, validate_cases
from .fixtures import load_fixture


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
