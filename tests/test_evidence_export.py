"""Portable evidence preserves numbers and original hashes, but cannot be accepted."""

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "export", ROOT / "scripts/export_evidence.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def test_nested_host_identity_removal_preserves_measurements_and_unique_keys():
    original = {
        "model": str(Path("/") / "data0" / "shared" / "model"),
        "hardware": {"gpu": "GPU-fixture", "cpu_affinity": [3, 17]},
        "runtime": {"packages": {"nvidia-curand": "10.4.0.35"}},
        "rows": [{"output_tps": 90.1, "ttft_s": [1.2, 1.3], "passed": True}],
        "extensions": {
            "/tmp/first/kernel.so": "a" * 64,
            "/tmp/second/kernel.so": "b" * 64,
        },
        "stdout": "arbitrary raw log",
        "note": "GPU-" + "abcdef01... / " + "GPU-" + "abcdef0123…",
        "server_command": ["serve", "--listen", "127.0.0.1:54321"],
        "last_lines": ["arbitrary diagnostic tail"],
    }
    raw = json.dumps(original).encode()
    summary = export.derive(raw, "original.json")
    assert summary["original"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert summary["data"]["rows"] == original["rows"]
    assert summary["data"]["runtime"]["packages"] == original["runtime"]["packages"]
    assert summary["data"]["hardware"] == {"cpu_affinity_count": 2}
    assert len(summary["data"]["extensions"]) == 2
    assert "GPU-" + "abcdef" not in json.dumps(summary)
    assert "arbitrary raw log" not in json.dumps(summary)
    assert "server_command" not in summary["data"]
    assert "last_lines" not in summary["data"]
    assert original["model"] not in json.dumps(summary)
    assert "/tmp/first" not in json.dumps(summary)


def test_formal_comparison_explicitly_rejects_derived_evidence():
    import sys

    sys.path.insert(0, str(ROOT / "benchmarks"))
    from ttft import compare

    summary = export.derive(b'{"output_tps": 90}', "original.json")
    with pytest.raises(ValueError, match="original raw evidence"):
        compare(summary, summary)


def test_redaction_collisions_fail_instead_of_losing_values():
    name = "/tmp/kernel.so"
    alias = "<local-path>#" + hashlib.sha256(name.encode()).hexdigest()[:12]
    for data in ({name: 1, alias: 2}, {"cpu_affinity": [1], "cpu_affinity_count": 2}):
        with pytest.raises(ValueError, match="colli"):
            export.derive(json.dumps(data).encode(), "original.json")


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_cli_never_overwrites_original_through_aliases(tmp_path, alias):
    source = tmp_path / "original.json"
    raw = b'{"output_tps":90}'
    source.write_bytes(raw)
    destination = tmp_path / "summary.json"
    if alias == "same":
        destination = source
    elif alias == "symlink":
        destination.symlink_to(source)
    else:
        destination.hardlink_to(source)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/export_evidence.py"),
            "--input",
            str(source),
            "--output",
            str(destination),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "keep the original input unchanged" in result.stderr
    assert source.read_bytes() == raw
