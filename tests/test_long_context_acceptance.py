"""CPU checks for the real long-context MTP acceptance log gate."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/long-context-acceptance.py"
spec = importlib.util.spec_from_file_location("long_context_acceptance", SCRIPT)
long_context = importlib.util.module_from_spec(spec)
spec.loader.exec_module(long_context)


def _lines(counts):
    return "".join(
        f"2026 INFO request_id={i} proposed_draft_tokens={count} request completed\n"
        for i, count in enumerate(counts)
    )


def test_long_context_drafts_count_each_request_from_new_log_region(tmp_path):
    log = tmp_path / "server.log"
    old = _lines([0])
    log.write_text(old + _lines([4, 3, 2, 1]))
    assert long_context.read_draft_counts(log, len(old.encode())) == [4, 3, 2, 1]


@pytest.mark.parametrize(
    ("lines", "error"),
    [
        (_lines([4, 3, 2]), "expected 4 completed"),
        (_lines([4, 3, 2, 1, 1]), "expected 4 completed"),
        (_lines([4, 3, 0, 1]), "no positive proposed_draft_tokens"),
        (
            "request completed\n" + _lines([4, 3, 2]),
            "no positive proposed_draft_tokens",
        ),
    ],
)
def test_long_context_draft_gate_rejects_false_evidence(tmp_path, lines, error):
    log = tmp_path / "server.log"
    log.write_text(lines)
    with pytest.raises(ValueError, match=error):
        long_context.read_draft_counts(log, 0)


def test_long_context_rejects_truncated_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(_lines([4, 3, 2, 1]))
    with pytest.raises(ValueError, match="truncated"):
        long_context.read_draft_counts(log, log.stat().st_size + 1)


def test_long_context_script_help_smoke():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--server-log" in result.stdout


@pytest.mark.parametrize(
    ("text", "count", "cached", "repeat", "error"),
    [
        ('{"n": 1, "label": "verified"}', 131072, 130000, 1, "incorrect content"),
        ('{"n": 123, "label": "verified"}', 131071, 130000, 1, "below 131072"),
        ('{"n": 123, "label": "verified"}', 131072, 129999, 1, "fewer than 130000"),
    ],
)
def test_long_context_response_gates_survive_python_optimized_mode(
    text, count, cached, repeat, error
):
    code = (
        "import importlib.util,json,sys; "
        "spec=importlib.util.spec_from_file_location('long_context',sys.argv[1]); "
        "module=importlib.util.module_from_spec(spec); "
        "spec.loader.exec_module(module); "
        "module.validate_response(**json.loads(sys.argv[2]))"
    )
    payload = json.dumps(
        {"text": text, "count": count, "cached": cached, "repeat": repeat}
    )
    result = subprocess.run(
        [sys.executable, "-O", "-c", code, str(SCRIPT), payload],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert error in result.stderr
