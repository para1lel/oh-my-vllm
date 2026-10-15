"""The translation lane stages a read-only book and observes honest progress."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agentic_corpus import MANIFEST_NAME, STAGED_DIRECTORY, Corpus
from benchmarks.agentic_translation import (
    TARGET_LANGUAGE,
    TRANSLATION_DIRECTORY,
    objective,
    observe,
    output_name,
    prepare_workspace,
    require_intact,
)
from tests.support.corpus import synthetic_book


def prepared_lane(tmp_path):
    book, identity, manifest = synthetic_book(tmp_path)
    corpus = Corpus(book, identity)
    workspace = tmp_path / "lane/workspace"
    workspace.mkdir(parents=True)
    staged = prepare_workspace(workspace, corpus)
    return workspace, staged, manifest, corpus


def client_for(workspace, staged):
    return SimpleNamespace(workspace=workspace, workspace_manifest=staged)


def test_workspace_stages_the_book_volumes_specification_and_output(tmp_path):
    workspace, staged, manifest, _corpus = prepared_lane(tmp_path)
    assert staged["kind"] == "native_goal_translation"
    assert staged["target_language"] == TARGET_LANGUAGE
    assert staged["source_root"] is None
    assert (
        staged["specification_sha256"]
        == hashlib.sha256((workspace / "TASK.md").read_bytes()).hexdigest()
    )
    assert [volume["name"] for volume in staged["corpus"]["volumes"]] == [
        f"pg{record['gutenberg_id']}.txt" for record in manifest["volumes"]
    ]
    assert sorted(entry.name for entry in workspace.iterdir()) == [
        "TASK.md",
        "corpus",
        "notes",
        "translation",
    ]
    assert sorted(entry.name for entry in (workspace / STAGED_DIRECTORY).iterdir()) == [
        "book.txt",
        MANIFEST_NAME,
        "pg731.txt",
        "pg732.txt",
    ]
    assert json.dumps(staged).find(str(tmp_path)) == -1
    text = (workspace / "TASK.md").read_text()
    assert "read-only" in text and "pg731.md" in text and "book.txt" in text
    # The workload writes whole files, so the guard rejects the edit tool.
    assert "rejects edit" in text


def test_objectives_resume_the_same_book_without_an_hour_requirement():
    first, second = objective(0), objective(1)
    assert f"{STAGED_DIRECTORY}/{MANIFEST_NAME}" in first
    assert TRANSLATION_DIRECTORY in first and "notes/progress.md" in first
    assert "compaction" in first
    assert "notes/progress.md" in second and f"{STAGED_DIRECTORY}/" in second
    for text in (first, second):
        assert "hour" not in text
        assert "3600" not in text


def test_observation_binds_progress_and_read_only_inputs(tmp_path):
    workspace, staged, manifest, _corpus = prepared_lane(tmp_path)
    volume = manifest["volumes"][0]["gutenberg_id"]
    output = workspace / TRANSLATION_DIRECTORY / output_name(f"pg{volume}.txt")
    output.write_text("第一章 译文\n", encoding="utf-8")
    (workspace / "notes/progress.md").write_text("started\n", encoding="utf-8")
    observed = observe(client_for(workspace, staged))
    assert observed["expected_volumes"] == len(manifest["volumes"])
    assert observed["output_volumes"] == 1
    assert observed["missing_output_volumes"] == ["pg732.txt"]
    assert observed["translated_characters"] == len("第一章 译文\n")
    assert observed["book_unchanged"] is True
    assert observed["progress_notes_present"] is True
    assert observed["corpus_unchanged"] is True
    assert observed["specification_unchanged"] is True
    assert observed["task_complete"] is False
    assert observed["observation_only"]
    assert require_intact(client_for(workspace, staged), _corpus) == {
        "staged_inputs_verified": True,
        "specification_verified": True,
    }


@pytest.mark.parametrize(
    "damage,message",
    [
        ("book", "accepted book bytes differ"),
        ("volume", "corpus volume differs"),
        ("manifest", "differs from the retained download"),
        ("specification", "lane specification changed during collection"),
    ],
)
def test_changed_read_only_inputs_fail_the_gate(tmp_path, damage, message):
    workspace, staged, manifest, corpus = prepared_lane(tmp_path)
    client = client_for(workspace, staged)
    path = {
        "book": workspace / STAGED_DIRECTORY / "book.txt",
        "volume": workspace / STAGED_DIRECTORY / "pg732.txt",
        "manifest": workspace / STAGED_DIRECTORY / MANIFEST_NAME,
        "specification": workspace / "TASK.md",
    }[damage]
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match=message):
        require_intact(client, corpus)
    observed = observe(client)
    assert (
        observed["corpus_unchanged"] is False
        or observed["specification_unchanged"] is False
    )
    assert manifest["book_file"] == "book.txt"


GUARD = (
    Path(__file__).resolve().parents[2] / "benchmarks/agentic_translation_tool_guard.ts"
)
DRIVER = (
    Path(__file__).resolve().parents[1] / "fixtures/translation_tool_guard_driver.ts"
)


def guard_decision(tool: str, payload: dict) -> dict | None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the pinned extension guard")
    result = subprocess.run(
        [node, str(DRIVER), tool, json.dumps(payload)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "tool,payload,blocked",
    [
        ("read", {"path": "corpus/pg731.txt"}, False),
        ("read", {"path": "corpus/book.txt"}, False),
        ("write", {"path": "corpus/pg731.txt"}, True),
        ("write", {"path": "corpus/book.txt"}, True),
        ("edit", {"path": "corpus/new.txt"}, True),
        ("write", {"path": "TASK.md"}, True),
        ("write", {"path": "translation/pg731.md"}, False),
        ("read", {"path": "notes/progress.md"}, False),
        ("read", {"path": "../outside.txt"}, True),
        ("bash", {"command": "rm -rf corpus"}, True),
        ("goal", {"op": "create"}, True),
        ("goal", {"op": "complete"}, False),
    ],
)
def test_translation_guard_keeps_the_corpus_and_specification_read_only(
    tool, payload, blocked
):
    decision = guard_decision(tool, payload)
    assert (decision is not None) is blocked
    if blocked:
        assert decision["block"] is True
        assert decision["reason"]


@pytest.mark.parametrize(
    "payload",
    [
        {"i": "remove a corpus line", "input": "[corpus/pg731.txt#A1B2]\nREM"},
        {"i": "move the specification", "input": "[TASK.md#A1B2]\nMV notes/task.md"},
        {"i": "escape the workspace", "input": "[../outside.txt#A1B2]\nREM"},
        {
            "i": "insert into an output",
            "input": "[translation/pg731.md#A1B2]\nINS 译文",
        },
    ],
)
def test_translation_guard_rejects_pinned_hashline_edit_payloads(payload):
    """Pinned OMP edit calls carry {i, input} hashline sections, not a path."""
    decision = guard_decision("edit", payload)
    assert decision is not None and decision["block"] is True
    assert "write tool" in decision["reason"]


def test_translation_guard_driver_uses_the_shipped_guard():
    assert GUARD.is_file() and DRIVER.is_file()
