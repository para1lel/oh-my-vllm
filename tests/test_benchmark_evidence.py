"""Generic collector evidence remains valid without cross-mode comparisons."""

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import evidence


def checkpoint(directory):
    (directory / "config.json").write_text('{"hidden_size":5120}')
    (directory / "model.safetensors").write_bytes(b"synthetic target weights")
    return evidence.checkpoint_identity(directory)


def test_checkpoint_identity_changes_with_weights_and_configuration(tmp_path):
    first = checkpoint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"changed target weights")
    second = evidence.checkpoint_identity(tmp_path)
    assert first["tree_sha256"] != second["tree_sha256"]
    assert first["files"]["config.json"] == second["files"]["config.json"]
    (tmp_path / "config.json").write_text('{"hidden_size":7}')
    third = evidence.checkpoint_identity(tmp_path)
    assert third["tree_sha256"] != second["tree_sha256"]
    assert third["files"]["model.safetensors"] == second["files"]["model.safetensors"]


def test_checkpoint_identity_contains_nested_relative_files(tmp_path):
    checkpoint(tmp_path)
    nested = tmp_path / "tokenizer"
    nested.mkdir()
    (nested / "vocab.json").write_bytes(b"{}")
    expected = evidence.checkpoint_identity(tmp_path)
    manifest = hashlib.sha256()
    for name, row in sorted(expected["files"].items()):
        encoded = name.encode()
        manifest.update(len(encoded).to_bytes(8, "little"))
        manifest.update(encoded)
        manifest.update(bytes.fromhex(row["sha256"]))
    assert "tokenizer/vocab.json" in expected["files"]
    assert all(not Path(name).is_absolute() for name in expected["files"])
    assert expected["tree_sha256"] == manifest.hexdigest()


@pytest.mark.parametrize("missing", ["config.json", "model.safetensors"])
def test_checkpoint_requires_configuration_and_weights(tmp_path, missing):
    checkpoint(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(ValueError):
        evidence.checkpoint_identity(tmp_path)


def test_artifact_export_keeps_values_and_original_hash(tmp_path):
    raw, portable = tmp_path / "raw.json", tmp_path / "portable.json"
    artifact = {
        "rows": [{"elapsed_s": 1.25, "tokens": 17}],
        "command": ["/tmp/private/worker"],
    }
    evidence.write_artifact(raw, artifact, portable)
    derived = json.loads(portable.read_text())
    assert derived["artifact_kind"] == "portable-derived-evidence"
    assert derived["data"]["rows"] == artifact["rows"]
    assert "command" not in derived["data"]
    assert derived["original"]["sha256"] == hashlib.sha256(raw.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="must not replace"):
        evidence.write_artifact(raw, {}, raw)
    assert json.loads(raw.read_text()) == artifact


def test_artifact_export_rejects_nonfinite_numbers(tmp_path):
    path = tmp_path / "raw.json"
    with pytest.raises(ValueError):
        evidence.write_artifact(path, {"elapsed_s": float("nan")})
    assert not path.exists()


@pytest.mark.parametrize("structured", [False, True])
def test_loaded_checkpoint_proof_matches_actual_configuration_and_weights(
    tmp_path, structured
):
    expected = checkpoint(tmp_path)
    proof = {
        "path": str(tmp_path),
        "config_sha256": expected["files"]["config.json"]["sha256"],
        "weights_sha256": {
            "model.safetensors": expected["files"]["model.safetensors"]["sha256"]
        },
    }
    message = "DSPARK_CHECKPOINT_PROVENANCE " + json.dumps(proof)
    log = json.dumps({"message": message}) if structured else message
    assert evidence.loaded_draft_provenance(log, expected) == proof
    with pytest.raises(ValueError, match="one actually loaded"):
        evidence.loaded_draft_provenance(log + "\n" + log, expected)


@pytest.mark.parametrize(
    "corruption", ["path", "config", "weight", "unknown_weight", "empty_weights"]
)
def test_loaded_checkpoint_proof_rejects_changed_or_missing_identity(
    tmp_path, corruption
):
    expected = checkpoint(tmp_path)
    proof = {
        "path": str(tmp_path),
        "config_sha256": expected["files"]["config.json"]["sha256"],
        "weights_sha256": {
            "model.safetensors": expected["files"]["model.safetensors"]["sha256"]
        },
    }
    if corruption == "path":
        proof["path"] = str(tmp_path / "other")
    elif corruption == "config":
        proof["config_sha256"] = "0" * 64
    elif corruption == "weight":
        proof["weights_sha256"]["model.safetensors"] = "0" * 64
    elif corruption == "unknown_weight":
        proof["weights_sha256"]["unknown.safetensors"] = "0" * 64
    else:
        proof["weights_sha256"] = {}
    with pytest.raises(ValueError):
        evidence.loaded_draft_provenance(
            "DSPARK_CHECKPOINT_PROVENANCE " + json.dumps(proof), expected
        )


@pytest.mark.parametrize("mode", ["mtp", "dspark"])
def test_service_evidence_requires_actual_method_and_response_ids(tmp_path, mode):
    path = tmp_path / "service.log"
    header = f"INFO BENCH_CONFIG speculative_mode={mode}\n"
    line = (
        "request_id=17 verified_draft_tokens=14 "
        "accepted_draft_tokens=3 request completed\n"
    )
    path.write_text(header + line)
    result = evidence.read_service_evidence(
        path, len(header.encode()), mode, request_ids={17}, expected=1
    )
    assert result == [
        {"request_id": 17, "verified_draft_tokens": 14, "accepted_draft_tokens": 3}
    ]


@pytest.mark.parametrize(
    "failure",
    [
        "mode",
        "request_ids",
        "proposed_only",
        "too_many_accepted",
        "duplicate",
        "zero_verified",
        "missing_config",
        "duplicate_config",
        "count",
        "truncated",
    ],
)
def test_service_evidence_rejects_incomplete_or_mismatched_requests(tmp_path, failure):
    path = tmp_path / "service.log"
    header = "INFO BENCH_CONFIG speculative_mode=dspark\n"
    line = (
        "request_id=17 verified_draft_tokens=14 "
        "accepted_draft_tokens=3 request completed\n"
    )
    mode, ids, count, offset = "dspark", {17}, 1, 0
    if failure == "mode":
        mode = "mtp"
    elif failure == "request_ids":
        ids = {18}
    elif failure == "proposed_only":
        line = line.replace("verified_draft_tokens", "proposed_draft_tokens")
    elif failure == "too_many_accepted":
        line = line.replace("accepted_draft_tokens=3", "accepted_draft_tokens=15")
    elif failure == "duplicate":
        line += line
    elif failure == "zero_verified":
        line = line.replace(
            "verified_draft_tokens=14", "verified_draft_tokens=0"
        ).replace("accepted_draft_tokens=3", "accepted_draft_tokens=0")
    elif failure == "missing_config":
        header = ""
    elif failure == "duplicate_config":
        header += header
    elif failure == "count":
        count = 2
    else:
        offset = len((header + line).encode()) + 1
    path.write_text(header + line)
    with pytest.raises(ValueError):
        evidence.read_service_evidence(
            path, offset, mode, request_ids=ids, expected=count
        )


def test_runtime_source_identity_preserves_common_identity_and_source_bytes(
    tmp_path, monkeypatch
):
    directories = ("crates", "python", "benchmarks", "scripts")
    for directory in directories:
        (tmp_path / directory).mkdir()
    sources = {
        "crates/model.rs": "fn main() {}",
        "python/model.py": "x = 1",
        "scripts/run.sh": "true",
        "python/kernel.cu": "void kernel() {}",
    }
    for name, text in sources.items():
        (tmp_path / name).write_text(text)
    monkeypatch.setattr(evidence, "__file__", str(tmp_path / "benchmarks/evidence.py"))
    monkeypatch.setattr(
        evidence,
        "source_identity",
        lambda binary: {"git_commit": "c" * 40, "binary_sha256": "b" * 64},
    )
    first = evidence.runtime_source_identity(tmp_path / "worker")
    assert first["git_commit"] == "c" * 40
    assert set(first["runtime_sources_sha256"]) == set(sources)
    (tmp_path / "python/kernel.cu").write_text("void changed_kernel() {}")
    second = evidence.runtime_source_identity(tmp_path / "worker")
    assert (
        first["runtime_sources_sha256"]["python/kernel.cu"]
        != second["runtime_sources_sha256"]["python/kernel.cu"]
    )


def test_observed_configuration_requires_one_record_and_preserves_quoted_values():
    line = (
        'INFO BENCH_CONFIG speculative_mode=dspark allocator="cuda async" '
        "num_speculative_tokens=7"
    )
    assert evidence.observed_configuration(line) == {
        "speculative_mode": "dspark",
        "allocator": "cuda async",
        "num_speculative_tokens": "7",
    }
    for log in ("no observed configuration", line + "\n" + line):
        with pytest.raises(ValueError, match="one observed"):
            evidence.observed_configuration(log)


def test_runtime_source_identity_changes_when_accumulator_header_changes(
    tmp_path, monkeypatch
):
    header = tmp_path / "python/kernels/groupwise_accum.cuh"
    header.parent.mkdir(parents=True)
    header.write_text("struct Accumulator { int version = 1; };\n")
    monkeypatch.setattr(evidence, "__file__", str(tmp_path / "benchmarks/evidence.py"))
    monkeypatch.setattr(
        evidence, "source_identity", lambda binary: {"binary_sha256": "b" * 64}
    )
    before = evidence.runtime_source_identity(tmp_path / "worker")
    header.write_text("struct Accumulator { int version = 2; };\n")
    after = evidence.runtime_source_identity(tmp_path / "worker")
    key = "python/kernels/groupwise_accum.cuh"
    assert before["binary_sha256"] == after["binary_sha256"]
    assert before["runtime_sources_sha256"][key] != after["runtime_sources_sha256"][key]
