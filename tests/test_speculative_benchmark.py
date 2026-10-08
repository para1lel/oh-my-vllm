"""CPU evidence checks for the current-source native MTP4/DSpark comparison."""

import copy
import datetime
import json
import os

import pytest

from benchmarks import speculative as bench


def measurement(mode="mtp", batch_size=1, rate=100.0, **changes):
    row = {
        "speculative_mode": mode,
        "batch_size": batch_size,
        "input_len": 32768,
        "output_len": 4096,
        "output_tokens": batch_size * 4096,
        "elapsed_s": batch_size * 4096 / rate,
        "output_tps": rate,
        "ttft_s": [1.0] * batch_size,
        "preemptions": 0,
        "prefix_hit_tokens": 0,
        "verified_draft_tokens": 20,
        "proposed_draft_tokens": 27,
        "accepted_draft_tokens": 12,
        "steps": 10,
    }
    row.update(changes)
    return row


def rounds(mtp=None, dspark=None, batch_size=1):
    mtp = mtp or [[100.0] * 5] * 3
    dspark = dspark or [[101.0] * 5] * 3
    result = []
    for number in range(3):
        pairs = []
        for index in range(5):
            pairs.append(
                {
                    "order": list(
                        bench.MODES
                        if (number + index) % 2 == 0
                        else reversed(bench.MODES)
                    ),
                    "mtp": measurement("mtp", batch_size, mtp[number][index]),
                    "dspark": measurement("dspark", batch_size, dspark[number][index]),
                }
            )
        result.append(
            {
                "warmups": {
                    mode: [measurement(mode, batch_size)] * 2 for mode in bench.MODES
                },
                "pairs": pairs,
            }
        )
    return result


def records(batch_size=1):
    result = []
    for number, data in enumerate(rounds(batch_size=batch_size)):
        for mode in bench.MODES:
            for index, row in enumerate(data["warmups"][mode]):
                result.append(
                    row
                    | {"round": number, "phase": "warmup", "index": index, "slot": 0}
                )
        for index, pair in enumerate(data["pairs"]):
            for slot, mode in enumerate(pair["order"]):
                result.append(
                    pair[mode]
                    | {
                        "round": number,
                        "phase": "measured",
                        "index": index,
                        "slot": slot,
                    }
                )
    return result


def test_small_stable_gain_passes_and_ttft_is_only_reported():
    data = rounds(dspark=[[100.00001] * 5] * 3)
    for item in data:
        for pair in item["pairs"]:
            pair["dspark"]["ttft_s"] = [30.0]
    result = bench.compare_rounds(data, 1, bootstrap_samples=1000)
    assert result["passed"]
    assert result["gain_lower_95_tps"] > 0
    assert not result["ttft_gated"]
    assert result["rounds"][0]["dspark"]["ttft_s_median"] == 30.0
    assert result["rounds"][0]["dspark"]["verified_draft_tokens"] == 100
    assert result["rounds"][0]["dspark"]["acceptance_rate"] == 0.6


@pytest.mark.parametrize(
    "kind", ["tie", "one_regressing_round", "unstable", "noisy_positive_median"]
)
def test_inferior_or_variable_speed_is_reported_without_an_additional_gate(kind):
    mtp = [[100.0] * 5] * 3
    dspark = [[101.0] * 5] * 3
    if kind == "tie":
        dspark = mtp
    elif kind == "one_regressing_round":
        dspark = [[110.0] * 5, [110.0] * 5, [99.0] * 5]
    elif kind == "unstable":
        mtp = [[100.0] * 4 + [111.0]] * 3
        dspark = [[120.0] * 5] * 3
    else:
        dspark = [[100.2] * 3 + [95.0] * 2] * 3
    result = bench.compare_rounds(rounds(mtp, dspark), 1, bootstrap_samples=1000)
    assert result["passed"]
    assert not result["additional_tps_gate"]
    if kind == "unstable":
        assert result["rounds"][0]["tps_spread_exceeds_10_percent"]["mtp"]
    elif kind in ("tie", "one_regressing_round"):
        assert not result["rounds"][-1]["dspark_median_faster"]
    else:
        assert not result["paired_gain_positive"]


def test_ten_percent_spread_boundary_passes():
    result = bench.compare_rounds(
        rounds([[100.0] * 4 + [110.0]] * 3, [[120.0] * 5] * 3),
        1,
        bootstrap_samples=1000,
    )
    assert result["passed"]
    assert result["rounds"][0]["mtp"]["output_tps_spread"] == 0.10


@pytest.mark.parametrize(
    "changes",
    [
        {"verified_draft_tokens": 0},
        {"verified_draft_tokens": None},
        {"verified_draft_tokens": 2},
        {"accepted_draft_tokens": -1},
        {"preemptions": 1},
        {"prefix_hit_tokens": 784},
        {"output_tokens": 100},
        {"elapsed_s": 1.0},
        {"ttft_s": [float("nan")]},
        {"speculative_mode": "dspark"},
    ],
)
def test_raw_next_proposals_cannot_replace_actual_verified_drafts(changes):
    with pytest.raises(ValueError):
        bench.validate_measurement(measurement(**changes), "mtp", 1)


def test_pairing_warmups_and_execution_order_are_checked():
    data = records()
    organized = bench.organize_records(data, 1)
    assert len(data) == 42
    assert bench.compare_rounds(organized, 1, bootstrap_samples=1000)["passed"]
    for bad in (data[:-1], [*data, data[-1]], [data[1], data[0], *data[2:]]):
        with pytest.raises(ValueError, match="record"):
            bench.organize_records(bad, 1)
    bad = copy.deepcopy(organized)
    bad[0]["pairs"][0]["order"].reverse()
    with pytest.raises(ValueError, match="alternate"):
        bench.compare_rounds(bad, 1)
    bad = copy.deepcopy(organized)
    bad[2]["warmups"]["dspark"].pop()
    with pytest.raises(ValueError, match="two full warmups"):
        bench.compare_rounds(bad, 1)


def test_checkpoint_hashes_cover_weight_content_and_configuration(tmp_path):
    (tmp_path / "config.json").write_text('{"block_size":7}')
    (tmp_path / "model.safetensors").write_bytes(b"first weights")
    first = bench.checkpoint_identity(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"other weights")
    second = bench.checkpoint_identity(tmp_path)
    assert first["tree_sha256"] != second["tree_sha256"]
    assert first["files"]["config.json"] == second["files"]["config.json"]
    (tmp_path / "config.json").write_text('{"block_size":8}')
    assert bench.checkpoint_identity(tmp_path)["tree_sha256"] != second["tree_sha256"]


def test_raw_and_portable_artifacts_keep_numeric_records_without_host_paths(tmp_path):
    raw, portable = tmp_path / "raw.json", tmp_path / "portable.json"
    artifact = {
        "rows": rounds(),
        "command": ["/tmp/private/worker"],
        "python": "/env/python",
    }
    bench.write_artifact(raw, artifact, portable)
    derived = json.loads(portable.read_text())
    assert derived["artifact_kind"] == "portable-derived-evidence"
    assert derived["data"]["rows"] == artifact["rows"]
    assert "command" not in derived["data"]
    assert "python" not in derived["data"]
    with pytest.raises(ValueError, match="original raw"):
        bench.validate_artifact(derived)
    with pytest.raises(ValueError, match="must not replace"):
        bench.write_artifact(raw, {}, raw)
    assert json.loads(raw.read_text()) == artifact


def test_service_logs_require_actual_method_and_matching_request_ids(tmp_path):
    log = tmp_path / "service.log"
    header = "INFO BENCH_CONFIG speculative_mode=dspark comparison=false\n"
    line = (
        "request_id=17 verified_draft_tokens=14 "
        "accepted_draft_tokens=3 request completed\n"
    )
    log.write_text(header + line)
    rows = bench.read_service_evidence(
        log, len(header.encode()), "dspark", request_ids={17}
    )
    assert rows[0]["verified_draft_tokens"] == 14
    for content, mode, ids in (
        (header + line, "mtp", {17}),
        (header + line, "dspark", {18}),
        (
            header
            + line.replace("verified_draft_tokens=14", "proposed_draft_tokens=14"),
            "dspark",
            {17},
        ),
        (
            header
            + line.replace("accepted_draft_tokens=3", "accepted_draft_tokens=15"),
            "dspark",
            {17},
        ),
        (header + line + line, "dspark", {17}),
    ):
        log.write_text(content)
        with pytest.raises(ValueError):
            bench.read_service_evidence(log, 0, mode, request_ids=ids)


def artifact():
    source = {
        "git_commit": "a" * 40,
        "git_status": "",
        "binary_sha256": "b" * 64,
        "runtime_sources_sha256": {"crates/zmq-worker/src/main.rs": "c" * 64},
        "python_sources_sha256": {"python/oh_my_vllm/worker/model_runner.py": "d" * 64},
    }
    checkpoint = {
        "path": "/tmp/draft",
        "tree_sha256": "e" * 64,
        "files": {
            "config.json": {"sha256": "f" * 64},
            "model.safetensors": {"sha256": "1" * 64},
        },
    }
    checkpoints = {"target": copy.deepcopy(checkpoint), "draft": checkpoint}
    hardware = {
        "gpu": "GPU-test-shared",
        "gpu_info": "GPU-test-shared, B200, 200000 MiB, driver",
        "cpu_affinity": [0],
    }
    result = {
        "source": source,
        "source_end": copy.deepcopy(source),
        "checkpoints": checkpoints,
        "checkpoints_end": copy.deepcopy(checkpoints),
        "hardware": hardware,
        "hardware_end": copy.deepcopy(hardware),
        "runtime": {"python": "/tmp/python"},
        "protocol": {"dspark_confidence_threshold": 0.0},
        "requested_capacities": {"fa": 1400, "mamba": 128},
        "rows": [],
    }
    for batch in bench.BATCH_SIZES:
        result["rows"].append(
            {
                "batch_size": batch,
                "source": copy.deepcopy(source),
                "hardware": copy.deepcopy(hardware),
                "config": {
                    "max_num_seqs": "32",
                    "max_num_batched_tokens": "32768",
                    "block_size": "784",
                    "max_model_len": "65536",
                    "worker_fa_pool_blocks": "1400",
                    "worker_gdn_pool_blocks": "128",
                    "fa_pool_blocks": "1400",
                    "gdn_pool_blocks": "128",
                    "comparison": "true",
                    "speculative_mode": "comparison",
                    "dspark_confidence_threshold": "0",
                },
                "measurement_audit": {
                    "steady_state_verified": True,
                    "phase_count": 42,
                    "measured_intervals": [[index, index + 1] for index in range(30)],
                },
                "cuda_build_provenance": {"so_sha256": "2" * 64},
                "loaded_draft": {
                    "path": "/tmp/draft",
                    "config_sha256": "f" * 64,
                    "weights_sha256": {"model.safetensors": "1" * 64},
                },
                "memory": {"max_reserved_bytes": 128},
                "log_sha256": "3" * 64,
                "records": records(batch),
                "rounds": bench.organize_records(records(batch), batch),
            }
        )
    return result


def test_full_comparison_identity_and_loaded_checkpoint_are_required():
    original = artifact()
    assert all(item["passed"] for item in bench.validate_artifact(original))
    for failure in (
        "single_mode",
        "false_comparison",
        "source",
        "hardware",
        "checkpoint",
        "loaded_weights",
        "capture",
        "capacity",
        "incomplete",
        "missing_records",
        "changed_rounds",
        "wrong_phase_count",
        "confidence_threshold",
    ):
        bad = copy.deepcopy(original)
        if failure == "single_mode":
            bad["rows"][0]["config"]["speculative_mode"] = "mtp"
        elif failure == "false_comparison":
            bad["rows"][0]["config"]["comparison"] = "false"
        elif failure == "source":
            bad["source_end"]["binary_sha256"] = "4" * 64
        elif failure == "hardware":
            bad["hardware_end"]["gpu"] = "GPU-test-changed"
        elif failure == "checkpoint":
            bad["checkpoints_end"]["draft"]["files"]["model.safetensors"]["sha256"] = (
                "5" * 64
            )
        elif failure == "loaded_weights":
            bad["rows"][0]["loaded_draft"]["weights_sha256"]["model.safetensors"] = (
                "6" * 64
            )
        elif failure == "capture":
            bad["rows"][0]["measurement_audit"]["steady_state_verified"] = False
        elif failure == "capacity":
            bad["rows"][0]["config"]["worker_fa_pool_blocks"] = "1399"
        elif failure == "missing_records":
            del bad["rows"][0]["records"]
        elif failure == "changed_rounds":
            bad["rows"][0]["rounds"][0]["pairs"][0]["mtp"]["output_tps"] = 99
        elif failure == "wrong_phase_count":
            bad["rows"][0]["measurement_audit"]["phase_count"] = 41
        elif failure == "confidence_threshold":
            bad["rows"][0]["config"]["dspark_confidence_threshold"] = "0.05"
        else:
            bad["rows"].pop()
        with pytest.raises(ValueError):
            bench.validate_artifact(bad)


def test_loaded_checkpoint_proof_parses_worker_structured_log():
    original = artifact()
    proof = original["rows"][0]["loaded_draft"]
    log = json.dumps({"message": "DSPARK_CHECKPOINT_PROVENANCE " + json.dumps(proof)})
    assert bench.loaded_draft_provenance(log, original["checkpoints"]["draft"]) == proof
    with pytest.raises(ValueError, match="one actually loaded"):
        bench.loaded_draft_provenance(
            log + "\n" + log, original["checkpoints"]["draft"]
        )


def paired_log():
    """Give each full attempt ten seconds, with independently explicit phases."""
    lines = []
    for index, record in enumerate(records()):
        timestamp = datetime.datetime.fromtimestamp(
            1700000000 + index * 10, datetime.UTC
        ).isoformat()
        fields = " ".join(
            f'{key}="{record[key]}"'
            if isinstance(record[key], str)
            else f"{key}={record[key]}"
            for key in ("round", "phase", "index", "slot", "speculative_mode")
        )
        flag = str(record["phase"] == "measured").lower()
        lines.append(f"{timestamp} BENCH_PHASE {fields} measured={flag}")
    return lines


@pytest.mark.parametrize(
    "activity",
    [
        "Compilation started: IR unit model_features",
        "Capturing CUDA graph: family=dspark_context key=(8,) rows=8",
        "Capturing CUDA graph: family=dspark key=(True, 1, 40960) rows=7",
    ],
)
def test_paired_audit_permits_each_warmup_but_rejects_measured_compilation(
    tmp_path, activity
):
    cache = tmp_path / "cache"
    cache.mkdir()
    log = tmp_path / "worker.log"
    lines = paired_log()
    lines.insert(15, activity)
    log.write_text("\n".join(lines))
    result = bench.audit_paired(log, [str(cache)])
    assert result["steady_state_verified"]
    assert result["phase_count"] == 42
    assert len(result["measured_intervals"]) == 30
    lines.insert(22, activity)
    log.write_text("\n".join(lines))
    result = bench.audit_paired(log, [str(cache)])
    assert not result["steady_state_verified"]
    assert activity in result["problems"]


def test_paired_cache_audit_uses_the_same_phase_windows(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    artifact = cache / "compiled.bin"
    artifact.write_bytes(b"compiled")
    log = tmp_path / "worker.log"
    log.write_text("\n".join(paired_log()))
    for phase_index, expected in ((14, True), (17, True), (18, False), (38, False)):
        moment = 1700000000 + phase_index * 10 + 1
        os.utime(artifact, (moment, moment))
        result = bench.audit_paired(log, [str(cache)])
        assert result["steady_state_verified"] is expected
    assert not bench.audit_paired(log, [None])["steady_state_verified"]


@pytest.mark.parametrize("failure", ["missing", "false", "swapped", "no_time"])
def test_paired_phase_markers_cannot_hide_measured_work(tmp_path, failure):
    cache = tmp_path / "cache"
    cache.mkdir()
    lines = paired_log()
    if failure == "missing":
        lines.pop(14)
    elif failure == "false":
        # Keep thirty true flags, but move one measurement into a warmup slot.
        lines[4] = lines[4].replace("measured=true", "measured=false")
        lines[14] = lines[14].replace("measured=false", "measured=true")
    elif failure == "swapped":
        lines[14], lines[15] = lines[15], lines[14]
    else:
        lines[14] = lines[14].split(" ", 1)[1]
    log = tmp_path / "worker.log"
    log.write_text("\n".join(lines))
    assert not bench.audit_paired(log, [str(cache)])["steady_state_verified"]


def test_eager_diagnosis_is_rejected_before_collection(monkeypatch):
    monkeypatch.setenv("OH_MY_VLLM_ENFORCE_EAGER", "1")
    with pytest.raises(ValueError, match="disable eager"):
        bench.collect(None)
