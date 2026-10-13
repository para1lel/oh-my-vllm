"""Worker failure evidence must survive a racing GPU inventory."""

import json

import pytest

from benchmarks import agentic_soak_support as support


def recorded(directory):
    return [
        json.loads(row) for row in (directory / "memory.jsonl").read_text().splitlines()
    ]


def setup(monkeypatch, samples):
    iterator = iter(samples)
    monkeypatch.setattr(support, "diagnostics", lambda _: next(iterator))
    monkeypatch.setattr(
        support,
        "resource_checks",
        lambda sample: [
            {"name": "running worker", "passed": sample["status"] == "running"},
            {"name": "resource limit", "passed": sample.get("bounded", True)},
        ],
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "recovering", "restarts": 1},
        {"epoch": 2},
        {"bounded": False},
    ],
)
def test_failed_worker_snapshot_is_saved_without_gpu_lookup(
    monkeypatch, tmp_path, changes
):
    sample = {
        "status": "running",
        "epoch": 1,
        "worker_pid": 99,
        "restarts": 0,
    } | changes
    setup(monkeypatch, [sample])

    def forbidden(_):
        pytest.fail("GPU lookup must not mask the failed worker snapshot")

    monkeypatch.setattr(support, "gpu_processes", forbidden)
    with pytest.raises(ValueError, match="worker or resource checks failed"):
        support.resource_sample("local", 1, support.Evidence(tmp_path))
    rows = recorded(tmp_path)
    assert len(rows) == 1
    assert rows[0]["epoch"] == sample["epoch"]
    assert rows[0]["status"] == sample["status"]
    assert rows[0]["sampling_error"]


def test_gpu_inventory_race_keeps_original_and_replacement_snapshot(
    monkeypatch, tmp_path
):
    first = {"status": "running", "epoch": 1, "worker_pid": 99, "restarts": 0}
    fresh = first | {"status": "recovering", "restarts": 1}
    setup(monkeypatch, [first, fresh])

    def disappeared(_):
        raise ValueError("worker missing")

    monkeypatch.setattr(support, "gpu_processes", disappeared)
    with pytest.raises(ValueError, match=r"status=recovering.*restarts=1"):
        support.resource_sample("local", 1, support.Evidence(tmp_path))
    rows = recorded(tmp_path)
    assert len(rows) == 2
    assert rows[0]["sampling_error"] == "GPU process inventory: worker missing"
    assert rows[1]["observation"] == "gpu_inventory_failure_recheck"
    assert rows[1]["sampling_error"] == ["running worker"]


def test_real_null_worker_recovery_is_saved(monkeypatch, tmp_path):
    sample = {
        "status": "recovering",
        "epoch": 1,
        "worker_pid": 99,
        "restarts": 1,
        "worker": None,
    }
    monkeypatch.setattr(support, "diagnostics", lambda _: sample)

    def forbidden(_):
        pytest.fail("GPU lookup must not run for an invalidated worker")

    monkeypatch.setattr(support, "gpu_processes", forbidden)
    with pytest.raises(ValueError, match="worker_snapshot_present"):
        support.resource_sample("local", 1, support.Evidence(tmp_path))
    assert recorded(tmp_path)[0]["worker"] is None


def test_inventory_and_unavailable_http_recheck_both_survive(monkeypatch, tmp_path):
    first = {"status": "running", "epoch": 1, "worker_pid": 99, "restarts": 0}
    setup(monkeypatch, [first])

    def disappeared(_):
        raise ValueError("worker missing")

    monkeypatch.setattr(support, "gpu_processes", disappeared)
    with pytest.raises(ValueError, match="worker missing; worker recheck failed"):
        support.resource_sample("local", 1, support.Evidence(tmp_path))
    rows = recorded(tmp_path)
    assert len(rows) == 2
    assert rows[0]["sampling_error"] == "GPU process inventory: worker missing"
    assert rows[1]["observation"] == "gpu_inventory_failure_recheck"
