"""Intentional completion races stay distinct from genuine cancellation."""

import pytest

from benchmarks.agentic_soak_analysis import probe_outcomes


def records():
    probes = [
        {"request_id": 1, "api": "chat", "kind": "cancel"},
        {"request_id": 2, "api": "responses", "kind": "cancel"},
        {"request_id": 3, "api": "chat", "kind": "slow"},
        {"request_id": 4, "api": "responses", "kind": "cancel"},
    ]
    metrics = [
        {"request_id": p["request_id"], "api": p["api"], "status": status}
        for p, status in zip(
            probes, ("cancelled", "cancelled", "completed", "completed"), strict=True
        )
    ]
    return probes, metrics


def test_completion_race_is_retained_with_genuine_cancellation_per_api():
    probes, metrics = records()
    result = probe_outcomes(probes, metrics)
    assert result["genuine_cancellations"] == {"chat": 1, "responses": 1}
    assert len(result["outcomes"]) == len(probes)
    assert result["outcomes"][3]["outcome"] == "completed_before_cancel"


def test_failed_intentional_disconnect_does_not_count_as_cancellation():
    probes, metrics = records()
    metrics[0]["status"] = "failed"
    with pytest.raises(ValueError, match="unexpected server status failed"):
        probe_outcomes(probes, metrics)


def test_completion_races_do_not_replace_cancellation_coverage():
    probes, metrics = records()
    metrics[1]["status"] = "completed"
    with pytest.raises(ValueError, match="genuine cancellation per API"):
        probe_outcomes(probes, metrics)


def test_failed_or_cancelled_slow_client_is_rejected():
    probes, metrics = records()
    metrics[2]["status"] = "cancelled"
    with pytest.raises(ValueError, match="unexpected server status cancelled"):
        probe_outcomes(probes, metrics)


@pytest.mark.parametrize("invalid", ["missing", "api"])
def test_probe_requires_matching_server_identity(invalid):
    probes, metrics = records()
    if invalid == "missing":
        metrics.pop(0)
    else:
        metrics[0]["api"] = "responses"
    with pytest.raises(ValueError, match="matching server API metric"):
        probe_outcomes(probes, metrics)
