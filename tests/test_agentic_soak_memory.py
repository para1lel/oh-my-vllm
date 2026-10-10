"""Temporal late-half memory evidence controls completion and final auditing."""

import copy

from benchmarks.agentic_soak_analysis import (
    late_idle_samples,
    normal_run_complete,
    plateau,
)


def samples():
    return [
        {
            "phase": "normal",
            "observation": "drained",
            "collected_monotonic_s": 1000 + index * 300,
            "worker_snapshot_unix_s": 100_000 + index * 300,
            "epoch": 1,
            "worker_pid": 100,
            "worker": {"gpu": {"allocated_bytes": 1000}},
        }
        for index in range(12)
    ]


def test_last_half_uses_normal_monotonic_window_and_excludes_other_phases():
    rows = samples()
    warm = dict(rows[-1], phase="warmup", collected_monotonic_s=4590)
    active = dict(rows[-1], observation="active", collected_monotonic_s=4590)
    probe = dict(rows[-1], phase="intentional_probe", collected_monotonic_s=4590)
    selected = late_idle_samples([*rows, warm, active, probe], 1000, 4600)
    assert [row["collected_monotonic_s"] for row in selected] == [
        2800,
        3100,
        3400,
        3700,
        4000,
        4300,
    ]
    # A changed end moves the true midpoint; an old cutoff cannot remain valid.
    assert len(late_idle_samples(rows, 1000, 5200)) == 5
    for row in rows:
        row["worker_snapshot_unix_s"] += 1_000_000
    assert len(late_idle_samples(rows, 1000, 4600)) == 6


def test_complete_duration_and_count_still_wait_for_six_late_idle_snapshots():
    rows = samples()
    assert normal_run_complete(
        3600, 3600, 500, 3600, 500, idle_samples=rows, normal_started=1000
    )
    assert not normal_run_complete(
        4200, 4200, 500, 3600, 500, idle_samples=rows, normal_started=1000
    )
    assert not normal_run_complete(
        3600, 3600, 500, 3600, 500, idle_samples=rows[:-1], normal_started=1000
    )


def test_repeated_reads_of_the_same_worker_snapshot_do_not_count_twice():
    rows = samples()
    copies = [
        dict(row, collected_monotonic_s=row["collected_monotonic_s"] + 1)
        for row in rows
    ]
    assert len(late_idle_samples([*rows, *copies], 1000, 4600)) == 6
    for row in copies:
        row["worker_snapshot_unix_s"] = rows[0]["worker_snapshot_unix_s"]
    assert late_idle_samples([rows[0], *copies], 1000, 4600) == []


def test_late_growth_cannot_be_hidden_by_higher_early_memory():
    rows = samples()
    for index, row in enumerate(rows):
        row["worker"]["gpu"]["allocated_bytes"] = (
            1000 if index < 6 else 100 if index < 9 else 500
        )
    result = plateau(
        rows,
        ("worker", "gpu", "allocated_bytes"),
        absolute_bytes=100,
        normal_started=1000,
        normal_ended=4600,
    )
    assert result["passed"] is False
    assert result["initial_late_median_bytes"] == 100
    assert result["final_late_median_bytes"] == 500
    assert result["late_samples"] == 6
    assert result["late_cutoff_monotonic_s"] == 2800
    assert result["late_first_monotonic_s"] == 2800
    assert result["late_last_monotonic_s"] == 4300
    assert result["late_span_s"] == 1500
    insufficient = copy.deepcopy(rows)
    insufficient[-1]["phase"] = "warmup"
    assert (
        plateau(
            insufficient,
            ("worker", "gpu", "allocated_bytes"),
            absolute_bytes=100,
            normal_started=1000,
            normal_ended=4600,
        )["passed"]
        is False
    )
