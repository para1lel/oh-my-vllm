"""Native-goal lifecycle transitions keep the hour and the same conversation."""

import hashlib
from types import SimpleNamespace

import pytest

from benchmarks import agentic_goals as goals


@pytest.mark.parametrize("terminal", ["complete", "paused", "dropped", "settled"])
@pytest.mark.parametrize("tail", [0, 3])
def test_early_goal_stop_submits_next_goal_in_the_same_client(
    monkeypatch, terminal, tail
):
    now = [100.0]
    monkeypatch.setattr(
        goals,
        "time",
        SimpleNamespace(
            monotonic=lambda: now[0],
            time=lambda: now[0] + 1000,
        ),
    )

    class Client:
        lane, timeout, error = 0, 10, None
        process = SimpleNamespace(poll=lambda: None)
        stop = SimpleNamespace(
            is_set=lambda: False,
            wait=lambda duration: now.__setitem__(0, now[0] + duration),
        )

        def __init__(self):
            self.responses, self.turns, self.created, self.commands = {}, [], [], []
            self.all_commands = []
            self.tail_deadline = 0

        def _send(self, kind, **fields):
            identity = f"host-{len(self.created)}"
            self.created.append(fields)
            self.responses[identity] = {
                "success": True,
                "data": {
                    "goal": {"id": f"goal-{len(self.created)}", "status": "active"},
                },
            }
            return identity

        def _wait(self, mapping, identity, timeout):
            return mapping.pop(identity)

        def command(self, kind, **fields):
            self.all_commands.append((kind, fields))
            if kind == "goal":
                self.commands.append(fields["op"])
                if fields["op"] == "pause":
                    self.tail_deadline = now[0] + tail
            return {"kind": kind}

        def check_state(self):
            status = terminal if len(self.created) == 1 else "active"
            if self.commands and self.commands[-1] == "pause":
                status = "paused"
            return {
                "sessionId": "unchanged-session",
                "isSettled": status != "active" and now[0] >= self.tail_deadline,
                "goal": None
                if status in ("complete", "dropped")
                else {
                    "enabled": True,
                    "goal": {
                        "id": f"goal-{len(self.created)}",
                        "status": "active" if status == "settled" else status,
                    },
                },
            }

        def checkpoint_entries(self):
            pass

        def persist(self):
            pass

    client = Client()
    collection = goals.run_goal(client, {"monotonic": 100.0}, 6)
    assert len(client.created) == 2
    assert client.created[0]["objective"] != client.created[1]["objective"]
    assert [turn["goal_id"] for turn in client.turns] == ["goal-1", "goal-2"]
    assert client.commands == ["pause", "drop", "pause"]
    assert sum(kind == "abort" for kind, _ in client.all_commands) == 1
    assert client.all_commands[-2:] == [("goal", {"op": "pause"}), ("abort", {})]
    assert client.turns[0]["collection_abort"] is None
    assert client.turns[-1]["collection_abort"] == {"kind": "abort"}
    assert client.turns[-1]["active_ended_monotonic_s"] >= 106.0
    assert client.turns[0]["transition_reason"] != "collection_deadline"
    assert all("result" not in turn for turn in client.turns)
    assert collection["collection_deadline_monotonic_s"] == 106.0
    assert collection["controller_ended_monotonic_s"] >= 106.0 + tail
    assert client.turns[-1]["paused_tail_seconds"] >= tail


def test_new_task_has_no_project_snapshot_and_rejects_nonempty_workspace(tmp_path):
    manifest = goals.prepare_workspace(tmp_path, 2)
    assert manifest["source_root"] is None
    assert (
        manifest["specification_sha256"]
        == hashlib.sha256((tmp_path / "TASK.md").read_bytes()).hexdigest()
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == ["TASK.md", "notes"]
    assert len(set(goals.objective(2, index) for index in range(12))) == 12
    with pytest.raises(ValueError, match="empty"):
        goals.prepare_workspace(tmp_path, 2)


def test_uncommanded_goal_replacement_is_still_an_integrity_failure():
    with pytest.raises(ValueError, match="without a host command"):
        goals.goal_state({"goal": {"goal": {"id": "other"}}}, "owned")


def test_incomplete_goal_products_are_recorded_without_ending_service_acceptance(
    tmp_path,
):
    goals.prepare_workspace(tmp_path, 0)
    result = goals.artifacts(tmp_path)
    assert result["task_complete"] is False
    assert result["missing_products"] == ["implementation", "tests", "progress_notes"]
    assert "TASK.md" in result["files_sha256"]


def test_generated_task_test_timeout_stays_a_workload_observation(tmp_path):
    goals.prepare_workspace(tmp_path, 0)

    def execute(command):
        raise TimeoutError("generated tests did not finish")

    observed = goals.observe_artifacts(
        SimpleNamespace(workspace=tmp_path, sandbox=SimpleNamespace(execute=execute))
    )
    assert observed["task_complete"] is False
    assert len(observed["independent_checks"]) == 2
    assert all(check["returncode"] is None for check in observed["independent_checks"])
