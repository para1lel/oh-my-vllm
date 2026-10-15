"""Native RPC goal lifecycle shared by the accepted workload fixtures."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable

COLLECTION_FIELDS = ("workload_collection", "goal_collection")


def collection_record(session: dict) -> dict | None:
    """Read the neutral collection record, or the historical goal-only record."""
    for name in COLLECTION_FIELDS:
        value = session.get(name)
        if value:
            return value
    return None


def goal_state(state: dict, identity: str) -> tuple[dict, str | None]:
    """Terminal goals are queue transitions; identity replacement is an error."""
    mode = state.get("goal") or {}
    goal = mode.get("goal") or {}
    if goal and goal.get("id") != identity:
        raise ValueError("native goal identity changed without a host command")
    if not goal:
        return goal, "native_goal_exited"
    if goal.get("status") != "active":
        return goal, f"native_{goal.get('status')}"
    if mode.get("enabled") is not True:
        raise ValueError("native goal mode was disabled")
    return goal, "native_continuation_settled" if state.get("isSettled") else None


def settle(client) -> dict:
    """Pausing does not abort an in-flight turn or background tool delivery."""
    deadline = time.monotonic() + client.timeout
    while True:
        state = client.check_state()
        if state.get("isSettled"):
            return state
        if time.monotonic() >= deadline:
            raise TimeoutError("native goal did not drain after pause")
        client.stop.wait(0.2)


def run_goal(client, clock: dict, duration: float, objective: Callable[[int], str]):
    """Run queued native goals in one process/history until the collection deadline."""
    deadline = clock["monotonic"] + duration
    deadline_unix = clock.get("unix", time.time()) + duration
    index, next_checkpoint = 0, time.monotonic()
    while time.monotonic() < deadline:
        started, started_unix = time.monotonic(), time.time()
        text = objective(index)
        # OMP suppresses continuation when all retained todos are blocked.
        # Their prior state remains in the raw journal; each queued goal starts fresh.
        client.command("set_todos", phases=[])
        identity = client._send("goal", op="create", objective=text)
        response = client._wait(client.responses, identity, 60)
        if response.get("success") is not True:
            raise ValueError(f"native goal rejected: {response}")
        created = response.get("data", {}).get("goal", {})
        if created.get("status") != "active" or not created.get("id"):
            raise ValueError("native goal did not become active")
        client.goal_id = created["id"]
        settled_since = None
        reason = "collection_deadline"
        while time.monotonic() < deadline:
            if (
                client.stop.is_set()
                or client.error
                or client.process.poll() is not None
            ):
                raise ValueError("native goal collection interrupted")
            state = client.check_state()
            _goal, transition = goal_state(state, client.goal_id)
            now = time.monotonic()
            if transition == "native_continuation_settled":
                settled_since = settled_since or now
                # Native scheduling can briefly race a state read after creation.
                if now - settled_since < 2:
                    transition = None
            else:
                settled_since = None
            if transition:
                reason = transition
                break
            if now >= next_checkpoint:
                client.checkpoint_entries()
                client.persist()
                next_checkpoint = now + 30
            client.stop.wait(min(1, max(0, deadline - now)))
        active_ended = time.monotonic()
        # Stop continuation and cancel active work only after the collection deadline.
        paused = client.command("goal", op="pause")
        aborted = client.command("abort") if reason == "collection_deadline" else None
        state = settle(client)
        completed = time.monotonic()
        client.turns.append(
            {
                "prompt_id": identity,  # Host correlation, not a prompt_result.
                "index": index,
                "kind": "native_goal",
                "goal_id": client.goal_id,
                "created_goal": created,
                "objective_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "transition_reason": reason,
                "goal_pause": paused,
                "collection_abort": aborted,
                "goal_final": state.get("goal"),
                "started_unix_s": started_unix,
                "active_ended_monotonic_s": active_ended,
                "completed_unix_s": time.time(),
                "elapsed_s": completed - started,
                "paused_tail_seconds": completed - active_ended,
                "completed_monotonic_s": completed,
            }
        )
        client.checkpoint_entries()
        client.persist()
        if time.monotonic() < deadline:
            client.command("goal", op="drop")
        index += 1
    return {
        "collection_deadline_monotonic_s": deadline,
        "collection_deadline_unix_s": deadline_unix,
        "controller_ended_monotonic_s": time.monotonic(),
        "final_paused_tail_seconds": client.turns[-1]["paused_tail_seconds"],
    }
