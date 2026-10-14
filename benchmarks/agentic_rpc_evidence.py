"""Timestamp native control actions without changing RPC bytes."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path


def control_command(message: dict) -> bool:
    return message.get("type") == "abort" or (
        message.get("type") == "goal" and message.get("op") == "pause"
    )


def control_receipt(message: dict) -> bool:
    return (
        message.get("type") == "response"
        and message.get("command") in ("goal", "abort")
    ) or (
        message.get("type") == "message_end"
        and message.get("message", {}).get("role") == "assistant"
        and message["message"].get("stopReason") == "aborted"
    )


class ControlRpcEvidence:
    def __init__(self, directory: Path):
        self.path = directory / "control-rpc.jsonl"
        self.path.touch(exist_ok=False)
        self.lock = threading.Lock()

    def record(
        self, phase: str, message: dict, raw: bytes, *, line_number: int | None = None
    ):
        kind = message.get("type")
        if not (control_command(message) or control_receipt(message)):
            return
        with self.lock:
            record = {
                "phase": phase,
                "unix_s": time.time(),
                "monotonic_s": time.monotonic(),
                "type": kind,
                "id": message.get("id"),
                "raw_sha256": hashlib.sha256(raw).hexdigest(),
                "line_number": line_number,
            }
            with self.path.open("a") as output:
                output.write(json.dumps(record) + "\n")
