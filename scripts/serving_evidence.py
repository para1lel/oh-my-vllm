"""Response checks shared by real speculative-service acceptance scripts."""

import json
import os
import signal
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path


class ServiceAttempt:
    """Reserve one result path and keep responses before acceptance assertions."""

    def __init__(self, path: Path, identity: dict):
        self.path = path
        self.state = {"passed": False, **identity, "responses": []}
        self.lock = threading.RLock()
        self.previous_signals = {}
        self.cancelled = None
        self.cleaning = False
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects previous attempts, including failed ones.
        with path.open("x") as output:
            json.dump(self.state, output, indent=2, allow_nan=False)
            output.write("\n")

    def __enter__(self):
        # Each acceptance CLI owns its main thread. A second signal must not
        # interrupt response/error persistence while the first one unwinds.
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                self.previous_signals[signum] = signal.getsignal(signum)
                signal.signal(signum, self._terminate)
        return self

    def _terminate(self, signum, frame):
        first = self.cancelled is None
        self.cancelled = self.cancelled or signum
        if first and not self.cleaning:
            raise InterruptedError(f"Service acceptance cancelled by signal {signum}")

    def __exit__(self, kind, error, traceback):
        self.cleaning = True
        try:
            if error is not None or self.cancelled is not None:
                self.update(
                    passed=False,
                    error={
                        "type": kind.__name__ if kind else "InterruptedError",
                        "message": str(error) if error else "Acceptance cancelled",
                    },
                )
            else:
                self.update()
                if self.cancelled is not None:
                    self.update(
                        passed=False,
                        error={
                            "type": "InterruptedError",
                            "message": "Acceptance cancelled during persistence",
                        },
                    )
        finally:
            restore_error = None
            for signum, previous in self.previous_signals.items():
                try:
                    signal.signal(signum, previous)
                except BaseException as caught:
                    restore_error = restore_error or caught
            if restore_error is not None:
                self.update(
                    passed=False,
                    error={
                        "type": type(restore_error).__name__,
                        "message": str(restore_error),
                    },
                )
                raise restore_error
        if error is None and self.cancelled is not None:
            # A first cancellation can arrive while the previous handlers are
            # restored, after the earlier persistence check.
            self.update(
                passed=False,
                error={
                    "type": "InterruptedError",
                    "message": "Acceptance cancelled during persistence",
                },
            )
            raise InterruptedError("Service acceptance cancelled during persistence")
        return False

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=self.path.parent, delete=False
                ) as output:
                    temporary = Path(output.name)
                    json.dump(self.state, output, indent=2, allow_nan=False)
                    output.write("\n")
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def record(self, response: dict):
        with self.lock:
            self.state["responses"].append(response)
            self.update()

    @staticmethod
    def _request(url, body, method):
        return urllib.request.Request(
            url,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
            method=method,
        )

    def request(self, url, body=None, method=None, *, timeout=120):
        record = {
            "url": url,
            "request": body,
            "method": method or ("POST" if body is not None else "GET"),
        }
        raw = b""
        try:
            with urllib.request.urlopen(
                self._request(url, body, method), timeout=timeout
            ) as response:
                record["status"] = response.status
                raw = response.read()
            value = json.loads(raw)
            record["response"] = value
            return value
        except urllib.error.HTTPError as error:
            record["status"] = error.code
            with error:
                raw = error.read()
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        except BaseException as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record["response_text"] = raw.decode("utf-8", errors="replace")
            self.record(record)

    @contextmanager
    def stream(self, url, body, *, timeout=120):
        record = {"url": url, "request": body, "method": "POST", "stream": True}
        chunks = []
        try:
            with urllib.request.urlopen(
                self._request(url, body, "POST"), timeout=timeout
            ) as response:
                record["status"] = response.status

                def lines():
                    for line in response:
                        chunks.append(line)
                        yield line

                yield lines()
        except urllib.error.HTTPError as error:
            record["status"] = error.code
            with error:
                chunks.append(error.read())
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        except BaseException as error:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            record["response_text"] = b"".join(chunks).decode("utf-8", errors="replace")
            self.record(record)


def response_text(result: dict, api: str) -> str:
    if api == "chat":
        choices = result.get("choices", [])
        if len(choices) != 1:
            raise ValueError("Chat response must have one choice")
        return choices[0].get("message", {}).get("content") or ""
    return "".join(
        part.get("text", "")
        for item in result.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )


def validate_constraint_response(result: dict, api: str, kind: str) -> None:
    """Enforce completion and grammar results even with Python assertions off."""
    if api not in ("chat", "responses") or kind not in (
        "json_object",
        "json_schema",
        "tool",
    ):
        raise ValueError("unknown constrained response case")
    if api == "chat":
        choices = result.get("choices", [])
        if len(choices) != 1 or choices[0].get("finish_reason") != (
            "tool_calls" if kind == "tool" else "stop"
        ):
            raise ValueError("Chat constrained response did not complete")
        if kind == "tool":
            calls = choices[0].get("message", {}).get("tool_calls", [])
            if len(calls) != 1 or calls[0].get("function", {}).get("name") != "report":
                raise ValueError("Chat constrained response has wrong tool calls")
            output = calls[0]["function"].get("arguments", "")
        else:
            output = response_text(result, api)
    else:
        if result.get("status") != "completed":
            raise ValueError("Responses constrained response did not complete")
        if kind == "tool":
            calls = [
                item
                for item in result.get("output", [])
                if item.get("type") == "function_call"
            ]
            if len(calls) != 1 or calls[0].get("name") != "report":
                raise ValueError("Responses constrained response has wrong tool calls")
            output = calls[0].get("arguments", "")
        else:
            output = response_text(result, api)
    try:
        value = json.loads(output)
    except (ValueError, TypeError) as error:
        raise ValueError("constrained response is not JSON") from error
    if not isinstance(value, dict) or (
        kind != "json_object" and value != {"n": 123, "label": "verified"}
    ):
        raise ValueError("constrained response has incorrect content")


def stream_identity(event: dict, api: str, previous: int | None = None):
    """Track the request identity before the first actual SSE text delta."""
    if api == "chat":
        has_output = any(
            choice.get("delta", {}).get("content")
            or choice.get("delta", {}).get("reasoning_content")
            for choice in event.get("choices", [])
        )
        rid = event.get("id")
    else:
        rid = event.get("response", {}).get("id")
        has_output = event.get("type") in (
            "response.output_text.delta",
            "response.reasoning_summary_text.delta",
        ) and bool(event.get("delta"))
    current = int(rid.rsplit("_", 1)[1]) if rid else previous
    if has_output and current is None:
        raise ValueError("SSE output arrived without a request identity")
    return current, bool(has_output)
