"""Four persistent OMP RPC clients and a source-audit follow-up curriculum."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from benchmarks.agentic_context import isolated_environment, session_configuration
from benchmarks.agentic_rpc_evidence import ControlRpcEvidence
from benchmarks.agentic_sdk_trace import SdkTrace

ROOT = Path(__file__).resolve().parents[1]
EVENTS = [
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_end",
    "tool_execution_start",
    "tool_execution_end",
    "auto_compaction_start",
    "auto_compaction_end",
    "auto_retry_start",
    "auto_retry_end",
    "session_settled",
]
DOMAINS = (
    ("scheduling and logical KV ownership", ("crates/scheduler/", "crates/kv-cache/")),
    (
        "HTTP protocols and request lifetimes",
        ("crates/zmq-worker/", "python/oh_my_vllm/worker/serving"),
    ),
    (
        "GPU computation, graphs and compiler caches",
        ("python/oh_my_vllm/worker/", "python/oh_my_vllm/kernels/"),
    ),
    ("tests, requirements and documented behavior", ("tests/", "docs/", "benchmarks/")),
)


def curriculum(lane: int) -> list[str]:
    """Read real owned files in a deterministic order, without padded prompts."""
    prefixes = DOMAINS[lane][1]
    paths = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    return sorted(
        path
        for path in paths
        if path.startswith(prefixes)
        and (ROOT / path).is_file()
        and Path(path).suffix in (".rs", ".py", ".cu", ".cuh", ".md")
    )


def prepare_workspace(directory: Path) -> dict:
    """Copy audit inputs outside the repository; writes cannot change source."""
    paths = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    ).splitlines()
    selected = [
        path
        for path in paths
        if path.startswith(("crates/", "python/", "tests/", "docs/", "benchmarks/"))
        and (ROOT / path).is_file()
        and Path(path).suffix in (".rs", ".py", ".cu", ".cuh", ".md", ".toml")
    ]
    for relative in selected:
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (directory / "notes").mkdir()
    return {"files": selected, "source_root": str(ROOT)}


def followup(lane: int, index: int, paths: list[str]) -> str:
    if not paths:
        raise ValueError("audit curriculum has no source files")
    selected = [paths[(2 * index + offset) % len(paths)] for offset in range(2)]
    angle = (
        "trace ownership and successful completion",
        "trace cancellation and failure",
        "check resource bounds and eviction",
        "compare code, tests and documentation",
    )[index % 4]
    note = f"notes/item-{index:05d}.md"
    return (
        f"Continue your persistent audit of {DOMAINS[lane][0]}. Work item {index}. "
        f"For this item, {angle}. Inspect {' and '.join(selected)} "
        "with read/grep tools. Make a fresh successful read call on one selected "
        "source file in this work item. Read at most 180 lines per call. "
        "Follow a relevant caller or test when needed to establish one invariant. "
        "Take the next necessary tool action directly. "
        "Once the evidence is sufficient, record one checked invariant and a "
        f"relevant test, with file and line citations, in {note}. "
        "Keep earlier item notes unchanged. Update notes/progress.md with a brief "
        "index of item notes, the current item, open questions and next steps; "
        "link earlier notes instead of copying their findings into each update. "
        "Audit inputs are copied snapshots: write only under notes/. "
        "Give a concise final answer after saving the note. "
        "After automatic context compaction, read notes/progress.md and the "
        "current item note, then continue this same work item. "
        "The controller will supply the next work item after this one finishes."
    )


class RpcClient:
    """Correlate commands while streaming events directly to bounded disk files."""

    def __init__(
        self,
        omp: str,
        api: str,
        lane: int,
        directory: Path,
        proxy: str,
        stop: threading.Event,
        timeout: float,
        *,
        goal_mode: bool = False,
        sandbox_image=None,
    ):
        self.sandbox = None
        try:
            self._initialize(
                omp,
                api,
                lane,
                directory,
                proxy,
                stop,
                timeout,
                goal_mode=goal_mode,
                sandbox_image=sandbox_image,
            )
        except BaseException:
            if self.sandbox:
                self.sandbox.close()
            if hasattr(self, "sdk_trace"):
                self.sdk_trace.close()
            if hasattr(self, "stderr"):
                self.stderr.close()
            raise

    def _initialize(
        self,
        omp,
        api,
        lane,
        directory,
        proxy,
        stop,
        timeout,
        *,
        goal_mode=False,
        sandbox_image=None,
    ):
        self.directory, self.api, self.lane = directory, api, lane
        self.goal_mode, self.goal_id = goal_mode, None
        self.sandbox = None
        if goal_mode or sandbox_image is not None:
            if sandbox_image is None:
                raise ValueError("native goal clients require an actual sandbox image")
            from benchmarks.agentic_sandbox import AgentSandbox, host_endpoint

            self.sandbox = AgentSandbox(sandbox_image, directory)
            proxy = host_endpoint(proxy, sandbox_image.gateway)
        self.stop, self.timeout = stop, timeout
        self.condition = threading.Condition()
        self.responses, self.results = {}, {}
        self.sequence, self.error, self.session_id = 0, None, None
        self.current_prompt, self.compacting = None, False
        self.events, self.states, self.turns = [], [], []
        self.control_evidence = ControlRpcEvidence(directory)
        self.leaf_id, self.entry_count = None, 0
        (directory / "entries.jsonl").touch(exist_ok=False)
        self.state = directory / "state"
        self.state.mkdir()
        models, settings = session_configuration(proxy, api)
        settings["telemetry"] = {"otlpExportEnabled": True}
        if goal_mode:
            settings["goal"] = {"enabled": True, "continuationModes": ["rpc"]}
        for name, value in (("models.yml", models), ("settings.yml", settings)):
            (self.state / name).write_text(json.dumps(value, indent=2) + "\n")
        self.configuration_hashes = {
            name: hashlib.sha256((self.state / name).read_bytes()).hexdigest()
            for name in ("models.yml", "settings.yml")
        }
        self.workspace = (
            self.sandbox.root / "workspace" if self.sandbox else directory / "workspace"
        )
        if not self.sandbox:
            self.workspace.mkdir()
        if goal_mode:
            from benchmarks.agentic_goals import prepare_workspace as prepare_goal

            self.workspace_manifest = prepare_goal(self.workspace, lane)
        else:
            self.workspace_manifest = prepare_workspace(self.workspace)
        tool_guard = directory / "tool-guard.ts"
        guard = "agentic_goal_tool_guard.ts" if goal_mode else "agentic_tool_guard.ts"
        shutil.copyfile(ROOT / "benchmarks" / guard, tool_guard)
        if self.sandbox:
            for name in ("models.yml", "settings.yml"):
                shutil.copyfile(self.state / name, self.sandbox.root / "state" / name)
            shutil.copyfile(tool_guard, self.sandbox.root / "tool-guard.ts")
            tool_guard = self.sandbox.root / "tool-guard.ts"
        self.workspace_manifest["tool_guard_sha256"] = hashlib.sha256(
            tool_guard.read_bytes()
        ).hexdigest()
        command = [
            omp,
            "--cwd",
            str(self.workspace),
            "--mode",
            "rpc",
            "--thinking",
            "max",
            "--model",
            f"oh-my-vllm-{api}/qwen3.8-27b-fp8",
            "--tools",
            "read,grep,write,edit,bash" if goal_mode else "read,grep,write",
            "--no-lsp",
            "--no-extensions",
            "--no-skills",
            "--no-rules",
            "--no-title",
            "--extension",
            str(tool_guard),
            "--no-ui",
            "--approval-mode",
            "yolo",
            "--session-dir",
            str((self.sandbox.root if self.sandbox else directory) / "sessions"),
        ]
        self.stderr = (directory / "stderr.log").open("x")
        self.sdk_trace = SdkTrace(directory / "sdk-trace")
        try:
            environment = (
                isolated_environment(self.state) | self.sdk_trace.environment()
            )
            if self.sandbox:
                command = self.sandbox.command(command, self.sdk_trace.environment())
                # Docker must use the same host daemon selection as inspect/build.
                # Only explicitly selected --env values enter the guest.
                environment = os.environ.copy()
            self.process = subprocess.Popen(
                command,
                cwd=self.workspace,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr,
                env=environment,
                start_new_session=True,
            )
        except BaseException:
            self.sdk_trace.close()
            self.stderr.close()
            if self.sandbox:
                self.sandbox.close()
            raise
        try:
            stat = Path("/proc", str(self.process.pid), "stat").read_text()
            self.process_record = {
                "pid": self.process.pid,
                "pgid": self.process.pid,
                "start_ticks": int(stat[stat.rfind(")") + 2 :].split()[19]),
                "started_unix_s": time.time(),
                "command": command,
            }
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
        except BaseException:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
            self.stderr.close()
            self.sdk_trace.close()
            if self.sandbox:
                self.sandbox.close()
            raise

    def _read(self):
        try:
            with gzip.open(
                self.directory / "rpc.jsonl.gz", "xb", compresslevel=1
            ) as raw:
                for line_number, line in enumerate(self.process.stdout, 1):
                    if len(line) > 16 << 20:
                        raise ValueError("RPC event exceeds 16 MiB bound")
                    raw.write(line)
                    event = json.loads(line)
                    self.control_evidence.record(
                        "received", event, line, line_number=line_number
                    )
                    event["collected_unix_s"] = time.time()
                    event["collected_monotonic_s"] = time.monotonic()
                    with self.condition:
                        kind = event.get("type")
                        if kind == "response":
                            self.responses[event.get("id")] = event
                        elif kind == "prompt_result":
                            self.results[event.get("id")] = event
                        else:
                            event["prompt_id"] = self.current_prompt
                            if len(self.events) >= 20000:
                                raise ValueError(
                                    "persistent evidence event limit exceeded"
                                )
                            if (
                                kind == "message_end"
                                and event.get("message", {}).get("role") == "assistant"
                            ):
                                message = event["message"]
                                event["message_sha256"] = hashlib.sha256(
                                    json.dumps(message, sort_keys=True).encode()
                                ).hexdigest()
                                # Raw RPC retains the full reasoning and text.
                                # The in-memory audit needs identity and usage.
                                event["message"] = {
                                    key: value
                                    for key, value in message.items()
                                    if key != "content"
                                }
                            self.events.append(event)
                            if kind == "auto_compaction_start":
                                self.compacting = True
                            elif kind == "auto_compaction_end":
                                self.compacting = False
                        self.condition.notify_all()
                if self.process.poll() is None:
                    self.process.wait(timeout=5)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            with self.condition:
                self.condition.notify_all()

    def context(self) -> dict:
        with self.condition:
            return {
                "lane": self.lane,
                "session_id": self.session_id,
                "prompt_id": self.current_prompt,
                "purpose_hint": "compaction" if self.compacting else "task",
            }

    def _send(self, kind: str, **fields) -> str:
        self.sequence += 1
        identity = f"lane-{self.lane}-{self.sequence}"
        command = {"id": identity, "type": kind, **fields}
        raw = (json.dumps(command) + "\n").encode()
        if kind == "prompt" or (kind == "goal" and fields.get("op") == "create"):
            with self.condition:
                self.current_prompt = identity
        with (self.directory / "commands.jsonl").open("a") as output:
            output.write(raw.decode())
        self.control_evidence.record("dispatch_started", command, raw)
        self.process.stdin.write(raw)
        self.process.stdin.flush()
        self.control_evidence.record("dispatch_completed", command, raw)
        return identity

    def _wait(self, mapping: dict, identity: str, timeout: float):
        deadline = time.monotonic() + timeout
        with self.condition:
            while identity not in mapping:
                if self.stop.is_set() or self.error or self.process.poll() is not None:
                    raise ValueError(f"persistent RPC stopped: {self.error}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("persistent OMP command exceeded its deadline")
                self.condition.wait(min(remaining, 0.2))
            return mapping.pop(identity)

    def command(self, kind: str, **fields) -> dict:
        identity = self._send(kind, **fields)
        result = self._wait(self.responses, identity, 60)
        if result.get("success") is not True:
            raise ValueError(f"RPC {kind} failed: {result}")
        return result.get("data", {})

    def check_state(self) -> dict:
        state = self.command("get_state")
        identity = state.get("sessionId")
        if not identity or (self.session_id and identity != self.session_id):
            raise ValueError("persistent session identity changed")
        if (
            state.get("thinkingLevel") != "max"
            or state.get("autoCompactionEnabled") is not True
        ):
            raise ValueError("persistent thinking or compaction configuration differs")
        if state.get("model", {}).get("contextWindow") != 262144:
            raise ValueError("persistent model context window differs")
        if state.get("model", {}).get("tokenizer") != "qwen3":
            raise ValueError("persistent model tokenizer differs")
        model = state["model"]
        expected_api = (
            "openai-completions" if self.api == "chat" else "openai-responses"
        )
        if (
            model.get("api") != expected_api
            or model.get("provider") != f"oh-my-vllm-{self.api}"
            or model.get("id") != "qwen3.8-27b-fp8"
        ):
            raise ValueError("persistent runtime provider/API differs")
        self.session_id = identity
        self.states.append({"collected_unix_s": time.time(), **state})
        return state

    def initialize(self):
        self.command(
            "set_event_filter",
            events=EVENTS + (["goal_updated"] if self.goal_mode else []),
        )
        self.command("set_auto_retry", enabled=False)
        self.check_state()
        if self.sandbox:
            self.sandbox.inspect()
        available = self.command("get_available_models")["models"]
        if any(
            model.get("provider") != f"oh-my-vllm-{self.api}" for model in available
        ):
            raise ValueError("persistent client has nonlocal fallback models")

    def prompt(self, message: str, index: int):
        started = time.monotonic()
        started_unix = time.time()
        identity = self._send("prompt", message=message)
        self.current_prompt = identity
        accepted = self._wait(self.responses, identity, 60)
        if accepted.get("success") is not True:
            raise ValueError(f"persistent prompt rejected: {accepted}")
        result = self._wait(self.results, identity, self.timeout)
        if (
            result.get("status") != "completed"
            or result.get("agentInvoked") is not True
        ):
            raise ValueError(f"persistent prompt failed: {result}")
        while not self.check_state().get("isSettled"):
            if time.monotonic() - started >= self.timeout:
                raise TimeoutError("persistent post-prompt work did not settle")
            self.stop.wait(0.1)
        self.turns.append(
            {
                "prompt_id": identity,
                "index": index,
                "result": result,
                "started_unix_s": started_unix,
                "completed_unix_s": time.time(),
                "elapsed_s": time.monotonic() - started,
                "completed_monotonic_s": time.monotonic(),
            }
        )
        self.checkpoint_entries()
        self.persist()

    def checkpoint_entries(self):
        fields = {"since": self.leaf_id} if self.leaf_id else {}
        result = self.command("get_entries", **fields)
        with (self.directory / "entries.jsonl").open("a") as output:
            for entry in result.get("entries", []):
                output.write(json.dumps(entry) + "\n")
                self.entry_count += 1
        self.leaf_id = result.get("leafId", self.leaf_id)

    def persist(self):
        with self.condition:
            for name, value in (
                ("events", self.events),
                ("states", self.states),
                ("turns", self.turns),
            ):
                (self.directory / f"{name}.json").write_text(
                    json.dumps(value, indent=2) + "\n"
                )

    def collect(self):
        self.check_state()
        self.checkpoint_entries()
        self.persist()
        trace_capture_error = None
        try:
            self.sdk_trace.wait_for(
                {
                    event["message"]["responseId"]
                    for event in self.events
                    if event.get("type") == "message_end"
                    and event.get("message", {}).get("role") == "assistant"
                    and event["message"].get("responseId")
                }
            )
        except Exception as error:
            trace_capture_error = f"{type(error).__name__}: {error}"
        return {
            "lane": self.lane,
            "api": self.api,
            "session_id": self.session_id,
            "turns": len(self.turns),
            "process": self.process_record,
            "entries_file": "entries.jsonl",
            "entry_count": self.entry_count,
            "leaf_id": self.leaf_id,
            "workspace": self.workspace_manifest,
            "workspace_path": str(self.workspace),
            "goal_id": self.goal_id,
            "sdk_trace": {
                "directory": "sdk-trace",
                "source": "omp_sdk_otlp",
                "capture_error": trace_capture_error,
                "api_binding": "openai-completions"
                if self.api == "chat"
                else "openai-responses",
                "api_binding_source": "isolated_provider_configuration",
                "configuration_sha256": self.configuration_hashes,
                "environment": self.sdk_trace.environment(),
            },
        }

    def close(self):
        failures = []

        def attempt(action):
            try:
                action()
            except Exception as error:
                failures.append(f"{type(error).__name__}: {error}")

        attempt(self.persist)
        if self.process.poll() is None:
            attempt(self.process.stdin.close)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if self.sandbox:
                    attempt(self.sandbox.stop)
                if self.process.poll() is None:
                    attempt(lambda: os.killpg(self.process.pid, signal.SIGTERM))
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        attempt(lambda: os.killpg(self.process.pid, signal.SIGKILL))
                        self.process.wait()
        attempt(lambda: self.reader.join(timeout=10))
        attempt(self.process.stdout.close)
        attempt(self.stderr.close)
        self.process_record.update(
            returncode=self.process.returncode, stopped_unix_s=time.time(), reaped=True
        )
        attempt(
            lambda: (self.directory / "process.json").write_text(
                json.dumps(self.process_record) + "\n"
            )
        )
        attempt(lambda: self.sdk_trace.close(strict=False))
        if self.sandbox:
            attempt(self.sandbox.close)
        if self.reader.is_alive():
            failures.append("persistent RPC reader did not exit")
        if failures:
            raise ValueError("persistent cleanup failed: " + "; ".join(failures))
