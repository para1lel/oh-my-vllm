"""Exact CPU preflight through the production normalizer and serving adapter."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path

from benchmarks.agentic_soak_analysis import file_identity

CONTEXT_WINDOW = 262144
OUTPUT_BUDGET = 32768
COMPACTION_THRESHOLD = 196608
RESERVE_TOKENS = 65536


class ContextGuard:
    """Never forward an unchecked request or retain its generation state."""

    def __init__(self, model: Path, normalizer: Path, directory: Path):
        self.lock = threading.Lock()
        self.close_lock = threading.Lock()
        self.closed = False
        self.requests, self.replies = queue.Queue(), queue.Queue()
        self.identity = {
            "normalizer": file_identity(normalizer),
            "model_tokenizer_files": {
                path.name: file_identity(path)
                for path in sorted(model.iterdir())
                if path.name
                in (
                    "config.json",
                    "generation_config.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "chat_template.jinja",
                )
            },
        }
        self.stderr = (directory / "normalizer.stderr").open("x")
        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "benchmarks.agentic_context",
                str(model),
                str(normalizer),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self.exchange = threading.Thread(target=self._exchange, daemon=True)
        self.exchange.start()

    def _exchange(self):
        try:
            while (value := self.requests.get()) is not None:
                self.process.stdin.write(json.dumps(value) + "\n")
                self.process.stdin.flush()
                line = self.process.stdout.readline(16 << 20)
                if not line.endswith("\n"):
                    raise ValueError("CPU preflight response missing or too large")
                self.replies.put(json.loads(line))
        except Exception as error:
            self.replies.put({"error": f"{type(error).__name__}: {error}"})

    def check(self, body: dict, path: str) -> dict:
        api = "chat" if path.endswith("/chat/completions") else "responses"
        if body.get("previous_response_id"):
            raise ValueError("soak requires complete explicit history")
        if not self.lock.acquire(timeout=120):
            raise TimeoutError("CPU context preflight admission timed out")
        try:
            if self.process.poll() is not None:
                raise ValueError("CPU normalizer exited before request preflight")
            self.requests.put({"request": body, "api": api})
            try:
                proof = self.replies.get(timeout=120)
            except queue.Empty as error:
                self.close()
                raise TimeoutError("CPU context preflight timed out") from error
            if "error" in proof:
                raise ValueError(proof["error"])
            return proof
        finally:
            self.lock.release()

    def close(self):
        with self.close_lock:
            if self.closed:
                return
            self.closed = True
            self.replies.put({"error": "CPU context guard cancelled"})
            if self.process.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(self.process.pid, signal.SIGTERM)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self.process.pid, signal.SIGKILL)
                    self.process.wait()
            self.requests.put(None)
            self.exchange.join(timeout=5)
            self.process.stdin.close()
            self.process.stdout.close()
            self.stderr.close()


def cpu_worker(model: Path, normalizer: Path):
    """Keep all blocking tokenizer/compiler work in a killable owned process."""
    from oh_my_vllm.worker.serving import ServingAdapter

    config = json.loads((model / "config.json").read_text())
    adapter = ServingAdapter(
        str(model), config.get("text_config", config)["vocab_size"], CONTEXT_WINDOW
    )
    with subprocess.Popen(
        [str(normalizer)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    ) as process:
        for line in sys.stdin:
            try:
                process.stdin.write(line)
                process.stdin.flush()
                normalized = json.loads(process.stdout.readline())
                if "error" in normalized:
                    raise ValueError(normalized["error"])
                request = normalized["normalized"]
                if request["effort"] != "xhigh":
                    raise ValueError(
                        "every task and summary must use maximum reasoning effort"
                    )
                ids, params, generation = adapter.prepare_inputs(request)
                del params, generation
                count, budget = len(ids), request["max_tokens"]
                proof = {
                    "input_tokens": count,
                    "output_budget": budget,
                    "total_budget": count + budget,
                    "context_window": CONTEXT_WINDOW,
                    "reasoning_effort": request["effort"],
                    "input_ids_sha256": hashlib.sha256(
                        json.dumps(ids, separators=(",", ":")).encode()
                    ).hexdigest(),
                }
            except Exception as error:
                proof = {"error": f"{type(error).__name__}: {error}"}
            print(json.dumps(proof), flush=True)
        process.stdin.close()


CHAT_ROLES = (
    "default",
    "smol",
    "slow",
    "vision",
    "plan",
    "commit",
    "tiny",
    "memory",
    "task",
    "advisor",
)


def session_configuration(base_url: str, api: str) -> tuple[dict, dict]:
    """Use one isolated local model for every task and summarization role."""
    from benchmarks.agentic_soak_support import agentic

    name = f"oh-my-vllm-{api}"
    provider = agentic.configuration(base_url)["providers"][name]
    model = provider["models"][0]
    target = f"{name}/{model['id']}:max"
    model.update(
        contextWindow=CONTEXT_WINDOW,
        maxTokens=OUTPUT_BUDGET,
        compactionModel=target,
        tokenizer="qwen3",
    )
    model["thinking"]["defaultLevel"] = "max"
    model["compat"]["alwaysSendMaxTokens"] = True
    models = {"providers": {name: provider}}
    settings = {
        "defaultThinkingLevel": "max",
        "extendedContext": False,
        "contextPromotion": {"enabled": False},
        "modelRoles": {role: target for role in CHAT_ROLES},
        "compaction": {
            "enabled": True,
            "midTurnEnabled": True,
            "asyncEnabled": False,
            "methodOrder": ["soft"],
            "autoContinue": True,
            "thresholdTokens": COMPACTION_THRESHOLD,
            "thresholdPercent": -1,
            "reserveTokens": RESERVE_TOKENS,
            "keepRecentTokens": 20000,
        },
        "memory": {"backend": "off"},
        "autolearn": {"enabled": False},
        "providers": {"openaiWebsockets": "off", "openaiLiveSteering": False},
        "retry": {"enabled": False},
        "enabledModels": [f"{name}/{model['id']}"],
    }
    return models, settings


def isolated_environment(state: Path) -> dict:
    """Provide the local client tools without inherited provider credentials."""
    import os

    names = (
        "HOME",
        "PATH",
        "LANG",
        "LC_ALL",
        "TZ",
        "LD_LIBRARY_PATH",
        "TMPDIR",
        "TERM",
    )
    environment = {name: os.environ[name] for name in names if name in os.environ}
    environment.update(
        PI_CODING_AGENT_DIR=str(state), PI_CONFIG_FILES=str(state / "settings.yml")
    )
    return environment


if __name__ == "__main__":
    cpu_worker(Path(sys.argv[1]), Path(sys.argv[2]))
