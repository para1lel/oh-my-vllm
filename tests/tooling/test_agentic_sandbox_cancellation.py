"""SIGTERM during image preparation or build deletes copied task tools."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from benchmarks import agentic_sandbox


@pytest.mark.parametrize("stage", ["copy", "build"])
def test_oneoff_sigterm_releases_builder_and_copied_tools(tmp_path, monkeypatch, stage):
    spec = importlib.util.spec_from_file_location(
        "cancelled_agentic_acceptance",
        Path(__file__).resolve().parents[2] / "scripts/agentic-acceptance.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    executable = tmp_path / "omp"
    executable.write_text("fake")
    monkeypatch.setattr(
        agentic_sandbox.shutil, "which", lambda command: str(executable)
    )
    monkeypatch.setattr(
        agentic_sandbox.subprocess,
        "check_output",
        lambda *args, **kwargs: "omp/18.10.3\n",
    )

    def docker(*arguments, **kwargs):
        if arguments[0] == "info":
            return json.dumps({"SecurityOptions": ["name=rootless"]})
        if arguments[1] == "ls":
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(agentic_sandbox, "docker", docker)
    original_popen = subprocess.Popen
    original_copyfile = agentic_sandbox.shutil.copyfile
    processes, timers = [], []

    def cancelled_copy(source, destination):
        result = original_copyfile(source, destination)
        if stage == "copy":
            os.kill(os.getpid(), signal.SIGTERM)
        return result

    monkeypatch.setattr(agentic_sandbox.shutil, "copyfile", cancelled_copy)

    def slow_builder(command, **kwargs):
        assert command[:2] == ["docker", "build"]
        process = original_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], **kwargs
        )
        processes.append(process)
        timer = threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timers.append(timer)
        timer.start()
        return process

    monkeypatch.setattr(agentic_sandbox.subprocess, "Popen", slow_builder)
    directory = tmp_path / "evidence"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agentic-acceptance",
            "--api",
            "chat",
            "--output-dir",
            str(directory),
            "--sandbox-host",
            "test-host",
        ],
    )
    old_handler = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(SystemExit) as result:
            module.main()
        assert result.value.code != 0
    finally:
        for timer in timers:
            timer.cancel()
            timer.join()
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert len(processes) == (1 if stage == "build" else 0)
    assert all(process.poll() is not None for process in processes)
    assert not (directory / "sandbox-runtime/build").exists()
    assert json.loads((directory / "sandbox-runtime/cleanup.json").read_text())[
        "closed"
    ]
    assert json.loads((directory / "chat-http.json").read_text()) == []
    summary = json.loads((directory / "chat-run.json").read_text())
    assert not summary["passed"]
    assert "cancelled" in summary["error"]
    assert signal.getsignal(signal.SIGTERM) == old_handler
