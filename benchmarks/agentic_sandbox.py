"""Task-owned rootless Docker images, agent containers and disposable workspaces."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
import uuid
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from benchmarks.agentic_goals import OMP_VERSION
from benchmarks.common import _children, _process_record

LABEL = "oh-my-vllm.agentic-sandbox"
BASE = "ubuntu@sha256:561618e2c15bf2397621dd04f96926663a3b5616c189cf7e38db7e82f5c538ea"


def docker(*arguments: str, timeout: float = 60) -> str:
    return subprocess.check_output(
        ["docker", *arguments], text=True, stderr=subprocess.PIPE, timeout=timeout
    ).strip()


def host_endpoint(url: str, gateway: str) -> str:
    """Expose only collector endpoint addresses, without mounting its directory."""
    parts = urlsplit(url)
    if parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("sandbox collector endpoint must be on host loopback")
    if not gateway or any(character in gateway for character in " /\n\r"):
        raise ValueError("invalid sandbox host gateway")
    return urlunsplit(parts._replace(netloc=f"{gateway}:{parts.port}"))


def remove_directory(root: Path):
    """Restore owner access to agent-created directories before removal."""
    root.chmod(0o700)
    for current, directories, _files in os.walk(root, followlinks=False):
        Path(current).chmod(0o700)
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                path.chmod(0o700)
    shutil.rmtree(root)


class SandboxImage:
    """One temporary image per collector; never prune other Docker resources."""

    def __init__(self, omp: str, directory: Path, gateway: str, *, stop=None):
        if not gateway:
            raise ValueError("set --sandbox-host or OH_MY_VLLM_SANDBOX_HOST")
        self.directory, self.gateway = directory, gateway
        self.owner = uuid.uuid4().hex
        self.tag = f"oh-my-vllm-agentic:{self.owner}"
        self.identity = None
        directory.mkdir(exist_ok=False)
        info = json.loads(docker("info", "--format", "{{json .}}"))
        if not any("rootless" in item for item in info.get("SecurityOptions", [])):
            raise ValueError("agentic acceptance requires a rootless Docker daemon")
        executable = shutil.which(omp)
        if not executable:
            raise ValueError("OMP executable is missing")
        version = subprocess.check_output(
            [executable, "--version"], text=True, timeout=10
        )
        if version.strip() != f"omp/{OMP_VERSION}":
            raise ValueError(f"native goal acceptance requires OMP {OMP_VERSION}")
        context = directory / "build"
        try:
            context.mkdir()
            shutil.copyfile(executable, context / "omp")
            (context / "Dockerfile").write_text(
                f"FROM {BASE}\nLABEL {LABEL}={self.owner}\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "python3 python3-venv bash ca-certificates "
                "&& rm -rf /var/lib/apt/lists/*\n"
                "COPY omp /usr/local/bin/omp\n"
                "RUN chmod 755 /usr/local/bin/omp "
                "&& ln -s /usr/bin/python3 /usr/local/bin/python\n"
            )
            if stop and stop.is_set():
                raise InterruptedError("sandbox image preparation cancelled")
            with (directory / "build.log").open("x") as output:
                process = subprocess.Popen(
                    ["docker", "build", "--force-rm", "--tag", self.tag, str(context)],
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    env=os.environ | {"DOCKER_BUILDKIT": "0"},
                    start_new_session=True,
                )
                try:
                    deadline = time.monotonic() + 1800
                    while process.poll() is None:
                        if stop and stop.is_set():
                            raise InterruptedError("sandbox image build cancelled")
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "sandbox image build exceeded its deadline"
                            )
                        with suppress(subprocess.TimeoutExpired):
                            process.wait(timeout=0.2)
                    if process.returncode:
                        raise ValueError(
                            f"sandbox image build failed: {process.returncode}"
                        )
                finally:
                    if process.poll() is None:
                        with suppress(ProcessLookupError):
                            os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            with suppress(ProcessLookupError):
                                os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            self.identity = json.loads(docker("image", "inspect", self.tag))[0]
            if self.identity["Config"]["Labels"].get(LABEL) != self.owner:
                raise ValueError("sandbox image ownership differs")
            (directory / "image.json").write_text(
                json.dumps(
                    {
                        "id": self.identity["Id"],
                        "owner": self.owner,
                        "base": BASE,
                        "omp_version": OMP_VERSION,
                        "omp_sha256": hashlib.sha256(
                            Path(executable).read_bytes()
                        ).hexdigest(),
                        "rootless": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        except BaseException:
            self.close()
            raise
        finally:
            if context.exists():
                shutil.rmtree(context)

    def close(self):
        errors, removed, failed = [], [], set()

        def attempt(action, fallback=None):
            try:
                return action()
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                return fallback

        def owned_remove(kind, identity):
            labels = json.loads(docker(kind, "inspect", identity))[0]["Config"][
                "Labels"
            ]
            if labels.get(LABEL) != self.owner:
                raise ValueError(f"refusing to remove an unowned sandbox {kind}")
            docker(kind, "rm", "--force", identity)
            return True

        def inventory(kind):
            return docker(
                kind,
                "ls",
                "--all",
                "--quiet",
                "--filter",
                f"label={LABEL}={self.owner}",
            )

        containers = attempt(lambda: inventory("container"), "")
        for identity in containers.splitlines():
            attempt(lambda identity=identity: owned_remove("container", identity))
        while True:
            images = attempt(lambda: inventory("image"), "")
            candidates = [
                identity for identity in images.splitlines() if identity not in failed
            ]
            if not candidates:
                break
            identity = candidates[0]
            if attempt(lambda identity=identity: owned_remove("image", identity)):
                removed.append(identity)
            else:
                failed.add(identity)
        remaining_images = attempt(lambda: inventory("image"), "unknown")
        remaining_containers = attempt(lambda: inventory("container"), "unknown")
        if remaining_images or remaining_containers:
            errors.append("task-owned sandbox resources remain")
        (self.directory / "cleanup.json").write_text(
            json.dumps(
                {
                    "removed_images": removed,
                    "remaining_images": remaining_images.splitlines(),
                    "remaining_containers": remaining_containers.splitlines(),
                    "closed": not errors,
                    "errors": errors,
                }
            )
            + "\n"
        )
        if errors:
            raise ValueError("sandbox image cleanup failed: " + "; ".join(errors))


class AgentSandbox:
    """Only the disposable agent directory is mounted into the container."""

    def __init__(self, image: SandboxImage, directory: Path):
        self.image, self.directory = image, directory
        self.root = directory / "sandbox"
        self.root.mkdir(exist_ok=False)
        for name in ("home", "tmp", "state", "workspace", "sessions"):
            (self.root / name).mkdir()
        self.name = f"omv-agent-{image.owner}-{directory.name}"
        self.container_id = None
        self.closed = False
        self.stopped = False
        self.cleanup_record = {}

    def command(self, arguments: list[str], environment: dict) -> list[str]:
        command = [
            "docker",
            "run",
            "--interactive",
            "--name",
            self.name,
            "--label",
            f"{LABEL}={self.image.owner}",
            "--hostname",
            "agent-sandbox",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "128",
            "--memory",
            "4g",
            "--cpus",
            "2",
            "--workdir",
            str(self.root / "workspace"),
            "--mount",
            f"type=bind,source={self.root},target={self.root}",
        ]
        values = {
            "HOME": str(self.root / "home"),
            "TMPDIR": str(self.root / "tmp"),
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C.UTF-8",
            "CUDA_VISIBLE_DEVICES": "",
            "PI_CODING_AGENT_DIR": str(self.root / "state"),
            "PI_CONFIG_FILES": str(self.root / "state/settings.yml"),
        }
        for key, value in environment.items():
            if key.startswith("OTEL_"):
                values[key] = (
                    host_endpoint(value, self.image.gateway)
                    if key.endswith("ENDPOINT")
                    else value
                )
        for key, value in sorted(values.items()):
            command.extend(("--env", f"{key}={value}"))
        command.extend(
            (self.image.identity["Id"], "/usr/local/bin/omp", *arguments[1:])
        )
        return command

    def inspect(self):
        state = json.loads(docker("container", "inspect", self.name))[0]
        if state["Config"]["Labels"].get(LABEL) != self.image.owner:
            raise ValueError("sandbox container ownership differs")
        self.container_id = state["Id"]
        host_pid = state["State"]["Pid"]
        if not state["State"]["Running"] or not _process_record(host_pid):
            raise ValueError("sandbox agent is not running")
        if os.readlink(f"/proc/{host_pid}/ns/pid") == os.readlink("/proc/self/ns/pid"):
            raise ValueError("agent does not have a separate PID namespace")
        record = {
            "id": state["Id"],
            "name": self.name,
            "owner": self.image.owner,
            "image_id": state["Image"],
            "started_at": state["State"]["StartedAt"],
            "host_pid": state["State"]["Pid"],
            "host_process": _process_record(host_pid),
            "pid_namespace": os.readlink(f"/proc/{host_pid}/ns/pid"),
            "mounts": state["Mounts"],
            "security_options": state["HostConfig"]["SecurityOpt"],
            "cap_drop": state["HostConfig"]["CapDrop"],
            "devices": state["HostConfig"]["Devices"],
            "workspace": str(self.root / "workspace"),
        }
        (self.directory / "sandbox.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        return record

    def preserve_logs(self):
        """Retain native journals and raw tool-output logs, never generated source."""
        root = self.root / "sessions"
        files = {}
        with tarfile.open(
            self.directory / "native-session-logs.tar.gz", "w:gz"
        ) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_symlink() or not path.is_file():
                    continue
                relative = str(path.relative_to(root))
                files[relative] = {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                archive.add(path, arcname=relative, recursive=False)
        (self.directory / "native-session-logs.json").write_text(
            json.dumps(files, indent=2) + "\n"
        )

    def execute(self, arguments: list[str], timeout: float = 120) -> dict:
        result = subprocess.run(
            [
                "docker",
                "exec",
                "--workdir",
                str(self.root / "workspace"),
                self.name,
                "timeout",
                "--kill-after=5",
                str(timeout),
                *arguments,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 15,
        )
        return {
            "command": arguments,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def stop(self):
        if self.stopped:
            return
        owned = docker(
            "container", "ls", "--all", "--quiet", "--filter", f"name=^/{self.name}$"
        )
        removed = None
        processes = []
        if owned:
            state = json.loads(docker("container", "inspect", self.name))[0]
            if state["Config"]["Labels"].get(LABEL) != self.image.owner:
                raise ValueError("refusing to remove an unowned agent container")
            removed = state["Id"]
            pending = [state["State"]["Pid"]] if state["State"]["Running"] else []
            while pending:
                pid = pending.pop()
                record = _process_record(pid)
                if record:
                    processes.append(record)
                    pending.extend(_children(pid))
            try:
                (self.directory / "sandbox-diff.txt").write_text(
                    docker("container", "diff", removed)
                )
            except Exception as error:
                self.cleanup_record["diff_error"] = f"{type(error).__name__}: {error}"
            finally:
                docker("container", "rm", "--force", removed)
        remaining = docker(
            "container", "ls", "--all", "--quiet", "--filter", f"name=^/{self.name}$"
        )
        if remaining:
            raise ValueError("agent sandbox container remains after removal")
        for original in processes:
            current = _process_record(original["pid"])
            if (
                current
                and current["start_ticks"] == original["start_ticks"]
                and current["state"] != "Z"
            ):
                raise ValueError("owned sandbox process survived container removal")
        self.stopped = True
        self.cleanup_record.update(
            {
                "removed_container": removed,
                "processes": processes,
                "remaining_containers": [],
                "processes_released": True,
            }
        )

    def close(self):
        if self.closed:
            return
        failures = []
        try:
            self.stop()
        except Exception as error:
            failures.append(f"container cleanup: {error}")
        try:
            self.preserve_logs()
        except Exception as error:
            failures.append(f"native log archive: {error}")
        finally:
            remove_directory(self.root)
            self.closed = True
        (self.directory / "sandbox-cleanup.json").write_text(
            json.dumps(
                {
                    **self.cleanup_record,
                    "workspace_removed": not self.root.exists(),
                    "remaining_containers": [],
                    "closed": not failures,
                    "errors": failures,
                    "at_unix_s": time.time(),
                }
            )
            + "\n"
        )
        if failures:
            raise ValueError("sandbox cleanup failed: " + "; ".join(failures))
