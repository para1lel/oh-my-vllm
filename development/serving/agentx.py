import argparse
import contextlib
import json
import os
import signal
import subprocess
from pathlib import Path

from .agentx_data import export_run
from .benchmark import (
    ROOT,
    Service,
    artifact_identity,
    verify_artifact_identity,
)
from .device import selected_uuid

VERSION = "1.0.7"
DATASET = "semianalysis_cc_traces_weka_062126_256k"


def profile_command(python, model, artifacts, port, duration, seed):
    if duration < 900:
        raise ValueError("AgentX profiling must last at least 900 seconds")
    return [
        str(python),
        "-m",
        "aiperf",
        "profile",
        "--scenario",
        "inferencex-agentx-mvp",
        "--url",
        f"http://127.0.0.1:{port}",
        "--model",
        "qwen3.8-27b-fp8",
        "--tokenizer",
        str(model),
        "--public-dataset",
        DATASET,
        "--max-context-length",
        "262144",
        "--concurrency",
        "4",
        "--benchmark-duration",
        str(duration),
        "--benchmark-grace-period",
        "600",
        "--agentic-warmup-grace-period",
        "1800",
        "--trajectory-start-min-ratio",
        "0.0",
        "--trajectory-start-max-ratio",
        "1.0",
        "--random-seed",
        str(seed),
        "--streaming",
        "--use-server-token-count",
        "--extra-inputs",
        "ignore_eos:true",
        "--extra-inputs",
        "reasoning_effort:off",
        "--extra-inputs",
        f"seed:{seed}",
        "--cache-bust",
        "first_turn_prefix",
        "--system-idle-gap-cap-seconds",
        "10",
        "--export-level",
        "raw",
        "--no-gpu-telemetry",
        "--output-artifact-dir",
        str(artifacts),
        "--ui",
        "simple",
    ]


def run_harness(command, log_path):
    environment = dict(
        os.environ,
        TZ="UTC",
        AIPERF_DATASET_CONFIGURATION_TIMEOUT="1800",
        AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="1800",
        AIPERF_HTTP_TCP_USER_TIMEOUT="900000",
        TOKENIZERS_PARALLELISM="false",
    )
    # A proxy must not route traffic to the local inference service.
    environment["NO_PROXY"] = ",".join(
        filter(None, (environment.get("NO_PROXY"), "127.0.0.1", "localhost"))
    )
    environment["no_proxy"] = environment["NO_PROXY"]
    with log_path.open("x") as log:
        process = subprocess.Popen(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            result = process.wait(timeout=7200)
            if result:
                raise RuntimeError(f"AgentX exited with status {result}: {log_path}")
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            if process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--agentx-python", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--mamba-blocks", type=int, default=128)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    result = parser.parse_args()
    for name in ("model", "artifacts", "binary", "csv", "figure"):
        setattr(result, name, getattr(result, name).resolve())
    # Resolving the interpreter symlink would bypass its virtual environment.
    result.agentx_python = result.agentx_python.absolute()
    if result.artifacts.is_relative_to(ROOT):
        parser.error("put raw artifacts external to the repository")
    if result.duration < 900:
        parser.error("AgentX profiling must last at least 900 seconds")
    if result.csv.exists() or result.figure.exists():
        parser.error("use new CSV and figure paths")
    result.gpu_uuid = selected_uuid()
    result.prompt_seed = f"agentx-{result.seed}"
    return result


def main():
    args = arguments()
    version = subprocess.check_output(
        [
            str(args.agentx_python),
            "-c",
            "from importlib.metadata import version; print(version('agentx-harness'))",
        ],
        text=True,
    ).strip()
    if version != VERSION:
        raise ValueError(f"install agentx-harness=={VERSION}, found {version}")
    args.artifacts.mkdir(parents=True, exist_ok=False)
    identity = artifact_identity(args)
    identity["agentx"] = dict(version=VERSION, dataset=DATASET, seed=args.seed)
    if "B200" not in identity["device"]["name"]:
        raise ValueError("AgentX measurement requires one B200")
    (args.artifacts / "identity.json").write_text(json.dumps(identity, indent=2))
    service = Service(args, "mtp4")
    try:
        service.ready()
        service.verify_identity(identity["checkpoint"], identity["tokenizer"])
        command = profile_command(
            args.agentx_python,
            args.model,
            args.artifacts / "agentx",
            service.port,
            args.duration,
            args.seed,
        )
        (args.artifacts / "command.json").write_text(json.dumps(command, indent=2))
        print(
            json.dumps(dict(event="agentx_start", artifacts=str(args.artifacts))),
            flush=True,
        )
        run_harness(command, args.artifacts / "agentx.log")
    finally:
        service.close()
    verify_artifact_identity(args, identity)
    service.verify_identity(identity["checkpoint"], identity["tokenizer"])
    _, counts = export_run(args.artifacts / "agentx", args.csv)
    subprocess.run(
        [
            str(args.agentx_python),
            "-m",
            "development.serving.agentx_plot",
            "--csv",
            str(args.csv),
            "--figure",
            str(args.figure),
        ],
        check=True,
    )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
