import argparse
import json
import subprocess
from pathlib import Path

from oh_my_vllm.models.deepseek_v41.tokenizer import Tokenization

from .agentx import DATASET, VERSION, profile_command, run_harness
from .agentx_data import export_run
from .agentx_plot import plot
from .deepseek_benchmark import (
    ROOT,
    SERVED_MODEL,
    Service,
    artifact_identity,
    verify_artifact_identity,
)
from .device import selected_uuids


def service_max_num_seqs(concurrency, override=None):
    value = concurrency if override is None else override
    if type(value) is not int or not 1 <= value <= 64:
        raise ValueError("max number of service sequences must be between 1 and 64")
    return value


def workload_identity(args):
    return {
        "kind": "agentx",
        "version": VERSION,
        "dataset": DATASET,
        "seed": args.seed,
        "strategy": args.strategy,
        "concurrency": args.concurrency,
        "duration": args.duration,
        "grace_period": args.grace_period,
        "mode": "dspark5",
        "prompt_seed": args.prompt_seed,
    }


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--agentx-python", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--strategy", choices=("tp4", "tep4", "dep4"), required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--duration", type=int, default=900)
    parser.add_argument("--grace-period", type=float, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-context", type=int, default=1048576)
    parser.add_argument("--blocks", type=int)
    parser.add_argument("--mamba-blocks", type=int)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int)
    parser.add_argument(
        "--binary", type=Path, default=ROOT / "target/release/oh-my-vllm-zmq-worker"
    )
    result = parser.parse_args()
    for name in ("model", "artifacts", "binary", "csv", "figure"):
        setattr(result, name, getattr(result, name).resolve())
    result.agentx_python = result.agentx_python.absolute()
    if result.artifacts.is_relative_to(ROOT):
        parser.error("put raw artifacts external to the repository")
    if result.duration < 900:
        parser.error("AgentX profiling must last at least 900 seconds")
    if (
        result.concurrency < 1
        or result.cache_context < 1
        or result.max_num_batched_tokens < 1
    ):
        parser.error("concurrency and cache context must be positive")
    try:
        result.max_num_seqs = service_max_num_seqs(
            min(64, result.concurrency),
            result.max_num_seqs,
        )
    except ValueError as error:
        parser.error(str(error))
    if result.csv.exists() or result.figure.exists():
        parser.error("use new CSV and figure paths")
    uuids = selected_uuids()
    if len(uuids) != 4:
        parser.error("DeepSeek AgentX requires four selected GPUs")
    result.gpu_uuids = uuids
    result.prompt_seed = f"agentx-deepseek-{result.seed}"
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
    tokenizer = Tokenization(args.model)
    workload = workload_identity(args)
    identity = artifact_identity(
        args,
        tokenizer,
        args.gpu_uuids,
        workload=workload,
    )
    (args.artifacts / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
    service = Service(args, "dspark5", args.strategy)
    service_log = args.artifacts / f"dspark5-{args.strategy}-service.log"
    try:
        service.ready()
        command = profile_command(
            args.agentx_python,
            args.model,
            args.artifacts / "agentx",
            service.port,
            args.duration,
            args.seed,
            args.grace_period,
            model_name=SERVED_MODEL,
            max_context=args.cache_context,
            concurrency=args.concurrency,
        )
        (args.artifacts / "command.json").write_text(
            json.dumps(command, indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "event": "agentx_start",
                    "strategy": args.strategy,
                    "concurrency": args.concurrency,
                    "artifacts": str(args.artifacts),
                }
            ),
            flush=True,
        )
        run_harness(command, args.artifacts / "agentx.log")
    finally:
        service.close()
    verify_artifact_identity(
        args,
        tokenizer,
        args.gpu_uuids,
        identity,
        workload=workload,
    )
    _, counts = export_run(
        args.artifacts / "agentx",
        args.csv,
        max_context=args.cache_context,
        service_log=service_log,
    )
    plot(
        args.csv,
        args.figure,
        title=f"AgentX  |  DeepSeek DSpark5  |  {args.strategy.upper()}",
        subtitle=(
            f"{args.concurrency} session trees  /  Four B200 GPUs  /  "
            f"{counts['completed']:,} completed requests"
        ),
    )
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
