import datetime
import importlib.metadata
import os

from .check_deepseek_dep4 import _write_json
from .deepseek_reference_capture import CandidateCapture
from .deepseek_reference_contract import digest, load_reference
from .deepseek_reference_inputs import checker_identity


def rank_entry(rank, args, run_id, operation):
    # ExchangeEvidence imports CUDA code before it calls the supplied operation.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices[rank]
    from .check_deepseek_dep4 import _rank_worker

    _rank_worker(rank, args, run_id, operation)


def _candidate_identity(
    args, proof, worker, manifest, reference, config, runtime_identity
):
    if (
        worker.strategy != args.strategy
        or worker.args.parallel_strategy != args.strategy.upper()
    ):
        raise ValueError("candidate strategy differs from the requested strategy")
    if worker.tokenizer.identity["tokenizer.json"] != manifest["tokenizer_sha256"]:
        raise ValueError("candidate tokenizer differs from the reference inputs")
    encoded = worker.tokenizer.tokenizer.encode(
        manifest["text"], add_special_tokens=False
    ).ids
    if encoded[:132] != manifest["cases"][1]["tokens"]:
        raise ValueError(
            "candidate tokenizer does not reproduce the frozen text inputs"
        )
    identity = runtime_identity(
        proof.rank, strategy=args.strategy, batch_decode=False, decode_graph=False
    )
    if identity["gpu_uuid"] != args.devices[proof.rank]:
        raise ValueError("candidate GPU UUID differs from the ordered lease")
    identity.update(
        strategy=args.strategy,
        configuration=config,
        inference_configuration=digest(args.model / "inference/config.json"),
        checkpoint=worker.checkpoint_identity,
        tokenizer=worker.tokenizer.identity,
        runtime={
            name: importlib.metadata.version(name)
            for name in ("torch", "safetensors", "tokenizers")
        },
        checker=checker_identity(),
        reference=reference,
        execution="eager",
        graph_verified=False,
        dep4_exchange="worker-default" if args.strategy == "dep4" else None,
        math_flags=manifest["math_flags"],
    )
    _write_json(args.output / f"identity-rank{proof.rank}.json", identity)
    return identity


def collect_rank(args, proof):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices[proof.rank]
    os.environ["LOCAL_RANK"] = "0"
    os.environ["OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT"] = "256"
    os.environ["OH_MY_VLLM_DEEPSEEK_DECODE_GRAPH"] = "0"
    os.environ["OH_MY_VLLM_DEEPSEEK_BATCH_DECODE"] = "0"
    os.environ.pop("OH_MY_VLLM_EXECUTION_TRACE", None)
    import torch
    import torch.distributed as dist
    from oh_my_vllm.models.deepseek_v41.inference import Worker
    from oh_my_vllm.models.deepseek_v41.trace import runtime_identity

    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{args.output / 'rank-store'}",
        rank=proof.rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=240),
    )
    worker = None
    try:
        manifest, expected, reference = load_reference(args.reference)
        torch.backends.cuda.matmul.allow_tf32 = manifest["math_flags"][
            "cuda_matmul_allow_tf32"
        ]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = manifest[
            "math_flags"
        ]["allow_bf16_reduced_precision_reduction"]
        config = dict(
            model_path=str(args.model),
            block_size=128,
            tensor_parallel_size=4,
            max_model_len=256,
            max_num_seqs=1,
            max_num_batched_tokens=8192,
            num_gpu_blocks=3,
            mamba_blocks=3,
            num_speculative_tokens=5,
        )
        worker = Worker(
            config,
            raw_config=(args.model / "config.json").read_bytes(),
            strategy=args.strategy,
        )
        identity = _candidate_identity(
            args, proof, worker, manifest, reference, config, runtime_identity
        )
        _collect_endpoints(args, proof, worker, manifest, expected, torch, identity)
        dist.barrier()
    finally:
        if worker is not None:
            worker.close()
        dist.destroy_process_group()


def _collect_endpoints(args, proof, worker, manifest, expected, torch, identity):
    from safetensors.torch import save_file

    capture = CandidateCapture(worker, manifest, expected, args.output, proof.rank)
    with torch.inference_mode():
        if args.strategy == "dep4":
            worker.dep4_executor.execute(
                lambda: capture.run(args.cases) if proof.rank == 0 else None
            )
        else:
            capture.run(args.cases)
    save_file(
        capture.arrays, str(args.output / f"endpoints-rank{proof.rank}.safetensors")
    )
    save_file(
        capture.runtime_arrays,
        str(args.output / f"runtime-endpoints-rank{proof.rank}.safetensors"),
    )
    capture.write_progress()
    identity["owned_cases"] = (
        list(args.cases) if args.strategy != "dep4" or proof.rank == 0 else []
    )
    if args.strategy == "dep4":
        identity.update(
            dep4_target_rounds=worker.dep4_executor.target_rounds,
            dep4_target_active_ranks=worker.dep4_executor.target_active_ranks,
            dep4_padded=worker.dep4_executor.padded,
        )
    _write_json(args.output / f"identity-rank{proof.rank}.json", identity)
