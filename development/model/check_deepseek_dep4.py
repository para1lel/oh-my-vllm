"""Compare DEP4 exchange paths with complete target and restoration endpoints."""

import argparse
import datetime
import hashlib
import json
import multiprocessing
import os
import time
import uuid
from contextlib import contextmanager
from multiprocessing.connection import wait
from pathlib import Path
from unittest.mock import patch


class RankGroupFailure(RuntimeError):
    pass


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _rank_worker(rank, args, run_id, operation):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4")
    proof = ExchangeEvidence(args.output, rank, mode=args.mode, run_id=run_id)
    try:
        with proof.observe():
            operation(args, proof)
    except BaseException as error:
        try:
            proof.fail(error)
        finally:
            os._exit(1)


def supervise(args, *, operation=None):
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError("DEP4 check output must be a new or empty directory")
    run_id = uuid.uuid4().hex
    manifest = dict(
        run_id=run_id,
        mode=args.mode,
        outcome="initializing",
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        lengths=getattr(args, "lengths", None),
        context=getattr(args, "context", None),
        evidence="cross-path-consistency-not-frozen-reference",
    )
    _write_json(args.output / "run.json", manifest)
    for rank in range(4):
        ExchangeEvidence(args.output, rank, mode=args.mode, run_id=run_id).write()
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_rank_worker, args=(rank, args, run_id, operation or run)
        )
        for rank in range(4)
    ]
    try:
        for process in processes:
            process.start()
        pending = {
            process.sentinel: (rank, process) for rank, process in enumerate(processes)
        }
        deadline = time.monotonic() + args.timeout_seconds
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("DEP4 check exceeded its process deadline")
            for sentinel in wait(pending, timeout=min(remaining, 0.2)):
                rank, process = pending.pop(sentinel)
                process.join()
                if process.exitcode:
                    raise RankGroupFailure(
                        f"rank {rank} exited with code {process.exitcode}"
                    )
    except BaseException as error:
        shutdown_started = time.monotonic()
        for process in processes:
            if process.pid is not None and process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=3)
        stopped = time.monotonic()
        failure_times = []
        for rank in range(4):
            path = args.output / f"proof-rank{rank}.json"
            record = json.loads(path.read_text())
            if record["failure_time_monotonic"] is not None:
                failure_times.append(record["failure_time_monotonic"])
            record["outcome"] = "failed"
            record["group_error"] = dict(type=type(error).__name__, message=str(error))
            _write_json(path, record)
        manifest.update(
            outcome="failed",
            error=dict(type=type(error).__name__, message=str(error)),
            shutdown_seconds=stopped - shutdown_started,
            failure_to_group_exit_seconds=stopped - min(failure_times)
            if failure_times
            else None,
            child_exitcodes=[process.exitcode for process in processes],
            remaining_child_pids=[
                process.pid
                for process in processes
                if process.pid is not None and process.is_alive()
            ],
        )
        _write_json(args.output / "run.json", manifest)
        raise
    else:
        manifest["outcome"] = "completed"
        _write_json(args.output / "run.json", manifest)
    finally:
        for process in processes:
            if process.pid is not None and not process.is_alive():
                process.close()


class ExchangeEvidence:
    def __init__(self, output, rank, *, mode, run_id=None):
        self.output = Path(output)
        self.rank = rank
        self.mode = mode
        self.run_id = run_id or uuid.uuid4().hex
        self.counts = dict(
            fixed=0,
            legacy=0,
            fallback=0,
            shadow_legacy=0,
            fixed_waves=0,
            padded_projections=0,
            completed=0,
            failed=0,
            maximum_source_rows=0,
            maximum_local_rows=0,
        )
        self.outcome = "initializing"
        self.error = None
        self.failure_time = None
        self.shadow_differences = []

    def fail(self, error):
        self.outcome = "failed"
        self.failure_time = time.monotonic()
        self.error = dict(type=type(error).__name__, message=str(error))
        self.write()

    def write(self):
        self.output.mkdir(parents=True, exist_ok=True)
        record = dict(
            run_id=self.run_id,
            rank=self.rank,
            mode=self.mode,
            outcome=self.outcome,
            counts=self.counts,
            error=self.error,
            failure_time_monotonic=self.failure_time,
            shadow_differences=self.shadow_differences,
        )
        _write_json(self.output / f"proof-rank{self.rank}.json", record)

    def require_fixed(self):
        if self.mode not in ("fixed", "shadow"):
            return
        counts = self.counts
        if (
            counts["fixed"] == 0
            or counts["legacy"]
            or counts["fallback"]
            or counts["fixed_waves"] == 0
            or counts["padded_projections"] != 3 * counts["fixed_waves"]
        ):
            raise AssertionError(f"fixed exchange path was not proved: {counts}")

    def compare_shadow(self, exchange, original, moe, rows, routing, actual):
        import torch
        import torch.distributed as dist

        exchange.fixed_round = False
        try:
            self.counts["shadow_legacy"] += 1
            expected = original(exchange, moe, rows, routing)
        finally:
            exchange.fixed_round = True
        difference = tensor_difference(expected, actual, torch)
        if difference:
            self.shadow_differences.append(
                dict(layer=moe.layer_id, source_rows=rows.size(0), **difference)
            )
        self.write()
        failed = torch.tensor([bool(difference)], dtype=torch.int32, device=rows.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=exchange.group)
        if failed.item():
            raise AssertionError(
                "fixed and legacy expert sums differ on at least one rank"
            )

    @contextmanager
    def observe(self):
        from oh_my_vllm.models.deepseek_v41 import kernel
        from oh_my_vllm.models.deepseek_v41.dispatch import (
            ExpertExchange,
            FixedDispatchBuffers,
        )

        original_exchange = ExpertExchange.__call__
        original_dispatch = FixedDispatchBuffers.dispatch
        original_project = kernel.grouped_fp4_gemm_padded

        def exchange(instance, moe, rows, routing):
            fixed = instance.fixed_round
            self.counts["fixed" if fixed else "legacy"] += 1
            if not fixed and self.mode in ("fixed", "shadow"):
                self.counts["fallback"] += 1
            for key, count in (
                ("maximum_source_rows", instance.source_rows),
                ("maximum_local_rows", rows.size(0)),
            ):
                self.counts[key] = max(self.counts[key], count)
            self.write()
            try:
                if self.mode == "shadow" and fixed:
                    legacy_rows = rows.clone()
                    legacy_routing = tuple(value.clone() for value in routing)
                value = original_exchange(instance, moe, rows, routing)
                if self.mode == "shadow" and fixed:
                    self.compare_shadow(
                        instance,
                        original_exchange,
                        moe,
                        legacy_rows,
                        legacy_routing,
                        value,
                    )
                self.counts["completed"] += 1
                return value
            except BaseException:
                self.counts["failed"] += 1
                raise
            finally:
                self.write()

        def dispatch(instance, *args, **kwargs):
            self.counts["fixed_waves"] += 1
            return original_dispatch(instance, *args, **kwargs)

        def project(*args, **kwargs):
            self.counts["padded_projections"] += 1
            return original_project(*args, **kwargs)

        self.write()
        with (
            patch.object(ExpertExchange, "__call__", exchange),
            patch.object(FixedDispatchBuffers, "dispatch", dispatch),
            patch.object(kernel, "grouped_fp4_gemm_padded", project),
        ):
            try:
                yield self
                self.require_fixed()
                self.outcome = "completed"
            except BaseException as error:
                self.outcome = "failed"
                self.error = dict(type=type(error).__name__, message=str(error))
                raise
            finally:
                self.write()


def tensor_difference(expected, actual, torch):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return dict(reason="tensor metadata differs")
    if torch.equal(
        expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
    ):
        return None
    delta = actual.double() - expected.double()
    return dict(
        mismatches=int((actual != expected).sum()),
        max_abs=float(delta.abs().max()) if delta.numel() else 0.0,
        relative_l2=float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.double()).clamp_min(1e-12)
        ),
    )


def initialize_rank(args, proof):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file

    baseline = (
        load_file(str(args.baseline / "endpoints.safetensors"))
        if args.baseline
        else None
    )
    chains = (
        json.loads((args.baseline / "chains.json").read_text())
        if args.baseline
        else None
    )
    torch.cuda.set_device(proof.rank)
    dist.init_process_group(
        "nccl",
        init_method=f"file://{args.output / 'rank-store'}",
        rank=proof.rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=240),
    )
    proof.outcome = "running"
    proof.write()
    return baseline, chains


def run(args, proof):
    import torch
    import torch.distributed as dist
    from oh_my_vllm.models.deepseek_v41 import model as candidate
    from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
    from oh_my_vllm.models.deepseek_v41.dep4 import Dep4Executor
    from oh_my_vllm.models.deepseek_v41.loader import RankLoader
    from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts
    from safetensors.torch import load_file, save_file
    from tokenizers import Tokenizer

    if os.environ.get("OH_MY_VLLM_GPU_COUNT") != "4":
        raise RuntimeError("this check requires four GPUs from the repository wrapper")
    rank = local_rank = proof.rank
    loader = cache = network = None
    initial = {}
    arrays, chains, reports = {}, {}, []
    baseline, baseline_chains = initialize_rank(args, proof)
    strategy = "TEP4" if args.mode == "tep4" else "DEP4"
    try:
        with torch.inference_mode(), torch.device("cuda", local_rank):
            torch.set_default_dtype(torch.bfloat16)
            torch.set_num_threads(1)
            config = json.loads((args.model / "inference/config.json").read_text())
            config.update(
                max_batch_size=1,
                max_seq_len=args.context,
                temperature=0,
                parallel_strategy=strategy,
                lazy_cache=False,
            )
            print(
                json.dumps(dict(rank=rank, phase="construct", mode=args.mode)),
                flush=True,
            )
            network = candidate.Transformer(
                candidate.ModelArgs(**config),
                Tokenizer.from_file(str(args.model / "tokenizer.json")),
            ).eval()
            loader = RankLoader(
                args.model,
                json.loads((args.model / "config.json").read_text()),
                rank,
                strategy=strategy,
            )
            loader.load_into(network)
            blocks = (args.context + 127) // 128
            cache = DeepSeekCache(
                network,
                max_seq_len=args.context,
                fa_capacity=blocks + 1,
                mamba_capacity=blocks + 1,
            )
            initial = {name: value.clone() for name, value in network.named_buffers()}
            executor = (
                Dep4Executor(network, padded=args.mode in ("fixed", "shadow"))
                if strategy == "DEP4"
                else None
            )
            print(
                json.dumps(dict(rank=rank, phase="loaded", mode=args.mode)), flush=True
            )

            def check(label, value):
                value = value.detach().contiguous().cpu()
                arrays[label] = value
                difference = (
                    tensor_difference(baseline[label], value, torch)
                    if baseline is not None
                    else None
                )
                record = dict(
                    label=label, passed=difference is None, difference=difference
                )
                reports.append(record)
                (args.output / f"rank{rank}-progress.json").write_text(
                    json.dumps(reports, indent=2) + "\n"
                )
                print(json.dumps(dict(rank=rank, **record)), flush=True)

            def reset():
                cache.reset()
                for name, value in network.named_buffers():
                    value.copy_(initial[name])
                candidate.shared_attn = candidate.SharedAttentionRuntime()

            def target(ids, position, label=None, *, full=False):
                result = network(ids, position, full_logits=full)
                if label:
                    check(label + ".logits", result[1])
                    check(label + ".ids", result[1].argmax(-1))
                    check(label + ".hidden", result[2][:, -1:])
                return result

            def workload(request):
                prefix = f"request{request}"
                length = args.lengths[request]
                tokens = torch.tensor(
                    [
                        [
                            900 + request * 137 + (i * 17) % 100
                            for i in range(max(132, length))
                        ]
                    ],
                    dtype=torch.long,
                )
                row = dict(
                    request_id=request + 1,
                    fa_block_table=list(range(1, blocks + 1)),
                    mamba_block_table=list(range(1, blocks + 1)),
                )
                reset()
                cache.begin_step(
                    row, 0, start_pos=0, end_pos=length + 2, minimum_end=length + 2
                )
                result = target(tokens[:, :length], 0, prefix + ".prefill")
                if args.prefill_only:
                    return
                current = result[0].reshape(1, 1)
                for offset in range(2):
                    result = target(
                        current, length + offset, prefix + f".decode{offset}"
                    )
                    current = result[0].reshape(1, 1)
                endpoint = length + 2
                cache.commit(
                    request + 1, row, 0, accepted_end=endpoint, include_mtp=False
                )
                continuous = target(current, endpoint, prefix + ".continuous")
                for name, value in network.named_buffers():
                    value.copy_(initial[name])
                candidate.shared_attn = candidate.SharedAttentionRuntime()
                cache.restore(
                    request + 1, 0, row, start_pos=endpoint, use_prefix_state=False
                )
                restored = target(current, endpoint, prefix + ".restart")
                torch.testing.assert_close(restored[1], continuous[1], rtol=0, atol=0)
                if baseline_chains is None:
                    reset()
                    target(tokens[:, :125], 0)
                    result = target(tokens[:, 125:126], 125)
                    chain = [int(result[0].item())]
                    for position in range(126, 131):
                        result = target(
                            torch.tensor([[chain[-1]]], dtype=torch.long), position
                        )
                        chain.append(int(result[0].item()))
                else:
                    chain = baseline_chains[prefix]
                chains[prefix] = chain
                for name, values in (
                    ("accept5", chain),
                    ("reject0", [chain[0], (chain[1] + 1) % 129280, *chain[2:]]),
                ):
                    reset()
                    _, _, hidden = target(tokens[:, :125], 0)
                    network.forward_spec(tokens[:, 125], hidden, 0)
                    _, _, hidden = target(tokens[:, 125:126], 125)
                    inputs = torch.tensor([values], dtype=torch.long)
                    proposal = network.forward_spec(inputs[:, 0], hidden, 125)
                    for suffix, value in zip(
                        ("proposal_ids", "proposal_logits", "proposal_confidence"),
                        proposal,
                        strict=True,
                    ):
                        check(prefix + "." + name + "." + suffix, value)
                    cache.commit(
                        request + 1, row, 0, accepted_end=126, include_mtp=True
                    )
                    cache.begin_step(
                        row, 0, start_pos=126, end_pos=132, minimum_end=127
                    )
                    _, logits, hidden = target(
                        inputs, 126, prefix + "." + name + ".verify", full=True
                    )
                    verification = verify_greedy_drafts(
                        values[1:], logits[0].argmax(-1).tolist()
                    )
                    desired = 5 if name == "accept5" else 0
                    if verification.accepted_draft_tokens != desired:
                        raise AssertionError(
                            f"unexpected verification endpoint: {prefix}/{name}"
                        )
                    check(
                        prefix + "." + name + ".emitted",
                        torch.tensor(verification.token_ids, dtype=torch.long),
                    )
                    accepted_end = 126 + desired + 1
                    network.forward_spec_seed(hidden[:, : desired + 1], 126)
                    cache.commit(
                        request + 1,
                        row,
                        0,
                        start_pos=126,
                        accepted_end=accepted_end,
                        include_mtp=True,
                    )
                    cache.restore(
                        request + 1,
                        0,
                        row,
                        start_pos=accepted_end,
                        use_prefix_state=False,
                    )
                    correction = torch.tensor(
                        [[verification.token_ids[-1]]], dtype=torch.long
                    )
                    target(
                        correction,
                        accepted_end,
                        prefix + "." + name + ".restored_probe",
                    )

            def work():
                requests = (
                    range(3) if strategy == "TEP4" else ([rank] if rank < 3 else [])
                )
                for request in requests:
                    workload(request)

            if executor is None:
                work()
            else:
                executor.execute(work)
            save_file(arrays, str(args.output / f"endpoints-rank{rank}.safetensors"))
            gathered = [None] * 4
            dist.all_gather_object(
                gathered,
                dict(
                    rank=rank,
                    chains=chains,
                    passed=all(row["passed"] for row in reports),
                ),
            )
            if rank == 0:
                merged_chains = {
                    key: value
                    for record in gathered
                    for key, value in record["chains"].items()
                }
                (args.output / "chains.json").write_text(
                    json.dumps(merged_chains, indent=2) + "\n"
                )
                combined = {}
                owners = [0] if strategy == "TEP4" else range(3)
                for owner in owners:
                    combined.update(
                        load_file(
                            str(args.output / f"endpoints-rank{owner}.safetensors")
                        )
                    )
                save_file(combined, str(args.output / "endpoints.safetensors"))
            if not all(record["passed"] for record in gathered):
                raise AssertionError("complete endpoint comparison failed")
            proof.require_fixed()
            dist.barrier()
    except BaseException as error:
        try:
            proof.fail(error)
        finally:
            # Failed collectives cannot safely enter process-group cleanup.
            os._exit(1)
    finally:
        if cache is not None:
            cache.close()
        initial.clear()
        network = None
        if loader is not None:
            loader.close()
        torch.cuda.synchronize(local_rank)
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument(
        "--mode", choices=("tep4", "legacy", "fixed", "shadow"), required=True
    )
    parser.add_argument("--lengths", type=int, nargs=3, default=[600, 300, 17])
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    args = parser.parse_args()
    if min(args.lengths) < 1 or max(133, max(args.lengths) + 3) > args.context:
        parser.error(
            "context must cover the prompt, continuation, and verification endpoints"
        )
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    if "RANK" in os.environ:
        parser.error("launch this driver once through with-gpu.sh, without torchrun")
    args.output = args.output.resolve()
    supervise(args)


if __name__ == "__main__":
    main()
