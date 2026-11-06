import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

from .deepseek_batch_capture import TargetCapture, cache_state, cpu, differences, row
from .deepseek_batch_rounds import multiround
from .deepseek_batch_schedule import (
    execute_message,
    execute_single_round,
    require_scheduled_requests,
    require_single_execution,
    reset_graphs,
)


def rank_main(rank, devices, args, store):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    os.environ["OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT"] = str(args.context)
    os.environ.pop("OH_MY_VLLM_EXECUTION_TRACE", None)
    sys.path[:0] = [str(args.tree / "python"), str(args.tree)]
    import torch
    import torch.distributed as dist
    from oh_my_vllm.models.deepseek_v41 import batch
    from oh_my_vllm.models.deepseek_v41.inference import Worker

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.manual_seed(0)
    dist.init_process_group(
        "nccl",
        init_method="file://" + store,
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=240),
    )
    blocks = (args.context + 127) // 128
    config = dict(
        model_path=str(args.model),
        block_size=128,
        tensor_parallel_size=4,
        max_model_len=args.context,
        max_num_seqs=args.batch,
        max_num_batched_tokens=8192,
        num_gpu_blocks=max(64, blocks * args.batch + 1),
        mamba_blocks=max(32, blocks * args.batch + 1),
        num_speculative_tokens=5,
    )

    worker = None
    try:
        worker = Worker(
            config,
            raw_config=(args.model / "config.json").read_bytes(),
            strategy=args.strategy,
        )
        write_identity(worker, args, config, rank)
        runner = BatchRank(worker, args, torch, batch, rank, blocks)
        runner.prepare_chains()
        runner.compare()
        if args.suite != "single":
            multiround(
                worker, args, torch, batch, runner.prepare, runner.make_row, rank
            )
    finally:
        if worker is not None:
            worker.close()
        dist.destroy_process_group()


def write_identity(worker, args, config, rank):
    from oh_my_vllm.models.deepseek_v41.trace import runtime_identity

    identity = runtime_identity(
        rank,
        strategy=args.strategy,
        batch_decode=args.batch_decode,
        decode_graph=args.graph,
        decode_graph_limit=worker.decode_graph_limit,
    )
    identity.update(
        checkpoint=worker.checkpoint_identity,
        tokenizer=worker.tokenizer.identity,
        config=config,
        checker_modules={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "check_deepseek_batch.py",
                "deepseek_batch_capture.py",
                "deepseek_batch_rank.py",
                "deepseek_batch_rounds.py",
                "deepseek_batch_schedule.py",
            )
        },
        checker_sha256=hashlib.sha256(
            Path(__file__).with_name("check_deepseek_batch.py").read_bytes()
        ).hexdigest(),
        arguments={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        state_comparison="full target vectors; SHA256 covers all cache bytes",
        target_scope="target methods exclude proposal and MTP seed calls",
        diagnostic_only=not args.require_verify_continuation,
        comparison_scope="same-source batch/Graph equivalence",
        frozen_mtp_reference_verified=False,
        rust_preemption_recompute_verified=False,
        slot_eviction_scope="resident slot eviction and accepted endpoint restore",
    )
    (args.output / f"identity-rank{rank}.json").write_text(
        json.dumps(identity, indent=2) + "\n"
    )


class BatchRank:
    def __init__(self, worker, args, torch, batch, rank, blocks):
        from development.serving.deepseek_benchmark import exact_prompt

        self.worker = worker
        self.args = args
        self.torch = torch
        self.batch = batch
        self.rank = rank
        self.blocks = blocks
        self.round_evidence = {}
        self.prompts = {
            index: self.worker.tokenizer.prepare(
                dict(
                    messages=exact_prompt(
                        self.worker.tokenizer, length, f"decode-group-{index}"
                    ),
                    effort="off",
                )
            )
            for index, length in enumerate(self.args.lengths, 1)
        }

    def make_row(self, request_id, tokens, position):
        return row(request_id, tokens, position, blocks=self.blocks)

    def prepare(
        self,
        dspark,
        *,
        budgets=None,
        through_execute=False,
        same_signature=False,
        shared_pair=False,
    ):
        self.worker.decode_graph_enabled = False
        self.worker.batch_decode_enabled = False
        self.worker.dspark = dspark
        reset_graphs(self.worker, self.torch)
        for request_id in tuple(self.worker.requests):
            self.worker._release(request_id)
        self.worker.cache.reset()
        outputs = {}
        for request_id, prompt in self.prompts.items():
            aligned = same_signature or (shared_pair and request_id <= 2)
            tokens = next(iter(self.prompts.values())) if aligned else prompt
            self.worker.handle(
                dict(
                    type="register",
                    rpc_id=request_id,
                    request_id=request_id,
                    prompt_token_ids=tokens,
                    max_tokens=32 if budgets is None else budgets[request_id],
                )
            )
            item = self.make_row(request_id, tokens, 0)
            outputs[request_id] = (
                self.worker.execute(execute_message([item], rpc_id=request_id))[
                    "outputs"
                ][0]
                if through_execute
                else self.worker._execute_row(item)
            )
        return outputs

    def prepare_chains(self):
        self.chains = {}
        if self.args.suite != "multiround":
            outputs = self.prepare(False)
            self.chains = {
                key: list(value["token_ids"]) for key, value in outputs.items()
            }
            for _ in range(5):
                for key in self.prompts:
                    output = self.worker._execute_row(
                        self.make_row(
                            key,
                            [self.chains[key][-1]],
                            self.worker.requests[key].state_position,
                        )
                    )
                    self.chains[key].extend(output["token_ids"])
            if any(len(tokens) != 6 for tokens in self.chains.values()):
                raise AssertionError("verification requires six greedy-chain tokens")

    def round_rows(self, mode, outputs):
        rows = []
        for index, key in enumerate(self.prompts):
            tokens = list(
                outputs[key]["token_ids"] if mode == "ordinary" else self.chains[key]
            )
            if mode.startswith("reject"):
                position = int(mode.removeprefix("reject")) + 1
                tokens[position] = (tokens[position] + 1) % self.worker.args.vocab_size
            elif mode == "mixed" and index:
                position = min(5, 2 * index - 1)
                tokens[position] = (tokens[position] + 1) % self.worker.args.vocab_size
            rows.append(
                self.make_row(key, tokens, self.worker.requests[key].state_position)
            )
        return rows

    def endpoints(self, captured, outputs, before):
        logits = self.torch.cat([item[0] for item in captured], dim=0)
        hidden = (
            self.torch.cat([item[1] for item in captured], dim=0)
            if captured[0][1] is not None
            else None
        )
        targets = cpu(dict(logits=logits, hidden=hidden), self.torch)
        endpoints = {
            key: request.state_position for key, request in self.worker.requests.items()
        }
        state = cache_state(self.worker, self.torch)
        captured.clear()
        for request in self.worker.requests.values():
            self.worker._free_slot(request)
        # Restore and run through the same Eager single-request path.
        self.worker.decode_graph_enabled = False
        restored = {
            key: self.worker._execute_row(
                self.make_row(key, value["token_ids"][-1:], endpoints[key])
            )
            for key, value in outputs.items()
        }
        restored_targets = cpu(captured, self.torch)
        return dict(
            before=before,
            outputs=outputs,
            targets=targets,
            endpoints=endpoints,
            state=state,
            restored=restored,
            restored_targets=restored_targets,
            restored_state=cache_state(self.worker, self.torch),
        )

    def run(self, mode, execution):
        outputs = self.prepare(mode != "ordinary")
        self.worker.batch_decode_enabled = (
            self.args.batch_decode and execution != "reference"
        )
        before = cache_state(self.worker, self.torch)
        rows = self.round_rows(mode, outputs)
        self.worker.decode_graph_enabled = execution == "graph"
        captured = []
        original_target = self.worker._target
        original_batch = self.batch.target_decode_batch

        def capture_target(*values, **options):
            result = original_target(*values, **options)
            captured.append(
                tuple(
                    value.clone() if isinstance(value, self.torch.Tensor) else value
                    for value in result[1:]
                )
            )
            return result

        def capture_batch(*values, **options):
            result = original_batch(*values, **options)
            captured.append(
                (
                    result.logits.clone(),
                    result.hidden.clone() if result.hidden is not None else None,
                )
            )
            return result

        self.worker._target = capture_target
        self.batch.target_decode_batch = capture_batch
        try:
            with TargetCapture(self.worker, self.torch, self.batch) as scheduled:
                outputs = execute_single_round(
                    self.worker,
                    rows,
                    reference=execution == "reference",
                    batch_decode=self.args.batch_decode,
                )
            require_scheduled_requests(rows, list(outputs.values()), scheduled.targets)
            require_single_execution(execution, scheduled.calls)
            self.round_evidence[execution] = dict(
                scheduled_requests=len(rows),
                executed_request_ids=sorted(scheduled.targets),
                batch_decode_enabled=self.worker.batch_decode_enabled,
                target_executions=sorted(
                    {call["execution"] for call in scheduled.calls}
                ),
            )
            return self.endpoints(captured, outputs, before)
        finally:
            self.worker._target = original_target
            self.batch.target_decode_batch = original_batch

    def compare(self):
        reports = []
        executions = (
            ("reference", "eager", "graph")
            if self.args.graph
            else ("reference", "eager")
        )
        modes = (
            ()
            if self.args.suite == "multiround"
            else ("ordinary", "accept5", "reject0", "reject1", "reject3", "mixed")
        )
        for mode in modes:
            expected = self.run(mode, "reference")
            for execution in executions:
                actual = self.run(mode, execution)
                accepted = [
                    item["num_accepted_draft_tokens"]
                    for item in actual["outputs"].values()
                ]
                wanted = [
                    0
                    if mode == "ordinary"
                    else int(mode.removeprefix("reject"))
                    if mode.startswith("reject")
                    else min(5, 2 * index - 1) - 1
                    if mode == "mixed" and index
                    else 5
                    for index in range(self.args.batch)
                ]
                if accepted != wanted:
                    raise AssertionError(
                        f"{mode} acceptance differs: {accepted} != {wanted}"
                    )
                changed = differences(expected, actual, self.torch)
                report = dict(
                    rank=self.rank,
                    mode=mode,
                    execution=execution,
                    differences=changed,
                    accepted=accepted,
                    **self.round_evidence[execution],
                )
                reports.append(report)
                (self.args.output / f"rank{self.rank}.json").write_text(
                    json.dumps(reports, indent=2) + "\n"
                )
                print(json.dumps(report), flush=True)
                if changed:
                    self.torch.save(
                        dict(expected=expected, actual=actual),
                        self.args.output / f"failure-rank{self.rank}.pt",
                    )
                    raise AssertionError("batched checkpoint endpoints differ")
