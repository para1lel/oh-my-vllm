"""Compare real batched decode and verification with independent requests."""

import argparse
import datetime
import hashlib
import json
import math
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path


def cpu(value, torch):
    from oh_my_vllm.models.deepseek_v41.packed_cache import PackedPage

    if isinstance(value, PackedPage):
        return cpu(vars(value), torch)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(item, torch) for item in value]
    return value


def differences(expected, actual, torch, name="endpoint"):
    if isinstance(expected, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return [dict(name=name, reason="tensor metadata differs")]
        if not torch.equal(
            expected.contiguous().view(torch.uint8),
            actual.contiguous().view(torch.uint8),
        ):
            return [
                dict(
                    name=name,
                    count=int((expected != actual).sum()),
                    size=expected.numel(),
                    max_abs=float(
                        (expected.float() - actual.float()).abs().nan_to_num().max()
                    ),
                )
            ]
        return []
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return [dict(name=name, reason="keys differ")]
        return [
            item
            for key in expected
            for item in differences(
                expected[key], actual[key], torch, name + "/" + str(key)
            )
        ]
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return [
                dict(
                    name=name,
                    reason="length differs",
                    expected=len(expected),
                    actual=len(actual),
                )
            ]
        return [
            item
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
            for item in differences(left, right, torch, name + "/" + str(index))
        ]
    return (
        []
        if expected == actual
        else [dict(name=name, expected=expected, actual=actual)]
    )


def fingerprint(value, torch):
    from oh_my_vllm.models.deepseek_v41.packed_cache import PackedPage

    if isinstance(value, PackedPage):
        return fingerprint(vars(value), torch)
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        )
    if isinstance(value, dict):
        return {key: fingerprint(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item, torch) for item in value]
    return value


def cache_state(worker, torch, *, digest=False):
    with torch.inference_mode():
        return _read_cache_state(worker, torch, digest=digest)


def _read_cache_state(worker, torch, *, digest=False):
    snapshot = fingerprint if digest else cpu
    result = {}
    cache = worker.cache
    for request_id, request in worker.requests.items():
        slot, end = request.cache_slot, request.state_position
        if slot is None:
            continue
        for kind, buffers in (("window", cache._window), ("mtp", cache._mtp_window)):
            for index, value in enumerate(buffers):
                result[f"{request_id}/{kind}/{index}"] = snapshot(value[slot], torch)
        for kind, buffers in (
            ("compressed", cache._compressed),
            ("index", cache._index),
        ):
            for index, (value, ratio) in enumerate(buffers):
                result[f"{request_id}/{kind}/{index}"] = snapshot(
                    value[slot, : end // ratio], torch
                )
        for index, value in enumerate(cache._engram):
            result[f"{request_id}/engram/{index}"] = snapshot(value[slot, :end], torch)
        for index, (kv, score) in enumerate(cache._tails):
            result[f"{request_id}/tail/{index}"] = snapshot(kv[slot], torch)
            result[f"{request_id}/score/{index}"] = snapshot(score[slot], torch)
    for kind, pages in (("fa", cache.fa_pages), ("mamba", cache.mamba_pages)):
        for identity, page in pages.items():
            result[f"pages/{kind}/{identity}"] = snapshot(vars(page), torch)
    return result


def row(request_id, tokens, position, *, blocks=2):
    first = (request_id - 1) * blocks + 1
    table = list(range(first, first + blocks))
    return dict(
        request_id=request_id,
        token_ids=tokens,
        num_computed_tokens=position,
        fa_block_table=list(table),
        mamba_block_table=list(table),
    )


def target_signature(position, length, cycle, window, ratios, bucket):
    return dict(
        length=length,
        bucket=bucket(position + length),
        residue=position % cycle,
        window_prefix=min(window, position) if length > 1 else window,
        window_block=position // window,
        compressed_groups={str(ratio): position // ratio for ratio in ratios},
    )


class TargetCapture:
    def __init__(self, worker, torch, batch):
        self.worker, self.torch, self.batch = worker, torch, batch
        self.targets = {}
        self.calls = []

    def record(
        self, logits, hidden, positions, slots, execution, graphs_before, *, length
    ):
        owners = {
            request.cache_slot: key for key, request in self.worker.requests.items()
        }
        for index, slot in enumerate(slots):
            key = owners[slot]
            self.targets.setdefault(key, []).append(
                cpu(
                    dict(
                        logits=logits[index],
                        hidden=None if hidden is None else hidden[index],
                    ),
                    self.torch,
                )
            )
        ratios = {
            layer.attn.compress_ratio
            for layer in self.worker.model.layers
            if layer.attn.compress_ratio > 1
        }
        self.calls.append(
            dict(
                requests=[owners[slot] for slot in slots],
                slots=list(slots),
                positions=list(positions),
                batch_size=len(slots),
                length=length,
                signatures=[
                    target_signature(
                        value,
                        length,
                        self.worker.decode_cycle,
                        self.worker.args.window_size,
                        ratios,
                        self.batch.context_token_bucket,
                    )
                    for value in positions
                ],
                execution=execution,
                captured=len(self.worker.decode_graphs) - graphs_before,
                graphs=len(self.worker.decode_graphs),
            )
        )

    def __enter__(self):
        self.original_target = self.worker._target
        self.original_group = self.worker._target_batch_group
        self.original_batch = self.batch.target_decode_batch

        def target(tokens, position, slot, **options):
            before = len(self.worker.decode_graphs)
            result = self.original_target(tokens, position, slot, **options)
            self.record(
                result[1],
                result[2],
                (position,),
                (slot,),
                self.worker._target_execution,
                before,
                length=len(tokens),
            )
            return result

        def group(tokens, position, slots, **options):
            before = len(self.worker.decode_graphs)
            result = self.original_group(tokens, position, slots, **options)
            self.record(
                result[1],
                result[2],
                (position,) * len(tokens),
                slots.tolist(),
                self.worker._target_execution,
                before,
                length=len(tokens[0]),
            )
            return result

        def batched(worker, tokens, positions, slots, **options):
            before = len(worker.decode_graphs)
            result = self.original_batch(worker, tokens, positions, slots, **options)
            self.record(
                result.logits,
                result.hidden,
                positions,
                slots,
                result.execution,
                before,
                length=len(tokens[0]),
            )
            return result

        self.worker._target = target
        self.worker._target_batch_group = group
        self.batch.target_decode_batch = batched
        return self

    def __exit__(self, *_error):
        self.worker._target = self.original_target
        self.worker._target_batch_group = self.original_group
        self.batch.target_decode_batch = self.original_batch


def request_state(worker):
    return {
        key: dict(
            history=list(request.history),
            computed=request.state_position,
            finish=request.finish,
        )
        for key, request in worker.requests.items()
    }


def execute_message(rows, *, rpc_id, evicted=()):
    return dict(
        type="execute",
        rpc_id=rpc_id,
        scheduled=rows,
        finished_request_ids=[],
        preempted_request_ids=list(evicted),
    )


def reset_graphs(worker, torch):
    torch.cuda.synchronize(worker.device)
    for graph in worker.decode_graphs.values():
        graph.close()
    worker.decode_graphs.clear()
    for bindings in worker.decode_bindings.values():
        bindings.close()
    worker.decode_bindings.clear()
    worker.decode_engram.clear()


def continuation_rows(worker, outputs, make_row, step):
    keys = [key for key in outputs if not worker.requests[key].finish]
    if keys and step >= 5:
        offset = (step - 4) % len(keys)
        keys = keys[offset:] + keys[:offset]
        if step % 2:
            keys.reverse()
    rows = []
    for key in keys:
        request = worker.requests[key]
        start = request.state_position
        if len(request.history) - start != 1:
            raise AssertionError("next target must start at the uncomputed output")
        drafts = outputs[key]["new_draft_token_ids"]
        if len(drafts) > min(5, max(0, request.limit - request.generated - 1)):
            raise AssertionError(
                "returned proposal exceeds the remaining output budget"
            )
        rows.append(make_row(key, [request.history[start], *drafts], start))
    return rows


def execute_scheduled(worker, rows, step, reference, evicted):
    if reference:
        if evicted:
            worker.execute(execute_message([], rpc_id=1000 + step, evicted=evicted))
        return [
            worker.execute(execute_message([item], rpc_id=2000 + step))["outputs"][0]
            for item in rows
        ]
    return worker.execute(execute_message(rows, rpc_id=2000 + step, evicted=evicted))[
        "outputs"
    ]


def execute_single_round(worker, rows, *, reference, batch_decode):
    if reference:
        return {item["request_id"]: worker._execute_row(item) for item in rows}
    if batch_decode:
        return worker._execute_decode_group(rows)
    outputs = worker.execute(execute_message(rows, rpc_id=3000))["outputs"]
    if len(outputs) != len(rows):
        raise AssertionError("scheduled request count differs from the worker outputs")
    return {item["request_id"]: item for item in outputs}


def require_scheduled_requests(rows, outputs, targets):
    expected = [item["request_id"] for item in rows]
    returned = [item["request_id"] for item in outputs]
    if (
        len(expected) != len(set(expected))
        or sorted(returned) != sorted(expected)
        or set(targets) != set(expected)
        or any(len(records) != 1 for records in targets.values())
    ):
        raise AssertionError(
            "scheduled requests differ from outputs or target execution"
        )


def require_single_execution(execution, calls):
    required = "graph" if execution == "graph" else "eager"
    if not calls or any(call["execution"] != required for call in calls):
        raise AssertionError(
            f"requested {required} target execution differs from captured calls"
        )


def update_boundaries(calls, observed, crossings):
    for call in calls:
        if call["batch_size"] != len(call["requests"]):
            raise AssertionError("target batch size differs from the captured requests")
        for key, signature in zip(call["requests"], call["signatures"], strict=True):
            seen = observed.setdefault(key, {})
            values = dict(
                bucket=signature["bucket"],
                window=signature["window_block"],
                **{
                    f"compress/{ratio}": group
                    for ratio, group in signature["compressed_groups"].items()
                },
            )
            for name, value in values.items():
                seen.setdefault(name, set()).add(value)
                crossings[name] |= len(seen[name]) > 1


def require_multiround_gates(
    mode,
    execution,
    *,
    continued,
    max_batch,
    crossings,
    require_verify_continuation=True,
    require_boundaries=False,
    require_batch=True,
    scheduled_requests=None,
    required_requests=None,
):
    if required_requests is not None and scheduled_requests != required_requests:
        raise AssertionError("candidate did not schedule the full request batch")
    if mode.startswith("dspark") and require_verify_continuation and not continued:
        raise AssertionError("verification did not consume the preceding proposal")
    if execution != "reference" and require_batch and max_batch <= 1:
        raise AssertionError("candidate did not execute a multi-request target group")
    if mode != "remaining" and require_boundaries and not all(crossings.values()):
        raise AssertionError(
            f"actual target boundary coverage is incomplete: {crossings}"
        )


def write_round_report(args, rank, report):
    with (args.output / f"multiround-rank{rank}.jsonl").open("a") as output:
        output.write(json.dumps(report) + "\n")
    print(json.dumps(report), flush=True)


def multiround(worker, args, torch, batch, prepare, make_row, rank):
    executions = (
        ("reference", "eager", "graph", "graph_fallback")
        if args.graph
        else ("reference", "eager")
    )
    reports = []
    original_limit = worker.decode_graph_limit
    try:
        for mode, same_signature in (
            ("ordinary", True),
            ("dspark_aligned", True),
            ("dspark", False),
            ("remaining", True),
        ):
            for execution in executions:
                worker.decode_graph_limit = (
                    1 if execution == "graph_fallback" else original_limit
                )
                budgets = {
                    key: 2 + index % 2 if mode == "remaining" else 6 * args.rounds + 8
                    for index, key in enumerate(range(1, args.batch + 1))
                }
                outputs = prepare(
                    mode != "ordinary",
                    budgets=budgets,
                    through_execute=True,
                    same_signature=same_signature,
                    shared_pair=mode == "dspark",
                )
                worker.decode_graph_enabled = execution.startswith("graph")
                worker.batch_decode_enabled = (
                    args.batch_decode and execution != "reference"
                )
                previous_verify = set()
                continued = graph_calls = fallback_calls = 0
                graph_verifies = graph_reuses = max_batch = max_scheduled = 0
                observed = {}
                eviction_steps = 0
                remapped = False
                ratios = {
                    layer.attn.compress_ratio
                    for layer in worker.model.layers
                    if layer.attn.compress_ratio > 1
                }
                crossings = {
                    "window": False,
                    "bucket": False,
                    **{f"compress/{ratio}": False for ratio in ratios},
                }
                for step in range(args.rounds):
                    rows = continuation_rows(worker, outputs, make_row, step)
                    if not rows:
                        break
                    keys = [item["request_id"] for item in rows]
                    continued += sum(
                        key in previous_verify
                        and bool(outputs[key]["new_draft_token_ids"])
                        for key in keys
                    )
                    previous_verify = {
                        item["request_id"]
                        for item in rows
                        if len(item["token_ids"]) > 1
                    }
                    evicted = keys if step in (2, 5) else ()
                    eviction_steps += bool(evicted)
                    before_slots = {
                        key: worker.requests[key].cache_slot for key in keys
                    }
                    with TargetCapture(worker, torch, batch) as captured:
                        result = execute_scheduled(
                            worker, rows, step, execution == "reference", evicted
                        )
                    require_scheduled_requests(rows, result, captured.targets)
                    max_scheduled = max(max_scheduled, len(rows))
                    outputs = {item["request_id"]: item for item in result}
                    for key in keys:
                        request = worker.requests[key]
                        if request.state_position != len(request.history) - 1:
                            raise AssertionError(
                                "accepted frontier differs from the uncomputed output"
                            )
                    update_boundaries(captured.calls, observed, crossings)
                    max_batch = max(
                        [max_batch, *(call["batch_size"] for call in captured.calls)]
                    )
                    remapped |= any(
                        worker.requests[key].cache_slot != before_slots[key]
                        for key in keys
                    )
                    actual = dict(
                        outputs=outputs,
                        targets=captured.targets,
                        requests=request_state(worker),
                        state=cache_state(worker, torch, digest=True),
                    )
                    path = args.output / f"multiround-{mode}-rank{rank}-step{step}.pt"
                    if execution == "reference":
                        torch.save(actual, path)
                        changed = []
                    else:
                        expected = torch.load(path, weights_only=True)
                        changed = differences(expected, actual, torch)
                    for call in captured.calls:
                        if call["execution"] == "graph":
                            graph_calls += 1
                            graph_verifies += call["length"] > 1
                            graph_reuses += call["captured"] == 0
                        elif execution.startswith("graph"):
                            if call["graphs"] < worker.decode_graph_limit:
                                raise AssertionError(
                                    "target used Eager before the graph cache limit"
                                )
                            fallback_calls += 1
                    report = dict(
                        rank=rank,
                        suite="multiround",
                        mode=mode,
                        execution=execution,
                        step=step,
                        scheduled_requests=len(rows),
                        executed_request_ids=sorted(captured.targets),
                        batch_decode_enabled=worker.batch_decode_enabled,
                        differences=changed,
                        targets=captured.calls,
                        accepted=[item["num_accepted_draft_tokens"] for item in result],
                        computed={
                            key: worker.requests[key].state_position for key in keys
                        },
                        slot_evictions=list(evicted),
                        diagnostic_only=not args.require_verify_continuation,
                    )
                    reports.append(report)
                    write_round_report(args, rank, report)
                    if changed:
                        torch.save(
                            actual, args.output / f"multiround-failure-rank{rank}.pt"
                        )
                        raise AssertionError("multiround checkpoint endpoints differ")
                if execution.startswith("graph") and not graph_calls:
                    raise AssertionError("multiround run did not replay a target graph")
                if mode == "ordinary" and execution == "graph" and not graph_reuses:
                    raise AssertionError(
                        "ordinary decode did not reuse a captured graph"
                    )
                require_multiround_gates(
                    mode,
                    execution,
                    continued=continued,
                    max_batch=max_batch,
                    crossings=crossings,
                    require_verify_continuation=args.require_verify_continuation,
                    require_boundaries=args.require_boundaries,
                    require_batch=args.batch_decode,
                    scheduled_requests=max_scheduled,
                    required_requests=args.batch,
                )
                if mode.startswith("dspark"):
                    if execution == "graph" and not graph_verifies:
                        raise AssertionError(
                            "multiround run did not replay a verification graph"
                        )
                    if not remapped:
                        raise AssertionError(
                            "multiround run did not reuse a slot across requests"
                        )
                if mode == "remaining" and any(
                    not request.finish for request in worker.requests.values()
                ):
                    raise AssertionError("remaining-one/two cases did not finish")
                if (
                    execution == "graph_fallback"
                    and mode.startswith("dspark")
                    and not fallback_calls
                ):
                    raise AssertionError(
                        "bounded graph cache did not exercise Eager fallback"
                    )
                coverage = dict(
                    rank=rank,
                    suite="coverage",
                    mode=mode,
                    execution=execution,
                    continued_verifications=continued,
                    graph_calls=graph_calls,
                    graph_verifications=graph_verifies,
                    reused_graphs=graph_reuses,
                    fallback_calls=fallback_calls,
                    remapped=remapped,
                    boundaries=crossings,
                    max_target_batch=max_batch,
                    max_scheduled_requests=max_scheduled,
                    batch_decode_enabled=worker.batch_decode_enabled,
                    batch_required=args.batch_decode and execution != "reference",
                    same_signature=same_signature,
                    shared_pair=mode == "dspark",
                    diagnostic_only=not args.require_verify_continuation,
                    comparison_scope="same-source batch/Graph equivalence",
                    frozen_mtp_reference_verified=False,
                    rust_preemption_recompute_verified=False,
                    slot_eviction_endpoint_restore=eviction_steps > 0,
                    slot_eviction_steps=eviction_steps,
                )
                write_round_report(args, rank, coverage)
    finally:
        worker.decode_graph_limit = original_limit
    return reports


def rank_main(rank, devices, args, store):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    os.environ["OH_MY_VLLM_DEEPSEEK_CACHE_CONTEXT"] = str(args.context)
    os.environ.pop("OH_MY_VLLM_EXECUTION_TRACE", None)
    sys.path[:0] = [str(args.tree / "python"), str(args.tree)]
    import torch
    import torch.distributed as dist
    from oh_my_vllm.models.deepseek_v41 import batch
    from oh_my_vllm.models.deepseek_v41.inference import Worker

    from development.serving.deepseek_benchmark import exact_prompt

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

    def make_row(request_id, tokens, position):
        return row(request_id, tokens, position, blocks=blocks)

    worker = None
    try:
        worker = Worker(
            config,
            raw_config=(args.model / "config.json").read_bytes(),
            strategy=args.strategy,
        )
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
            checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        prompts = {
            index: worker.tokenizer.prepare(
                dict(
                    messages=exact_prompt(
                        worker.tokenizer, length, f"decode-group-{index}"
                    ),
                    effort="off",
                )
            )
            for index, length in enumerate(args.lengths, 1)
        }

        def prepare(
            dspark,
            *,
            budgets=None,
            through_execute=False,
            same_signature=False,
            shared_pair=False,
        ):
            worker.decode_graph_enabled = False
            worker.batch_decode_enabled = False
            worker.dspark = dspark
            reset_graphs(worker, torch)
            for request_id in tuple(worker.requests):
                worker._release(request_id)
            worker.cache.reset()
            outputs = {}
            for request_id, prompt in prompts.items():
                aligned = same_signature or (shared_pair and request_id <= 2)
                tokens = next(iter(prompts.values())) if aligned else prompt
                worker.handle(
                    dict(
                        type="register",
                        rpc_id=request_id,
                        request_id=request_id,
                        prompt_token_ids=tokens,
                        max_tokens=32 if budgets is None else budgets[request_id],
                    )
                )
                item = make_row(request_id, tokens, 0)
                outputs[request_id] = (
                    worker.execute(execute_message([item], rpc_id=request_id))[
                        "outputs"
                    ][0]
                    if through_execute
                    else worker._execute_row(item)
                )
            return outputs

        chains = {}
        if args.suite != "multiround":
            outputs = prepare(False)
            chains = {key: list(value["token_ids"]) for key, value in outputs.items()}
            for _ in range(5):
                for key in prompts:
                    output = worker._execute_row(
                        make_row(
                            key, [chains[key][-1]], worker.requests[key].state_position
                        )
                    )
                    chains[key].extend(output["token_ids"])
            if any(len(tokens) != 6 for tokens in chains.values()):
                raise AssertionError("verification requires six greedy-chain tokens")

        round_evidence = {}

        def run(mode, execution):
            outputs = prepare(mode != "ordinary")
            worker.batch_decode_enabled = args.batch_decode and execution != "reference"
            before = cache_state(worker, torch)
            rows = []
            for index, key in enumerate(prompts):
                tokens = list(
                    outputs[key]["token_ids"] if mode == "ordinary" else chains[key]
                )
                if mode.startswith("reject"):
                    position = int(mode.removeprefix("reject")) + 1
                    tokens[position] = (tokens[position] + 1) % worker.args.vocab_size
                elif mode == "mixed" and index:
                    position = min(5, 2 * index - 1)
                    tokens[position] = (tokens[position] + 1) % worker.args.vocab_size
                rows.append(make_row(key, tokens, worker.requests[key].state_position))
            worker.decode_graph_enabled = execution == "graph"
            captured = []
            original_target = worker._target
            original_batch = batch.target_decode_batch

            def capture_target(*values, **options):
                result = original_target(*values, **options)
                captured.append(
                    tuple(
                        value.clone() if isinstance(value, torch.Tensor) else value
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

            worker._target = capture_target
            batch.target_decode_batch = capture_batch
            try:
                with TargetCapture(worker, torch, batch) as scheduled:
                    outputs = execute_single_round(
                        worker,
                        rows,
                        reference=execution == "reference",
                        batch_decode=args.batch_decode,
                    )
                require_scheduled_requests(
                    rows, list(outputs.values()), scheduled.targets
                )
                require_single_execution(execution, scheduled.calls)
                round_evidence[execution] = dict(
                    scheduled_requests=len(rows),
                    executed_request_ids=sorted(scheduled.targets),
                    batch_decode_enabled=worker.batch_decode_enabled,
                    target_executions=sorted(
                        {call["execution"] for call in scheduled.calls}
                    ),
                )
                logits = torch.cat([item[0] for item in captured], dim=0)
                hidden = (
                    torch.cat([item[1] for item in captured], dim=0)
                    if captured[0][1] is not None
                    else None
                )
                targets = cpu(dict(logits=logits, hidden=hidden), torch)
                endpoints = {
                    key: request.state_position
                    for key, request in worker.requests.items()
                }
                state = cache_state(worker, torch)
                captured.clear()
                for request in worker.requests.values():
                    worker._free_slot(request)
                # Restore and run through the same Eager single-request path.
                worker.decode_graph_enabled = False
                restored = {
                    key: worker._execute_row(
                        make_row(key, value["token_ids"][-1:], endpoints[key])
                    )
                    for key, value in outputs.items()
                }
                restored_targets = cpu(captured, torch)
                return dict(
                    before=before,
                    outputs=outputs,
                    targets=targets,
                    endpoints=endpoints,
                    state=state,
                    restored=restored,
                    restored_targets=restored_targets,
                    restored_state=cache_state(worker, torch),
                )
            finally:
                worker._target = original_target
                batch.target_decode_batch = original_batch

        reports = []
        executions = (
            ("reference", "eager", "graph") if args.graph else ("reference", "eager")
        )
        modes = (
            ()
            if args.suite == "multiround"
            else ("ordinary", "accept5", "reject0", "reject1", "reject3", "mixed")
        )
        for mode in modes:
            expected = run(mode, "reference")
            for execution in executions:
                actual = run(mode, execution)
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
                    for index in range(args.batch)
                ]
                if accepted != wanted:
                    raise AssertionError(
                        f"{mode} acceptance differs: {accepted} != {wanted}"
                    )
                changed = differences(expected, actual, torch)
                report = dict(
                    rank=rank,
                    mode=mode,
                    execution=execution,
                    differences=changed,
                    accepted=accepted,
                    **round_evidence[execution],
                )
                reports.append(report)
                (args.output / f"rank{rank}.json").write_text(
                    json.dumps(reports, indent=2) + "\n"
                )
                print(json.dumps(report), flush=True)
                if changed:
                    torch.save(
                        dict(expected=expected, actual=actual),
                        args.output / f"failure-rank{rank}.pt",
                    )
                    raise AssertionError("batched checkpoint endpoints differ")
        if args.suite != "single":
            multiround(worker, args, torch, batch, prepare, make_row, rank)
    finally:
        if worker is not None:
            worker.close()
        dist.destroy_process_group()


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tree", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--strategy", choices=("tp4", "tep4"), default="tp4")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--batch", type=int, choices=(4, 16, 64), default=4)
    parser.add_argument("--lengths", type=int, nargs="+", default=(130, 134, 138, 142))
    parser.add_argument(
        "--suite", choices=("single", "multiround", "all"), default="all"
    )
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--graph", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--batch-decode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable target groups; disabling uses scheduled individual requests.",
    )
    parser.add_argument(
        "--require-verify-continuation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require verify continuation; disabling gives diagnostic results only.",
    )
    parser.add_argument("--require-boundaries", action="store_true")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    if not args.require_verify_continuation:
        print(
            "DIAGNOSTIC ONLY: continuation disabled; not valid for acceptance.",
            file=sys.stderr,
        )
    args.graph = args.strategy == "tp4" if args.graph is None else args.graph
    if args.rounds < 6:
        parser.error(
            "multiround checks need at least six steps for slot reuse and fallback"
        )
    args.lengths = tuple(
        args.lengths[index % len(args.lengths)] for index in range(args.batch)
    )
    output_tokens = 32 if args.suite == "single" else max(32, 6 * args.rounds + 8)
    if min(args.lengths) < 1 or max(args.lengths) + output_tokens > args.context:
        parser.error("context must cover every prompt and its full output budget")
    sys.path[:0] = [str(args.tree / "python"), str(args.tree)]
    from oh_my_vllm.models.deepseek_v41.model import context_token_bucket

    configuration = json.loads((args.model / "inference/config.json").read_text())
    cycle = math.lcm(
        *(ratio for ratio in configuration["compress_ratios"] if ratio > 1)
    )
    window = configuration["window_size"]
    for length in (1, 6):
        groups = {
            (
                start % cycle,
                context_token_bucket(start + length),
                min(window, start) if length > 1 else window,
            )
            for start in args.lengths
        }
        if args.suite != "multiround" and len(groups) != 1:
            parser.error(
                "single-round lengths must share a context bucket and compressor branch"
            )
    if args.output.resolve().is_relative_to(args.tree.resolve()):
        parser.error("checkpoint artifacts must be outside the source tree")
    args.output.mkdir(parents=True, exist_ok=False)
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 4:
        raise ValueError("four leased GPUs are required")
    context = multiprocessing.get_context("spawn")
    with tempfile.TemporaryDirectory(
        prefix="batch-check-", dir=os.environ.get("TMPDIR")
    ) as directory:
        processes = [
            context.Process(
                target=rank_main,
                args=(rank, devices, args, str(Path(directory) / "nccl")),
            )
            for rank in range(4)
        ]
        try:
            for process in processes:
                process.start()
            while any(process.is_alive() for process in processes):
                if any(process.exitcode not in (None, 0) for process in processes):
                    raise RuntimeError("a checkpoint rank failed")
                for process in processes:
                    process.join(timeout=0.1)
            if any(process.exitcode != 0 for process in processes):
                raise RuntimeError("a checkpoint rank failed")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                if process.pid is not None:
                    process.join()


if __name__ == "__main__":
    main()
