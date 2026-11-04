"""Compare real batched decode and verification with independent requests."""

import argparse
import datetime
import json
import math
import multiprocessing
import os
import sys
import tempfile
from pathlib import Path


def cpu(value, torch):
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


def cache_state(worker, torch):
    result = {}
    cache = worker.cache
    for request_id, request in worker.requests.items():
        slot, end = request.cache_slot, request.state_position
        if slot is None:
            continue
        for kind, buffers in (("window", cache._window), ("mtp", cache._mtp_window)):
            for index, value in enumerate(buffers):
                result[f"{request_id}/{kind}/{index}"] = cpu(value[slot], torch)
        for kind, buffers in (
            ("compressed", cache._compressed),
            ("index", cache._index),
        ):
            for index, (value, ratio) in enumerate(buffers):
                result[f"{request_id}/{kind}/{index}"] = cpu(
                    value[slot, : end // ratio], torch
                )
        for index, value in enumerate(cache._engram):
            result[f"{request_id}/engram/{index}"] = cpu(value[slot, :end], torch)
        for index, (kv, score) in enumerate(cache._tails):
            result[f"{request_id}/tail/{index}"] = cpu(kv[slot], torch)
            result[f"{request_id}/score/{index}"] = cpu(score[slot], torch)
    for kind, pages in (("fa", cache.fa_pages), ("mamba", cache.mamba_pages)):
        for identity, page in pages.items():
            result[f"pages/{kind}/{identity}"] = cpu(vars(page), torch)
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

        def prepare(dspark):
            worker.decode_graph_enabled = False
            worker.dspark = dspark
            for request_id in tuple(worker.requests):
                worker._release(request_id)
            worker.cache.fa_pages.clear()
            worker.cache.mamba_pages.clear()
            outputs = {}
            for request_id, tokens in prompts.items():
                worker.handle(
                    dict(
                        type="register",
                        rpc_id=request_id,
                        request_id=request_id,
                        prompt_token_ids=tokens,
                        max_tokens=32,
                    )
                )
                outputs[request_id] = worker._execute_row(
                    make_row(request_id, tokens, 0)
                )
            return outputs

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

        def run(mode, execution):
            outputs = prepare(mode != "ordinary")
            before = cache_state(worker, torch)
            rows = []
            for index, key in enumerate(prompts):
                tokens = list(
                    outputs[key]["token_ids"] if mode == "ordinary" else chains[key]
                )
                if mode == "reject0":
                    tokens[1] = (tokens[1] + 1) % worker.args.vocab_size
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
                if execution == "graph" and result.execution != "graph":
                    raise AssertionError("batch silently used Eager")
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
                if execution == "reference":
                    outputs = {
                        item["request_id"]: worker._execute_row(item) for item in rows
                    }
                else:
                    outputs = worker._execute_decode_group(rows)
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
            ("reference", "eager", "graph")
            if args.strategy == "tp4"
            else ("reference", "eager")
        )
        for mode in ("ordinary", "accept5", "reject0", "mixed"):
            expected = run(mode, "reference")
            for execution in executions:
                actual = run(mode, execution)
                accepted = [
                    item["num_accepted_draft_tokens"]
                    for item in actual["outputs"].values()
                ]
                wanted = [
                    0
                    if mode in ("ordinary", "reject0")
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
    finally:
        if worker is not None:
            worker.close()
        dist.destroy_process_group()


def main():
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
    args = parser.parse_args()
    args.lengths = tuple(
        args.lengths[index % len(args.lengths)] for index in range(args.batch)
    )
    if min(args.lengths) < 1 or max(args.lengths) + 32 > args.context:
        parser.error("context must cover every prompt and its 32-token output")
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
        if len(groups) != 1:
            parser.error("lengths must share a context bucket and compressor branch")
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
