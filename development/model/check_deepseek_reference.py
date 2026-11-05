"""Collect native Worker endpoints against the immutable independent MP4 reference."""

import argparse
import datetime
import importlib.metadata
import json
import os
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from .check_deepseek_batch import cpu, differences, fingerprint, row
from .check_deepseek_dep4 import _write_json, supervise
from .deepseek_reference_contract import (
    CASE_NAMES,
    acceptance_record,
    compare_tensor,
    digest,
    endpoint_schema,
    load_reference,
    selected_schema,
    validate_tensors,
)

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_SHA256 = "7f82d0e2dcca8243f7dcfea71c96b204f78e1ddbf720a8843a0e48baea82faed"


class CandidateCapture:
    def __init__(self, worker, manifest, expected, output, rank):
        import torch

        self.worker, self.torch = worker, torch
        self.manifest, self.expected = manifest, expected
        self.output, self.rank = output, rank
        self.arrays, self.runtime_arrays = {}, {}
        self.comparisons, self.states = [], []
        self.row = row(1, [], 0, blocks=2)

    def write_progress(self):
        _write_json(
            self.output / f"comparison-rank{self.rank}.json",
            dict(comparisons=self.comparisons, states=self.states),
        )

    def save(self, label, value, *, runtime=False):
        value = cpu(value, self.torch).contiguous()
        arrays = self.runtime_arrays if runtime else self.arrays
        if label in arrays:
            raise ValueError(f"candidate endpoint was collected twice: {label}")
        arrays[label] = value
        result = compare_tensor(label, self.expected[label], value)
        self.comparisons.append(dict(path="runtime" if runtime else "driver", **result))
        self.write_progress()

    def check(self, label, actual, expected):
        self.comparisons.append(
            dict(
                label=label,
                metric="exact",
                passed=actual == expected,
                actual=actual,
                expected=expected,
            )
        )
        self.write_progress()

    def reset(self):
        from oh_my_vllm.models.deepseek_v41 import model

        self.worker.cache.reset()
        self.worker.cache.activate_row(self.row, 0, end_pos=256)
        model.shared_attn = model.SharedAttentionRuntime()

    def target(self, tokens, position, label=None, *, runtime=False, full=False):
        from oh_my_vllm.models.deepseek_v41 import kernel

        result = self.worker._target(tokens, position, 0, full_logits=full)
        if kernel._SPARSE_NATIVE_DISABLED or kernel._SPARSE_NATIVE_FAILED:
            raise RuntimeError("candidate disabled native sparse attention")
        if label is not None:
            self.save(label + ".logits", result[1], runtime=runtime)
            self.save(label + ".hidden", result[2][:, -1:], runtime=runtime)
            self.save(label + ".argmax", result[0], runtime=runtime)
            self.check(
                label + ".sampled_ids",
                result[0].reshape(-1).tolist(),
                result[1].argmax(-1).reshape(-1).tolist(),
            )
        return result

    def proposal(self, predicted, hidden, position, label=None, *, runtime=False):
        from oh_my_vllm.models.deepseek_v41 import kernel

        result = self.worker.model.forward_spec(predicted.reshape(-1), hidden, position)
        if kernel._SPARSE_NATIVE_DISABLED or kernel._SPARSE_NATIVE_FAILED:
            raise RuntimeError("candidate disabled native sparse attention")
        if position == 0:
            if result is not None:
                raise ValueError("initial DSpark call must only seed the cache")
        elif label is not None:
            for suffix, value in zip(
                ("ids", "logits", "confidence"), result, strict=True
            ):
                self.save(label + "." + suffix, value, runtime=runtime)
        return result

    def base(self, case, *, record=False):
        self.reset()
        base = case["base_tokens"]
        predicted, _, hidden = self.target(base[:125], 0)
        self.check(case["name"] + ".base_known125", int(predicted.item()), base[125])
        self.proposal(predicted, hidden, 0)
        predicted, logits, hidden = self.target(base[125:], 125)
        self.check(
            case["name"] + ".base_known126", int(predicted.item()), case["chain"][0]
        )
        if record:
            self.save("authentic.base_logits", logits)
            self.save("authentic.base_hidden", hidden)
        self.proposal(predicted, hidden, 125, "authentic" if record else None)

    def ordinary(self, case):
        self.reset()
        prefix = case["prefill"]
        self.target(case["tokens"][:prefix], 0, f"{case['name']}.{prefix}")
        for position in range(prefix, len(case["tokens"])):
            label = (
                f"{case['name']}.{position + 1}"
                if position + 1 in case["endpoints"]
                else None
            )
            self.target(case["tokens"][position : position + 1], position, label)

    def verification(self, case):
        name, chain = case["name"], case["chain"]
        self.base(case, record=name == "authentic")
        rows = []
        for stage, token in enumerate(chain):
            _, logits, hidden = self.target([token], 126 + stage)
            rows.append(cpu(logits, self.torch))
            self.save(f"{name}.verify_hidden.{stage}", hidden)
        logits = self.torch.stack(rows, dim=1)
        self.save(name + ".verify_logits", logits)
        actual = self.verify(case, logits, "driver")
        self.base(case)
        for stage in range(actual["accepted_drafts"] + 1):
            predicted, logits, hidden = self.target([chain[stage]], 126 + stage)
            self.proposal(predicted, hidden, 126 + stage)
        self.save(name + ".accepted_logits", logits)
        self.save(name + ".accepted_hidden", hidden)
        predicted, _, hidden = self.target(
            [actual["correction"]],
            actual["accepted_endpoint"],
            name + ".restored_probe",
        )
        self.proposal(
            predicted, hidden, actual["accepted_endpoint"], name + ".restored_spec"
        )
        self.runtime_restoration(case)

    def verify(self, case, logits, path):
        from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts

        actual = acceptance_record(case["chain"], logits)
        production = verify_greedy_drafts(
            case["chain"][1:], logits.argmax(-1).reshape(-1).tolist()
        )
        self.check(
            path + "." + case["name"] + ".acceptance_rule",
            [production.accepted_draft_tokens, list(production.token_ids)],
            [actual["accepted_drafts"], actual["emitted_ids"]],
        )
        for field in (
            "accepted_drafts",
            "accepted_endpoint",
            "correction",
            "probe_endpoint",
        ):
            self.check(
                path + "." + case["name"] + "." + field, actual[field], case[field]
            )
        self.check(
            path + "." + case["name"] + ".emitted_ids",
            actual["emitted_ids"],
            [*case["chain"][1 : case["accepted_drafts"] + 1], case["correction"]],
        )
        return actual

    def commit(self, start, end):
        self.worker.cache.commit(
            1, self.row, 0, start_pos=start, accepted_end=end, include_mtp=True
        )

    def state(self, endpoint):
        cache = self.worker.cache

        def window(values, end):
            return [value[: min(end, value.size(0))] for value in values]

        def tails(values, end):
            return [
                (kv[: end % kv.size(0)], score[: end % kv.size(0)])
                for kv, score in values
            ]

        pages = {}
        for block, page in cache.fa_pages.items():
            if page.valid_tokens == 0:
                continue
            packed = {}
            for kind, histories, values in (
                ("compressed", cache._compressed, page.compressed),
                ("index", cache._index, page.index),
            ):
                packed[kind] = [
                    value[: page.valid_tokens // ratio]
                    for (_history, ratio), value in zip(histories, values, strict=True)
                ]
            pages[block] = dict(
                valid_tokens=page.valid_tokens,
                **packed,
                engram=[value[: page.valid_tokens] for value in page.engram],
            )
        states = {
            block: dict(
                endpoint=page.endpoint,
                window=window(page.window, page.endpoint),
                mtp_window=window(page.mtp_window, page.endpoint),
                engram=[value[: page.valid_tokens] for value in page.engram],
                tails=tails(page.tails, page.endpoint),
            )
            for block, page in cache.mamba_pages.items()
            if page.endpoint
        }
        return fingerprint(
            dict(
                endpoint=endpoint,
                window=window([value[0] for value in cache._window], endpoint),
                mtp_window=window([value[0] for value in cache._mtp_window], endpoint),
                tails=tails(
                    [(kv[0], score[0]) for kv, score in cache._tails], endpoint
                ),
                engram=[value[0, :endpoint] for value in cache._engram],
                fa_pages=pages,
                state_pages=states,
            ),
            self.torch,
        )

    def runtime_restoration(self, case):
        name, chain = case["name"], case["chain"]
        self.base(case)
        self.commit(0, 126)
        self.worker.cache.begin_step(
            self.row, 0, start_pos=126, end_pos=132, minimum_end=127
        )
        _, logits, hidden = self.target(chain, 126, full=True)
        self.save(name + ".verify_logits", logits, runtime=True)
        for stage in range(6):
            self.save(
                f"{name}.verify_hidden.{stage}",
                hidden[:, stage : stage + 1],
                runtime=True,
            )
        actual = self.verify(case, logits, "runtime")
        count, endpoint = actual["accepted_drafts"] + 1, actual["accepted_endpoint"]
        self.save(name + ".accepted_logits", logits[:, count - 1], runtime=True)
        self.save(name + ".accepted_hidden", hidden[:, count - 1 : count], runtime=True)
        self.worker._seed_mtp(hidden[:, :count], 126, 0, source_length=count)
        correction = self.torch.tensor([actual["correction"]], dtype=self.torch.long)
        self.proposal(correction, hidden[:, count - 1 : count], endpoint - 1)
        self.commit(126, endpoint)
        self.worker.cache.restore(
            1, 0, self.row, start_pos=endpoint, use_prefix_state=False
        )
        restored = self.state(endpoint)
        predicted, _, hidden = self.target(
            [actual["correction"]], endpoint, name + ".restored_probe", runtime=True
        )
        self.proposal(
            predicted, hidden, endpoint, name + ".restored_spec", runtime=True
        )
        self.base(case)
        self.commit(0, 126)
        for stage in range(count):
            position = 126 + stage
            self.worker.cache.begin_step(
                self.row, 0, start_pos=position, end_pos=position + 1
            )
            predicted, _, hidden = self.target([chain[stage]], position)
            self.proposal(predicted, hidden, position)
            self.commit(position, position + 1)
        fresh = self.state(endpoint)
        changed = differences(fresh, restored, self.torch)
        state = dict(
            case=name, endpoint=endpoint, passed=not changed, differences=changed
        )
        self.states.append(state)
        _write_json(
            self.output / f"state-{name}-rank{self.rank}.json",
            dict(**state, restored=restored, fresh_accepted_only=fresh),
        )
        self.write_progress()

    def run(self, cases):
        for case in self.manifest["cases"]:
            if case["name"] not in cases:
                continue
            if case["name"] in ("short", "window"):
                self.ordinary(case)
            else:
                self.verification(case)
            print(
                json.dumps(dict(rank=self.rank, case=case["name"], phase="collected")),
                flush=True,
            )
        validate_tensors(self.arrays, selected_schema(self.manifest, cases))


def rank_entry(rank, args, run_id, operation):
    # ExchangeEvidence imports CUDA code before it calls the supplied operation.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices[rank]
    from .check_deepseek_dep4 import _rank_worker

    _rank_worker(rank, args, run_id, operation)


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
    from safetensors.torch import save_file

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
            checker={
                path.name: digest(path)
                for path in (
                    Path(__file__),
                    Path(__file__).with_name("deepseek_reference_contract.py"),
                    ROOT / "development/model/check_deepseek_batch.py",
                    ROOT / "development/model/check_deepseek_dep4.py",
                )
            },
            reference=reference,
            execution="eager",
            graph_verified=False,
            dep4_exchange="worker-default" if args.strategy == "dep4" else None,
            math_flags=manifest["math_flags"],
        )
        _write_json(args.output / f"identity-rank{proof.rank}.json", identity)
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
        dist.barrier()
    finally:
        if worker is not None:
            worker.close()
        dist.destroy_process_group()


def coverage_matches(cases, comparisons, states, *, owns):
    schema = selected_schema(dict(tensors=endpoint_schema()), cases)
    required = {(label, "driver") for label in schema}
    required.update(
        (label, "runtime")
        for label in schema
        for name in cases
        if name not in ("short", "window")
        and label.startswith(name + ".")
        and label.removeprefix(name + ".").startswith(
            ("verify_", "accepted_", "restored_")
        )
    )
    observed = [
        (record["label"], record["path"]) for record in comparisons if "path" in record
    ]
    semantic = Counter()
    for label in schema:
        if label.endswith(".argmax"):
            probe = label.removesuffix(".argmax") + ".sampled_ids"
            semantic[probe] = 2 if ".restored_probe." in label else 1
    for name in cases:
        if name in ("short", "window"):
            continue
        for field in ("base_known125", "base_known126"):
            semantic[name + "." + field] = 4
        for path in ("driver", "runtime"):
            for field in (
                "acceptance_rule",
                "accepted_drafts",
                "accepted_endpoint",
                "correction",
                "probe_endpoint",
                "emitted_ids",
            ):
                semantic[path + "." + name + "." + field] = 1
    observed_semantic = Counter(
        record["label"] for record in comparisons if "path" not in record
    )
    state_names = [record["case"] for record in states]
    expected_states = {name for name in cases if name not in ("short", "window")}
    if not owns:
        return not comparisons and not states
    return (
        len(observed) == len(required)
        and set(observed) == required
        and observed_semantic == semantic
        and len(state_names) == len(expected_states)
        and set(state_names) == expected_states
        and all(
            record["endpoint"]
            == 127
            + (
                5
                if record["case"] == "authentic"
                else int(record["case"].split(".")[1])
            )
            for record in states
        )
        and all(record.get("passed") is True for record in (*comparisons, *states))
    )


def summarize(args, reference):
    ranks = []
    run = json.loads((args.output / "run.json").read_text())
    first_identity = json.loads((args.output / "identity-rank0.json").read_text())
    shared_identity = (
        "source_commit",
        "source_digest",
        "native_modules",
        "configuration",
        "inference_configuration",
        "checkpoint",
        "tokenizer",
        "runtime",
        "checker",
        "reference",
        "strategy",
        "execution_modes",
        "execution",
        "math_flags",
    )
    for rank in range(4):
        record = json.loads((args.output / f"comparison-rank{rank}.json").read_text())
        proof = json.loads((args.output / f"proof-rank{rank}.json").read_text())
        identity = json.loads((args.output / f"identity-rank{rank}.json").read_text())
        owns = args.strategy != "dep4" or rank == 0
        identity_matches = (
            identity.get("rank") == rank
            and identity.get("gpu_uuid") == args.devices[rank]
            and identity.get("strategy") == args.strategy
            and identity.get("reference") == reference
            and all(
                key in identity and identity[key] == first_identity.get(key)
                for key in shared_identity
            )
        )
        proof_matches = (
            run["outcome"] == "completed"
            and proof.get("run_id") == run["run_id"]
            and proof.get("rank") == rank
            and proof.get("mode") == args.mode
        )
        comparisons, states = record["comparisons"], record["states"]
        ranks.append(
            dict(
                rank=rank,
                passed=coverage_matches(args.cases, comparisons, states, owns=owns)
                and identity_matches
                and proof_matches
                and proof["outcome"] == "completed"
                and proof["counts"]["fallback"] == 0
                and proof["counts"]["failed"] == 0
                and (
                    args.strategy != "dep4"
                    or (
                        proof["counts"]["legacy"] > 0
                        and proof["counts"]["fixed"] == 0
                        and proof["counts"]["completed"] == proof["counts"]["legacy"]
                        and identity["dep4_target_rounds"] > 0
                        and identity["dep4_target_active_ranks"] > 0
                        and identity["owned_cases"]
                        == (list(args.cases) if owns else [])
                    )
                ),
                comparisons=len(comparisons),
                state_cases=len(states),
                coverage_complete=coverage_matches(
                    args.cases, comparisons, states, owns=owns
                ),
                identity_matches=identity_matches,
                proof_matches=proof_matches,
                failed=[
                    row
                    for row in (*comparisons, *states)
                    if row.get("passed") is not True
                ],
                gpu_uuid=identity["gpu_uuid"],
                exchange=proof["counts"],
            )
        )
    passed = all(rank["passed"] for rank in ranks)
    result = dict(
        reference=reference,
        strategy=args.strategy,
        execution="eager",
        graph_verified=False,
        batch=1,
        dep4_dense_owner_rank=0 if args.strategy == "dep4" else None,
        multiple_owners_verified=False,
        cases=list(args.cases),
        complete=tuple(args.cases) == CASE_NAMES,
        passed=passed,
        ranks=ranks,
        scope="independent full-vocabulary endpoints and native cache restoration",
        state_oracle=(
            "fresh accepted-only candidate replay; "
            "independent endpoint fixtures stay read-only"
        ),
        rust_service_verified=False,
    )
    _write_json(args.output / "acceptance.json", result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "ranks"}),
        flush=True,
    )
    if not passed:
        raise AssertionError(
            "independent reference or fresh-replay state comparison failed"
        )


def checkpoint_stats(model, names):
    return {
        name: dict(
            bytes=(model / name).stat().st_size,
            mtime_ns=(model / name).stat().st_mtime_ns,
            inode=(model / name).stat().st_ino,
            ctime_ns=(model / name).stat().st_ctime_ns,
            device=(model / name).stat().st_dev,
        )
        for name in sorted(names)
    }


def bundle_members(root):
    path = root / "bundle-manifest.json"
    if digest(path) != BUNDLE_SHA256:
        raise ValueError("reviewed recovery bundle SHA256 differs")
    return {
        record["path"]: record["sha256"]
        for record in json.loads(path.read_text())["files"]
    }


def checkpoint_inputs(args):
    from development.serving.deepseek_benchmark import checkpoint_identity

    members = bundle_members(args.reference)
    if (
        digest(args.checkpoint_digests)
        != members["provenance/current-original-checkpoint-sha256.json"]
    ):
        raise ValueError(
            "checkpoint digest record is not the reviewed recovery bundle member"
        )
    manifest = json.loads(args.checkpoint_digests.read_text())
    expected = {name: value["sha256"] for name, value in manifest["files"].items()}
    before = checkpoint_stats(
        args.model, (*expected, "config.json", "model.safetensors.index.json")
    )
    observed = checkpoint_identity(args.model)
    after = checkpoint_stats(args.model, before)
    if after != before:
        raise ValueError("checkpoint files changed during the SHA256 check")
    if {
        name: value
        for name, value in observed.items()
        if name not in ("config", "index")
    } != expected:
        raise ValueError(
            "candidate checkpoint bytes differ from the reference conversion inputs"
        )
    return observed, before


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--model", type=Path)
    result.add_argument("--strategy", choices=("tp4", "tep4", "dep4"), default="tp4")
    result.add_argument("--cases", nargs="+", choices=CASE_NAMES, default=CASE_NAMES)
    result.add_argument("--validate-only", action="store_true")
    result.add_argument("--checkpoint-digests", type=Path)
    result.add_argument("--evidence", type=Path, action="append", default=[])
    result.add_argument("--timeout-seconds", type=float, default=1800)
    return result


def main():
    arguments = parser()
    args = arguments.parse_args()
    args.reference, args.output = args.reference.resolve(), args.output.resolve()
    if (
        args.output.is_relative_to(ROOT)
        or args.reference.is_relative_to(args.output)
        or args.output.is_relative_to(args.reference)
    ):
        arguments.error(
            "reports must be external and separate from the read-only reference"
        )
    if args.output.exists():
        arguments.error("output must be a new external directory")
    if len(set(args.cases)) != len(args.cases) or args.timeout_seconds <= 0:
        arguments.error("cases must be distinct and timeout-seconds must be positive")
    args.cases = tuple(name for name in CASE_NAMES if name in args.cases)
    manifest, _, reference = load_reference(args.reference)
    members = bundle_members(args.reference)
    evidence = []
    for path in (args.reference / "bundle-manifest.json", *args.evidence):
        sha256 = digest(path)
        member = next(
            (name for name, value in members.items() if value == sha256), None
        )
        if sha256 != BUNDLE_SHA256 and member is None:
            raise ValueError("evidence is not a reviewed recovery bundle member")
        evidence.append(dict(path=str(path), sha256=sha256, bundle_member=member))
    if args.validate_only:
        args.output.mkdir(parents=True)
        _write_json(
            args.output / "validation.json",
            dict(
                reference=reference,
                cases=list(CASE_NAMES),
                tensors=len(manifest["tensors"]),
                evidence=evidence,
                candidate_executed=False,
            ),
        )
        print(
            json.dumps(
                dict(
                    reference=reference,
                    tensors=len(manifest["tensors"]),
                    candidate_executed=False,
                )
            )
        )
        return
    if args.model is None or args.checkpoint_digests is None:
        arguments.error(
            "candidate collection requires --model and --checkpoint-digests"
        )
    args.model = args.model.resolve()
    devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    selected = os.environ.get("OH_MY_VLLM_GPU_UUIDS", "").split(",")
    if (
        os.environ.get("OH_MY_VLLM_GPU_COUNT") != "4"
        or devices != selected
        or len(set(devices)) != 4
        or any(not value.startswith("GPU-") for value in devices)
    ):
        arguments.error("use with-gpu.sh with four explicit ordered GPU UUIDs")
    if os.environ.get("OH_MY_VLLM_DEEPSEEK_NATIVE_SPARSE", "1") != "1":
        arguments.error("native sparse attention must be enabled")
    if "RANK" in os.environ:
        arguments.error("launch this entry once, without torchrun")
    observed, stats = checkpoint_inputs(args)
    args.devices = devices
    args.mode = "legacy" if args.strategy == "dep4" else args.strategy
    try:
        with patch("development.model.check_deepseek_dep4._rank_worker", rank_entry):
            supervise(args, operation=collect_rank)
    finally:
        if args.output.exists():
            _write_json(
                args.output / "candidate-inputs.json",
                dict(
                    reference=reference,
                    checkpoint=observed,
                    checkpoint_stats=stats,
                    checkpoint_hash_method="full SHA256 of all current shards",
                    checkpoint_manifest_sha256=digest(args.checkpoint_digests),
                    evidence=evidence,
                    arguments=json.loads(json.dumps(vars(args), default=str)),
                ),
            )
    if checkpoint_stats(args.model, stats) != stats:
        raise ValueError("checkpoint files changed during candidate execution")
    summarize(args, reference)


if __name__ == "__main__":
    main()
