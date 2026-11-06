import datetime
import hashlib
import json
import os
from pathlib import Path

from .deepseek_dep4_evidence import (
    _write_json,
    checker_identity,
    request_owners,
    tensor_difference,
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

    if os.environ.get("OH_MY_VLLM_GPU_COUNT") != "4":
        raise RuntimeError("this check requires four GPUs from the repository wrapper")
    runner = Dep4Rank(args, proof, torch)
    try:
        with torch.inference_mode(), torch.device("cuda", proof.rank):
            runner.load()
            runner.execute()
            runner.save()
    except BaseException as error:
        try:
            proof.fail(error)
        finally:
            # Failed collectives cannot safely enter process-group cleanup.
            os._exit(1)
    finally:
        runner.close()
        dist.destroy_process_group()


class Dep4Rank:
    def __init__(self, args, proof, torch):
        self.args = args
        self.proof = proof
        self.torch = torch
        self.rank = self.local_rank = self.proof.rank
        self.loader = self.cache = self.network = None
        self.initial = {}
        self.arrays, self.chains, self.reports = {}, {}, []
        self.baseline, self.baseline_chains = initialize_rank(self.args, self.proof)
        self.strategy = "TEP4" if self.args.mode == "tep4" else "DEP4"

    def load(self):
        from oh_my_vllm.models.deepseek_v41 import model as candidate
        from oh_my_vllm.models.deepseek_v41.cache import DeepSeekCache
        from oh_my_vllm.models.deepseek_v41.dep4 import Dep4Executor
        from oh_my_vllm.models.deepseek_v41.loader import RankLoader
        from tokenizers import Tokenizer

        self.candidate = candidate
        self.torch.set_default_dtype(self.torch.bfloat16)
        self.torch.set_num_threads(1)
        self.config = json.loads(
            (self.args.model / "inference/config.json").read_text()
        )
        self.config.update(
            max_batch_size=1,
            max_seq_len=self.args.context,
            temperature=0,
            parallel_strategy=self.strategy,
            lazy_cache=False,
        )
        print(
            json.dumps(dict(rank=self.rank, phase="construct", mode=self.args.mode)),
            flush=True,
        )
        self.network = self.candidate.Transformer(
            self.candidate.ModelArgs(**self.config),
            Tokenizer.from_file(str(self.args.model / "tokenizer.json")),
        ).eval()
        self.loader = RankLoader(
            self.args.model,
            json.loads((self.args.model / "config.json").read_text()),
            self.rank,
            strategy=self.strategy,
        )
        self.loader.load_into(self.network)
        self.blocks = (self.args.context + 127) // 128
        self.cache = DeepSeekCache(
            self.network,
            max_seq_len=self.args.context,
            fa_capacity=self.blocks + 1,
            mamba_capacity=self.blocks + 1,
        )
        self.initial = {
            name: value.clone() for name, value in self.network.named_buffers()
        }
        self.executor = (
            Dep4Executor(self.network, padded=self.args.mode in ("fixed", "shadow"))
            if self.strategy == "DEP4"
            else None
        )
        print(
            json.dumps(dict(rank=self.rank, phase="loaded", mode=self.args.mode)),
            flush=True,
        )

    def check(self, label, value):
        value = value.detach().contiguous().cpu()
        self.arrays[label] = value
        difference = (
            tensor_difference(self.baseline[label], value, self.torch)
            if self.baseline is not None
            else None
        )
        record = dict(label=label, passed=difference is None, difference=difference)
        self.reports.append(record)
        (self.args.output / f"rank{self.rank}-progress.json").write_text(
            json.dumps(self.reports, indent=2) + "\n"
        )
        print(json.dumps(dict(rank=self.rank, **record)), flush=True)

    def reset(self, row):
        self.cache.reset()
        for name, value in self.network.named_buffers():
            value.copy_(self.initial[name])
        from oh_my_vllm.models.deepseek_v41 import model_context

        model_context.shared_attn = model_context.SharedAttentionRuntime()
        self.cache.activate_row(row, 0, end_pos=self.args.context)

    def target(self, ids, position, label=None, *, full=False):
        result = self.network(ids, position, full_logits=full)
        if label:
            self.check(label + ".logits", result[1])
            self.check(label + ".ids", result[1].argmax(-1))
            self.check(label + ".hidden", result[2][:, -1:])
        return result

    def continuation(self, request, prefix, tokens, row, length):
        self.reset(row)
        self.cache.begin_step(
            row, 0, start_pos=0, end_pos=length + 2, minimum_end=length + 2
        )
        result = self.target(tokens[:, :length], 0, prefix + ".prefill")
        if self.args.prefill_only:
            return
        current = result[0].reshape(1, 1)
        for offset in range(2):
            result = self.target(current, length + offset, prefix + f".decode{offset}")
            current = result[0].reshape(1, 1)
        endpoint = length + 2
        self.cache.commit(request + 1, row, 0, accepted_end=endpoint, include_mtp=False)
        continuous = self.target(current, endpoint, prefix + ".continuous")
        for name, value in self.network.named_buffers():
            value.copy_(self.initial[name])
        from oh_my_vllm.models.deepseek_v41 import model_context

        model_context.shared_attn = model_context.SharedAttentionRuntime()
        self.cache.restore(
            request + 1, 0, row, start_pos=endpoint, use_prefix_state=False
        )
        restored = self.target(current, endpoint, prefix + ".restart")
        self.torch.testing.assert_close(restored[1], continuous[1], rtol=0, atol=0)

    def greedy_chain(self, prefix, tokens, row):
        if self.baseline_chains is None:
            self.reset(row)
            self.target(tokens[:, :125], 0)
            result = self.target(tokens[:, 125:126], 125)
            chain = [int(result[0].item())]
            for position in range(126, 131):
                result = self.target(
                    self.torch.tensor([[chain[-1]]], dtype=self.torch.long), position
                )
                chain.append(int(result[0].item()))
        else:
            chain = self.baseline_chains[prefix]
        self.chains[prefix] = chain
        return chain

    def verification(self, request, prefix, tokens, row, chain):
        from oh_my_vllm.models.deepseek_v41.speculation import verify_greedy_drafts

        for name, values in (
            ("accept5", chain),
            ("reject0", [chain[0], (chain[1] + 1) % 129280, *chain[2:]]),
        ):
            self.reset(row)
            _, _, hidden = self.target(tokens[:, :125], 0)
            self.network.forward_spec(tokens[:, 125], hidden, 0)
            _, _, hidden = self.target(tokens[:, 125:126], 125)
            inputs = self.torch.tensor([values], dtype=self.torch.long)
            proposal = self.network.forward_spec(inputs[:, 0], hidden, 125)
            for suffix, value in zip(
                ("proposal_ids", "proposal_logits", "proposal_confidence"),
                proposal,
                strict=True,
            ):
                self.check(prefix + "." + name + "." + suffix, value)
            self.cache.commit(request + 1, row, 0, accepted_end=126, include_mtp=True)
            self.cache.begin_step(row, 0, start_pos=126, end_pos=132, minimum_end=127)
            _, logits, hidden = self.target(
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
            self.check(
                prefix + "." + name + ".emitted",
                self.torch.tensor(verification.token_ids, dtype=self.torch.long),
            )
            accepted_end = 126 + desired + 1
            self.network.forward_spec_seed(hidden[:, : desired + 1], 126)
            self.cache.commit(
                request + 1,
                row,
                0,
                start_pos=126,
                accepted_end=accepted_end,
                include_mtp=True,
            )
            self.cache.restore(
                request + 1,
                0,
                row,
                start_pos=accepted_end,
                use_prefix_state=False,
            )
            correction = self.torch.tensor(
                [[verification.token_ids[-1]]], dtype=self.torch.long
            )
            self.target(
                correction,
                accepted_end,
                prefix + "." + name + ".restored_probe",
            )

    def workload(self, request):
        prefix = f"request{request}"
        length = self.args.lengths[request]
        tokens = self.torch.tensor(
            [[900 + request * 137 + (i * 17) % 100 for i in range(max(132, length))]],
            dtype=self.torch.long,
        )
        row = dict(
            request_id=request + 1,
            fa_block_table=list(range(1, self.blocks + 1)),
            mamba_block_table=list(range(1, self.blocks + 1)),
        )
        self.continuation(request, prefix, tokens, row, length)
        if self.args.prefill_only:
            return
        chain = self.greedy_chain(prefix, tokens, row)
        self.verification(request, prefix, tokens, row, chain)

    def identity(self):
        self.owners = request_owners(self.args.lengths)
        self.requests = (
            self.owners
            if self.strategy == "TEP4"
            else ([self.rank] if self.rank in self.owners else [])
        )

        from oh_my_vllm.models.deepseek_v41.trace import runtime_identity

        identity = runtime_identity(
            self.rank,
            strategy=self.strategy.lower(),
            batch_decode=False,
            decode_graph=False,
        )
        gpu_uuid = str(self.torch.cuda.get_device_properties(self.local_rank).uuid)
        identity.update(
            gpu_uuid=gpu_uuid if gpu_uuid.startswith("GPU-") else "GPU-" + gpu_uuid,
            strategy=self.strategy.lower(),
            mode=self.args.mode,
            execution="eager",
            batch_decode_enabled=False,
            owner_ranks=self.args.owner_ranks,
            scheduled_request_ids=[request + 1 for request in self.requests],
            configuration=self.config,
            checkpoint_index=self.loader.index_sha256,
            tokenizer_sha256=hashlib.sha256(
                (self.args.model / "tokenizer.json").read_bytes()
            ).hexdigest(),
            checker_modules=checker_identity(),
            checker_sha256=hashlib.sha256(
                Path(__file__).with_name("check_deepseek_dep4.py").read_bytes()
            ).hexdigest(),
            comparison_scope="same-source parallel-strategy equivalence",
            frozen_reference_verified=False,
        )
        _write_json(self.args.output / f"identity-rank{self.rank}.json", identity)
        return identity

    def execute(self):
        identity = self.identity()
        completed_requests = []

        def work():
            for request in self.requests:
                self.workload(request)
                completed_requests.append(request + 1)

        if self.executor is None:
            work()
        else:
            self.executor.execute(work)
        if completed_requests != identity["scheduled_request_ids"]:
            raise AssertionError("scheduled requests differ from executed requests")
        identity["executed_request_ids"] = completed_requests
        _write_json(self.args.output / f"identity-rank{self.rank}.json", identity)

    def save(self):
        import torch.distributed as dist
        from safetensors.torch import load_file, save_file

        save_file(
            self.arrays,
            str(self.args.output / f"endpoints-rank{self.rank}.safetensors"),
        )
        gathered = [None] * 4
        dist.all_gather_object(
            gathered,
            dict(
                rank=self.rank,
                chains=self.chains,
                passed=all(row["passed"] for row in self.reports),
            ),
        )
        if self.rank == 0:
            merged_chains = {
                key: value
                for record in gathered
                for key, value in record["chains"].items()
            }
            (self.args.output / "chains.json").write_text(
                json.dumps(merged_chains, indent=2) + "\n"
            )
            combined = {}
            output_ranks = [0] if self.strategy == "TEP4" else self.owners
            for owner in output_ranks:
                combined.update(
                    load_file(
                        str(self.args.output / f"endpoints-rank{owner}.safetensors")
                    )
                )
            save_file(combined, str(self.args.output / "endpoints.safetensors"))
        if not all(record["passed"] for record in gathered):
            raise AssertionError("complete endpoint comparison failed")
        self.proof.require_fixed()
        dist.barrier()

    def close(self):
        if self.cache is not None:
            self.cache.close()
        self.initial.clear()
        self.network = None
        if self.loader is not None:
            self.loader.close()
        self.torch.cuda.synchronize(self.local_rank)
