import json

from .check_deepseek_batch import cpu, differences, fingerprint, row
from .check_deepseek_dep4 import _write_json
from .deepseek_reference_contract import (
    acceptance_record,
    compare_tensor,
    selected_schema,
    validate_tensors,
)


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
        from oh_my_vllm.models.deepseek_v41 import model_context

        self.worker.cache.reset()
        self.worker.cache.activate_row(self.row, 0, end_pos=256)
        model_context.shared_attn = model_context.SharedAttentionRuntime()

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
