from types import SimpleNamespace
from unittest.mock import Mock

import torch
from oh_my_vllm.models.deepseek_v41.inference import Worker
from oh_my_vllm.models.deepseek_v41.quantization import fp4_act_quant
from oh_my_vllm.models.deepseek_v41.request import Request
from oh_my_vllm.models.deepseek_v41.worker_limits import DEFAULT_PREFILL_ROWS

from tests.test_deepseek_cache import FakeModel, cache_for

"""CPU checks for adapter control and cache state.

GPU checks cover real MTP history, batched model computation, and Graph execution.
"""


class _PositionModel(FakeModel):
    def __init__(self, *, value_offset=1, predictions=None, batch_size=2, context=32):
        super().__init__(batch_size=batch_size, context=context)
        self.value_offset, self.predictions = value_offset, predictions

    def __call__(self, inputs, *, start_pos, cache_slots_cpu, **kwargs):
        self._state_export.bind(cache_slots_cpu)
        attention = self.layers[0].attn
        positions = torch.arange(start_pos, start_pos + inputs.size(1))
        offsets = torch.tensor(cache_slots_cpu).view(-1, 1, 1) * 100
        values = (
            (positions + self.value_offset).float().view(1, -1, 1) + offsets
        ).expand(-1, -1, 3)
        self._state_export.window(attention.window_kv_cache, values, start_pos)
        self._state_export.tail(
            attention.compressor.kv_state,
            attention.compressor.score_state,
            values,
            values + 100,
            start_pos,
        )
        for index, slot in enumerate(cache_slots_cpu):
            attention.window_kv_cache[slot, positions[-4:].remainder(4)] = values[
                index, -4:
            ]
            self.engram_hash.cache[slot, positions] = positions
        logits = (
            None
            if self.predictions is None
            else torch.nn.functional.one_hot(
                torch.tensor([self.predictions]), 16
            ).float()
        )
        return None, logits, values

    def forward_spec_seed(self, hidden, *, start_pos, cache_slots):
        self._state_export.bind(cache_slots.tolist())
        window = self.mtp[0].attn.window_kv_cache
        self._state_export.window(window, hidden, start_pos)
        positions = torch.arange(start_pos, start_pos + hidden.size(1))
        for index, slot in enumerate(cache_slots.tolist()):
            window[slot, positions[-4:].remainder(4)] = hidden[index, -4:]

    def forward_spec(self, input_ids, hidden, *, start_pos, cache_slots):
        self.forward_spec_seed(hidden, start_pos=start_pos, cache_slots=cache_slots)
        return None


class _ChainModel(_PositionModel):
    """A CPU control fixture whose proposal does not read the MTP window."""

    def __init__(self, *, acceptance=5, batch_size=2):
        super().__init__(batch_size=batch_size, context=64)
        self.acceptance = acceptance
        self.targets, self.proposals = [], []

    @staticmethod
    def prediction(token, previous, position):
        return (token + 2 * previous + position) % 120 + 1

    def __call__(
        self,
        inputs,
        *,
        start_pos,
        cache_slots_cpu,
        full_logits=False,
        logit_start=0,
        routing_trace=None,
        **kwargs,
    ):
        hidden, predictions = [], []
        attention = self.layers[0].attn
        for row, slot in zip(inputs.tolist(), cache_slots_cpu, strict=True):
            self.targets.append(dict(start=start_pos, tokens=row, verify=full_logits))
            previous = (
                int(self.engram_hash.cache[slot, start_pos - 1]) if start_pos else 0
            )
            values, targets = [], []
            for position, token in enumerate(row, start_pos):
                values.append([position + 1, token, previous])
                targets.append(self.prediction(token, previous, position))
                self.engram_hash.cache[slot, position] = token
                previous = token
            values = torch.tensor([values], dtype=torch.float32)
            self._state_export.bind((slot,))
            self._state_export.window(attention.window_kv_cache, values, start_pos)
            kv, score = attention.compressor.kv_state, attention.compressor.score_state
            tail = torch.cat((kv[slot : slot + 1, : start_pos % 2], values), dim=1)
            self._state_export.tail(kv, score, tail, tail + 100, start_pos)
            positions = torch.arange(start_pos, start_pos + len(row))
            attention.window_kv_cache[slot, positions[-4:].remainder(4)] = values[
                0, -4:
            ]
            completed = values[:, positions.remainder(2) == 1]
            for history, width in (
                (attention.compress_kv_cache, 3),
                (attention.indexer.k_cache, 2),
            ):
                if completed.size(1):
                    features = torch.nn.functional.pad(
                        completed[:, :, :width], (0, 32 - width)
                    )
                    encoded, scales = fp4_act_quant(
                        features, history.group_size, scale_dtype=history.scale_dtype
                    )
                    history.write(torch.tensor([slot]), start_pos // 2, encoded, scales)
            kv[slot].zero_()
            score[slot].fill_(-torch.inf)
            if (start_pos + len(row)) % 2:
                kv[slot, 0] = values[0, -1]
                score[slot, 0] = values[0, -1] + 100
            hidden.append(values)
            predictions.append(targets)
        if routing_trace is not None:
            routing_trace.record(
                0, torch.arange(6, dtype=torch.int32).expand(inputs.numel(), -1)
            )
        logits = torch.nn.functional.one_hot(torch.tensor(predictions), 128).float()
        logits = logits[:, logit_start:] if full_logits else logits[:, -1]
        return None, logits, torch.cat(hidden)

    def forward_spec(self, input_ids, hidden, *, start_pos, cache_slots):
        if tuple(hidden.shape) != (1, 1, 3) or hidden[0, 0, 0] != start_pos + 1:
            raise ValueError("proposal did not receive the accepted boundary hidden")
        self.proposals.append(
            dict(start=start_pos, token=int(input_ids.item()), hidden=hidden.clone())
        )
        self.forward_spec_seed(hidden, start_pos=start_pos, cache_slots=cache_slots)
        if start_pos == 0:
            return None
        token, previous = int(input_ids.item()), int(hidden[0, 0, 1])
        drafts = []
        for position in range(start_pos + 1, start_pos + 6):
            predicted = self.prediction(token, previous, position)
            drafts.append(predicted)
            previous, token = token, predicted
        accepted = (
            self.acceptance
            if isinstance(self.acceptance, int)
            else self.acceptance[int(cache_slots.item())]
        )
        if accepted < 5:
            drafts[accepted] = (drafts[accepted] + 1) % 128
        return torch.tensor([[int(input_ids.item()), *drafts]]), None, None


def _worker():
    worker = Worker.__new__(Worker)
    worker.device = torch.device("cpu")
    worker.runtime_max_context = 64
    worker.dspark = False
    worker.args = SimpleNamespace(window_size=384)
    worker.rank = 0
    worker.worker_epoch = 7
    worker.last_step_id = 0
    worker.prefill_rows = DEFAULT_PREFILL_ROWS
    worker.decode_graph_enabled = False
    worker.decode_graph_limit = 32
    worker.decode_graphs = {}
    worker.decode_bindings = {}
    worker.decode_engram = {}
    worker.batch_decode_enabled = False
    worker.routing_trace_enabled = False
    worker.target_call = 0
    worker._target_execution = "eager"
    worker.cache = SimpleNamespace(
        exports=SimpleNamespace(steps={}), mtp_checkpoint_ends=lambda *args: []
    )
    worker._input_tokens = lambda rows: torch.tensor(rows, dtype=torch.long)
    return worker


def _chain_worker(*, prompts=((10, 11, 12),), acceptance=5, limit=24, **options):
    worker = _worker()
    worker.model = _ChainModel(acceptance=acceptance, batch_size=len(prompts) + 1)
    worker.device = torch.device("cpu")
    worker.dspark = True
    worker.args = SimpleNamespace(window_size=4)
    worker.slots = set(range(len(prompts) + 1))
    tokenizer = SimpleNamespace(
        valid_tokens=128,
        grammar=None,
        eos=set(),
        decode_bytes=lambda tokens: b"x" * len(tokens),
    )
    worker.requests = {
        key: Request(
            tokenizer, prompt, dict(max_tokens=limit, sampling={"seed": 0}, **options)
        )
        for key, prompt in enumerate(prompts, 1)
    }
    worker.trace = SimpleNamespace(record=Mock())
    worker.cache = cache_for(
        worker.model,
        block_tokens=4,
        max_seq_len=64,
    )
    return worker


def _chain_row(worker, tokens, *, key=1, start=None):
    request = worker.requests[key]
    blocks = list(range(16 * (key - 1) + 1, 16 * key + 1))
    return dict(
        request_id=key,
        token_ids=list(tokens),
        num_computed_tokens=request.state_position if start is None else start,
        fa_block_table=blocks,
        mamba_block_table=list(blocks),
    )


def _chain_state(worker, key=1):
    cache, request = worker.cache, worker.requests[key]
    slot, end = request.cache_slot, request.state_position
    values = [value[slot] for value in (*cache._window, *cache._mtp_window)]
    values += [
        value[slot, : end // ratio]
        for value, ratio in (*cache._compressed, *cache._index)
    ]
    values += [value[slot] for tail in cache._tails for value in tail]
    values += [value[slot, :end] for value in cache._engram]
    return [value.clone() for value in values]


def _chain_batch(worker, tokens, positions, slots, **options):
    """Supply batch result metadata through separate CPU model calls."""
    results = [
        worker.model(
            torch.tensor([row]),
            start_pos=start,
            cache_slots_cpu=(slot,),
            full_logits=len(row) > 1,
        )
        for row, start, slot in zip(tokens, positions, slots, strict=True)
    ]
    return SimpleNamespace(
        logits=torch.cat([result[1] for result in results]),
        hidden=torch.cat([result[2] for result in results]),
        execution="eager",
        routes=None,
        bindings=None,
        call=1,
    )


def _verification_worker(predictions):
    worker = _worker()
    worker.model = _PositionModel(value_offset=0, predictions=predictions)
    worker.device = torch.device("cpu")
    worker.dspark = True
    worker.args = SimpleNamespace(window_size=4)
    worker.slots = {1}
    worker.requests = {
        1: SimpleNamespace(
            cache_slot=0,
            history=[10] * 20,
            prompt_len=20,
            state_position=0,
            generated=0,
            limit=8,
            finish=None,
            thinking_count=0,
            append=lambda tokens: (tokens, "ok"),
        )
    }
    worker.trace = SimpleNamespace(record=lambda *args, **kwargs: None)
    worker.cache = cache_for(worker.model, block_tokens=4, max_seq_len=32)
    return worker, {
        "request_id": 1,
        "token_ids": [10] * 20 + [1, 2, 3, 4, 5],
        "num_computed_tokens": 0,
        "fa_block_table": [1, 2, 3, 4, 5, 6, 7],
        "mamba_block_table": [0, 0, 0, 3, 5, 6, 7],
    }


def _wire_message(worker, message):
    result = {
        "type": "execute",
        "worker_epoch": worker.worker_epoch,
        "step_id": worker.last_step_id + 1,
        "num_batched_tokens": sum(
            len(row["token_ids"]) for row in message["scheduled"]
        ),
        **message,
    }
    scheduled = []
    for row in message["scheduled"]:
        history = row.get("prefill_token_ids")
        if history is None:
            history = worker.requests[row["request_id"]].history
        start = row["num_computed_tokens"]
        end = start + len(row["token_ids"])
        drafts = max(0, end - len(history))
        sampling = {"kind": "none" if end < len(history) else "target"}
        if drafts:
            sampling = {"kind": "verify", "drafts": drafts}
        state = {
            "kind": "deepseek_resident_v1",
            "fa_block_table": row.get("fa_block_table", []),
            "mamba_block_table": row.get("mamba_block_table", []),
            "fa_block_copies": row.get("fa_block_copies", []),
            "prefix_state_slot": row.get("prefix_state_slot"),
            "cache_checkpoint": row.get("cache_checkpoint"),
        }
        converted = {
            "request_id": row["request_id"],
            "lease_id": 11,
            "purpose": "advance",
            "action": "forward",
            "sampling": sampling,
            "token_ids": row["token_ids"],
            "forward_start": start,
            "forward_end": end,
            "accepted_end": len(history),
            "endpoint": end,
            "model_state": state,
        }
        if row.get("prefill_token_ids") is not None:
            converted["prefill_token_ids"] = row["prefill_token_ids"]
        scheduled.append(converted)
    result["scheduled"] = scheduled
    return result


class DeepSeekSpeculationFixture:
    def _run_chain(self, mode, *, acceptance):
        worker = _chain_worker(acceptance=acceptance)
        request = worker.requests[1]
        if mode == "greedy":
            worker._next_proposals = lambda *_args: []
        output = worker._execute_row(_chain_row(worker, request.history))
        states = {request.state_position: _chain_state(worker)}
        rounds = []
        while not request.finish:
            drafts = output["new_draft_token_ids"]
            output = worker._execute_row(
                _chain_row(worker, [request.history[-1], *drafts])
            )
            rounds.append((len(drafts), output))
            self.assertEqual(request.state_position, len(request.history) - 1)
            states[request.state_position] = _chain_state(worker)
            if drafts:
                self.assertTrue(
                    torch.all(
                        worker.model.engram_hash.cache[
                            request.cache_slot, request.state_position :
                        ]
                        == 0
                    )
                )
        return worker, states, rounds
