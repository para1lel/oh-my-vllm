from types import SimpleNamespace
from unittest.mock import patch

import torch
from oh_my_vllm.models.dspark import DSparkConfig
from oh_my_vllm.worker.batch_plan import PlannedRequest
from oh_my_vllm.worker.dspark import DSpark
from oh_my_vllm.worker.protocol import RequestOutput, ScheduledRequest


class TinyDraft:
    def __init__(self):
        self.config = DSparkConfig(
            hidden_size=8,
            intermediate_size=12,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=11,
            markov_rank=2,
            mask_token_id=10,
            target_layer_ids=(0, 1),
        )
        self.injections = []
        self.calls = []
        self.confidence = [0.9, 0.8, 0.1, 0.95, 0.95, 0.95, 0.95]

    def inject(self, features, positions, slots, caches):
        self.injections.append((features.clone(), positions.clone(), slots.clone()))

    def greedy(self, tokens, positions, tables, lengths, caches):
        self.calls.append(
            (tokens.clone(), positions.clone(), tables.clone(), lengths.clone())
        )
        candidates = (tokens[:, :1] + torch.arange(1, 8)[None, :]) % 11
        confidence = torch.tensor(self.confidence).expand(len(tokens), -1)
        return candidates, confidence, torch.ones_like(candidates, dtype=torch.bool)

    def backbone(self, tokens, positions, tables, lengths, caches):
        self.calls.append(
            (tokens.clone(), positions.clone(), tables.clone(), lengths.clone())
        )
        hidden = torch.arange(7).view(1, 7, 1).expand(len(tokens), 7, 8).bfloat16()
        base = torch.linspace(-1, 1, 11).view(1, 1, 11).expand(len(tokens), 7, 11)
        return hidden, base

    def step(self, hidden, base, previous):
        position = int(hidden[0, 0])
        logits = base.clone()
        logits[0, (int(previous[0]) + 1) % 11] += 3
        return logits, torch.tensor([self.confidence[position]])


def make_worker(*, threshold=0, max_tokens=4096):
    model = TinyDraft()
    target = SimpleNamespace(embedding=torch.zeros(11, 8, dtype=torch.bfloat16))
    with patch("oh_my_vllm.worker.dspark.DSparkModel", return_value=model):
        worker = DSpark(target, ".", 8, max_tokens, confidence_threshold=threshold)
    return worker, model


def proposal(
    worker,
    *,
    start=0,
    count=2,
    scheduled=None,
    pages=None,
    rid=1,
    output=True,
    finished=False,
    samplers=None,
    serving=None,
    history=None,
):
    history = history or [position % 11 for position in range(start + count + 1)]
    tokens = history[start : start + count] if scheduled is None else scheduled
    request = ScheduledRequest(rid, tokens, start, pages or [1], [])
    plan = PlannedRequest(request, 0, [], [], [], False)
    features = torch.arange(len(tokens) * 16).view(-1, 16).bfloat16()
    result = RequestOutput(
        rid,
        [history[start + count]] if output else [],
        finish_reason="length" if finished else None,
    )
    return worker.propose(
        [plan],
        [0],
        [count],
        features,
        {rid: history},
        [result],
        samplers=samplers,
        serving=serving,
    )
