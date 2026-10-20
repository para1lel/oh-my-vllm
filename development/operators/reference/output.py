import numpy as np
import torch

from .common import rms
from .decisions import cdf_margin
from .random import uniform


def distribution(logits, history, mask, parameters):
    rows, vocabulary = logits.shape
    tokens = torch.arange(vocabulary, device=logits.device)
    permitted = ((mask[:, tokens // 8].long() >> (tokens % 8)) & 1).bool()
    generated = history[:, 1]
    present = history.sum(dim=1) > 0
    repetition = parameters[:, 6:7]
    repeated = torch.where(logits > 0, logits / repetition, logits * repetition)
    scores = torch.where(present, repeated, logits)
    scores = (
        scores - generated * parameters[:, 4:5] - (generated > 0) * parameters[:, 5:6]
    )
    scores = scores / torch.where(parameters[:, :1] > 0, parameters[:, :1], 1)
    scores = scores.masked_fill(~permitted, -torch.inf)
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    sorted_scores = torch.gather(scores, 1, order)
    probability = torch.zeros_like(logits)
    for row in range(rows):
        if parameters[row, 0] == 0:
            probability[row, order[row, 0]] = 1
            continue
        topk = int(parameters[row, 1])
        cutoff = sorted_scores[row, topk - 1] if topk > 0 else -torch.inf
        mass = (sorted_scores[row] - sorted_scores[row, 0]).exp()
        mass = torch.where(
            (sorted_scores[row] >= cutoff) & (mass >= parameters[row, 3]), mass, 0
        )
        before = mass.cumsum(dim=0) - mass
        kept = torch.where(before < parameters[row, 2] * mass.sum(), mass, 0)
        probability[row, order[row]] = kept / kept.sum()
    return probability, order


def sample(logits, history, mask, parameters, rng):
    probability, order = distribution(logits, history, mask, parameters)
    counters = rng.cpu().numpy().copy()
    selected, edges = [], []
    for row, (seed, counter) in enumerate(counters.tolist()):
        if parameters[row, 0] == 0:
            token = int(order[row, 0])
            edges.append([0.0, 1.0])
        else:
            cumulative = probability[row, order[row]].cumsum(dim=0)
            draw = uniform(seed, counter)
            cdf_margin(probability[row, order[row]], draw)
            offset = int(torch.searchsorted(cumulative, draw))
            edges.append(
                [
                    float(cumulative[offset - 1]) if offset else 0.0,
                    float(cumulative[offset]),
                ]
            )
            token = int(order[row, min(offset, logits.shape[1] - 1)])
            counters[row, 1] += np.uint64(1)
        selected.append(token)
    tokens = torch.tensor(selected, dtype=torch.int64, device=logits.device)
    selected_probability = probability[
        torch.arange(len(tokens), device=logits.device), tokens
    ]
    return {
        "probabilities": probability,
        "tokens": tokens,
        "selected_probability": selected_probability,
        "selection_cdf": np.asarray(edges, dtype=np.float64),
        "rng": counters,
    }


def project(value, residual, weights, norm="model.language_model.norm.weight"):
    hidden = rms(value + residual, weights.tensor(norm))
    return hidden, weights.linear(hidden, "lm_head")


def reference(case, inputs, state, weights):
    hidden = rms(
        inputs["value"] + inputs["residual"],
        weights.tensor("model.language_model.norm.weight"),
    )
    if case.boundary == "output.hidden":
        return {"hidden": hidden, "hidden_cache": hidden[inputs["slots"] >= 0]}
    logits = weights.linear(hidden, "lm_head")
    args = (logits, inputs["history"], inputs["mask"], inputs["parameters"])
    if case.boundary == "output.distribution":
        return {"logits": logits, "probabilities": distribution(*args)[0]}
    return {"logits": logits, **sample(*args, state["rng"])}
