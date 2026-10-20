import numpy as np
import torch

from ..bindings import EMBEDDING
from . import fa, ffn, output
from .common import rms
from .decisions import acceptance_margin, cdf_margin
from .random import accept_uniform


def input_projection(hidden, tokens, weights):
    embedded = weights.tensor(EMBEDDING, rows=tokens.cpu().numpy())
    packed = torch.cat(
        (
            rms(embedded, weights.tensor("mtp.pre_fc_norm_embedding.weight")),
            rms(hidden, weights.tensor("mtp.pre_fc_norm_hidden.weight")),
        ),
        dim=-1,
    )
    return weights.linear(packed, "mtp.fc")


def context(case, inputs, state, weights):
    rows = inputs["hidden_rows"].long()
    hidden = torch.where(
        rows[:, None] >= 0,
        inputs["hidden"][rows.clamp_min(0)],
        inputs["boundary_hidden"][inputs["boundary_rows"].long()],
    )
    projected = input_projection(hidden, inputs["tokens"], weights)
    zero = torch.zeros_like(projected)
    fa.compute(
        projected,
        zero,
        inputs,
        state["kv"],
        weights,
        "mtp.layers.0.",
        variant="kv_only",
        first_token=1,
    )
    endpoints = inputs["sequence_rows"][1:].long() - 1
    return {
        "value": projected[endpoints],
        "residual": zero[endpoints],
        **fa.written_cache(state, inputs, weights, "mtp.layers.0."),
    }


def draft_step(hidden, tokens, inputs, state, weights, *, first):
    value = hidden if first else input_projection(hidden, tokens, weights)
    value, residual = fa.compute(
        value,
        torch.zeros_like(value),
        inputs,
        state["kv"],
        weights,
        "mtp.layers.0.",
        variant="query_only" if first else "full",
        first_token=1,
    )
    value, residual = ffn.compute(value, residual, weights, "mtp.layers.0.")
    normalized, logits = output.project(value, residual, weights, "mtp.norm.weight")
    result = output.sample(
        logits, inputs["history"], inputs["mask"], inputs["parameters"], state["rng"]
    )
    return normalized, {"hidden": normalized, "logits": logits, **result}


def draft(case, inputs, state, weights):
    first = case.options["variant"] == "query_only"
    _, sampled = draft_step(
        inputs["hidden"], inputs["tokens"], inputs, state, weights, first=first
    )
    indices = (
        torch.empty(0, dtype=torch.int64, device=inputs["hidden"].device)
        if first
        else inputs["slots"].long()
    )
    return {
        **sampled,
        **fa.written_cache(state, inputs, weights, "mtp.layers.0.", indices=indices),
    }


def chain(case, inputs, state, weights):
    hidden, tokens = inputs["hidden"], inputs["tokens"]
    selected, probabilities, slots, hidden_rows, logits = [], [], [], [], []
    for stage in range(4):
        metadata = {
            name.removeprefix(f"stage{stage}."): value
            for name, value in inputs.items()
            if name.startswith(f"stage{stage}.")
        }
        metadata.update(
            {name: inputs[name] for name in ("history", "mask", "parameters")}
        )
        hidden, sampled = draft_step(
            hidden, tokens, metadata, state, weights, first=stage == 0
        )
        tokens = sampled["tokens"]
        selected.append(tokens)
        probabilities.append(sampled["probabilities"])
        hidden_rows.append(sampled["hidden"])
        logits.append(sampled["logits"])
        if stage:
            slots.append(metadata["slots"].long())
    indices = torch.stack(slots, dim=1).reshape(-1)
    return {
        "tokens": torch.stack(selected),
        "hidden": torch.stack(hidden_rows),
        "logits": torch.stack(logits),
        "probabilities": torch.stack(probabilities),
        "rng": state["rng"],
        **fa.written_cache(state, {}, weights, "mtp.layers.0.", indices=indices),
    }


def accept(case, inputs, state, weights):
    batch = case.shape["batch"]
    kept = np.full((batch, 5), -1, np.int64)
    counts, slots = np.zeros(batch, np.int32), np.zeros(batch, np.int32)
    counters = state["rng"].cpu().numpy().copy()
    greedy_target = inputs["target"].dtype == torch.int64
    for request, (seed, counter) in enumerate(counters.tolist()):
        accepted = attempts = 0
        drafts = inputs["drafts"][request]
        for stage, token in enumerate(drafts[:4].tolist()):
            if token < 0:
                break
            p = (
                float(inputs["target"][request, stage] == token)
                if greedy_target
                else float(inputs["target"][request, stage, token])
            )
            q = 1 if drafts[4] else float(inputs["proposal"][request, stage, token])
            threshold = accept_uniform(seed, counter + attempts * 2)
            acceptance_margin(p, q, threshold)
            attempts += 1
            if q == 0 or p / q < threshold:
                break
            kept[request, accepted] = token
            accepted += 1
        if greedy_target:
            token = int(inputs["target"][request, accepted])
        else:
            probability = inputs["target"][request, accepted].clone()
            if accepted < 4 and drafts[accepted] >= 0:
                if drafts[4]:
                    probability[int(drafts[accepted])] -= 1
                else:
                    probability -= inputs["proposal"][request, accepted]
                probability = probability.clamp_min(0)
            cumulative = probability.cumsum(dim=0)
            draw = accept_uniform(seed, counter + attempts * 2)
            cdf_margin(probability, draw)
            token = int(torch.searchsorted(cumulative, draw * cumulative[-1]))
        kept[request, accepted] = token
        counts[request] = accepted + 1
        slots[request] = int(inputs["private_slots"][request, accepted])
        counters[request, 1] += np.uint64(2 * (attempts + 1))
    return {"tokens": kept, "counts": counts, "candidate_slots": slots, "rng": counters}


def reference(case, inputs, state, weights):
    return {
        "mtp.context": context,
        "mtp.draft": draft,
        "mtp.chain": chain,
        "mtp.accept": accept,
    }[case.boundary](case, inputs, state, weights)
