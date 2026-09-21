#!/data0/shared/dongwu.chen/conda-envs/oh-my-vllm/bin/python
"""Actual independent-model operator probes; opt-in eager diagnostics only."""

import json
import math
import os
import runpy
import sys

import torch
from oh_my_vllm.kernels import gdn
from oh_my_vllm.kernels.attention import PagedAttention
from oh_my_vllm.kernels.mtp_attention import MTPAttention
from oh_my_vllm.worker.model_runner import OhMyVllmWorker

# Small FP64 matrix-vector references are faster without large thread pools.
torch.set_num_threads(1)


def cpu(tensor):
    return tensor.detach().cpu().double().clone()


coverage = set()


def check(name, actual, expected, atol=0.03, rtol=0.03):
    coverage.add(name)
    actual = cpu(actual).reshape(expected.shape)
    error = (actual - expected).abs().max().item()
    print(
        json.dumps(
            {
                "probe": name,
                "max_abs_error": error,
                "atol": atol,
                "rtol": rtol,
                "actual_nonfinite": int((~actual.isfinite()).sum()),
                "reference_nonfinite": int((~expected.isfinite()).sum()),
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    if name.endswith("_state"):
        # A recurrent state contains many near-zero entries; measure global
        # error as well as its worst coordinate relative to the state scale.
        rms = expected.square().mean().sqrt().item()
        nrmse = (actual - expected).square().mean().sqrt().item() / max(rms, 1e-12)
        relative_max = error / max(expected.abs().max().item(), 1e-12)
        print(
            json.dumps(
                {
                    "probe_state_metrics": name,
                    "nrmse": nrmse,
                    "relative_max": relative_max,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        assert nrmse <= 0.01 and relative_max <= 0.02
    else:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


def recurrence(q, k, v, g, beta, state, normalize):
    q, k, v, g, beta, state = map(cpu, (q, k, v, g, beta, state))
    if normalize:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    repeats = v.shape[-2] // q.shape[-2]
    q = q.repeat_interleave(repeats, dim=-2) / math.sqrt(q.shape[-1])
    k = k.repeat_interleave(repeats, dim=-2)
    output = []
    state = state[0]
    for index in range(q.shape[1]):
        state = state * g[0, index].exp()[:, None, None]
        residual = v[0, index] - torch.einsum("hvk,hk->hv", state, k[0, index])
        state = state + torch.einsum(
            "hv,hk->hvk", residual * beta[0, index, :, None], k[0, index]
        )
        output.append(torch.einsum("hvk,hk->hv", state, q[0, index]))
    return torch.stack(output)[None], state[None]


original_execute = OhMyVllmWorker.execute_model
seen = set()
selected_cache = None


def execute(self, output):
    seen.clear()
    return original_execute(self, output)


original_plan = PagedAttention.plan
original_attention = PagedAttention.__call__
original_mtp_attention = MTPAttention.__call__


def plan(self, starts, page_starts, pages, last, *args):
    self.probe_metadata = (
        starts.tolist(),
        page_starts.tolist(),
        pages.tolist(),
        last.tolist(),
    )
    return original_plan(self, starts, page_starts, pages, last, *args)


def attention_reference(query, cache, tables, lengths, first=0):
    q = cpu(query)
    results = []
    for row, (table, end) in enumerate(zip(tables, lengths, strict=True)):
        positions = torch.arange(first, end, device=cache.device)
        pages = torch.tensor(table, device=cache.device)[positions // 784]
        k = cpu(cache[pages, 0, positions % 784]).repeat_interleave(6, 1)
        v = cpu(cache[pages, 1, positions % 784]).repeat_interleave(6, 1)
        scores = torch.einsum("hd,khd->hk", q[row], k) / 16
        results.append(torch.einsum("hk,khd->hd", scores.softmax(-1), v))
    return torch.stack(results)


def attention(self, query, cache):
    global selected_cache
    result = original_attention(self, query, cache)
    if selected_cache is None:
        selected_cache = cache.data_ptr()
    if cache.data_ptr() != selected_cache:
        return result
    starts, page_starts, pages, last = self.probe_metadata
    tables, lengths = [], []
    for seq, end in enumerate(starts[1:]):
        table = pages[page_starts[seq] : page_starts[seq + 1]]
        length = (len(table) - 1) * 784 + last[seq]
        for position in range(length - (end - starts[seq]), length):
            tables.append(table)
            lengths.append(position + 1)
    name = "gqa_prefill" if len(query) > 5 else "gqa_decode"
    check(name, result, attention_reference(query, cache, tables, lengths))
    return result


def mtp_attention(self, query, cache):
    result = original_mtp_attention(self, query, cache)
    if self.decode_mode and "mtp_gqa_decode" not in seen:
        seen.add("mtp_gqa_decode")
        check(
            "mtp_gqa_decode",
            result,
            attention_reference(
                query, cache, self.tables.tolist(), self.lengths.tolist(), first=1
            ),
        )
    return result


original_prefill, original_recurrent = gdn.prefill, gdn.recurrent


def prefill(q, k, v, g, beta, initial, starts):
    if "prefill" in seen:
        return original_prefill(q, k, v, g, beta, initial, starts)
    seen.add("prefill")
    assert initial.shape[0] == 1, "probe requires a single request"
    ref, state = recurrence(
        q[None], k[None], v[None], g[None], beta[None], initial, True
    )
    result = original_prefill(q, k, v, g, beta, initial, starts)
    check("gdn_prefill", result[0], ref[0])
    check("gdn_prefill_state", result[1], state)
    return result


def recurrent(q, k, v, g, beta, pool, starts, reads, writes):
    if "recurrent" in seen:
        return original_recurrent(q, k, v, g, beta, pool, starts, reads, writes)
    seen.add("recurrent")
    assert reads.numel() == 1, "probe requires a single request"
    initial = pool[reads.long()].clone()
    ref, _ = recurrence(q[None], k[None], v[None], g[None], beta[None], initial, True)
    states = [
        recurrence(
            q[None, :i],
            k[None, :i],
            v[None, :i],
            g[None, :i],
            beta[None, :i],
            initial,
            True,
        )[1]
        for i in range(1, len(q) + 1)
    ]
    result = original_recurrent(q, k, v, g, beta, pool, starts, reads, writes)
    name = "gdn_mtp" if len(q) > 1 else "gdn_decode"
    check(name, result, ref[0])
    for index, target in enumerate(writes.tolist()):
        if target >= 0:
            check(name + "_state", pool[target], states[index][0])
    return result


if os.environ.get("OH_MY_VLLM_ENFORCE_EAGER") != "1":
    raise RuntimeError("FP64 probe requires OH_MY_VLLM_ENFORCE_EAGER=1")
OhMyVllmWorker.execute_model = execute
PagedAttention.plan = plan
PagedAttention.__call__ = attention
MTPAttention.__call__ = mtp_attention
gdn.prefill = prefill
gdn.recurrent = recurrent
sys.argv = [sys.argv[2], *sys.argv[3:]]
runpy.run_module("oh_my_vllm.worker.zmq_bridge", run_name="__main__")
required = {
    "gqa_prefill",
    "gqa_decode",
    "gdn_prefill",
    "gdn_prefill_state",
    "gdn_decode",
    "gdn_decode_state",
}
if os.environ.get("OH_MY_VLLM_PROBE_MTP") == "1":
    required -= {"gdn_decode", "gdn_decode_state"}
    required |= {"gdn_mtp", "gdn_mtp_state", "mtp_gqa_decode"}
if not required <= coverage:
    raise RuntimeError(f"Actual-path probe coverage missing: {required - coverage}")
print(
    json.dumps({"probe_coverage_pass": sorted(coverage)}), file=sys.stderr, flush=True
)
