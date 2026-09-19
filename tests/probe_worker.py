#!/data0/shared/dongwu.chen/conda-envs/vllm/bin/python
"""Opt-in actual-path FP64 probe; only for single-request eager smoke runs.

This wraps the actual selected backend calls without replacing their computation.
It synchronizes and copies small layer inputs to CPU, so never benchmark with it.
"""

import json
import math
import os
import runpy
import sys

import torch
from oh_my_vllm.worker.model_runner import OhMyVllmWorker
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn
from vllm.v1.attention.backends.flashinfer import FlashInferImpl

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
        json.dumps({"probe": name, "max_abs_error": error, "atol": atol, "rtol": rtol}),
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


original_fa = FlashInferImpl.forward
selected_fa = None
keys, values = [], []
request_start = 0
original_execute = OhMyVllmWorker.execute_model


def execute(self, output):
    global request_start
    if output.scheduled:
        assert len(output.scheduled) == 1, "probe requires a single request"
        request_start = output.scheduled[0].num_computed_tokens
    return original_execute(self, output)


def fa_forward(
    self, layer, query, key, value, kv_cache, attn_metadata, output, **kwargs
):
    global selected_fa, keys, values
    if attn_metadata is None:
        return original_fa(
            self, layer, query, key, value, kv_cache, attn_metadata, output, **kwargs
        )
    if selected_fa is None:
        selected_fa = self
    if self is not selected_fa:
        return original_fa(
            self, layer, query, key, value, kv_cache, attn_metadata, output, **kwargs
        )
    count = attn_metadata.num_actual_tokens
    q, k, v = map(cpu, (query[:count], key[:count], value[:count]))
    start = request_start
    if keys:
        keys = [torch.cat(keys)[:start]]
        values = [torch.cat(values)[:start]]
    keys.append(k)
    values.append(v)
    k, v = torch.cat(keys), torch.cat(values)
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    scores = torch.einsum("qhd,khd->hqk", q, k) * self.scale
    mask = torch.arange(k.shape[0])[None, :] > (start + torch.arange(count))[:, None]
    scores.masked_fill_(mask[None], -torch.inf)
    ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)
    result = original_fa(
        self, layer, query, key, value, kv_cache, attn_metadata, output, **kwargs
    )
    check(
        "gqa_prefill" if attn_metadata.num_prefills > 0 else "gqa_decode",
        output[:count],
        ref,
    )
    return result


original_prefill = gdn.fi_chunk_gated_delta_rule
last_prefill_context = None


def prefill(**kwargs):
    global last_prefill_context
    context = get_forward_context()
    if context.attn_metadata is None or context is last_prefill_context:
        return original_prefill(**kwargs)
    last_prefill_context = context
    ref, state = recurrence(
        *(kwargs[name] for name in ("q", "k", "v", "g", "beta", "initial_state")),
        kwargs.get("use_qk_l2norm_in_kernel", True),
    )
    result = original_prefill(**kwargs)
    check("gdn_prefill", result[0], ref)
    check("gdn_prefill_state", result[1], state)
    return result


original_decode = gdn.fused_recurrent_gated_delta_rule_packed_decode
last_decode_context = None


def decode(**kwargs):
    global last_decode_context
    context = get_forward_context()
    if context.attn_metadata is None or context is last_decode_context:
        return original_decode(**kwargs)
    last_decode_context = context
    state = kwargs["initial_state"][kwargs["ssm_state_indices"].long()]
    _, heads, value_dim, key_dim = state.shape
    mixed = kwargs["mixed_qkv"]
    qk_dim = (mixed.shape[-1] - heads * value_dim) // 2
    q, k, v = mixed.split([qk_dim, qk_dim, heads * value_dim], dim=-1)
    q = q.reshape(1, 1, -1, key_dim)
    k = k.reshape_as(q)
    v = v.reshape(1, 1, heads, value_dim)
    decay = -cpu(kwargs["A_log"]).exp() * torch.nn.functional.softplus(
        cpu(kwargs["a"]) + cpu(kwargs["dt_bias"])
    )
    beta = cpu(kwargs["b"]).sigmoid()
    ref, expected_state = recurrence(q, k, v, decay[None], beta[None], state, True)
    result = original_decode(**kwargs)
    check("gdn_decode", kwargs["out"], ref)
    check(
        "gdn_decode_state",
        kwargs["initial_state"][kwargs["ssm_state_indices"].long()],
        expected_state,
    )
    return result


original_mtp = gdn.ops.fused_gdn_decode_post_conv_mtp
last_mtp_context = None


def mtp(**kwargs):
    global last_mtp_context
    context = get_forward_context()
    if context.attn_metadata is None or context is last_mtp_context:
        return original_mtp(**kwargs)
    last_mtp_context = context
    indices = kwargs["state_indices"]
    assert indices.shape[0] == 1
    source = indices[0, int(kwargs["num_accepted_tokens"][0]) - 1]
    initial = kwargs["state"][source : source + 1].clone()
    _, heads, value_dim, key_dim = initial.shape
    count = int(kwargs["cu_seqlens"][1])
    mixed = kwargs["mixed_qkv"][:count]
    qk_dim = (mixed.shape[-1] - heads * value_dim) // 2
    q, k, v = mixed.split([qk_dim, qk_dim, heads * value_dim], dim=-1)
    q = q.reshape(1, count, -1, key_dim)
    k = k.reshape_as(q)
    v = v.reshape(1, count, heads, value_dim)
    decay = -cpu(kwargs["A_log"]).exp() * torch.nn.functional.softplus(
        cpu(kwargs["a"][:count]) + cpu(kwargs["dt_bias"])
    )
    beta = cpu(kwargs["b"][:count]).sigmoid()
    ref, _ = recurrence(q, k, v, decay[None], beta[None], initial, True)
    states = []
    for end in range(1, count + 1):
        _, state = recurrence(
            q[:, :end],
            k[:, :end],
            v[:, :end],
            decay[None, :end],
            beta[None, :end],
            initial,
            True,
        )
        states.append(state)
    # The fused kernel explicitly rounds recurrent output to BF16 before RMSNorm.
    ref = ref[0].bfloat16().double()
    ref = ref / (ref.square().mean(-1, keepdim=True) + kwargs["norm_eps"]).sqrt()
    ref = (
        ref
        * cpu(kwargs["norm_weight"])
        * torch.nn.functional.silu(cpu(kwargs["output_gate"][:count]))
    )
    result = original_mtp(**kwargs)
    check("gdn_mtp_decode", kwargs["out"][:count], ref)
    for index, state in enumerate(states):
        check("gdn_mtp_state", kwargs["state"][indices[0, index]], state)
    return result


OhMyVllmWorker.execute_model = execute
gdn.ops.fused_gdn_decode_post_conv_mtp = mtp

FlashInferImpl.forward = fa_forward
gdn.fi_chunk_gated_delta_rule = prefill
gdn.fused_recurrent_gated_delta_rule_packed_decode = decode
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
    required |= {"gdn_mtp_decode", "gdn_mtp_state"}
if not required <= coverage:
    raise RuntimeError(f"Actual-path probe coverage missing: {required - coverage}")
print(
    json.dumps({"probe_coverage_pass": sorted(coverage)}), file=sys.stderr, flush=True
)
