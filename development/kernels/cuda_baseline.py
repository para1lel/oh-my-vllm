"""Complete repository CUDA chains used only for paired tuning diagnostics."""

from importlib import import_module

import torch


def operator(module, name):
    if module == "fp8" and name in ("gated_norm_linear", "add_norm_linear"):
        from oh_my_vllm.kernels import fp8, normalization

        if name == "gated_norm_linear":

            def gated(x, gamma, gate, weight, scale):
                normalized = normalization.rms_norm(x, gamma, gate=gate)
                return fp8.linear(normalized.flatten(1), weight, scale)

            return gated

        def residual(x, residual, gamma, weight, scale):
            summed, normalized = normalization.add_rms_norm(x, residual, gamma)
            return summed, fp8.linear(normalized, weight, scale)

        return residual
    if (module, name) == ("gdn", "prefill"):
        from oh_my_vllm.kernels.cuda_backend import compiled
        from oh_my_vllm.kernels.gdn import normalize_qk

        def recurrent(q, k, v, decay, beta, initial, starts):
            q, k = normalize_qk(q, k)
            output, final = torch.empty_like(v), torch.empty_like(initial)
            compiled().gdn_prefill(q, k, v, decay, beta, initial, starts, output, final)
            return output, final

        return recurrent
    return getattr(import_module(f"oh_my_vllm.kernels.{module}"), name)
