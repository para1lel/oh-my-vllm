"""Project-owned adaptations of V2's GPU runner integration points."""

from dataclasses import replace

from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler


def clear_warmup_cache(runner):
    """Start serving with empty KV/state storage after dummy warmup writes.

    Partial attention pages can read masked padding lanes, where NaN times zero
    is still NaN. No requests exist yet, so clearing cache tensors is safe and
    adds no per-step work. Preserve their addresses for captured CUDA graphs.
    """
    for entry in runner.kv_caches:
        tensors = entry if isinstance(entry, (list, tuple)) else [entry]
        for tensor in tensors:
            tensor.zero_()


class RustDraftTokensHandler(DraftTokensHandler):
    """Return actual drafts to Rust even for unconstrained requests.

    V2 normally returns -1 placeholders for those requests because its scheduler
    only needs draft counts. Our unsigned wire protocol and Rust token history
    require the real IDs. Reuse V2's asynchronous copy/event implementation;
    the copied batch flag affects only draft transfer, never grammar or sampling.
    """

    def set_draft_tokens(self, input_batch, draft_tokens):
        super().set_draft_tokens(
            replace(input_batch, has_structured_output_reqs=True), draft_tokens
        )
