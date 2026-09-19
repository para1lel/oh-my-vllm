"""oh_my_vllm.worker.model_runner — thin wrapper around vLLM's GPUWorker.

The Rust scheduler owns request queuing, KV block allocation, and token
budgeting.  This module owns the GPU: model loading, forward passes, and
sampling.  The two sides talk over ZMQ using msgpack-serialised messages
defined in protocol.py.

For Qwen3.5-27B-FP8 the model has two attention backends:
  - 16 full-attention layers → FlashAttentionBackend / tml_fa4
  - 48 GatedDeltaNet layers → GDNAttentionBackend (FlashInfer)

We do not touch the backend selection; vLLM's model registry handles it via
the Qwen3.5 config.  The only thing we adapt is the message envelope between
the Rust SchedulerOutput and vLLM's SchedulerOutput dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput


# ---------------------------------------------------------------------------
# Data types mirroring crates/scheduler/src/output.rs
# ---------------------------------------------------------------------------


@dataclass
class ScheduledRequest:
    """Rust scheduler's view of one request for one step."""

    request_id: int
    token_ids: list[int]
    num_computed_tokens: int
    fa_block_table: list[int]
    mamba_block_table: list[int]


@dataclass
class SchedulerOutput:
    """Top-level output from the Rust scheduler for one step."""

    scheduled: list[ScheduledRequest] = field(default_factory=list)
    finished_request_ids: list[int] = field(default_factory=list)
    preempted_request_ids: list[int] = field(default_factory=list)
    num_batched_tokens: int = 0


@dataclass
class RequestOutput:
    """Per-request result returned to the Rust scheduler after one step."""

    request_id: int
    next_token_id: int
    num_accepted_draft_tokens: int = 0
    new_draft_token_ids: list[int] = field(default_factory=list)


@dataclass
class WorkerOutput:
    """Batch result returned to the Rust scheduler."""

    outputs: list[RequestOutput]


# ---------------------------------------------------------------------------
# Adapter: Rust SchedulerOutput → vLLM SchedulerOutput
# ---------------------------------------------------------------------------


def _to_vllm_scheduler_output(
    rust_output: SchedulerOutput,
    sampling_params_map: dict,  # request_id (int) -> vllm SamplingParams
    prompt_token_ids_map: dict,  # request_id (int) -> list[int]
) -> VllmSchedulerOutput:
    """Convert the Rust scheduler output into a vLLM SchedulerOutput.

    vLLM identifies requests by str req_id; we use str(int) for round-trip
    compatibility with the Rust u64 request ids.
    """
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.sched.output import (
        NewRequestData,
        ResumedRequestData,
    )
    from vllm.v1.core.sched.output import (
        SchedulerOutput as VllmOut,
    )

    new_reqs: list[NewRequestData] = []
    resumed_reqs: list[ResumedRequestData] = []
    num_scheduled: dict[str, int] = {}

    for sr in rust_output.scheduled:
        req_id_str = str(sr.request_id)
        num_scheduled[req_id_str] = len(sr.token_ids)

        # block_ids is a tuple of per-group lists: (fa_blocks, mamba_blocks)
        block_ids = (sr.fa_block_table, sr.mamba_block_table)

        if sr.num_computed_tokens == 0 and sr.request_id in prompt_token_ids_map:
            # First scheduling of this request.
            sp = sampling_params_map.get(sr.request_id, SamplingParams())
            new_reqs.append(
                NewRequestData(
                    req_id=req_id_str,
                    prompt_token_ids=prompt_token_ids_map[sr.request_id],
                    mm_features=[],
                    sampling_params=sp,
                    pooling_params=None,
                    block_ids=block_ids,
                    num_computed_tokens=sr.num_computed_tokens,
                    lora_request=None,
                )
            )
        else:
            resumed_reqs.append(
                ResumedRequestData(
                    req_id=req_id_str,
                    block_ids=block_ids,
                    num_computed_tokens=sr.num_computed_tokens,
                )
            )

    return VllmOut(
        scheduled_new_reqs=new_reqs,
        scheduled_resumed_reqs=resumed_reqs,
        scheduled_running_reqs=[],
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=rust_output.num_batched_tokens,
        spec_token_ids={},
        num_lookahead_slots=0,
        running_queue_size=0,
        finished_req_ids=set(str(i) for i in rust_output.finished_request_ids),
        free_encoder_input_ids=[],
        preempted_req_ids=set(str(i) for i in rust_output.preempted_request_ids),
    )


# ---------------------------------------------------------------------------
# Worker wrapper
# ---------------------------------------------------------------------------


class OhMyVllmWorker:
    """Wraps vLLM's GPUWorker, adapting the Rust scheduler's message format.

    Usage::

        from vllm.config import VllmConfig
        from oh_my_vllm.worker.model_runner import OhMyVllmWorker

        worker = OhMyVllmWorker(vllm_config)
        worker.init_device()
        worker.load_model()
        num_blocks = worker.determine_num_available_blocks()
        worker.initialize_cache(*num_blocks)

        # main loop
        rust_output: SchedulerOutput = ...  # from ZMQ
        worker_out: WorkerOutput = worker.execute_model(rust_output)
        # send worker_out back over ZMQ
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        sampling_params_map: dict | None = None,
        prompt_token_ids_map: dict | None = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        from vllm.v1.worker.gpu_worker import GPUWorker

        self._worker = GPUWorker(
            vllm_config, local_rank=0, rank=0, distributed_init_method="env://"
        )
        # Mutable maps updated by the Rust side as requests arrive / finish.
        self.sampling_params_map: dict = sampling_params_map or {}
        self.prompt_token_ids_map: dict = prompt_token_ids_map or {}
        # 0 means MTP spec decode is disabled.
        self._num_speculative_tokens = num_speculative_tokens

    def init_device(self) -> None:
        self._worker.init_device()

    def load_model(self) -> None:
        self._worker.load_model()

    def determine_num_available_blocks(self) -> tuple[int, int]:
        return self._worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int = 0) -> None:
        self._worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def register_request(
        self,
        request_id: int,
        prompt_token_ids: list[int],
        sampling_params=None,
    ) -> None:
        """Called by the ZMQ bridge when a new request arrives from Rust."""
        from vllm.sampling_params import SamplingParams

        self.prompt_token_ids_map[request_id] = prompt_token_ids
        self.sampling_params_map[request_id] = (
            sampling_params if sampling_params is not None else SamplingParams()
        )

    def unregister_request(self, request_id: int) -> None:
        """Called by the ZMQ bridge when a request is finished or aborted."""
        self.prompt_token_ids_map.pop(request_id, None)
        self.sampling_params_map.pop(request_id, None)

    def execute_model(self, rust_output: SchedulerOutput) -> WorkerOutput:
        """Run one forward step and return per-request next tokens."""
        if not rust_output.scheduled:
            return WorkerOutput(outputs=[])

        vllm_out = _to_vllm_scheduler_output(
            rust_output,
            self.sampling_params_map,
            self.prompt_token_ids_map,
        )
        model_runner_output = self._worker.execute_model(vllm_out)

        if model_runner_output is None:
            return WorkerOutput(outputs=[])

        req_ids = list(vllm_out.num_scheduled_tokens.keys())
        raw_sampled: list[list[int]] = model_runner_output.sampled_token_ids or []

        if self._num_speculative_tokens > 0:
            from oh_my_vllm.worker.spec_decode import parse_mtp_output

            parsed = parse_mtp_output(raw_sampled, self._num_speculative_tokens)
            outputs: list[RequestOutput] = [
                RequestOutput(
                    request_id=int(req_id),
                    next_token_id=next_tok,
                    num_accepted_draft_tokens=num_accepted,
                    new_draft_token_ids=new_drafts,
                )
                for req_id, (next_tok, num_accepted, new_drafts) in zip(
                    req_ids, parsed, strict=False
                )
            ]
        else:
            outputs = [
                RequestOutput(
                    request_id=int(req_id),
                    next_token_id=int(toks[0]) if toks else 0,
                )
                for req_id, toks in zip(req_ids, raw_sampled, strict=False)
            ]

        # Clean up finished requests from our local maps.
        for rid in rust_output.finished_request_ids:
            self.unregister_request(rid)

        return WorkerOutput(outputs=outputs)
