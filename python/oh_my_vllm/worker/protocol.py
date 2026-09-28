"""Framework-owned Rust/Python execution messages, independent of GPU libraries."""

from dataclasses import dataclass, field


@dataclass
class ScheduledRequest:
    request_id: int
    token_ids: list[int]
    num_computed_tokens: int
    fa_block_table: list[int]
    mamba_block_table: list[int]
    prefill_token_ids: list[int] | None = None


@dataclass
class SchedulerOutput:
    scheduled: list[ScheduledRequest] = field(default_factory=list)
    finished_request_ids: list[int] = field(default_factory=list)
    preempted_request_ids: list[int] = field(default_factory=list)
    num_batched_tokens: int = 0


@dataclass
class RequestOutput:
    request_id: int
    token_ids: list[int]
    error: str | None = None
    num_accepted_draft_tokens: int = 0
    new_draft_token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str | None = None
    reasoning_tokens: int = 0


@dataclass
class WorkerOutput:
    outputs: list[RequestOutput]
