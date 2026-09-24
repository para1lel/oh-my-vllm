"""Validate Rust allocations and select explicit hybrid state destinations.

This module never allocates logical cache pages. CPU planning makes invalid or
aliased GPU addresses fail before kernels run. MTP candidates are physical rows
of the Rust-owned Mamba table; accepted rows become the next step's source.
"""

from dataclasses import dataclass

from oh_my_vllm.worker.protocol import ScheduledRequest

BLOCK = 784


@dataclass
class PlannedRequest:
    request: ScheduledRequest
    source: int
    writes: list[int]
    drafts: list[int]
    sample_indices: list[int]
    prefill: bool

    def commit(self, sampled_count: int) -> tuple[int, int, list[tuple[int, int]]]:
        """Return committed input count, source row and checkpoint copies.

        Crossing a page boundary during verification must preserve the state at
        that exact boundary, even if later draft tokens were also accepted.
        Copies are (source, destination) and run after all model layers finish.
        """
        if self.sample_indices:
            if not 1 <= sampled_count <= len(self.drafts) + 1:
                raise ValueError("invalid retained sample count")
        elif sampled_count:
            raise ValueError("intermediate prefill cannot produce samples")
        count = len(self.writes) - len(self.drafts) + max(0, sampled_count - 1)
        source = self.writes[count - 1]
        if source <= 0:
            raise ValueError("committed input has no recurrent state snapshot")
        start = self.request.num_computed_tokens
        end = start + count
        copies = []
        for boundary in range((start // BLOCK + 1) * BLOCK, end + 1, BLOCK):
            destination = self.request.mamba_block_table[boundary // BLOCK - 1]
            if destination == 0:
                # Chunked prefill intentionally retains only its final checkpoint.
                continue
            state = self.writes[boundary - start - 1]
            if state <= 0:
                raise ValueError("cacheable boundary has no saved state")
            if state != destination:
                copies.append((state, destination))
        return count, source, copies


def plan_request(
    request: ScheduledRequest,
    history: list[int],
    source: int | None,
    capacity: int,
    speculative_tokens: int,
) -> PlannedRequest:
    start = request.num_computed_tokens
    tokens = request.token_ids
    end = start + len(tokens)
    if not tokens or not 0 <= start < len(history):
        raise ValueError("scheduled input must begin in accepted history")
    if capacity <= 1 or speculative_tokens not in (0, 4):
        raise ValueError("invalid cache capacity or speculative token count")
    known = min(len(tokens), len(history) - start)
    if tokens[:known] != history[start : start + known]:
        raise ValueError("scheduled input disagrees with accepted history")
    drafts = tokens[known:]
    if len(drafts) > speculative_tokens:
        raise ValueError("too many speculative tokens")
    if drafts and known != 1:
        raise ValueError("verification must start with exactly one committed token")
    admission = source is None
    if any(not 0 <= p < capacity for p in request.mamba_block_table):
        raise ValueError("Mamba table contains a row outside the allocated pool")
    if admission:
        if start % BLOCK:
            raise ValueError("admission cache hit must be block aligned")
        if start and len(request.mamba_block_table) < start // BLOCK:
            raise ValueError("Mamba table is missing the prefix checkpoint")
        if request.prefill_token_ids != history:
            raise ValueError("admission/resumption requires complete accepted history")
        source = 0 if start == 0 else request.mamba_block_table[(start - 1) // BLOCK]
    elif request.prefill_token_ids is not None:
        raise ValueError("running request cannot repeat admission history")
    if not 0 <= source < capacity or (start > 0 and source == 0):
        raise ValueError("missing or invalid recurrent source")
    pages = (end + BLOCK - 1) // BLOCK
    if len(request.fa_block_table) < pages:
        raise ValueError("FA table does not cover scheduled input")
    if any(not 0 < p < capacity for p in request.fa_block_table):
        raise ValueError("FA page is null or outside the allocated pool")
    base = pages - 1
    prefill = len(tokens) > 1 and not drafts
    count = 1 if prefill else len(tokens)
    if len(request.mamba_block_table) < base + count:
        raise ValueError("Mamba table does not cover candidate states")
    destinations = request.mamba_block_table[base : base + count]
    if any(not 0 < p < capacity for p in destinations):
        raise ValueError("candidate state is null or outside the allocated pool")
    if admission and start and source in destinations:
        raise ValueError("admission cannot overwrite a shared prefix checkpoint")
    if len(set(destinations)) != len(destinations):
        raise ValueError("candidate state destinations overlap")
    writes = [-1] * (len(tokens) - 1) + destinations if prefill else destinations
    sample_indices = list(range(known - 1, len(tokens))) if end >= len(history) else []
    return PlannedRequest(request, source, writes, drafts, sample_indices, prefill)


def validate_batch(plans: list[PlannedRequest]) -> None:
    """Shared reads are allowed; writes cannot alias another request's state."""
    # PY-01: FA pages touched by each request's tail block must be disjoint.
    # Shared prefix pages are read-only; only the writable tail page is checked.
    if __debug__:
        fa_page_owners: dict[int, int] = {}
        for plan in plans:
            rid = plan.request.request_id
            req = plan.request
            end = req.num_computed_tokens + len(plan.writes)
            # The tail (highest index) page is the only writable FA page.
            tail_page_idx = (end - 1) // BLOCK
            if tail_page_idx < len(req.fa_block_table):
                tail_page = req.fa_block_table[tail_page_idx]
                if tail_page > 0:
                    if tail_page in fa_page_owners:
                        raise AssertionError(
                            f"FA page {tail_page} is writable by both request "
                            f"{fa_page_owners[tail_page]} and request {rid}"
                        )
                    fa_page_owners[tail_page] = rid
    owners = {}
    for plan in plans:
        rid = plan.request.request_id
        destinations = {p for p in plan.writes if p > 0}
        start = plan.request.num_computed_tokens
        end = start + len(plan.writes)
        destinations.update(
            plan.request.mamba_block_table[b // BLOCK - 1]
            for b in range((start // BLOCK + 1) * BLOCK, end + 1, BLOCK)
            if plan.request.mamba_block_table[b // BLOCK - 1] > 0
        )
        for destination in destinations:
            if destination in owners:
                raise ValueError("two requests write the same recurrent state")
            owners[destination] = rid
    for plan in plans:
        owner = owners.get(plan.source)
        if owner is not None and owner != plan.request.request_id:
            raise ValueError("candidate state overwrites another request's source")
