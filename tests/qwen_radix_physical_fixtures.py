from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from oh_my_vllm.models.qwen38.cache import Cache, PageState
from oh_my_vllm.models.qwen38.cache_budget import LAYOUT_ID, CacheLayout


def config(**changes):
    return {
        "max_num_seqs": 4,
        "max_num_batched_tokens": 32768,
        "max_model_len": 262144,
        "num_speculative_tokens": 0,
        "num_gpu_blocks": None,
        "mamba_blocks": None,
        **changes,
    }


def reference(end, slot, version=1):
    return {"end": end, "slot": slot, "version": version}


def row(
    start=0,
    end=3,
    *,
    purpose="advance",
    action="forward",
    sampling="none",
    deepest=0,
    contiguous=0,
    resume=None,
    pages=None,
    spans=None,
    exports=None,
):
    if pages is None:
        pages = [
            {
                "page_index": 0,
                "slot": 1,
                "version": 1,
                "valid_tokens": start,
                "access": "write",
            }
        ]
    if spans is None:
        spans = [{"start": start, "end": end, "mode": "write"}] if start < end else []
    if exports is None:
        exports = [
            {
                **reference(end, 2 if purpose == "finalize" else 5),
                "role": "candidate_tail"
                if purpose == "finalize"
                else "private_endpoint",
            }
        ]
    return {
        "request_id": 17,
        "lease_id": 4,
        "purpose": purpose,
        "action": action,
        "sampling": {"kind": sampling},
        "token_ids": list(range(start, end)),
        "forward_start": start,
        "forward_end": end,
        "accepted_end": end,
        "endpoint": end,
        "model_state": {
            "kind": "qwen_radix_v1",
            "layout_fingerprint": LAYOUT_ID,
            "checkpoint_hit_end": deepest,
            "kv_contiguous_end": contiguous,
            "resume": reference(0, 0, 0) if resume is None else resume,
            "private_checkpoints": [
                {"slot": slot, "version": 1} for slot in range(5, 11)
            ],
            "page_table": pages,
            "kv_spans": spans,
            "page_copies": [],
            "exports": exports,
            "endpoint_source": {"kind": "forward"},
            "mtp_endpoint": None,
        },
    }


class Device:
    def __init__(self):
        self.copies = []
        self.allocations = []
        self.events = []

    def ensure_healthy(self):
        self.events.append("healthy")

    def wait_pinned_safe(self):
        self.events.append("joined")

    def synchronize_compute(self):
        self.events.append("compute_complete")

    def zeros(self, shape, dtype):
        value = SimpleNamespace(shape=shape, dtype=dtype, name=len(self.allocations))
        self.allocations.append(value)
        return value

    def view(self, value, shape, offset=0):
        if isinstance(value, np.ndarray):
            return value.reshape(-1)[offset : offset + int(np.prod(shape))].reshape(
                shape
            )
        return (value.name, shape, offset)

    def copy(self, destination, source):
        self.copies.append((destination, source))
        if isinstance(destination, np.ndarray):
            np.copyto(destination, source)


def cache():
    return Cache(Device(), config(mamba_blocks=4, num_gpu_blocks=256))


def request(**changes):
    values = dict(
        position=lambda position: (position,) * 3,
        state_slot=None,
        state_position=0,
        computed_end=0,
        kv_valid_end=0,
        endpoints={},
        drafts=[],
        mtp_position=0,
        history=[0, 1, 2],
        finish="length",
        thinking_count=7,
        random_offset=11,
        proposal_offset=13,
        accept_offset=17,
        append=Mock(side_effect=AssertionError("compute-only work must not append")),
    )
    values.update(changes)
    return SimpleNamespace(**values)


def worker(owner):
    from oh_my_vllm.models.qwen38.inference import Worker

    value = Worker.__new__(Worker)
    value.cache = cache()
    value.device = value.cache.device
    value.requests = {17: owner}
    value.config = config()
    value.speculation = None
    value.epoch, value.last_step = 1, 0
    value.sampler = Mock(
        side_effect=AssertionError("compute-only work must not sample")
    )
    value.model = SimpleNamespace(
        forward=Mock(return_value=((None, None), None, False))
    )
    value.trace = SimpleNamespace(record=Mock())
    return value


def execute_message(item):
    return {
        "rpc_id": 19,
        "worker_epoch": 1,
        "step_id": 1,
        "scheduled": [item],
        "finished_request_ids": [],
        "preempted_request_ids": [],
        "num_batched_tokens": len(item["token_ids"]),
    }


def context_batch(lengths, start=8192):
    owners, scheduled, plans, tables = [], [], [], []
    starts = [0]
    for index, count in enumerate(lengths):
        end = start + count
        owner = request(
            state_position=end,
            mtp_position=start - 1,
            history=list(range(end + 1)),
            finish=None,
            generated=1,
            limit=262144,
        )
        pages = tuple(
            {"slot": index * 512 + page + 1, "access": "write"} for page in range(2048)
        )
        owners.append(owner)
        scheduled.append({"forward_start": start})
        plans.append(SimpleNamespace(end=end, write_mask=(True,) * count, pages=pages))
        tables.append([page["slot"] for page in pages])
        starts.append(starts[-1] + count)
    return SimpleNamespace(
        scheduled=scheduled,
        requests=owners,
        plans=plans,
        schedule=SimpleNamespace(
            sequence_rows=tuple(starts), read_slots=tuple(range(5, 5 + len(lengths)))
        ),
        metadata=(None, None, np.asarray(tables, np.int32)),
    )


def tiny_cache():
    value = Cache.__new__(Cache)
    value.device = Device()
    value.blocks = 3
    value.pages = {1: PageState(1, 0, 128, 128)}
    value.layout = CacheLayout(4, 4, 32768, 16)
    value.fa = (
        (
            np.arange(4 * 2 * 128 * 4 * 256, dtype=np.uint32)
            .astype(np.uint8)
            .reshape(4, 2, 128, 4, 256),
            np.arange(4 * 2 * 128 * 4, dtype=np.float32).reshape(4, 2, 128, 4),
        ),
    )
    return value
