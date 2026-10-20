from collections import deque
from dataclasses import dataclass

import numpy as np

from ..operators.gdn.api import Schedule


@dataclass
class Batch:
    scheduled: list
    requests: list
    schedule: Schedule
    tokens: np.ndarray
    metadata: tuple
    writes: tuple

    @property
    def rows(self):
        return len(self.tokens)

    @property
    def max_queries(self):
        return max(np.diff(self.schedule.sequence_rows))


class Cache:
    def __init__(self, device, config):
        self.device = device
        self.blocks = config["num_gpu_blocks"]
        self.mamba_blocks = config["mamba_blocks"] or self.blocks
        self.parallel = config["max_num_seqs"]
        self.private_start = self.mamba_blocks
        self.scratch_start = self.private_start + self.parallel * 6
        self.slots = (
            self.scratch_start
            + (min(32768, config["max_num_batched_tokens"]) + 783) // 784
            + self.parallel
        )
        if self.slots > 65536:
            raise ValueError("GDN cache exceeds the state slot limit")
        self.fa = tuple(
            (device.zeros((self.blocks, 2, 784, 4, 256), "bfloat16"),)
            for _ in range(16 + bool(config["num_speculative_tokens"]))
        )
        self.state = tuple(
            device.zeros((self.slots, 48, 128, 128), "float32") for _ in range(48)
        )
        self.convolution = tuple(
            device.zeros((self.slots, 10240, 3), "bfloat16") for _ in range(48)
        )
        self.hidden = device.zeros((self.slots, 5120), "bfloat16")
        self.indices = {}
        self.free_indices = deque(range(self.parallel))

    def release(self, request_id):
        index = self.indices.pop(request_id, None)
        if index is not None:
            self.free_indices.append(index)

    def private_slots(self, request_id, read):
        if request_id not in self.indices:
            if not self.free_indices:
                raise ValueError("worker request capacity is exhausted")
            self.indices[request_id] = self.free_indices.popleft()
        first = self.private_start + self.indices[request_id] * 6
        return [slot for slot in range(first, first + 6) if slot != read]

    def copy_prefix(self, copies):
        for source, destination, length in copies:
            if (
                not 0 < source < self.blocks
                or not 0 < destination < self.blocks
                or source == destination
                or not 0 < length < 784
            ):
                raise ValueError("invalid FA prefix copy")
            for (cache,) in self.fa:
                for plane in range(2):
                    views = [
                        self.device.view(
                            cache,
                            (length, 4, 256),
                            (block * 2 + plane) * 784 * 4 * 256,
                        )
                        for block in (source, destination)
                    ]
                    self.device.module.copy_bytes(views[1], views[0])

    def batch(self, scheduled, requests, verify):
        starts, reads, writes, checkpoints, checkpoint_slots = (
            [0],
            [],
            [],
            [],
            [],
        )
        tokens, positions, slots, lengths = [], [], [], []
        tables = np.zeros((len(scheduled), 335), dtype=np.int32)
        scratch = self.scratch_start
        for index, (row, request) in enumerate(zip(scheduled, requests, strict=True)):
            start = row["num_computed_tokens"]
            count = len(row["token_ids"])
            restored = row.get("prefix_state_slot")
            if restored is not None and (
                request.state_position != start or request.state_slot is None
            ):
                if not start or not 0 < restored < self.mamba_blocks:
                    raise ValueError("invalid prefix state slot")
                read = restored
            elif request.state_position == start and request.state_slot is not None:
                read = request.state_slot
            elif start:
                read = row["mamba_block_table"][(start - 1) // 784]
                if not read:
                    raise ValueError("prefix has no GDN checkpoint")
            else:
                read = 0
            private = iter(self.private_slots(row["request_id"], read))
            reads.append(read)
            table = row["fa_block_table"]
            if len(table) > 335:
                raise ValueError("FA block table exceeds the context limit")
            tables[index, : len(table)] = table
            first = len(tokens)
            cached_endpoint = row.get("cache_checkpoint")
            for offset, token in enumerate(row["token_ids"]):
                position = start + offset
                tokens.append(token)
                positions.append(request.position(position))
                block = table[position // 784]
                if not 0 < block < self.blocks:
                    raise ValueError("FA write has no private block")
                slots.append(block * 784 + position % 784)
                lengths.append(position + 1)
                slot = -1
                checkpoint = (position + 1) % 784 == 0
                if verify or checkpoint or offset + 1 == count:
                    logical = (
                        row["mamba_block_table"][(position + 1) // 784 - 1]
                        if checkpoint
                        else 0
                    )
                    if (
                        cached_endpoint is not None
                        and position + 1 == cached_endpoint[0]
                    ):
                        logical = cached_endpoint[1]
                        if not 0 < logical < self.mamba_blocks:
                            raise ValueError("invalid prefix checkpoint slot")
                    if logical:
                        slot = logical
                    elif offset + 1 == count or verify:
                        slot = next(private)
                    else:
                        slot = scratch
                        scratch += 1
                    if not verify:
                        checkpoints.append(len(tokens) - 1)
                        checkpoint_slots.append(slot)
                writes.append(slot if verify else -1)
            starts.append(first + count)
        if scratch > self.slots:
            raise ValueError("prefill checkpoint scratch capacity is exhausted")
        schedule = Schedule(
            tuple(starts),
            tuple(reads),
            tuple(writes),
            tuple(checkpoints),
            tuple(checkpoint_slots),
            tuple(row["num_computed_tokens"] for row in scheduled),
            self.slots,
            verify,
        )
        actual_writes = schedule.validate()
        metadata = (
            np.asarray(positions, dtype=np.int64).T.copy(),
            np.asarray(slots, dtype=np.int32),
            tables,
            np.asarray(starts, dtype=np.int32),
            np.asarray(lengths, dtype=np.int32),
        )
        return Batch(
            scheduled,
            requests,
            schedule,
            np.asarray(tokens, dtype=np.int64),
            metadata,
            actual_writes,
        )

    def commit(self, request, batch, index, accepted):
        first = batch.schedule.sequence_rows[index]
        count = batch.schedule.sequence_rows[index + 1] - first
        selected = count - 1 if accepted is None else accepted
        slot = batch.writes[first + selected]
        if slot < 0:
            raise ValueError("accepted state has no private export")
        request.state_slot = slot
        request.state_position = (
            batch.scheduled[index]["num_computed_tokens"] + selected + 1
        )
