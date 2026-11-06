class CacheState:
    def _state_destinations(self, row, start_pos, end_pos, mamba_blocks):
        destinations = {}
        for index, block in enumerate(mamba_blocks):
            endpoint = (index + 1) * self.block_tokens
            if block and start_pos < endpoint <= end_pos:
                destinations[block] = endpoint
        tail = mamba_blocks[(end_pos - 1) // self.block_tokens]
        if not tail:
            raise ValueError("DeepSeek accepted endpoint has no state page")
        destinations[tail] = end_pos
        checkpoint = row.get("cache_checkpoint")
        if checkpoint is not None:
            if (
                not isinstance(checkpoint, (list, tuple))
                or len(checkpoint) != 2
                or type(checkpoint[0]) is not int
                or not 0 < checkpoint[0] <= self.max_seq_len
            ):
                raise ValueError("invalid DeepSeek cache checkpoint")
            endpoint, block = checkpoint
            self._validate_block_id(block, self.mamba_capacity, "checkpoint")
            if not block:
                raise ValueError("DeepSeek checkpoint must name a physical page")
            if start_pos < endpoint <= end_pos:
                if block in destinations and destinations[block] != endpoint:
                    raise ValueError("DeepSeek state page has conflicting endpoints")
                destinations[block] = endpoint
        return destinations

    def begin_step(self, row, slot, *, start_pos, end_pos, minimum_end=None):
        if not 0 <= start_pos < end_pos <= self.max_seq_len:
            raise ValueError("DeepSeek state export range is out of bounds")
        fa, mamba = self._validate_row(row, end=end_pos)
        minimum_end = end_pos if minimum_end is None else minimum_end
        if type(minimum_end) is not int or not start_pos < minimum_end <= end_pos:
            raise ValueError("invalid DeepSeek minimum accepted endpoint")
        if end_pos - minimum_end > 5:
            raise ValueError("DeepSeek verification exceeds five draft tokens")
        endpoints = set(
            self._state_destinations(row, start_pos, end_pos, mamba).values()
        )
        endpoints.update(range(minimum_end, end_pos + 1))
        self._ensure_pages(fa, mamba)
        self.page_table.bind(slot, fa)
        for page_index in range(
            start_pos // self.block_tokens, (end_pos - 1) // self.block_tokens + 1
        ):
            page = self.fa_pages[fa[page_index]]
            prefix = max(0, start_pos - page_index * self.block_tokens)
            page.valid_tokens = min(page.valid_tokens, prefix)
        self.exports.begin(slot, start_pos, endpoints, minimum_end=minimum_end)

    def mtp_checkpoint_ends(self, slots, first, last, final_end):
        return sorted(
            {
                end
                for slot in slots
                for end in self.exports.endpoints(slot)
                if first < end <= last
                and end < final_end
                and end < self.exports.steps[slot].minimum_end
            }
        )

    def export_decoded_step(self, slots, start_pos, end_pos):
        """Export the target state a graph replay wrote outside Python."""
        self.exports.export_decoded_step(
            self._window, self._tails, slots, start_pos, end_pos
        )

    def _captured_state(self, slot, endpoint, accepted_end, include_mtp):
        if slot in self.exports.steps:
            return (
                [
                    self.exports.window_state(slot, endpoint, value)
                    for value in self._window
                ],
                [
                    self.exports.window_state(slot, endpoint, value)
                    for value in self._mtp_window
                ]
                if include_mtp
                else (),
                [
                    self.exports.tail_state(slot, endpoint, kv)
                    for kv, _score in self._tails
                ],
            )
        if endpoint != accepted_end:
            raise ValueError("DeepSeek intermediate endpoint was not exported")
        return (
            [value[slot] for value in self._window],
            [value[slot] for value in self._mtp_window] if include_mtp else (),
            [(kv[slot], score[slot]) for kv, score in self._tails],
        )

    def commit(
        self, request_id, row, slot, *, start_pos=0, accepted_end, include_mtp=False
    ):
        if type(start_pos) is not int or type(accepted_end) is not int:
            raise ValueError("DeepSeek cache positions must be integers")
        if not 0 <= start_pos <= accepted_end <= self.max_seq_len:
            raise ValueError("DeepSeek cache positions are out of range")
        fa_blocks, mamba_blocks = self._validate_row(row, end=accepted_end)
        if accepted_end == 0:
            self.exports.discard(slot)
            return
        destinations = self._state_destinations(
            row, start_pos, accepted_end, mamba_blocks
        )
        prefix_state = row.get("prefix_state_slot")
        if prefix_state is not None and start_pos:
            self._validate_block_id(prefix_state, self.mamba_capacity, "prefix")
            prefix = self.mamba_pages.get(prefix_state)
            if prefix is None or prefix.endpoint != start_pos:
                raise ValueError(
                    "DeepSeek prefix state endpoint differs from the schedule"
                )
            if prefix_state in destinations and destinations[prefix_state] != start_pos:
                raise ValueError(
                    "DeepSeek prefix state requires a private continuation"
                )
        states = {
            end: self._captured_state(slot, end, accepted_end, include_mtp)
            for end in set(destinations.values())
        }
        first_page = start_pos // self.block_tokens
        last_page = (accepted_end - 1) // self.block_tokens
        for page_index in range(first_page, last_page + 1):
            page_start = page_index * self.block_tokens
            page_end = min(accepted_end, page_start + self.block_tokens)
            block = fa_blocks[page_index]
            if not block:
                continue
            page = self._fa_page(block)
            page.valid_tokens = max(page.valid_tokens, page_end - page_start)
        for block, endpoint in destinations.items():
            page = self._mamba_page(block)
            window, mtp_window, tails = states[endpoint]
            for source, destination in zip(window, page.window, strict=True):
                destination.copy_(source)
            if include_mtp:
                for source, destination in zip(
                    mtp_window, page.mtp_window, strict=True
                ):
                    destination.copy_(source)
            for (source_kv, source_score), (target_kv, target_score) in zip(
                tails, page.tails, strict=True
            ):
                target_kv.copy_(source_kv)
                target_score.copy_(source_score)
            page_start = (endpoint - 1) // self.block_tokens * self.block_tokens
            for source, destination in zip(self._engram, page.engram, strict=True):
                destination.zero_()
                destination[: endpoint - page_start].copy_(
                    source[slot, page_start:endpoint]
                )
        proposed_end = max(self.exports.endpoints(slot), default=accepted_end)
        for page_index in range(
            first_page, (proposed_end - 1) // self.block_tokens + 1
        ):
            block = fa_blocks[page_index]
            valid = max(
                0,
                min(self.block_tokens, accepted_end - page_index * self.block_tokens),
            )
            for history in self.histories:
                history.discard_after(block, valid)
            self.fa_pages[block].valid_tokens = valid
        for block, endpoint in destinations.items():
            page = self.mamba_pages[block]
            page.endpoint = endpoint
            page.valid_tokens = (
                endpoint - (endpoint - 1) // self.block_tokens * self.block_tokens
            )
        self.exports.discard(slot)

    def restore(self, request_id, slot, row, *, start_pos, use_prefix_state=True):
        if type(start_pos) is not int or not 0 <= start_pos <= self.max_seq_len:
            raise ValueError("DeepSeek restore position is out of range")
        fa_blocks, mamba_blocks = self._validate_row(row, end=start_pos)
        state_source = row.get("prefix_state_slot") if use_prefix_state else None
        if state_source is not None:
            self._validate_block_id(state_source, self.mamba_capacity, "prefix")
            if state_source == 0:
                raise ValueError("DeepSeek prefix state must name a physical page")
        state = None
        if start_pos:
            last_page = (start_pos - 1) // self.block_tokens
            if state_source is None:
                state_source = next(
                    (
                        block
                        for block in reversed(mamba_blocks[: last_page + 1])
                        if block
                    ),
                    None,
                )
            state = self.mamba_pages.get(state_source)
            if state is None:
                raise ValueError("DeepSeek restore state page is unavailable")
            if state.endpoint != start_pos:
                raise ValueError(
                    "DeepSeek restore state endpoint differs from the schedule"
                )
            for index, block in enumerate(fa_blocks[: last_page + 1]):
                if block:
                    page = self.fa_pages.get(block)
                    required = min(
                        self.block_tokens, start_pos - index * self.block_tokens
                    )
                    if page is None or page.valid_tokens < required:
                        raise ValueError(
                            "DeepSeek restore FA page is unavailable or incomplete"
                        )
        self.clear_slot(slot)
        self.page_table.bind(slot, fa_blocks)
        if not start_pos:
            return
        for targets, sources in (
            (self._window, state.window),
            (self._mtp_window, state.mtp_window),
        ):
            for value, source in zip(targets, sources, strict=True):
                value[slot].copy_(source)
        for (kv, score), (source_kv, source_score) in zip(
            self._tails, state.tails, strict=True
        ):
            kv[slot].copy_(source_kv)
            score[slot].copy_(source_score)
        page_start = (start_pos - 1) // self.block_tokens * self.block_tokens
        for value, source in zip(self._engram, state.engram, strict=True):
            value[slot, page_start:start_pos] = source[: start_pos - page_start]

    def copy_prefix(self, copies):
        for source, destination, length in copies:
            if (
                type(source) is not int
                or type(destination) is not int
                or type(length) is not int
                or source <= 0
                or destination <= 0
                or source == destination
                or not 0 < length <= self.block_tokens
            ):
                raise ValueError("invalid DeepSeek FA prefix copy")
            self._validate_block_id(source, self.fa_capacity, "FA source")
            self._validate_block_id(destination, self.fa_capacity, "FA destination")
            source_page = self.fa_pages.get(source)
            if source_page is None or source_page.valid_tokens < length:
                raise ValueError(
                    "DeepSeek prefix copy references an unknown or incomplete page"
                )
            destination_page = self._fa_page(destination)
            for spec, sources, targets in (
                (self._compressed, source_page.compressed, destination_page.compressed),
                (self._index, source_page.index, destination_page.index),
            ):
                for (_value, ratio), source_value, target in zip(
                    spec, sources, targets, strict=True
                ):
                    count = min(target.shape[0], length // ratio)
                    target[:count].copy_(source_value[:count])
            for source_value, target in zip(
                source_page.engram, destination_page.engram, strict=True
            ):
                target[:length].copy_(source_value[:length])
            destination_page.valid_tokens = length
