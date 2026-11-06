from dataclasses import dataclass

from .cache_budget import LAYOUT_ID, PAGE_TOKENS


@dataclass(frozen=True)
class Endpoint:
    end: int
    slot: int
    version: int

    def wire(self):
        return {"end": self.end, "slot": self.slot, "version": self.version}


def integer(value, name, maximum=0xFFFFFFFFFFFFFFFF):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"invalid Qwen {name}")
    return value


def endpoint(value, slots, context):
    result = Endpoint(
        integer(value["end"], "endpoint end", context),
        integer(value["slot"], "endpoint slot", slots - 1),
        integer(value["version"], "endpoint version"),
    )
    if (result.end == 0) != (result.slot == 0):
        raise ValueError("only the empty checkpoint can use null slot zero")
    return result


@dataclass(frozen=True)
class SequencePlan:
    row: dict
    read: Endpoint
    pages: tuple
    exports: tuple
    write_mask: tuple

    @property
    def start(self):
        return self.row["forward_start"]

    @property
    def end(self):
        return self.row["forward_end"]


def validate_work(row, layout):
    start = integer(row["forward_start"], "forward start", layout.max_context)
    end = integer(row["forward_end"], "forward end", layout.max_context)
    integer(row["accepted_end"], "accepted end", layout.max_context)
    integer(row["endpoint"], "endpoint", layout.max_context)
    if end < start or end - start != len(row["token_ids"]):
        raise ValueError("Qwen forward range differs from token rows")
    purpose, action, sampling = row["purpose"], row["action"], row["sampling"]
    if purpose not in ("advance", "finalize"):
        raise ValueError("invalid Qwen work purpose")
    if action not in ("forward", "head_only", "export_only"):
        raise ValueError("invalid Qwen work action")
    kind = sampling["kind"]
    if kind not in ("none", "target", "verify"):
        raise ValueError("invalid Qwen sampling mode")
    if purpose == "finalize" and kind != "none":
        raise ValueError("finalize must not sample")
    if action != "forward" and (start != end or row["token_ids"]):
        raise ValueError("endpoint work must have zero forward rows")
    if action == "forward" and start == end:
        raise ValueError("forward work requires token rows")
    if action == "head_only" and (kind != "target" or purpose != "advance"):
        raise ValueError("head-only requires target sampling")
    if action == "export_only" and kind != "none":
        raise ValueError("export-only must not sample")
    if kind == "verify" and not 1 <= integer(sampling["drafts"], "draft count", 4) <= 4:
        raise ValueError("verification needs one to four drafts")
    if kind == "verify" and end - start != sampling["drafts"] + 1:
        raise ValueError("verification row count differs from draft count")


def page_table(state, layout):
    pages = [None] * layout.table_width
    physical = set()
    for page in state["page_table"]:
        index = integer(page["page_index"], "page index", layout.table_width - 1)
        slot = integer(page["slot"], "page slot", layout.pages)
        integer(page["version"], "page version")
        integer(page["valid_tokens"], "page valid tokens", PAGE_TOKENS)
        if not slot or pages[index] is not None or slot in physical:
            raise ValueError("Qwen page table contains null or duplicate pages")
        if page["access"] not in ("read", "write"):
            raise ValueError("invalid Qwen page access")
        pages[index] = page
        physical.add(slot)
    return tuple(pages)


def write_mask(state, pages, start, end):
    mask = []
    cursor = start
    for span in state["kv_spans"]:
        first, last = span["start"], span["end"]
        if (
            first != cursor
            or not first < last <= end
            or span["mode"] not in ("reuse", "write")
        ):
            raise ValueError("KV spans must partition the forward range")
        write = span["mode"] == "write"
        for position in range(first, last):
            page = pages[position // PAGE_TOKENS]
            if page is None:
                raise ValueError("forward attention has a missing KV page")
            if write and page["access"] != "write":
                raise ValueError("KV append requires an exclusive write page")
            if not write and page["valid_tokens"] <= position % PAGE_TOKENS:
                raise ValueError("KV reuse exceeds the valid page prefix")
            mask.append(write)
        cursor = last
    if cursor != end:
        raise ValueError("KV spans do not cover all forward rows")
    return tuple(mask)


def sequence_plan(row, layout):
    validate_work(row, layout)
    state = row["model_state"]
    if state["kind"] != "qwen_radix_v1" or state["layout_fingerprint"] != LAYOUT_ID:
        raise ValueError("Qwen model state has a different physical layout")
    integer(state["checkpoint_hit_end"], "checkpoint hit", layout.max_context)
    contiguous = integer(state["kv_contiguous_end"], "KV prefix", layout.max_context)
    read = endpoint(state["resume"], layout.slots, layout.max_context)
    if read.end > contiguous:
        raise ValueError("resume must name a checkpoint at or before the KV prefix")
    validate_private(state, layout)
    pages = page_table(state, layout)
    exports = tuple(
        endpoint(item, layout.slots, layout.max_context) for item in state["exports"]
    )
    if any(item.slot == 0 for item in exports):
        raise ValueError("checkpoint exports cannot overwrite the null slot")
    if len({item.slot for item in exports}) != len(exports):
        raise ValueError("checkpoint export slots must be distinct")
    return SequencePlan(
        row,
        read,
        pages,
        exports,
        write_mask(state, pages, row["forward_start"], row["forward_end"]),
    )


def validate_private(state, layout):
    private = state["private_checkpoints"]
    if len(private) != 6 or len({item["slot"] for item in private}) != 6:
        raise ValueError("a Qwen request requires six distinct private checkpoints")
    for item in private:
        slot = integer(item["slot"], "private slot", layout.scratch_start - 1)
        integer(item["version"], "private slot version")
        if slot < layout.private_start:
            raise ValueError("private checkpoint lies outside its physical pool")
    refs = {(item["slot"], item["version"]) for item in private}
    for export in state["exports"]:
        if export["role"] == "private_endpoint":
            if (export["slot"], export["version"]) not in refs:
                raise ValueError("private export has no matching request lease")
        elif (
            export["role"] not in ("periodic", "candidate_tail")
            or not 0 < export["slot"] <= layout.checkpoints
        ):
            raise ValueError("persistent export lies outside its physical pool")


def write_ranges(mask):
    starts = []
    for index, write in enumerate((*mask, False)):
        if (write and (not index or not mask[index - 1])) or (
            not write and index and mask[index - 1]
        ):
            starts.append(index)
    return tuple(zip(starts[::2], starts[1::2], strict=True))
