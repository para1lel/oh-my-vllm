"""Union final live physical KV, checkpoint, and working-state writes.

Only committed target inputs enter persistent KV. MTP proposal rows are
temporary; committed MTP forward rows persist. Rejected GDN candidate snapshots
are temporary. A checkpoint copy changes the physical location of one value,
and repeated writes to a working slot leave only its final live version.
"""

from collections import defaultdict

BLOCK = 784


class WriteLedger:
    def __init__(self):
        self.ranges = defaultdict(list)
        self.checkpoints = set()
        self.working = {}
        self.state_width = None
        self.page_owners = {}
        self.checkpoint_owners = {}

    def own_page(self, page, request_id, logical):
        owner = request_id, logical
        previous = self.page_owners.get(page)
        if previous is not None and previous != owner:
            for family in ("target", "mtp", "dspark", "mtp_hidden"):
                self.ranges.pop((family, page), None)
        self.page_owners[page] = owner

    @staticmethod
    def retained_end(request):
        # Rust registers prefixes before this worker response. Accepted drafts
        # beyond its known history have not been registered on terminal release.
        known = request["context"] + request["query"] - request["drafts"]
        return known // BLOCK * BLOCK

    def tokens(self, family, table, start, count, request_id):
        if type(start) is not int or type(count) is not int or start < 0 or count < 0:
            raise ValueError("invalid persistent token interval")
        end = start + count
        for logical in range(start // BLOCK, (end + BLOCK - 1) // BLOCK):
            if (
                logical >= len(table)
                or type(table[logical]) is not int
                or table[logical] <= 0
            ):
                raise ValueError("persistent write has no physical page")
            self.own_page(table[logical], request_id, logical)
            self.ranges[(family, table[logical])].append(
                (
                    max(start, logical * BLOCK) % BLOCK,
                    min(end, (logical + 1) * BLOCK) - logical * BLOCK,
                )
            )

    def invalidate(self, request):
        """Observe allocations and writes, including another phase's work.

        A consumed working version is no longer a phase-final live value. Rust
        can evict a checkpoint into an unused speculative reservation. Page
        reassignment invalidates every target/draft family at that address.
        """
        rid = request["request_id"]
        self.working.pop(rid, None)
        for logical, page in enumerate(request["fa_block_table"]):
            self.own_page(page, rid, logical)
        base = (request["context"] + request["query"] - 1) // BLOCK
        reserved = {
            row
            for row in request["mamba_block_table"][base:]
            if row > 0 and row != request.get("state_source")
        }
        written = {row for row in request["state_row_writes"] if row > 0}
        invalid = reserved | written
        self.checkpoints.difference_update(invalid)
        for row in invalid:
            self.checkpoint_owners.pop(row, None)
        for owner, row in tuple(self.working.items()):
            if row in written:
                self.working.pop(owner)

    def target(self, requests, state_dtype):
        width = {"torch.float32": 4, "torch.bfloat16": 2}.get(state_dtype)
        if width is None or self.state_width not in (None, width):
            raise ValueError("phase changed persistent state precision")
        self.state_width = width
        for request in requests:
            count = request["kept_rows"]
            start = request["context"]
            writes = request["state_row_writes"]
            self.invalidate(request)
            if not 0 < count <= len(writes) or writes[count - 1] <= 0:
                raise ValueError("committed state has no recorded destination")
            self.working[request["request_id"]] = writes[count - 1]
            table = request["mamba_block_table"]
            for boundary in range(
                (start // BLOCK + 1) * BLOCK, start + count + 1, BLOCK
            ):
                index = boundary // BLOCK - 1
                if index >= len(table):
                    raise ValueError("checkpoint destination is missing")
                if table[index] > 0:
                    if writes[boundary - start - 1] <= 0:
                        raise ValueError("checkpoint has no source snapshot")
                    if writes[boundary - start - 1] != table[index] and not request.get(
                        "checkpoint_copies_completed", True
                    ):
                        continue
                    self.checkpoints.add(table[index])
                    self.checkpoint_owners[table[index]] = (
                        request["request_id"],
                        boundary,
                    )
            rid = request["request_id"]
            self.tokens("target", request["fa_block_table"], start, count, rid)
            if request.get("terminal", False):
                self.working.pop(rid, None)
                end = self.retained_end(request)
                for logical in range(end // BLOCK, len(request["fa_block_table"])):
                    page = request["fa_block_table"][logical]
                    for family in ("target", "mtp", "dspark", "mtp_hidden"):
                        self.ranges.pop((family, page), None)
                for row, (owner, boundary) in tuple(self.checkpoint_owners.items()):
                    if owner == rid and boundary > end:
                        self.checkpoint_owners.pop(row)
                        self.checkpoints.discard(row)
            if request.get("mtp", False):
                # Prefix re-admission reads the boundary target hidden vector.
                for boundary in range(
                    (start // BLOCK + 1) * BLOCK, start + count + 1, BLOCK
                ):
                    if not request.get("checkpoint_copies_completed", True):
                        continue
                    if request.get("terminal", False) and boundary > end:
                        continue
                    page = request["fa_block_table"][boundary // BLOCK - 1]
                    self.own_page(page, rid, boundary // BLOCK - 1)
                    self.ranges[("mtp_hidden", page)] = [(0, 1)]

    def draft(self, operations, target_requests):
        targets = {r["request_id"]: r for r in target_requests}
        for operation in operations:
            if operation["kind"] == "dspark_inject":
                for rid, count in zip(
                    operation["requests"], operation["queries"], strict=True
                ):
                    request = targets[rid]
                    start = request["context"]
                    if request.get("terminal", False):
                        end = self.retained_end(request)
                        count = max(0, min(start + count, end) - start)
                    self.tokens("dspark", request["fa_block_table"], start, count, rid)
            elif operation["kind"] == "mtp_forward" and operation["persistent"]:
                for rid, start, count, table in zip(
                    operation["requests"],
                    operation["contexts"],
                    operation["queries"],
                    operation["tables"],
                    strict=True,
                ):
                    request = targets[rid]
                    if request.get("terminal", False):
                        end = self.retained_end(request)
                        count = max(0, min(start + count, end) - start)
                    self.tokens("mtp", table, start, count, rid)

    def bytes(self):
        per_token = {
            "target": 16 * 2 * 4 * 256 * 2,
            "mtp": 2 * 4 * 256 * 2,
            "dspark": 5 * 2 * 8 * 128 * 2,
            "mtp_hidden": 5120 * 2,
        }
        total = 0
        for (family, _), intervals in self.ranges.items():
            right = 0
            for left, end in sorted(intervals):
                total += max(0, end - max(left, right)) * per_token[family]
                right = max(right, end)
        state_rows = self.checkpoints | set(self.working.values())
        if state_rows:
            total += (
                len(state_rows)
                * 48
                * (48 * 128 * 128 * self.state_width + 10240 * 3 * 2)
            )
        return total
