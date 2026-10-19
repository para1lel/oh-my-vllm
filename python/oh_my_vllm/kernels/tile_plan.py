import math
from dataclasses import dataclass


class TileExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Capacity:
    tasks: int
    edges: int
    queue: int
    rows: int
    batch: int = 1
    pages: int = 0
    slots: int = 0
    workers: int = 0

    def validate(self):
        for name, value in vars(self).items():
            minimum = 1 if name in {"tasks", "queue", "rows", "batch"} else 0
            if type(value) is not int or not minimum <= value <= 2**31 - 1:
                raise ValueError(f"invalid tile capacity: {name}")


@dataclass(frozen=True)
class StepMetadata:
    epoch: int
    query_lengths: tuple[int, ...]
    valid_rows: int
    page_tables: tuple[tuple[int, ...], ...] = ()
    read_slots: tuple[int, ...] = ()
    write_slots: tuple[int, ...] = ()

    def validate(self, capacity, previous_epoch):
        if type(self.epoch) is not int or not previous_epoch < self.epoch < 2**63:
            raise ValueError("tile epoch must increase")
        if (
            type(self.valid_rows) is not int
            or not 0 <= self.valid_rows <= capacity.rows
        ):
            raise ValueError("invalid active row count")
        if (
            not 0 < len(self.query_lengths) <= capacity.batch
            or any(
                type(value) is not int or not 0 <= value <= capacity.rows
                for value in self.query_lengths
            )
            or sum(self.query_lengths) != self.valid_rows
        ):
            raise ValueError("query lengths differ from active rows")
        if self.page_tables and (
            len(self.page_tables) != len(self.query_lengths)
            or any(
                type(page) is not int or not 0 <= page < capacity.pages
                for pages in self.page_tables
                for page in pages
            )
        ):
            raise ValueError("invalid page table")
        if self.read_slots and (
            len(self.read_slots) != len(self.query_lengths)
            or any(
                type(slot) is not int or not -1 <= slot < capacity.slots
                for slot in self.read_slots
            )
        ):
            raise ValueError("invalid state read slots")
        if self.write_slots and (
            len(self.write_slots) != self.valid_rows
            or any(
                type(slot) is not int or not 0 <= slot < capacity.slots
                for slot in self.write_slots
            )
            or len(set(self.write_slots)) != len(self.write_slots)
            or set(self.write_slots) & set(self.read_slots)
        ):
            raise ValueError("state writes must use distinct private slots")


@dataclass(frozen=True)
class Task:
    kind: str
    inputs: tuple[int, ...]
    output: int
    start: int
    count: int
    depends: tuple[int, ...] = ()
    factor: float = 1.0

    def __post_init__(self):
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "depends", tuple(self.depends))


@dataclass(frozen=True)
class Graph:
    tasks: tuple[Task, ...]
    roots: tuple[int, ...]
    successors: tuple[tuple[int, ...], ...]


KINDS = {"copy": 0, "scale": 1, "add": 2}


def validate_task(task, capacity, buffers, size):
    if task.kind not in KINDS or len(task.inputs) != (2 if task.kind == "add" else 1):
        raise ValueError("unknown task kind or input count")
    if any(
        type(index) is not int or not 0 <= index < buffers
        for index in (*task.inputs, task.output)
    ):
        raise ValueError("invalid task buffer")
    if (
        type(task.start) is not int
        or type(task.count) is not int
        or task.start < 0
        or not 0 < task.count <= capacity.rows - task.start
    ):
        raise ValueError("task range exceeds row capacity")
    if len(set(task.depends)) != len(task.depends) or any(
        type(parent) is not int or not 0 <= parent < size for parent in task.depends
    ):
        raise ValueError("invalid or repeated dependency")
    if not math.isfinite(task.factor) or abs(task.factor) > 3.402823466e38:
        raise ValueError("task scale must be finite FP32")


def check_hazards(tasks, ancestors):
    accesses = {}
    for index, task in enumerate(tasks):
        for buffer in set((*task.inputs, task.output)):
            entries = accesses.setdefault(buffer, [])
            for previous, other in entries:
                overlap = (
                    task.start < other.start + other.count
                    and other.start < task.start + task.count
                )
                if (
                    overlap
                    and buffer in (task.output, other.output)
                    and not (
                        ancestors[index] & (1 << previous)
                        or ancestors[previous] & (1 << index)
                    )
                ):
                    raise ValueError(
                        "overlapping data, state, or lifetime accesses "
                        "need a dependency"
                    )
            entries.append((index, task))


def compile_graph(tasks, capacity, *, buffers):
    capacity.validate()
    tasks = tuple(tasks)
    if not tasks or len(tasks) > capacity.tasks or buffers < 1:
        raise ValueError("invalid graph size")
    successors = [[] for _ in tasks]
    indegrees = []
    for index, task in enumerate(tasks):
        validate_task(task, capacity, buffers, len(tasks))
        indegrees.append(len(task.depends))
        for parent in task.depends:
            successors[parent].append(index)
    if sum(indegrees) > capacity.edges:
        raise ValueError("graph exceeds edge capacity")
    roots = tuple(index for index, count in enumerate(indegrees) if count == 0)
    if len(roots) > capacity.queue:
        raise ValueError("initial ready tasks exceed queue capacity")
    pending, ancestors, visited = list(roots), [0] * len(tasks), 0
    while pending:
        parent = pending.pop()
        visited += 1
        for child in successors[parent]:
            ancestors[child] |= ancestors[parent] | (1 << parent)
            indegrees[child] -= 1
            if indegrees[child] == 0:
                pending.append(child)
    if visited != len(tasks):
        raise ValueError("tile graph has a dependency cycle")
    check_hazards(tasks, ancestors)
    return Graph(tasks, roots, tuple(tuple(children) for children in successors))
