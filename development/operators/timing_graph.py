from contextlib import AbstractContextManager, ExitStack
from time import perf_counter

from torch.cuda.graphs import register_graph_instantiate_hook


def _graph_node_count(graph):
    data = graph.get_graph_data()
    nodes = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(nodes, list) or not nodes:
        raise RuntimeError("CUDA Graph has no nodes")
    return len(nodes)


def _capture_nodes(captures):
    def capture(graph):
        try:
            captures.append(_graph_node_count(graph))
        except BaseException as failure:
            captures.append(failure)

    return capture


def _verified_nodes(captures):
    if len(captures) != 1:
        raise RuntimeError("expected exactly one CUDA Graph instantiation")
    captured = captures[0]
    if isinstance(captured, BaseException):
        raise RuntimeError(f"cannot inspect CUDA Graph nodes: {captured}") from captured
    return captured


class TimingGraph(AbstractContextManager):
    def __init__(self, prepared, events, repeats):
        self.device = prepared.device
        self.events = events
        self.resources = ExitStack()
        started = perf_counter()
        prepared.reset()
        self.device.synchronize()
        captures = []
        handle = register_graph_instantiate_hook(_capture_nodes(captures))
        try:
            try:
                self.resources.enter_context(prepared.graph_lease())
                replay = self.resources.enter_context(self.device.graph(prepared.run))
            finally:
                handle.remove()
            nodes = _verified_nodes(captures)
        except BaseException:
            self.resources.close()
            raise
        self.replay = replay
        self.repeats = repeats
        prepared.setup.setdefault("timing_graphs", []).append(
            {
                "seconds": perf_counter() - started,
                "repeats": repeats,
                "production_nodes": nodes,
                "graph_validation": "get-graph-data",
            }
        )

    def __call__(self):
        self.device.ensure_healthy()
        self.events.begin()
        for _ in range(self.repeats):
            self.replay()
        self.events.record_end()

    def __exit__(self, *_):
        self.resources.close()
