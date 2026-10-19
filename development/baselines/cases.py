import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from development.kernels.workload import modeled_cases


@dataclass(frozen=True)
class CoreCase:
    operator: str
    queries: tuple[int, ...]
    contexts: tuple[int, ...]
    positions: tuple[int, ...]
    phase: str = "auto"

    @property
    def batch(self):
        return len(self.queries)

    @property
    def rows(self):
        return sum(self.queries)

    @property
    def verify(self):
        if self.phase != "auto":
            return self.phase == "recurrent"
        return max(self.queries) <= 5

    @property
    def identity(self):
        data = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(data).hexdigest()[:16]


def add_case(cases, case, origin):
    if case.operator not in {"fa_attention", "gdn_attention"}:
        raise ValueError("core case must select FA or GDN")
    if case.phase not in {"auto", "prefill", "recurrent"}:
        raise ValueError("core phase must select prefill or recurrent execution")
    if not case.queries or any(value < 1 for value in case.queries):
        raise ValueError("core case must have nonempty sequences")
    if len(case.contexts) != case.batch or len(case.positions) != case.batch:
        raise ValueError("core case sequence metadata differs in length")
    if any(
        context < query or position < 0
        for query, context, position in zip(
            case.queries, case.contexts, case.positions, strict=True
        )
    ):
        raise ValueError("invalid core context or position")
    entry = cases.setdefault(case, set())
    entry.add(origin)


def modeled_manifest():
    cases = {}
    for item in modeled_cases():
        shape = item.shape
        if shape.operator not in {"fa_attention", "gdn_attention"}:
            continue
        queries = shape.rows // shape.batch
        contexts = max(queries, shape.context)
        case = CoreCase(
            shape.operator,
            (queries,) * shape.batch,
            (contexts,) * shape.batch,
            (shape.position,) * shape.batch,
            "recurrent" if queries <= 5 else "prefill",
        )
        add_case(cases, case, json.dumps({"shape": asdict(shape)}, sort_keys=True))
    return cases


TRACE_POLICY = (
    "all HTTP prefills; first/middle/final decode per configuration and batch pattern"
)

HTTP_CONFIGURATIONS = {
    (mode, batch, prompt, output)
    for mode, prompts in (
        ("ordinary", (32768,)),
        ("mtp4", (32768, 65536, 131072)),
    )
    for batch in (1, 2, 4)
    for prompt in prompts
    for output in (64, 8192)
}


def http_requests(paths):
    requests = {}
    configurations = set()
    for path in paths:
        result = json.loads(Path(path).read_text())
        configuration = tuple(
            result[name] for name in ("mode", "batch", "input_tokens", "output_tokens")
        )
        if configuration not in HTTP_CONFIGURATIONS:
            raise ValueError("HTTP result is outside the required 24 configurations")
        if configuration in configurations:
            raise ValueError("HTTP result files repeat a configuration")
        configurations.add(configuration)
        ids = [int(row["id"].rsplit("_", 1)[1]) for row in result["responses"]]
        if len(set(ids)) != configuration[1]:
            raise ValueError("HTTP result request IDs differ from its batch")
        for request in ids:
            key = (configuration[0], request)
            if key in requests:
                raise ValueError("HTTP result files repeat a request ID")
            requests[key] = configuration
    return requests


def selected_events(path, requests):
    from development.serving.trace import validate

    events = [
        (number, json.loads(line))
        for number, line in enumerate(Path(path).read_text().splitlines(), 1)
        if line.strip()
    ]
    identities = [event for _, event in events if event["operation"] == "identity"]
    if len(identities) != 1:
        raise ValueError("trace must contain one serving identity")
    proposals = identities[0]["config"]["num_speculative_tokens"]
    if proposals not in {0, 4}:
        raise ValueError("trace must use ordinary or MTP4 serving")
    mode = "mtp4" if proposals else "ordinary"
    configurations = {}
    for _, event in events:
        for row in event.get("requests", ()):
            key = (mode, row["id"])
            if key in requests:
                configurations.setdefault(requests[key], set()).add(row["id"])
    for configuration, request_ids in configurations.items():
        validate([event for _, event in events], request_ids, configuration[3])
    prefills, decode = [], {}
    for number, event in events:
        operation = event["operation"]
        if operation not in {"target", "draft"}:
            continue
        rows = [row for row in event["requests"] if (mode, row["id"]) in requests]
        if not rows:
            continue
        configurations = {requests[(mode, row["id"])] for row in rows}
        if len(configurations) != 1 or len(rows) != len(event["requests"]):
            raise ValueError("trace event crosses HTTP configuration boundaries")
        configuration = next(iter(configurations))
        event = {**event, "requests": rows}
        item = (number, event, configuration)
        if operation == "target" and not event["verify"]:
            prefills.append(item)
        else:
            pattern = tuple(row.get("count", 1) for row in rows)
            key = (configuration, operation, event.get("stage"), pattern)
            decode.setdefault(key, []).append(item)
    selected = [(item, "all_prefills") for item in prefills]
    for items in decode.values():
        for index in sorted({0, len(items) // 2, len(items) - 1}):
            selected.append(
                (items[index], f"decode_representative:{index}/{len(items)}")
            )
    return sorted(selected, key=lambda item: item[0][0])


def add_trace(cases, path, requests):
    seen = set()
    for (number, event, configuration), selection in selected_events(path, requests):
        operation, rows = event["operation"], event["requests"]
        seen.update((configuration[0], row["id"]) for row in rows)
        if any(row.get("prompt", configuration[2]) != configuration[2] for row in rows):
            raise ValueError("trace prompt lengths differ from the HTTP configuration")
        if operation == "target":
            queries = tuple(row["count"] for row in rows)
            positions = tuple(row["start"] for row in rows)
        else:
            queries = (1,) * len(rows)
            positions = tuple(row["position"] for row in rows)
        contexts = tuple(
            position + count for position, count in zip(positions, queries, strict=True)
        )
        operators = (
            ("fa_attention", "gdn_attention")
            if operation == "target"
            else ("fa_attention",)
        )
        for operator in operators:
            phase = (
                "recurrent"
                if operation == "draft" or event.get("verify")
                else "prefill"
            )
            add_case(
                cases,
                CoreCase(operator, queries, contexts, positions, phase),
                json.dumps(
                    {
                        "trace": Path(path).name,
                        "line": number,
                        "http_configuration": configuration,
                        "selection": selection,
                    },
                    sort_keys=True,
                ),
            )
    return seen


def manifest(traces=(), http_results=()):
    if bool(traces) != bool(http_results):
        raise ValueError("supply both serving traces and their HTTP result files")
    cases = modeled_manifest()
    requests = http_requests(http_results)
    seen = set()
    for trace in traces:
        seen.update(add_trace(cases, trace, requests))
    if seen != set(requests):
        raise ValueError("trace coverage is incomplete for the HTTP result request IDs")
    return [
        {"id": case.identity, "case": asdict(case), "origins": sorted(origins)}
        for case, origins in sorted(
            cases.items(),
            key=lambda item: (
                item[0].operator,
                item[0].queries,
                item[0].contexts,
                item[0].positions,
                item[0].phase,
            ),
        )
    ]


def from_record(record):
    return CoreCase(
        record["operator"],
        tuple(record["queries"]),
        tuple(record["contexts"]),
        tuple(record["positions"]),
        record.get("phase", "auto"),
    )
