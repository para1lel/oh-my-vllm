import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

CASE_PATH = Path(__file__).with_name("cases.json")
BOUNDARIES = {
    "fa",
    "gdn",
    "ffn",
    "output.hidden",
    "output.distribution",
    "output.sample",
    "mtp.context",
    "mtp.draft",
    "mtp.chain",
    "mtp.accept",
    "vision.input",
    "vision.block",
    "vision.merge",
}
CASE_COUNTS = {
    (boundary, purpose): 3
    if boundary.startswith("vision.") or boundary == "mtp.chain"
    else 5
    for boundary in BOUNDARIES
    for purpose in ("numerics", "performance")
}


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Case:
    id: str
    boundary: str
    purpose: str
    shape: dict
    mode: str
    weights: tuple
    options: dict
    outputs: tuple
    fixture_sha256: str | None = None

    def definition(self):
        value = asdict(self)
        value.pop("fixture_sha256")
        return json.loads(json.dumps(value, allow_nan=False))

    @property
    def identity(self):
        return digest(self.definition())


def validate_cases(cases):
    cases = tuple(cases)
    counts = Counter((case.boundary, case.purpose) for case in cases)
    if any(count > 5 for count in counts.values()):
        raise ValueError("each boundary permits at most five fixed cases per purpose")
    identities = set()
    for case in cases:
        if case.id in identities:
            raise ValueError("duplicate case identity")
        identities.add(case.id)
        if not case.id or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for char in case.id
        ):
            raise ValueError("invalid case identity")
        if case.boundary not in BOUNDARIES or case.purpose not in {
            "numerics",
            "performance",
        }:
            raise ValueError("unknown case boundary or purpose")
        if case.mode not in {"eager", "graph"}:
            raise ValueError("unknown execution mode")
        if not case.shape or any(
            type(value) is not int or value < 1 for value in case.shape.values()
        ):
            raise ValueError("case dimensions must be positive integers")
        if not case.outputs or len(set(case.outputs)) != len(case.outputs):
            raise ValueError("case outputs must be unique and nonempty")
        if {"batch", "rows", "queries"} <= case.shape.keys() and (
            case.shape["rows"] != case.shape["batch"] * case.shape["queries"]
        ):
            raise ValueError("case query rows differ from batch dimensions")
        if case.boundary.startswith("vision.") and case.mode != "eager":
            raise ValueError("vision cases execute eagerly")
        if case.fixture_sha256 is not None and (
            len(case.fixture_sha256) != 64
            or any(c not in "0123456789abcdef" for c in case.fixture_sha256)
        ):
            raise ValueError("invalid fixture identity")
    if counts != CASE_COUNTS:
        raise ValueError("fixed case coverage is incomplete")
    return tuple(cases)


def load_cases(path=CASE_PATH):
    data = json.loads(Path(path).read_text())
    if set(data) != {"version", "cases"} or data["version"] != "operator-cases-v1":
        raise ValueError("unsupported operator case catalog")
    return validate_cases(
        Case(
            **{
                **entry,
                "weights": tuple(entry["weights"]),
                "outputs": tuple(entry["outputs"]),
            }
        )
        for entry in data["cases"]
    )
