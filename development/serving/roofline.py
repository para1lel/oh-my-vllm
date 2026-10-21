from .accounting import VERSION
from .kernels import reference
from .trace import canonical_passes


def lower_bound(events, manifest):
    passes = canonical_passes(events, manifest)
    calculation = reference(passes)
    acceptance = {
        row["id"]: dict(tokens=0, accepted=0, proposed=0)
        for row in manifest["requests"]
    }
    for logical in passes:
        for row in logical["requests"]:
            for name in ("tokens", "accepted", "proposed"):
                acceptance[row["id"]][name] += row[name]
    return dict(
        **calculation,
        acceptance=acceptance,
        accounting=VERSION,
        passes=len(passes),
    )
