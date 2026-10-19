from .accounting import VERSION, reference_seconds, reference_work
from .trace import canonical_passes


def lower_bound(events, manifest):
    passes = canonical_passes(events, manifest)
    work = reference_work(passes)
    acceptance = {
        row["id"]: dict(tokens=0, accepted=0, proposed=0)
        for row in manifest["requests"]
    }
    for logical in passes:
        for row in logical["requests"]:
            for name in ("tokens", "accepted", "proposed"):
                acceptance[row["id"]][name] += row[name]
    return dict(
        seconds=reference_seconds(work),
        work=work,
        acceptance=acceptance,
        accounting=VERSION,
        passes=len(passes),
    )
