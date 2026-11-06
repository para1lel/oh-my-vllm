import json
from collections import Counter

from .check_deepseek_dep4 import _write_json
from .deepseek_reference_contract import CASE_NAMES, endpoint_schema, selected_schema


def coverage_matches(cases, comparisons, states, *, owns):
    schema = selected_schema(dict(tensors=endpoint_schema()), cases)
    required = {(label, "driver") for label in schema}
    required.update(
        (label, "runtime")
        for label in schema
        for name in cases
        if name not in ("short", "window")
        and label.startswith(name + ".")
        and label.removeprefix(name + ".").startswith(
            ("verify_", "accepted_", "restored_")
        )
    )
    observed = [
        (record["label"], record["path"]) for record in comparisons if "path" in record
    ]
    semantic = Counter()
    for label in schema:
        if label.endswith(".argmax"):
            probe = label.removesuffix(".argmax") + ".sampled_ids"
            semantic[probe] = 2 if ".restored_probe." in label else 1
    for name in cases:
        if name in ("short", "window"):
            continue
        for field in ("base_known125", "base_known126"):
            semantic[name + "." + field] = 4
        for path in ("driver", "runtime"):
            for field in (
                "acceptance_rule",
                "accepted_drafts",
                "accepted_endpoint",
                "correction",
                "probe_endpoint",
                "emitted_ids",
            ):
                semantic[path + "." + name + "." + field] = 1
    observed_semantic = Counter(
        record["label"] for record in comparisons if "path" not in record
    )
    state_names = [record["case"] for record in states]
    expected_states = {name for name in cases if name not in ("short", "window")}
    if not owns:
        return not comparisons and not states
    return (
        len(observed) == len(required)
        and set(observed) == required
        and observed_semantic == semantic
        and len(state_names) == len(expected_states)
        and set(state_names) == expected_states
        and all(
            record["endpoint"]
            == 127
            + (
                5
                if record["case"] == "authentic"
                else int(record["case"].split(".")[1])
            )
            for record in states
        )
        and all(record.get("passed") is True for record in (*comparisons, *states))
    )


def _rank_summaries(args, reference, run, first_identity, shared_identity, ranks):
    for rank in range(4):
        record = json.loads((args.output / f"comparison-rank{rank}.json").read_text())
        proof = json.loads((args.output / f"proof-rank{rank}.json").read_text())
        identity = json.loads((args.output / f"identity-rank{rank}.json").read_text())
        owns = args.strategy != "dep4" or rank == 0
        identity_matches = (
            identity.get("rank") == rank
            and identity.get("gpu_uuid") == args.devices[rank]
            and identity.get("strategy") == args.strategy
            and identity.get("reference") == reference
            and all(
                key in identity and identity[key] == first_identity.get(key)
                for key in shared_identity
            )
        )
        proof_matches = (
            run["outcome"] == "completed"
            and proof.get("run_id") == run["run_id"]
            and proof.get("rank") == rank
            and proof.get("mode") == args.mode
        )
        comparisons, states = record["comparisons"], record["states"]
        ranks.append(
            dict(
                rank=rank,
                passed=coverage_matches(args.cases, comparisons, states, owns=owns)
                and identity_matches
                and proof_matches
                and proof["outcome"] == "completed"
                and proof["counts"]["fallback"] == 0
                and proof["counts"]["failed"] == 0
                and (
                    args.strategy != "dep4"
                    or (
                        proof["counts"]["legacy"] > 0
                        and proof["counts"]["fixed"] == 0
                        and proof["counts"]["completed"] == proof["counts"]["legacy"]
                        and identity["dep4_target_rounds"] > 0
                        and identity["dep4_target_active_ranks"] > 0
                        and identity["owned_cases"]
                        == (list(args.cases) if owns else [])
                    )
                ),
                comparisons=len(comparisons),
                state_cases=len(states),
                coverage_complete=coverage_matches(
                    args.cases, comparisons, states, owns=owns
                ),
                identity_matches=identity_matches,
                proof_matches=proof_matches,
                failed=[
                    row
                    for row in (*comparisons, *states)
                    if row.get("passed") is not True
                ],
                gpu_uuid=identity["gpu_uuid"],
                exchange=proof["counts"],
            )
        )


def summarize(args, reference):
    ranks = []
    run = json.loads((args.output / "run.json").read_text())
    first_identity = json.loads((args.output / "identity-rank0.json").read_text())
    shared_identity = (
        "source_commit",
        "source_digest",
        "native_modules",
        "configuration",
        "inference_configuration",
        "checkpoint",
        "tokenizer",
        "runtime",
        "checker",
        "reference",
        "strategy",
        "execution_modes",
        "execution",
        "math_flags",
    )
    _rank_summaries(args, reference, run, first_identity, shared_identity, ranks)
    passed = all(rank["passed"] for rank in ranks)
    result = dict(
        reference=reference,
        strategy=args.strategy,
        execution="eager",
        graph_verified=False,
        batch=1,
        dep4_dense_owner_rank=0 if args.strategy == "dep4" else None,
        multiple_owners_verified=False,
        cases=list(args.cases),
        complete=tuple(args.cases) == CASE_NAMES,
        passed=passed,
        ranks=ranks,
        scope="independent full-vocabulary endpoints and native cache restoration",
        state_oracle=(
            "fresh accepted-only candidate replay; "
            "independent endpoint fixtures stay read-only"
        ),
        rust_service_verified=False,
    )
    _write_json(args.output / "acceptance.json", result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "ranks"}),
        flush=True,
    )
    if not passed:
        raise AssertionError(
            "independent reference or fresh-replay state comparison failed"
        )
