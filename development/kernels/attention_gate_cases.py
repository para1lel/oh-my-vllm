"""Gate shapes follow full Q/K preparation and selected endpoint preparation."""

import hashlib
import json


def attention_gate_cases(matrix):
    merged = {}
    for case in matrix:
        source = case["configuration"]
        if source["operation"] not in ("prepare_attention", "prepare_query"):
            continue
        configuration = {
            "operation": "attention_gate",
            "tokens": source["tokens"],
            "heads": 24,
            "width": 256,
            "packed_width": 14336
            if source["operation"] == "prepare_attention"
            else 12288,
        }
        key = json.dumps(configuration, sort_keys=True)
        row = merged.setdefault(
            key,
            {
                "id": "attention_gate-" + hashlib.sha256(key.encode()).hexdigest()[:10],
                "configuration": configuration,
                "workloads": [],
            },
        )
        for workload in case["workloads"]:
            if workload not in row["workloads"]:
                row["workloads"].append(workload)
    return list(merged.values())
