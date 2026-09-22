"""Static maxima from the twelve accepted workloads; no runtime shape tracing.

32144 = floor(32768/784)*784. A short batch's next admission combines
624 + 32144 rows. Long ordinary batch2 combines144 +32144; batch4 can also
include two decode rows. Prefix hits leave624 rows/request. This mirrors
Scheduler::aligned_prefill and the worker's prefill-before-decode ordering.
"""

import hashlib
import json

# Each tuple is one legal maximum prefill schedule (prefills, then active decodes).
PREFILL = {
    (32768, 1): (32144,),
    (32768, 2): (624, 32144),
    (32768, 4): (624, 32144),
    (131072, 1): (32144,),
    (131072, 2): (144, 32144),
    (131072, 4): (144, 32144, 1, 1),
}


def cases():
    merged = {}

    def add(workload, operation, **configuration):
        config = dict(operation=operation, **configuration)
        key = json.dumps(config, sort_keys=True)
        if key not in merged:
            merged[key] = dict(
                id=operation + "-" + hashlib.sha256(key.encode()).hexdigest()[:10],
                configuration=config,
                workloads=[],
            )
        if workload not in merged[key]["workloads"]:
            merged[key]["workloads"].append(workload)

    for mode, length in (
        ("ordinary", 32768),
        ("mtp", 32768),
        ("prefix", 32768),
        ("ordinary", 131072),
    ):
        for batch in (1, 2, 4):
            workload = f"{mode}-{length}-{batch}"
            counts = (624,) * batch if mode == "prefix" else PREFILL[length, batch]
            for phase, shape in (
                ("prefill", counts),
                ("decode", (5 if mode == "mtp" else 1,) * batch),
            ):
                n = sum(shape)
                add(workload, "norm", tokens=n, width=5120)
                add(workload, "add_norm", tokens=n, width=5120)
                add(workload, "gated_norm", tokens=n, heads=48, width=128)
                for heads in (4, 24):
                    add(workload, "norm_rope", tokens=n, heads=heads)
                for width in (5120, 6144):
                    add(workload, "quant", tokens=n, width=width, column=n <= 32)
                add(workload, "silu_quant", tokens=n, width=17408, column=n <= 32)
                add(workload, "gates", tokens=n)
                add(workload, "append", tokens=n)
                add(workload, "convolution", counts=shape)
                if phase == "prefill":
                    add(workload, "qk", tokens=sum(c for c in shape if c > 1))
                else:
                    add(
                        workload,
                        "recurrent",
                        counts=shape,
                        state_dtype="bfloat16" if mode == "mtp" else "float32",
                    )
            # Three active sequences have a distinct GDN tile/thread dispatch.
            if batch == 4:
                add(
                    workload,
                    "recurrent",
                    counts=(5 if mode == "mtp" else 1,) * 3,
                    state_dtype="bfloat16" if mode == "mtp" else "float32",
                )
            if mode == "mtp":
                # Proposal graphs, catch-up graphs and graph-cache exhaustion
                # use different metadata dtypes/grouping. Batch3 also exercises
                # the largest grouped shape retaining64 splits (batch4 uses16).
                for active in (batch, 3) if batch == 4 else (batch,):
                    for queries, grouped, table_dtype, length_dtype, extent in (
                        (1, False, "int64", "int64", 36864),
                        (1, False, "int32", "int64", 36864),
                        (5, True, "int32", "int64", 36864),
                        (5, False, "int32", "int32", 262144),
                    ):
                        add(
                            workload,
                            "attention",
                            batch=active,
                            queries=queries,
                            length=36864,
                            grouped=grouped,
                            table_dtype=table_dtype,
                            length_dtype=length_dtype,
                            max_tokens=extent,
                        )
    return list(merged.values())


# The quantized checkpoint's MLP uses fused SiLU quantization, and Q/K use
# fused RMS/RoPE. Standalone silu/rotary remain covered by existing correctness
# tests, but are not called in the twelve production performance workloads.
UNUSED = {"silu": "fused into silu_quant", "rope": "fused into norm_rope"}
