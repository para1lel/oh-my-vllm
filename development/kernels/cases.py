"""Static maxima from twelve existing workloads and three DSpark workloads.

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
        ("dspark", 32768),
        ("prefix", 32768),
        ("ordinary", 131072),
    ):
        for batch in (1, 2, 4):
            workload = f"{mode}-{length}-{batch}"
            counts = (624,) * batch if mode == "prefix" else PREFILL[length, batch]
            for phase, shape in (
                ("prefill", counts),
                (
                    "decode",
                    (8 if mode == "dspark" else 5 if mode == "mtp" else 1,) * batch,
                ),
            ):
                n = sum(shape)
                add(workload, "norm", tokens=n, width=5120)
                add(workload, "add_norm", tokens=n, width=5120)
                add(
                    workload, "add_norm_fp8_linear", tokens=n, width=5120, columns=34816
                )
                add(workload, "gated_norm", tokens=n, heads=48, width=128)
                # Layer.full_attention is shared by target and MTP. The actual
                # projection chain now fuses Q/K RMS/RoPE, V.contiguous(), and
                # append. Batch.positions/fa_slots and DraftGraph buffers are
                # int64; int32 attention table/length metadata is unrelated.
                add(workload, "prepare_attention", tokens=n, index_dtype="int64")
                for width in (5120, 6144):
                    add(workload, "quant", tokens=n, width=width, column=n <= 32)
                add(workload, "silu_quant", tokens=n, width=17408, column=n <= 32)
                if n > 32:
                    # Other owned regular FP8 projections. Target and MTP use
                    # the same dimensions; <=32 keeps the unchanged TRT backend.
                    for columns, width, silu in (
                        (2048, 5120, False),
                        (16384, 5120, False),
                        (14336, 5120, False),
                        (5120, 6144, False),
                        (5120, 17408, True),
                    ):
                        add(
                            workload,
                            "fp8_linear",
                            tokens=n,
                            width=width,
                            columns=columns,
                            silu=silu,
                        )
                    add(workload, "prepare_context", tokens=n, index_dtype="int64")
                    if mode == "mtp" and batch == 1:
                        # Initial MTP context excludes token position zero.
                        add(
                            workload,
                            "prepare_context",
                            tokens=32143,
                            index_dtype="int64",
                        )
                        add(
                            workload,
                            "fp8_linear",
                            tokens=32143,
                            width=5120,
                            columns=2048,
                            silu=False,
                        )
                add(workload, "gates", tokens=n)
                add(workload, "convolution", counts=shape)
                if phase == "prefill":
                    add(
                        workload, "gdn_prefill", counts=tuple(c for c in shape if c > 1)
                    )
                    add(workload, "qk", tokens=sum(c for c in shape if c > 1))
                else:
                    add(
                        workload,
                        "recurrent",
                        counts=shape,
                        state_dtype="bfloat16"
                        if mode in ("mtp", "dspark")
                        else "float32",
                    )
            # Three active sequences have a distinct GDN tile/thread dispatch.
            # Partial target prefill can contain one endpoint plus the other
            # active verification rows. MTP context keeps at most one endpoint
            # per request. Pure decode uses the existing full preparation path.
            selected_rows = 1 + 8 * (batch - 1) if mode == "dspark" else batch
            add(workload, "prepare_query", tokens=selected_rows, index_dtype="int64")
            if batch == 4:
                add(
                    workload,
                    "recurrent",
                    counts=(8 if mode == "dspark" else 5 if mode == "mtp" else 1,) * 3,
                    state_dtype="bfloat16" if mode in ("mtp", "dspark") else "float32",
                )
            if mode in ("mtp", "dspark"):
                # Proposal graphs, catch-up graphs and graph-cache exhaustion
                # use different metadata dtypes/grouping. Batch3 also exercises
                # the largest grouped shape retaining64 splits (batch4 uses16).
                for active in (batch, 3) if batch == 4 else (batch,):
                    for queries, grouped, table_dtype, length_dtype, extent in (
                        (1, False, "int64", "int64", 36864),
                        (1, False, "int32", "int64", 36864),
                        (8 if mode == "dspark" else 5, True, "int32", "int64", 36864),
                        (8 if mode == "dspark" else 5, False, "int32", "int32", 262144),
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
                            **({"first": 0} if mode == "dspark" else {}),
                        )
            if mode == "dspark":
                # Context injection follows retained target feature rows. The
                # draft always computes seven bidirectional rows, even when a
                # request can retain fewer proposals near its output boundary.
                prefill_rows = sum(PREFILL[length, batch])
                for tokens in (prefill_rows, 8 * batch, 7 * batch):
                    add(workload, "dspark_rms_norm", tokens=tokens)
                for tokens in (prefill_rows, 8 * batch):
                    add(
                        workload,
                        "dspark_norm_rope",
                        tokens=tokens,
                        heads=8,
                        position_base=262144 - tokens,
                        draft=False,
                    )
                    add(workload, "dspark_append", tokens=tokens)
                for heads in (8, 32):
                    add(
                        workload,
                        "dspark_norm_rope",
                        tokens=7 * batch,
                        heads=heads,
                        position_base=262143,
                        draft=True,
                    )
                for active in (batch, 3) if batch == 4 else (batch,):
                    for context in (36864, 262144):
                        add(workload, "dspark_attention", batch=active, length=context)
                        # Preserve all old target gates, then add the new
                        # eight-query target verifier at the context ceiling.
                        add(
                            workload,
                            "attention",
                            batch=active,
                            queries=8,
                            length=262144,
                            grouped=True,
                            table_dtype="int32",
                            length_dtype="int64",
                            max_tokens=262144,
                            first=0,
                        )
    return list(merged.values())


# The quantized checkpoint's MLP uses fused SiLU quantization, and Q/K use
# fused attention preparation. Standalone entries retain existing correctness
# tests, but are not called in the original production performance workloads.
UNUSED = {
    "silu": "fused into silu_quant",
    "rope": "fused into prepare_attention",
    "norm_rope": "Q/K chain fused into prepare_attention",
    "append": "KV write and V layout conversion fused into prepare_attention",
}
