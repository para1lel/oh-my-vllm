"""Static supplemental maxima for the three long-MTP production workloads.

The scheduler divides 32768 by admitted long requests, then rounds down to
784. Batch one retains its ordinary 32144-row chunk. All equal cold prompts
advance together, including the final full block and 144-token tail.
The first MTP context excludes position zero per sequence. Later chunks do
not. MTP catch-up includes the final sampled anchor; target KV does not.
No measured trace selects these cases. The original matrix supplies unchanged
operation geometry, while this module derives new legal rows and contexts.
"""

import hashlib
import json

from .short_prefill_cases import short_prefill_cases

BLOCK = 784
STEP_BUDGET = 32768
INPUT = 131072
OUTPUT = 4096


def long_prefill_cases(original):
    rows = {}

    def add(workload, configuration):
        key = json.dumps(configuration, sort_keys=True)
        row = rows.setdefault(
            key,
            {
                "id": configuration["operation"]
                + "-"
                + hashlib.sha256(key.encode()).hexdigest()[:10],
                "configuration": configuration,
                "workloads": [],
            },
        )
        if workload not in row["workloads"]:
            row["workloads"].append(workload)

    # Batch-one cold prefill supplies each existing operation's fixed widths,
    # precision, fused-chain boundary and metadata types. It excludes attention
    # length cases, endpoint-only query preparation, and the 32143-row context.
    templates = []
    for row in original:
        config = row["configuration"]
        if "mtp-32768-1" not in row["workloads"]:
            continue
        if config.get("tokens") == 32144 or config.get("counts") == (32144,):
            templates.append(config)

    for batch in (1, 2, 4):
        workload = f"mtp-{INPUT}-{batch}"
        quota = STEP_BUDGET // batch // BLOCK * BLOCK
        counts = (quota,) * batch
        tokens = sum(counts)
        for template in templates:
            config = dict(template)
            if "tokens" in config:
                config["tokens"] = tokens
            if "counts" in config:
                config["counts"] = counts
            add(workload, config)

        # Context-only normalization, quantization, KV projection and append
        # execute on shifted cold rows as well as subsequent full chunks.
        cold = tokens - batch
        for config in (
            dict(operation="norm", tokens=cold, width=5120),
            dict(operation="quant", tokens=cold, width=5120, column=False),
            dict(
                operation="fp8_linear",
                tokens=cold,
                width=5120,
                columns=2048,
                silu=False,
            ),
            dict(operation="prepare_context", tokens=cold, index_dtype="int64"),
        ):
            add(workload, config)

        # The last aligned checkpoint is167 blocks. B1 reaches it with a
        # three-block2352-row chunk after four full chunks. Its medium norm,
        # convolution, Q/K and one-SM clustered GEMM paths differ from32144.
        boundary = INPUT // BLOCK * BLOCK
        last_aligned = boundary % quota or quota
        if batch == 1:
            medium_ops = {
                "norm",
                "add_norm",
                "add_norm_fp8_linear",
                "gated_norm",
                "gated_norm_fp8_linear",
                "silu_quant",
                "fp8_linear",
                "gates",
                "convolution",
                "gdn_prefill",
                "qk",
            }
            for template in templates:
                if template["operation"] not in medium_ops:
                    continue
                if (
                    template["operation"] == "fp8_linear"
                    and template["columns"] == 2048
                ):
                    continue
                config = dict(template)
                if "tokens" in config:
                    config["tokens"] = last_aligned
                if "counts" in config:
                    config["counts"] = (last_aligned,)
                add(workload, config)

        # Each target tail has144 rows. Restoring the checkpoint's boundary
        # hidden row gives MTP145 context rows, ending at INPUT+1. All context
        # rows use RMS; only selected endpoints use the subsequent MLP.
        tail = INPUT - boundary
        add(workload, dict(operation="norm", tokens=(tail + 1) * batch, width=5120))
        tail_ops = {"add_norm", "add_norm_fp8_linear", "gdn_prefill", "qk"}
        if batch == 1:
            tail_ops.add("gated_norm_fp8_linear")
        else:
            tail_ops.update({"gated_norm", "silu_quant", "gates", "convolution"})
        for template in templates:
            operation = template["operation"]
            projection = operation == "fp8_linear" and (
                template["silu"] or (batch == 1 and template["columns"] == 16384)
            )
            if operation not in tail_ops and not projection:
                continue
            config = dict(template)
            if "tokens" in config:
                config["tokens"] = tail * batch
            if "counts" in config:
                config["counts"] = (tail,) * batch
            add(workload, config)

        # Remaining output can clip verification to anchor+two draft rows.
        # Three active single-row proposals also reach the <=3 warp path.
        for config in (
            dict(operation="add_norm_fp8_linear", tokens=3, width=5120, columns=34816),
            dict(operation="quant", tokens=3, width=5120, column=True),
            dict(operation="quant", tokens=3, width=6144, column=True),
            dict(operation="silu_quant", tokens=3, width=17408, column=True),
        ):
            add(workload, config)

        # These are owned first=1 MTP paths. Target first=0 attention uses its
        # independently checked Torch/CUDA route and has final length 135167.
        # Proposal: generated <=4091, anchor+three forwarded draft rows.
        # Catch-up: the final accepted anchor can reach length 135168.
        for active in (batch, 3) if batch == 4 else (batch,):
            configurations = [
                (1, False, "int64", "int64", INPUT + OUTPUT - 2),
                (1, False, "int32", "int64", INPUT + OUTPUT),
                (5, True, "int32", "int64", INPUT + OUTPUT),
                (5, False, "int32", "int32", INPUT + OUTPUT),
            ]
            if active == 4:
                # A single-row fallback at batch4 reaches the dual-KV-buffer
                # branch that the five-query, twenty-row shape cannot cover.
                configurations.append((1, False, "int32", "int32", INPUT + OUTPUT))
            if active == batch:
                # Selected context endpoints use this metadata at prefill end.
                # Equal cold prompts finish together, without a batch3 endpoint.
                configurations.append((1, False, "int32", "int32", INPUT + 1))
            for queries, grouped, table, lengths, length in configurations:
                add(
                    workload,
                    dict(
                        operation="attention",
                        batch=active,
                        queries=queries,
                        length=length,
                        grouped=grouped,
                        table_dtype=table,
                        length_dtype=lengths,
                        max_tokens=INPUT + OUTPUT,
                        first=1,
                    ),
                )
    short_prefill_cases(add)
    # Five accepted target inputs can end exactly at an FA page boundary.
    # The next context restores its hidden row and retains five new rows.
    # draft_context projects all six KV rows, with column-major small scales;
    # attention and the MLP still process only one endpoint per sequence.
    for batch in (1, 2, 4):
        config = dict(operation="quant", tokens=6 * batch, width=5120, column=True)
        for mode, length in (("mtp", 32768), ("prefix", 32768), ("mtp", INPUT)):
            add(f"{mode}-{length}-{batch}", config)
    return list(rows.values())
