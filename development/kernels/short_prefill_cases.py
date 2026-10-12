"""Static maxima for legal FIFO mixed steps in the twelve short workloads.

Short prompts retain serial FIFO admission. At batch4, the third step has
624 +31360 +D target rows, then the fifth has624 +784 +D +D. Batch2 ends
with624 +D. D is1/5/8 for ordinary/MTP4/DSpark. The worker orders prefill
rows before decode rows. GDN prefill excludes D, while convolution includes
it. MTP restores one boundary hidden row in the final624-row context.
Prefix contexts restore that row independently for every request.

These shapes derive from the fixed32768 budget and784 block boundary,
not observed traces. Select the maximum legal rows within each distinct
normalization, GEMM, quantization and state path. Preserve historical gates.
"""


def short_prefill_cases(add):
    tail = 32768 % 784
    next_chunk = 32768 // 784 * 784 - 784
    regular = (
        (16384, 5120, False),
        (14336, 5120, False),
        (5120, 6144, False),
        (5120, 17408, True),
    )
    for mode, decode in (("ordinary", 1), ("mtp", 5), ("dspark", 8)):
        for batch in (1, 2, 4):
            workload = f"{mode}-32768-{batch}"
            if batch == 1:
                if mode == "mtp":
                    add(workload, dict(operation="norm", tokens=tail + 1, width=5120))
                if mode == "dspark":
                    add(workload, dict(operation="dspark_rms_norm", tokens=tail))
                continue
            counts = (tail, decode) if batch == 2 else (tail, 784, decode, decode)
            rows = sum(counts)
            context_rows = rows + (mode == "mtp")
            add(workload, dict(operation="norm", tokens=context_rows, width=5120))
            add(workload, dict(operation="add_norm", tokens=rows, width=5120))
            add(
                workload, dict(operation="gated_norm", tokens=rows, heads=48, width=128)
            )
            add(workload, dict(operation="gates", tokens=rows))
            add(workload, dict(operation="convolution", counts=counts))
            add(
                workload,
                dict(operation="silu_quant", tokens=rows, width=17408, column=False),
            )
            for operation, shape in (
                ("add_norm_fp8_linear", dict(width=5120, columns=34816)),
                ("gated_norm_fp8_linear", dict(columns=5120)),
            ):
                add(workload, dict(operation=operation, tokens=rows, **shape))
                if batch == 4:
                    add(
                        workload,
                        dict(
                            operation=operation,
                            tokens=tail + next_chunk + decode,
                            **shape,
                        ),
                    )
            for columns, width, silu in regular:
                config = dict(
                    operation="fp8_linear",
                    tokens=rows,
                    columns=columns,
                    width=width,
                    silu=silu,
                )
                add(workload, config)
                if batch == 4 and (columns == 16384 or silu):
                    add(workload, dict(config, tokens=tail + next_chunk + decode))
            if batch == 4:
                add(workload, dict(operation="gdn_prefill", counts=(tail, 784)))
                add(workload, dict(operation="qk", tokens=tail + 784))
            if mode == "dspark":
                add(workload, dict(operation="dspark_rms_norm", tokens=rows))

    # Every seeded-prefix context has625 rows: restored boundary+624 tail.
    for batch in (1, 2, 4):
        workload = f"prefix-32768-{batch}"
        rows = (tail + 1) * batch
        for config in (
            dict(operation="norm", tokens=rows, width=5120),
            dict(operation="quant", tokens=rows, width=5120, column=False),
            dict(
                operation="fp8_linear",
                tokens=rows,
                columns=2048,
                width=5120,
                silu=False,
            ),
            dict(operation="prepare_context", tokens=rows, index_dtype="int64"),
        ):
            add(workload, config)
