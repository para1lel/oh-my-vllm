"""Check local MoE composition, not model acceptance or collective execution.

Encoded packets borrow reusable workspace buffers. Copy local slices before
encoding again, and retain external packet buffers through graph destruction.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np

from development.operators.performance_inputs import bf16
from development.operators.storage import decode, equal_bits

from .reference_quantization import activation_fp8


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def selected_experts(draft=False):
    return (
        ([0, 31, 127], [32, 63, 64])
        if draft
        else ([0, 1, 95, 96, 191, 383], [2, 3, 94, 97, 190, 382])
    )


def split_inputs(rows):
    host = np.zeros((rows, 5120), np.float32)
    host[:, 0] = np.resize(np.array([1, 2, 3, 4, 6, 8], np.float32), rows)
    return host, ((np.arange(rows) // 2) % 2).astype(np.uint8)


def packet_arrays(host, mask):
    rows = len(host)
    quantized = activation_fp8(host) if rows else None
    return (
        quantized.encoded if rows else np.empty((0, 5120), np.uint8),
        quantized.scales if rows else np.empty((0, 160), np.uint8),
        np.zeros((rows, 160), np.uint8),
        np.asarray(selected_experts(), np.int32)[mask],
        np.full((rows, 6), 0.25, np.float32),
    )


def expected_outputs(host, mask, selected, start, count, width, shared_width):
    local_weight = np.array(
        [
            sum(
                1 + expert % 2
                for expert in selected[m]
                if start <= expert < start + count
            )
            for m in mask
        ],
        np.float32,
    )
    route = 1.5 / len(selected[0])
    return (
        (10 * route * host[:, 0] * width * local_weight)[:, None],
        (10 * host[:, 0] * shared_width)[:, None],
    )


def check_cpu():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU check must hide CUDA")
    tags, _ = split_inputs(6)
    for factor in (1, 2.5, 10):
        values = np.zeros((6, 32), np.float32)
        values[:, 0] = factor * tags[:, 0]
        require(
            np.array_equal(activation_fp8(values).dequantized, values),
            "synthetic tags change under FP8 quantization",
        )
    for width in (576, 2304):
        for factor in (2.5, 5, 10):
            values = factor * tags[:, 0] * width
            require(
                np.array_equal(decode(bf16(values), "bfloat16"), values),
                "synthetic down projection changes under BF16 rounding",
            )
    for counts in ((0, 1, 2, 3), (0, 1, 127, 256)):
        host, mask = split_inputs(sum(counts))
        packet = packet_arrays(host, mask)
        require(
            [value.shape for value in packet]
            == [(sum(counts), width) for width in (5120, 160, 160, 6, 6)],
            "packet domain shapes differ",
        )
        for rank in range(4):
            routed, shared = expected_outputs(
                host, mask, selected_experts(), rank * 96, 96, 2304, 2304
            )
            require(
                routed.dtype == np.float32
                and routed.shape == shared.shape == (sum(counts), 1),
                "oracle output domains differ",
            )
        empty = packet_arrays(host[:0], mask[:0])
        require(all(len(field) == 0 for field in empty), "empty owner packet differs")
    print(json.dumps(dict(oracle_passed=True)), flush=True)


def source_identity():
    root = Path(__file__).resolve().parents[3]
    paths = [
        Path(__file__),
        root / "python/oh_my_vllm/models/deepseek_v41/moe.py",
        root / "python/oh_my_vllm/models/deepseek_v41/workspace.py",
        Path(__file__).with_name("reference_quantization.py"),
        Path(__file__).with_name("quantization.py"),
        *(
            root / "development/operators" / f"{name}.py"
            for name in ("check_deepseek_exact", "performance_inputs", "storage")
        ),
        *(root / "development/serving" / f"{name}.py" for name in ("device", "gpu")),
    ]
    for family in (
        "activation",
        "linear",
        "projection",
        "quantization",
        "routing",
        "runtime",
    ):
        paths.extend((root / "python/oh_my_vllm/operators" / family).rglob("*.py"))
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def weight(device, bits, outputs, width, code, scale, sparse):
    from oh_my_vllm.operators.projection.api import Weights
    from oh_my_vllm.operators.projection.workspace import prepare_weights

    raw = np.zeros((outputs, width * bits // 8), np.uint8)
    if sparse:
        raw[:, 0] = code
    else:
        raw.fill(code | (code << 4) if bits == 4 else code)
    scales = np.full(
        (outputs if bits == 4 else outputs // 32, width // 32), scale, np.uint8
    )
    return prepare_weights(
        device, Weights(device.upload(raw), device.upload(scales)), bits=bits
    )


def owner_weights(device, strategy, rank, draft):
    from oh_my_vllm.models.deepseek_v41.placement import RankPlan
    from oh_my_vllm.models.deepseek_v41.weights import RankWeights

    experts = 128 if draft else 384
    name = "mtp.0" if draft else "layers.0"
    prefix = f"{name}.ffn"
    width = 576 if strategy == "TP4" else 2304
    shared_width = 2304 if strategy == "DEP4" else 576
    plan = RankPlan(MappingProxyType({}), MappingProxyType({}), 0, 0, strategy, rank)
    owner = RankWeights(device, plan, 1 << 30, 0)
    owner.tensors[f"{prefix}.gate.weight"] = device.upload(
        bf16(np.zeros((experts, 5120), np.float32)), "bfloat16"
    )
    selected = selected_experts(draft)
    for suffix, ids in zip(("bias", "bias_vl"), selected, strict=True):
        bias = np.zeros(experts, np.float32)
        bias[ids] = np.arange(len(ids), 0, -1, dtype=np.float32)
        owner.tensors[f"{prefix}.gate.{suffix}"] = device.upload(bias)
    count = experts if strategy == "TP4" else experts // 4
    start = 0 if strategy == "TP4" else rank * count
    # Exact dyadic weights: w1*x=16*x[0], w3*x=x[0], w2 sums channels.
    routed = {
        "w1": weight(device, 4, width, 5120, 6, 129, True),
        "w3": weight(device, 4, width, 5120, 2, 127, True),
        "w2": weight(device, 4, 5120, width, 2, 127, False),
    }
    odd_down = weight(device, 4, 5120, width, 4, 127, False)
    for expert in range(start, start + count):
        for projection, value in routed.items():
            owner.packed[f"{prefix}.experts.{expert}.{projection}.weight"] = (
                odd_down if projection == "w2" and expert % 2 else value
            )
    for projection, outputs, columns, code, sparse in (
        ("w1", shared_width, 5120, 0x58, True),
        ("w3", shared_width, 5120, 0x38, True),
        ("w2", 5120, shared_width, 0x38, False),
    ):
        owner.packed[f"{prefix}.shared_experts.{projection}.weight"] = weight(
            device, 8, outputs, columns, code, 127, sparse
        )
    return owner, name, selected, start, count, width, shared_width


def check_case(device, strategy, rank, draft, rows):
    from oh_my_vllm.models.deepseek_v41.moe import prepare

    from development.operators.check_deepseek_exact import native_identity, rejected

    owner, name, selected, start, count, width, shared_width = owner_weights(
        device, strategy, rank, draft
    )
    workspace = None
    try:
        workspace = prepare(owner, name, rows=rows)
        host = np.zeros((rows, 5120), np.float32)
        host[:, 0] = 1 + np.arange(rows) % 2
        mask = (np.arange(rows) % 2).astype(np.uint8)
        value = device.upload(bf16(host), "bfloat16")
        image_mask = device.upload(mask)
        # The clamped gate result rounds to exactly ten in BF16.
        # Top-k probabilities are equal; routes are 1.5/6 or 1.5/3.
        routed, shared = expected_outputs(
            host, mask, selected, start, count, width, shared_width
        )

        def check(invalid=False, values=None):
            flags = [
                device.download(status, np.uint8) for status in workspace.outputs.status
            ]
            if invalid:
                require(
                    flags[0][0, 0] == 1 and flags[1][0] == 1,
                    "invalid MoE input accepted",
                )
                require(
                    not flags[0][1:].any() and not flags[1][1:].any(),
                    "invalid MoE row affected other rows",
                )
            else:
                require(
                    not any(flag.any() for flag in flags), "valid MoE status differs"
                )
            if values is None:
                values = workspace.outputs.routed, workspace.outputs.shared
            for output, expected, dtype in zip(
                values, (routed, shared), (np.float32, np.uint16), strict=True
            ):
                actual = device.download(output, dtype)
                if dtype == np.uint16:
                    actual = decode(actual, "bfloat16")
                if invalid:
                    actual, expected = actual[1:], expected[1:]
                require(
                    np.array_equal(actual, np.broadcast_to(expected, actual.shape)),
                    "MoE output differs",
                )

        def operation():
            return workspace.run(value, image_mask)

        operation()
        check()
        if not draft and rows == 2:
            packet = workspace.encode_routes(value, image_mask)
            shared_branch = workspace.run_shared(packet)
            routed_branch = workspace.run_routed(packet)
            check(values=(routed_branch.value, shared_branch.value))
            require(
                shared_branch.status[0] is packet.status
                and routed_branch.status[0] is packet.status
                and all(
                    not device.download(flag, np.uint8).any()
                    for branch in (shared_branch, routed_branch)
                    for flag in branch.status
                ),
                "equal-domain staged status differs",
            )
            if strategy != "DEP4":
                rejected(lambda: prepare(owner, name, rows=rows, routed_rows=rows + 1))
        if (strategy, draft, rows) == ("TP4", False, 2):
            rejected(lambda: prepare(owner, name, rows=rows, byte_limit=1))
            rejected(lambda: prepare(owner, "layers.40", rows=rows))
            rejected(lambda: workspace.run(workspace.outputs.shared, image_mask))

            def overlapping(shape, dtype):
                if tuple(shape) == (rows, 5120) and dtype == "bfloat16":
                    return value
                return device.empty(shape, dtype)

            rejected(lambda: prepare(owner, name, rows=rows, allocate=overlapping))
            operation()
            check()
        with workspace.graph_lease(), device.graph(operation) as replay:
            replay()
            check()
            if rows:
                invalid = bf16(host)
                invalid[0, 0] = 0x7FC0
                device.module.copy_bytes(value, device.upload(invalid, "bfloat16"))
                replay()
                check(invalid=True)
                device.module.copy_bytes(value, device.upload(bf16(host), "bfloat16"))
                replay()
                check()
            workspace.close()
            owner.close()
            replay()
            check()
        rejected(operation)
        rejected(lambda: prepare(owner, name, rows=rows))
        return dict(
            strategy=strategy,
            rank=rank,
            draft=draft,
            rows=rows,
            workspace_bytes=workspace.allocated_bytes,
            native=[native_identity(part) for part in workspace.parts],
        )
    finally:
        device.synchronize_compute()
        if workspace is not None:
            workspace.close()
        owner.close()


def download_packet(device, packet):
    return tuple(
        device.download(value, dtype)
        for value, dtype in zip(
            packet, (np.uint8, np.uint8, np.uint8, np.int32, np.float32), strict=True
        )
    )


def check_split_case(device, rank, counts):
    from oh_my_vllm.models.deepseek_v41.moe import prepare
    from oh_my_vllm.operators.routing.api import EncodedDispatchInputs

    from development.operators.check_deepseek_exact import native_identity, rejected

    rows, routed_rows = counts[rank], sum(counts)
    first, last = sum(counts[:rank]), sum(counts[: rank + 1])
    owner, name, selected, start, count, width, shared_width = owner_weights(
        device, "DEP4", rank, False
    )
    workspace = None
    try:
        workspace = prepare(owner, name, rows=rows, routed_rows=routed_rows)
        require(
            workspace.plan.rows == rows
            and workspace.dispatch.plan.rows == routed_rows
            and workspace.gate.plan.rows == rows
            and all(part.plan.rows == rows for part in workspace.shared.values())
            and all(
                part.plan.capacity == routed_rows * 6
                for part in workspace.routed.values()
            ),
            "local/routed preparation domains differ",
        )
        host, mask = split_inputs(routed_rows)
        reference = packet_arrays(host, mask)
        value = device.upload(bf16(host[first:last]), "bfloat16")
        image_mask = device.upload(mask[first:last])
        gathered = EncodedDispatchInputs(*(device.upload(field) for field in reference))
        destinations = [
            device.view(field, (rows, field.shape[1]), first * field.shape[1])
            for field in gathered
        ]
        expected_routed, _ = expected_outputs(
            host, mask, selected, start, count, width, shared_width
        )
        _, expected_shared = expected_outputs(
            host[first:last],
            mask[first:last],
            selected,
            start,
            count,
            width,
            shared_width,
        )
        remote_row = next(row for row in range(routed_rows) if not first <= row < last)
        # Only check mutation here; frozen routing checks cover coefficient accuracy.
        coefficients = device.download(
            workspace.encode_routes(value, image_mask).coefficients, np.float32
        )
        packet = shared = routed = None

        def operation():
            nonlocal packet, shared, routed
            packet = workspace.encode_routes(value, image_mask)
            for destination, source in zip(destinations, packet, strict=True):
                device.module.copy_bytes(destination, source)
            shared = workspace.run_shared(packet)
            routed = workspace.run_routed(gathered)

        def check(local_bad=False, remote_bad=False):
            require(
                shared.status[0] is packet.status
                and routed.status[0] is gathered.status,
                "branch did not retain supplied packet status by identity",
            )
            local_status = np.zeros((rows, 160), np.uint8)
            global_status = np.zeros((routed_rows, 160), np.uint8)
            dispatch_status = np.zeros(routed_rows, np.uint8)
            bad_row = first if local_bad else remote_row
            if local_bad:
                local_status[0, 0] = 1
            if local_bad or remote_bad:
                global_status[bad_row, 0] = 1
                dispatch_status[bad_row] = 1
            shared_flags = [device.download(flag, np.uint8) for flag in shared.status]
            routed_flags = [device.download(flag, np.uint8) for flag in routed.status]
            require(
                len(shared_flags) == 2
                and equal_bits(shared_flags[0], local_status)
                and equal_bits(
                    shared_flags[1], np.zeros((rows, shared_width // 32), np.uint8)
                ),
                "shared local status differs",
            )
            expected_flags = [
                global_status,
                dispatch_status,
                *(np.zeros(count, np.uint8) for _ in range(3)),
                np.zeros((routed_rows * 6, width // 32), np.uint8),
                dispatch_status,
            ]
            for name, actual, wanted in zip(
                ("input", "dispatch", "w1", "w3", "w2", "activation", "combine"),
                routed_flags,
                expected_flags,
                strict=True,
            ):
                require(equal_bits(actual, wanted), f"routed {name} status differs")
            actual_shared = decode(device.download(shared.value, np.uint16), "bfloat16")
            actual_routed = device.download(routed.value, np.float32)
            wanted_routed = np.broadcast_to(expected_routed, (routed_rows, 5120)).copy()
            if local_bad or remote_bad:
                wanted_routed[bad_row] = 0
            require(
                equal_bits(actual_routed, wanted_routed),
                "split routed output differs",
            )
            skip = 1 if local_bad else 0
            require(
                np.array_equal(
                    actual_shared[skip:],
                    np.broadcast_to(expected_shared, (rows, 5120))[skip:],
                ),
                "split shared output differs",
            )
            local_fields = download_packet(device, packet)
            global_fields = download_packet(device, gathered)
            remote = np.ones(routed_rows, bool)
            remote[first:last] = False
            for index, (local, global_field, original) in enumerate(
                zip(local_fields, global_fields, reference, strict=True)
            ):
                wanted = global_status if index == 2 else original
                wanted_local = coefficients if index == 4 else original[first:last]
                require(
                    equal_bits(local[skip:], wanted_local[skip:])
                    and equal_bits(global_field[first:last], local)
                    and equal_bits(global_field[remote], wanted[remote]),
                    "local/gathered packet changed or owner offset differs",
                )

        operation()
        check()
        rejected(lambda: workspace.run(value, image_mask))
        rejected(lambda: workspace.run_shared(gathered))
        rejected(lambda: workspace.run_routed(packet))
        if counts == (0, 1, 2, 3) and rank == 1:
            sentinel = np.full(workspace.gate.outputs.value.shape, 17, np.float32)
            device.module.copy_bytes(
                workspace.gate.outputs.value, device.upload(sentinel)
            )
            rejected(lambda: workspace.run(value, image_mask))
            require(
                equal_bits(
                    device.download(workspace.gate.outputs.value, np.float32), sentinel
                ),
                "unequal-domain run performed gate computation before rejection",
            )
            for local_rows, global_rows in ((1, 0), (0, -1), (0, 385)):
                rejected(
                    lambda local_rows=local_rows, global_rows=global_rows: prepare(
                        owner, name, rows=local_rows, routed_rows=global_rows
                    )
                )
            rejected(
                lambda: prepare(
                    owner, name, rows=rows, routed_rows=routed_rows, byte_limit=1
                )
            )
            aliases = (
                workspace.dispatch.outputs.encoded,
                workspace.routed["w2"].packed_scales,
                workspace.routed_quantization.outputs.status,
                workspace.dispatch.source_rows,
                workspace.dispatch.outputs.coefficients,
            )
            for index, alias in enumerate(aliases):
                fields = list(gathered)
                fields[index] = device.view(alias, gathered[index].shape)
                rejected(
                    lambda fields=fields: workspace.run_routed(
                        EncodedDispatchInputs(*fields)
                    )
                )
            fields = list(packet)
            fields[0] = device.view(
                workspace.shared["w1"].packed_input_scales, packet.encoded.shape
            )
            rejected(lambda: workspace.run_shared(EncodedDispatchInputs(*fields)))
            for index in (3, 4):
                fields = list(packet)
                fields[index] = gathered[index]
                rejected(
                    lambda fields=fields: workspace.run_shared(
                        EncodedDispatchInputs(*fields)
                    )
                )
            check()
        if routed_rows == 6:
            normal_status = device.upload(reference[2])
            invalid_status = reference[2].copy()
            invalid_status[remote_row, 0] = 1
            invalid_status = device.upload(invalid_status)
            restored_value = device.upload(bf16(host[first:last]), "bfloat16")
            invalid_value = bf16(host[first:last])
            if rows:
                invalid_value[0, 0] = 0x7FC0
            invalid_value = device.upload(invalid_value, "bfloat16")
            with workspace.graph_lease(), device.graph(operation) as replay:
                replay()
                check()
                device.module.copy_bytes(gathered.status, invalid_status)
                replay()
                check(remote_bad=True)
                device.module.copy_bytes(gathered.status, normal_status)
                replay()
                check()
                if rows:
                    device.module.copy_bytes(value, invalid_value)
                    replay()
                    check(local_bad=True)
                    device.module.copy_bytes(value, restored_value)
                    replay()
                    check()
                workspace.close()
                owner.close()
                replay()
                check()
            rejected(operation)
        return dict(
            strategy="DEP4",
            rank=rank,
            rows=rows,
            routed_rows=routed_rows,
            owner_offset=first,
            graph=routed_rows == 6,
            workspace_bytes=workspace.allocated_bytes,
            native=[native_identity(part) for part in workspace.parts],
        )
    finally:
        device.synchronize_compute()
        if workspace is not None:
            workspace.close()
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-cpu", action="store_true")
    args = parser.parse_args()
    if args.check_cpu:
        check_cpu()
        return
    require(args.output is not None, "--output is required for GPU checks")
    from oh_my_vllm.models.deepseek_v41.moe import prepare  # noqa: F401
    from oh_my_vllm.operators.runtime.device import Device

    from development.operators.check_deepseek_exact import caller_stream
    from development.serving.device import device_identity, selected_uuid

    sources = source_identity()
    device = Device()
    try:
        with caller_stream(device):
            checks = [
                check_case(device, strategy, rank, draft, rows)
                for strategy, rank, draft, rows in (
                    ("TP4", 0, False, 2),
                    ("TEP4", 1, False, 2),
                    ("DEP4", 3, False, 2),
                    ("TP4", 0, True, 1),
                    ("TEP4", 1, True, 2),
                    ("DEP4", 2, True, 2),
                    ("TP4", 0, False, 0),
                    ("TEP4", 0, False, 384),
                )
            ]
            checks.extend(
                check_split_case(device, rank, (0, 1, 2, 3)) for rank in range(4)
            )
            checks.append(check_split_case(device, 1, (0, 1, 127, 256)))
        require(source_identity() == sources, "MoE sources changed during execution")
        report = dict(
            passed=True,
            device=device_identity(selected_uuid()),
            checks=checks,
            sources=sources,
        )
        with args.output.open("x") as output:
            json.dump(report, output, indent=2)
            output.write("\n")
    finally:
        device.synchronize_compute()
        device.close()


if __name__ == "__main__":
    main()
