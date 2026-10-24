"""Check synthetic layer-0 composition; no model acceptance or collectives.

Ranks execute sequentially. Completed inputs come from independent literals,
and downloaded local contributions must sum to those literals. KV, exchanged
packets, completed branches, and their row identities stay caller-owned.
"""

import argparse
import hashlib
import json
import os
import weakref
from fractions import Fraction
from pathlib import Path

import numpy as np

from development.operators.performance_inputs import bf16
from development.operators.storage import decode, equal_bits

from .check_attention import owner_weights as attention_weights
from .check_attention import quarter_turns
from .check_moe import download_packet, packet_arrays, require, selected_experts, weight
from .check_moe import owner_weights as moe_weights
from .reference_quantization import activation_fp8

ROOT = Path(__file__).resolve().parents[3]
ATTN_PRE = np.float32(0.5) + np.float32(1e-6)
FFN_PRE = np.float32(1) + np.float32(1e-6)
FINAL_BITS = np.array([0x421A, 0x420E], np.uint16)  # 38.5, 35.5.
COMPLETE_ROUTED = np.array([14.0625, 11.25], np.float32)


def check_cpu():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU check must hide CUDA")
    # Uniform Sinkhorn reduces to c/(4*c+eps). Prove an invariant enclosure,
    # rather than replaying its iterations or assuming exactly one quarter.
    eps = Fraction(float(np.float32(1e-6)))
    low = Fraction(1, 4) - Fraction(1, 2**20)
    high = Fraction(1, 4) - Fraction(3, 2**24)

    def normalize(c, upper):
        error = Fraction(1, 2**24)
        return c / (4 * c + eps + (-error if upper else error)) + (
            Fraction(1 if upper else -1, 2**26)
        )

    initial = Fraction(1, 4) + eps
    require(
        normalize(initial - Fraction(1, 2**26), False) >= low
        and normalize(initial + Fraction(1, 2**26), True) <= high
        and normalize(low, False) >= low
        and normalize(high, True) <= high,
        "analytic uniform Sinkhorn enclosure failed",
    )

    def cell(first, last, bits):
        neighbors = decode(np.array([bits - 1, bits, bits + 1], np.uint16), "bfloat16")
        left, center, right = (Fraction(float(x)) for x in neighbors)
        require(
            (left + center) / 2 < first <= last < (center + right) / 2,
            f"literal {bits:#06x} is not inside its BF16 rounding cell",
        )

    # 2^-17 bounds the few FP32 adds/products at these particular magnitudes.
    # It is a proof bound for literals, never a candidate comparison tolerance.
    error = Fraction(1, 2**17)
    cell(15 + 15 * low - error, 15 + 15 * high + error, 0x4196)
    collapsed = Fraction(75, 4) * 4 * Fraction(float(ATTN_PRE))
    cell(collapsed - error, collapsed + error, 0x4216)
    require(Fraction(10 * 576 * 10, 4 * 4096) == Fraction(225, 64), "TP route")
    require(Fraction(10 * 576, 4096) == Fraction(45, 32), "TP shared")
    require(4 * Fraction(225, 64) == Fraction(225, 16), "complete route")
    require(4 * Fraction(45, 32) == Fraction(45, 8), "complete shared")
    # 19.6875 is the midpoint; 19.75 has the even BF16 significand.
    require(
        Fraction(225, 16) + Fraction(45, 8) == Fraction(315, 16)
        and Fraction(315, 16) == (Fraction(157, 8) + Fraction(79, 4)) / 2
        and (0x419E & 1) == 0,
        "branch tie-even literal",
    )
    cell(
        Fraction(79, 4) + 75 * low - error, Fraction(79, 4) + 75 * high + error, 0x421A
    )
    cell(
        Fraction(135, 8) + 75 * low - error,
        Fraction(135, 8) + 75 * high + error,
        0x420E,
    )
    for initial, expanded, branch, final in (
        (Fraction(77, 2), Fraction(107, 2), Fraction(79, 4), 0x4292),
        (Fraction(71, 2), Fraction(101, 2), Fraction(135, 8), 0x4287),
    ):
        cell(
            branch + 4 * expanded * low - 2 * error,
            branch + 4 * expanded * high + 2 * error,
            final,
        )
        cell(
            15 + 4 * initial * low - 2 * error,
            15 + 4 * initial * high + 2 * error,
            0x4256 if initial == Fraction(77, 2) else 0x424A,
        )
    require(
        not equal_bits(np.full(4, ATTN_PRE), np.full(4, FFN_PRE)),
        "next-pre literals must discriminate the two sublayers",
    )
    for counts in ((2, 2, 2, 2), (0, 1, 2, 3), (0, 0, 0, 0)):
        rows = sum(counts)
        values = np.ones((rows, 5120), np.float32)
        mask = (np.arange(rows) % 2).astype(np.uint8)
        packet = packet_arrays(values, mask)
        if rows:
            require(
                np.array_equal(activation_fp8(values).dequantized, values),
                "normalized packet must survive FP8 exactly",
            )
        for strategy in ("TP4", "TEP4", "DEP4"):
            sums = np.zeros(rows, np.float32)
            for rank in range(4):
                sums += routed_literal(strategy, rank, mask)
            require(
                equal_bits(sums, COMPLETE_ROUTED[mask]),
                f"{strategy} routed completion literals",
            )
        require(all(len(field) == rows for field in packet), "packet domain")
    return dict(oracle_passed=True, gpu_executed=False, collectives_executed=False)


def routed_literal(strategy, rank, mask):
    if strategy == "TP4":
        return COMPLETE_ROUTED[mask] / np.float32(4)
    multipliers = np.array(
        [
            sum(
                1 + expert % 2
                for expert in group
                if rank * 96 <= expert < (rank + 1) * 96
            )
            for group in selected_experts()
        ],
        np.float32,
    )
    return multipliers[mask] * np.float32(1.40625)


def owner_weights(device, strategy, rank):
    owner, name, _, start, count, width, shared_width = moe_weights(
        device, strategy, rank, False
    )
    donor = attention_weights(device, rank)
    owner.tensors.update(donor.tensors)
    owner.packed.update(donor.packed)
    donor.close()
    # softplus(64) rounds to 64, sqrt to 8; six equal scores yield exactly 1/4.
    gate = np.zeros((384, 5120), np.float32)
    gate[:, 0] = 64
    owner.tensors[f"{name}.ffn.gate.weight"] = device.upload(bf16(gate), "bfloat16")
    if strategy == "DEP4":
        owner.packed[f"{name}.attn.wq_b.weight"] = weight(
            device, 8, 64 * 512, 1280, 0x38, 127, True
        )
        owner.packed[f"{name}.attn.wo_b.weight"] = weight(
            device, 8, 5120, 8192, 0x57, 127, True
        )
        down = np.zeros((8192, 4096), np.float32)
        down[np.arange(8) * 1024, 448] = 1
        owner.tensors[f"{name}.attn.wo_a.weight"] = device.upload(
            bf16(down), "bfloat16"
        )
        owner.tensors[f"{name}.attn.attn_sink"] = device.upload(
            np.full(64, -1000, np.float32)
        )
    down = [weight(device, 4, 5120, width, code, 115, False) for code in (2, 4)]
    for expert in range(start, start + count):
        owner.packed[f"{name}.ffn.experts.{expert}.w2.weight"] = down[expert % 2]
    owner.packed[f"{name}.ffn.shared_experts.w2.weight"] = weight(
        device, 8, 5120, shared_width, 0x38, 115, False
    )
    for sublayer in ("attn", "ffn"):
        base = np.zeros(24, np.float32)
        if sublayer == "ffn":
            base[:4] = 1000
        for suffix, value in (
            ("fn", np.zeros((24, 20480), np.float32)),
            ("scale", np.zeros(3, np.float32)),
            ("base", base),
        ):
            owner.tensors[f"{name}.hc_{sublayer}_{suffix}"] = device.upload(value)
        owner.tensors[f"{name}.{sublayer}_norm.weight"] = device.upload(
            bf16(np.ones(5120, np.float32)), "bfloat16"
        )
    return owner, name


def constant(device, tensor, expected, message):
    dtype = {"bfloat16": np.uint16, "float32": np.float32, "uint8": np.uint8}
    actual = device.download(tensor, dtype[str(tensor.dtype)])
    expected = np.asarray(expected, dtype=actual.dtype)
    if expected.ndim == 1:
        expected = expected.reshape((-1,) + (1,) * (actual.ndim - 1))
    require(equal_bits(actual, np.broadcast_to(expected, actual.shape).copy()), message)
    return actual


def native_parts(workspace):
    result = []
    for part in workspace.parts:
        if hasattr(part, "source_id"):
            from development.operators.check_deepseek_exact import native_identity

            result.append(native_identity(part))
        if hasattr(part, "parts"):
            result.extend(native_parts(part))
    return result


def check_case(device, strategy, rank, counts):
    from oh_my_vllm.models.deepseek_v41.block import prepare
    from oh_my_vllm.operators.routing.api import EncodedDispatchInputs

    from development.operators.check_deepseek_exact import rejected

    split = strategy == "DEP4"
    rows = counts[rank]
    total = sum(counts) if split else rows
    first = sum(counts[:rank]) if split else 0
    last = first + rows
    masks = (np.arange(total) % 2).astype(np.uint8)
    local_masks = masks[first:last]
    owner, name = owner_weights(device, strategy, rank)
    workspace = None
    try:
        workspace = prepare(owner, name, rows=rows, routed_rows=total, slots=1)
        require(
            workspace.plan.rows == rows
            and workspace.moe.gate.plan.rows == rows
            and workspace.moe.dispatch.plan.rows == total,
            "local/global preparation domains differ",
        )
        residual_host = np.broadcast_to(
            np.array([1, 2, 4, 8], np.float32)[None, :, None], (rows, 4, 5120)
        ).copy()
        residual = device.upload(bf16(residual_host), "bfloat16")
        incoming = device.upload(
            np.broadcast_to(np.array([1, 0, 0, 0], np.float32), (rows, 4)).copy()
        )
        coefficients = device.upload(quarter_turns(np.zeros(rows, np.int32)))
        image_mask = device.upload(local_masks)
        indices = device.upload(np.arange(rows, dtype=np.int32).reshape(rows, 1))
        window = device.empty((rows, 512), "uint8")
        scales = device.empty((rows, 16), "uint8")
        compressed = device.empty((0, 256), "uint8")
        compressed_scales = device.empty((0, 32), "uint8")
        complete_attention = device.upload(np.full((rows, 5120), 15, np.float32))
        routed_host = np.broadcast_to(
            COMPLETE_ROUTED[local_masks, None], (rows, 5120)
        ).copy()
        complete_routed = device.upload(routed_host)
        shared_host = np.full((rows, 5120), 5.625, np.float32)
        complete_shared = device.upload(bf16(shared_host), "bfloat16")
        complete_shared_fp32 = device.upload(shared_host)
        reference = packet_arrays(np.ones((total, 5120), np.float32), masks)
        gathered = EncodedDispatchInputs(*(device.upload(field) for field in reference))
        remote_check = split and rank == 1
        if remote_check:
            remote_row = next(row for row in range(total) if not first <= row < last)
            bad_status_host = reference[2].copy()
            bad_status_host[remote_row, 0] = 1
            bad_status = device.upload(bad_status_host)
            normal_status = device.upload(reference[2])
        destinations = [
            device.view(field, (rows, field.shape[1]), first * field.shape[1])
            for field in gathered
        ]
        encoded = attended = packet = shared = routed = output = None

        def attend():
            return workspace.attend(
                indices, window, scales, compressed, compressed_scales
            )

        def operation(
            shared_complete=complete_shared, residual_input=residual, pre_input=incoming
        ):
            nonlocal encoded, attended, packet, shared, routed, output
            encoded = workspace.begin(residual_input, pre_input, coefficients)
            device.module.copy_bytes(window, encoded.window)
            device.module.copy_bytes(scales, encoded.window_scales)
            attended = attend()
            packet, shared = workspace.ffn(complete_attention, image_mask)
            if split:
                for destination, source in zip(destinations, packet, strict=True):
                    device.module.copy_bytes(destination, source)
            routed = workspace.run_routed(gathered if split else packet)
            output = workspace.finish(complete_routed, shared_complete)

        def check(reused=False, remote_bad=False):
            constant(
                device,
                workspace.hc_attn.outputs.collapsed,
                np.where(local_masks == 0, 0x431A, 0x430E) if reused else 0x3F80,
                "initial collapse",
            )
            constant(device, workspace.attn_norm.outputs.value, 0x3F80, "attention RMS")
            constant(
                device,
                workspace.hc_attn.outputs.next_pre,
                ATTN_PRE,
                "attention next-pre",
            )
            constant(
                device,
                attended.value,
                0x4170 if split else bf16(np.float32(2**rank)),
                "rank attention",
            )
            constant(
                device,
                workspace.hc_attn.outputs.expanded,
                np.where(local_masks == 0, 0x4256, 0x424A) if reused else 0x4196,
                "attention HC expansion",
            )
            constant(
                device,
                workspace.hc_ffn.outputs.collapsed,
                np.where(local_masks == 0, 0x42D6, 0x42CA) if reused else 0x4216,
                "FFN collapse must use attention next-pre",
            )
            constant(device, workspace.ffn_norm.outputs.value, 0x3F80, "FFN RMS")
            constant(device, shared.value, 0x40B4 if split else 0x3FB4, "rank shared")
            wanted_routed = routed_literal(strategy, rank, masks)
            if remote_bad:
                wanted_routed[remote_row] = 0
            constant(device, routed.value, wanted_routed, "rank routed")
            constant(
                device,
                output.residual,
                np.where(local_masks == 0, 0x4292, 0x4287)
                if reused
                else FINAL_BITS[local_masks],
                "final HC expansion",
            )
            constant(device, output.next_pre, FFN_PRE, "must return FFN next-pre")
            require(
                output.residual is workspace.hc_ffn.outputs.expanded,
                "extra residual buffer",
            )
            require(
                output.next_pre is workspace.hc_ffn.outputs.next_pre,
                "extra next-pre buffer",
            )
            require(output.status, "block must expose status")
            flags = [device.download(status, np.uint8) for status in output.status]
            for flag in flags:
                wanted = np.zeros_like(flag)
                if remote_bad:
                    if flag.shape == (total, 160):
                        wanted = bad_status_host
                    elif flag.shape == (total,):
                        wanted[remote_row] = 1
                require(equal_bits(flag, wanted), "block status differs")
            if remote_bad:
                require(
                    any(equal_bits(flag, bad_status_host) for flag in flags),
                    "final status omitted the supplied global packet status",
                )
            for actual, wanted in zip(
                download_packet(device, packet), reference, strict=True
            ):
                require(
                    equal_bits(actual, wanted[first:last]),
                    "local packet or row identity",
                )
            if split:
                for index, (actual, wanted) in enumerate(
                    zip(download_packet(device, gathered), reference, strict=True)
                ):
                    if remote_bad and index == 2:
                        wanted = bad_status_host
                    require(
                        equal_bits(actual, wanted),
                        "global packet or remote rows changed",
                    )

        rejected(attend)
        rejected(lambda: workspace.ffn(complete_attention, image_mask))
        rejected(lambda: workspace.run_routed(gathered))
        rejected(lambda: workspace.finish(complete_routed, complete_shared))
        if rank == 0:
            rejected(
                lambda: prepare(
                    owner, name, rows=rows, routed_rows=total, slots=1, byte_limit=1
                )
            )
            if rows:
                rejected(
                    lambda: prepare(
                        owner,
                        name,
                        rows=rows,
                        routed_rows=total,
                        slots=1,
                        byte_limit=workspace.allocated_bytes - 1,
                    )
                )
        sizes = {"uint8": 1, "bfloat16": 2, "float32": 4, "int32": 4, "uint64": 8}
        payload = sum(
            value.numel() * sizes[str(value.dtype)] for value in workspace._allocations
        )
        require(
            workspace.allocated_bytes
            == payload + (384 if strategy == "TP4" else 96) * 48,
            "block bytes must include every child buffer and grouped address",
        )
        if rows:
            rejected(
                lambda: workspace.begin(
                    workspace.hc_attn.outputs.expanded, incoming, coefficients
                )
            )
        operation()
        check()
        contributions = (
            device.download(attended.value, np.uint16),
            device.download(routed.value, np.float32),
            device.download(shared.value, np.uint16),
        )
        rejected(attend)
        rejected(lambda: workspace.ffn(complete_attention, image_mask))
        rejected(lambda: workspace.run_routed(gathered))
        rejected(lambda: workspace.finish(complete_routed, complete_shared))
        if rows:
            operation(complete_shared, output.residual, output.next_pre)
            check(reused=True)
            workspace.begin(residual, incoming, coefficients)
            attended = attend()
            rejected(
                lambda: workspace.ffn(workspace.hc_ffn.outputs.collapsed, image_mask)
            )
        if split:
            workspace.begin(residual, incoming, coefficients)
            attended = attend()
            local_packet, _ = workspace.ffn(complete_attention, image_mask)
            rejected(lambda: workspace.run_routed(local_packet))
        operation(complete_shared_fp32)
        check()
        with workspace.graph_lease(), device.graph(operation) as replay:
            replay()
            check()
            if remote_check:
                device.module.copy_bytes(gathered.status, bad_status)
                replay()
                check(remote_bad=True)
                device.module.copy_bytes(gathered.status, normal_status)
                replay()
                check()
            workspace.close()
            owner.close()
            replay()
            check()
        rejected(operation)
        rejected(lambda: prepare(owner, name, rows=rows, routed_rows=total, slots=1))
        return (
            dict(
                strategy=strategy,
                rank=rank,
                rows=rows,
                routed_rows=total,
                owner_offset=first,
                graph=True,
                remote_status_injection=remote_check,
                workspace_bytes=workspace.allocated_bytes,
                native=native_parts(workspace),
            ),
            contributions,
            weakref.ref(workspace),
        )
    finally:
        device.synchronize_compute()
        if workspace is not None:
            workspace.close()
        owner.close()


def source_identity():
    paths = {
        Path(__file__),
        Path(__file__).with_name("check_attention.py"),
        Path(__file__).with_name("check_moe.py"),
    }
    paths.update((ROOT / "python/oh_my_vllm/models/deepseek_v41").glob("*.py"))
    for family in (
        "activation",
        "hyperconnection",
        "linear",
        "projection",
        "quantization",
        "routing",
        "runtime",
        "sparse_attention",
        "transform",
    ):
        paths.update(
            path
            for path in (ROOT / "python/oh_my_vllm/operators" / family).rglob("*")
            if path.suffix in (".py", ".cu", ".cuh", ".cpp", ".hpp")
        )
    paths.update(
        ROOT / f"development/operators/{name}.py"
        for name in ("performance_inputs", "storage", "check_deepseek_exact")
    )
    paths.update(ROOT / f"development/serving/{name}.py" for name in ("device", "gpu"))
    paths.update(
        Path(__file__).with_name(name + ".py")
        for name in ("reference_quantization", "quantization")
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", "--check-cpu", action="store_true")
    parser.add_argument(
        "--output", type=Path, help="explicit report path for GPU checks"
    )
    args = parser.parse_args()
    if args.cpu:
        print(json.dumps(check_cpu()), flush=True)
        return
    if args.output is None:
        parser.error("--output is required for GPU checks")
    from oh_my_vllm.operators.runtime.device import Device

    from development.operators.check_deepseek_exact import (
        caller_stream,
        native_identity,
    )
    from development.serving.device import device_identity, selected_uuid

    sources = source_identity()
    device = Device()
    checks = []
    try:
        device_native_before = native_identity(device)
        with caller_stream(device):
            for strategy, counts in (
                ("TP4", (2, 2, 2, 2)),
                ("TEP4", (2, 2, 2, 2)),
                ("DEP4", (0, 1, 2, 3)),
            ):
                contributions = []
                for rank in range(4):
                    report, parts, reference = check_case(
                        device, strategy, rank, counts
                    )
                    require(
                        reference() is None,
                        "closed block workspace requires cyclic garbage collection",
                    )
                    checks.append(report)
                    contributions.append(parts)
                total = sum(counts) if strategy == "DEP4" else counts[0]
                masks = (np.arange(total) % 2).astype(np.uint8)
                require(
                    equal_bits(
                        sum(parts[1] for parts in contributions),
                        np.broadcast_to(
                            COMPLETE_ROUTED[masks, None], (total, 5120)
                        ).copy(),
                    ),
                    "sequential routed partials do not sum to independent completion",
                )
                if strategy in ("TP4", "TEP4"):
                    for index, expected in ((0, 15), (2, 5.625)):
                        actual = sum(
                            decode(parts[index], "bfloat16") for parts in contributions
                        )
                        require(
                            np.all(actual == expected),
                            f"{strategy} attention/shared partial completion",
                        )
            report, _, reference = check_case(device, "TP4", 0, (0, 0, 0, 0))
            require(
                reference() is None,
                "closed empty block workspace requires cyclic garbage collection",
            )
            checks.append(report)
        require(source_identity() == sources, "block sources changed during execution")
        device_native_after = native_identity(device)
        require(
            device_native_after == device_native_before,
            "Device native identity changed during execution",
        )
        report = dict(
            passed=True,
            collectives_executed=False,
            checkpoint_used=False,
            device=device_identity(selected_uuid()),
            device_native=dict(before=device_native_before, after=device_native_after),
            checks=checks,
            sources=sources,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    finally:
        device.synchronize_compute()
        device.close()


if __name__ == "__main__":
    main()
