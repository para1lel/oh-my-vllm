"""Check exact four-rank block composition, not full-model acceptance."""

import argparse
import hashlib
import json
import os
import time
import weakref
from fractions import Fraction
from pathlib import Path

import numpy as np

from development.operators.performance_inputs import bf16
from development.operators.storage import decode, equal_bits

from .check_attention import fatal, quarter_turns, rendezvous, supervise
from .check_attention import owner_weights as attention_weights
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


def check_case(device, strategy, rank, counts, directory, case_id):
    from oh_my_vllm.models.deepseek_v41 import block, block_collectives
    from oh_my_vllm.operators.runtime.collectives import Communicator

    from development.operators.check_deepseek_exact import rejected
    from development.operators.collective_peer import wait_ready

    split = strategy == "DEP4"
    rows = counts[rank]
    total = sum(counts) if split else rows
    first = sum(counts[:rank]) if split else 0
    masks = (np.arange(total) % 2).astype(np.uint8)
    local_masks = masks[first : first + rows]
    owner, name = owner_weights(device, strategy, rank)
    workspace = block.prepare(owner, name, rows=rows, routed_rows=total, slots=1)
    communicator = Communicator(4, rank, rendezvous(directory, case_id, rank))
    wait_ready(communicator, timeout=120)
    exchange = block_collectives.prepare(workspace, communicator, counts=counts)
    residual = device.upload(
        bf16(
            np.broadcast_to(
                np.array([1, 2, 4, 8], np.float32)[None, :, None], (rows, 4, 5120)
            ).copy()
        ),
        "bfloat16",
    )
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
    reference = packet_arrays(np.ones((total, 5120), np.float32), masks)
    injected = np.zeros((rows, 160), np.uint8)
    remote_row = sum(counts[:1])
    if split and rows and rank == 1:
        injected[0, 0] = 1
    normal_status = device.upload(np.zeros_like(injected))
    status = device.upload(injected)
    output = None

    def operation(capture=False, reuse=False):
        nonlocal output
        try:
            encoded = workspace.begin(
                output.residual if reuse else residual,
                output.next_pre if reuse else incoming,
                coefficients,
            )
            device.module.copy_bytes(window, encoded.window)
            device.module.copy_bytes(scales, encoded.window_scales)
            workspace.attend(indices, window, scales, compressed, compressed_scales)
            workspace.ffn(exchange.attention(capture=capture), image_mask)
            if split:
                device.module.copy_bytes(exchange.local_packet.status, normal_status)
            workspace.run_routed(exchange.packet(capture=capture))
            output = exchange.finish(capture=capture)
        except BaseException as error:
            fatal(error)

    def check(reused=False, remote_bad=False):
        constant(device, workspace.attn_norm.outputs.value, 0x3F80, "attention RMS")
        constant(
            device, workspace.hc_attn.outputs.next_pre, ATTN_PRE, "attention next-pre"
        )
        constant(
            device,
            workspace.attention.outputs.value,
            0x4170 if split else bf16(np.float32(2**rank)),
            "rank attention",
        )
        constant(
            device,
            workspace.hc_attn.outputs.expanded,
            np.where(local_masks == 0, 0x4256, 0x424A) if reused else 0x4196,
            "attention collective and HC expansion",
        )
        constant(
            device,
            workspace.hc_ffn.outputs.collapsed,
            np.where(local_masks == 0, 0x42D6, 0x42CA) if reused else 0x4216,
            "FFN collapse uses attention next-pre",
        )
        constant(device, workspace.ffn_norm.outputs.value, 0x3F80, "FFN RMS")
        expected = (
            np.where(local_masks == 0, 0x4292, 0x4287)
            if reused
            else FINAL_BITS[local_masks].copy()
        )
        if remote_bad and rank == 1:
            expected[0] = 0x41C3
        constant(device, output.residual, expected, "final HC expansion")
        constant(device, output.next_pre, FFN_PRE, "FFN next-pre")
        require(output.residual is workspace.outputs.residual, "extra residual buffer")
        flags = [device.download(value, np.uint8) for value in output.status]
        bad = reference[2].copy()
        if remote_bad:
            bad[remote_row, 0] = 1
        if remote_bad:
            require(
                any(equal_bits(value, bad) for value in flags),
                "final status omitted exchanged remote packet status",
            )
        else:
            require(all(not value.any() for value in flags), "nonzero block status")
        for index, (actual, wanted) in enumerate(
            zip(download_packet(device, exchange.global_packet), reference, strict=True)
        ):
            require(
                equal_bits(actual, bad if index == 2 else wanted),
                "global packet fields or owner-major row order differ",
            )
        if not split:
            require(
                all(
                    any(value is status for status in output.status)
                    for value in exchange.widen_status
                ),
                "widen status omitted",
            )

    rejected(
        lambda: workspace.attend(indices, window, scales, compressed, compressed_scales)
    )
    rejected(lambda: workspace.run_routed(exchange.global_packet))
    if total:
        rejected(
            lambda: block_collectives.prepare(
                workspace,
                communicator,
                counts=counts,
                byte_limit=exchange.allocated_bytes - 1,
            )
        )
    if strategy != "DEP4":
        rejected(
            lambda: block_collectives.prepare(
                workspace, communicator, counts=(1, 2, 3, 4)
            )
        )
    rejected(
        lambda: block_collectives.prepare(
            workspace, communicator, counts=(0, 0, 0, 385)
        )
    )
    rejected(
        lambda: block.prepare(
            owner, name, rows=rows, routed_rows=total, slots=1, byte_limit=1
        )
    )
    if rows:
        rejected(
            lambda: workspace.begin(
                workspace.hc_attn.outputs.expanded, incoming, coefficients
            )
        )
    operation()
    wait_ready(communicator)
    check()
    operation(reuse=True)
    wait_ready(communicator)
    check(reused=True)
    exchange.prepare_capture()
    with (
        exchange.graph_lease(),
        device.graph(lambda: operation(capture=True)) as replay,
    ):

        def invoke():
            replay()
            communicator.record_completion()
            wait_ready(communicator)

        try:
            invoke()
            check()
            if split and total:
                # Capture binds the normal source; change its contents before replay.
                device.module.copy_bytes(normal_status, status)
                invoke()
                check(remote_bad=True)
                device.module.zero(normal_status)
                invoke()
                check()
            exchange.close()
            workspace.close()
            owner.close()
            invoke()
            check()
        except BaseException as error:
            fatal(error)
    end = time.monotonic() + 30
    while not communicator.abort_ready():
        require(time.monotonic() < end, "collective Graph lease did not release")
        time.sleep(0.001)
    wait_ready(communicator)
    communicator.abort()
    require(communicator.poll() == "aborted", "communicator did not drain")
    rejected(lambda: exchange.attention())
    return (
        dict(
            strategy=strategy,
            rank=rank,
            rows=rows,
            routed_rows=total,
            owner_offset=first,
            graph=True,
            checkpoint_used=False,
            workspace_bytes=workspace.allocated_bytes,
            communication_bytes=exchange.allocated_bytes,
            native=native_parts(workspace) + native_parts(exchange),
        ),
        weakref.ref(workspace),
        weakref.ref(exchange),
    )


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
        for name in (
            "performance_inputs",
            "storage",
            "check_deepseek_exact",
            "collective_peer",
        )
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


def run_rank(args):
    from oh_my_vllm.operators.runtime.collectives import local_device
    from oh_my_vllm.operators.runtime.device import Device

    from development.operators.check_deepseek_exact import (
        caller_stream,
        native_identity,
    )
    from development.operators.collective_peer import loaded_native_identity

    sources = source_identity()
    count, identity = local_device()
    require(
        count == 1 and identity == os.environ["CUDA_VISIBLE_DEVICES"],
        "rank device differs",
    )
    device = Device()
    with caller_stream(device):
        try:
            checks = []
            for case_id, (strategy, counts) in enumerate(
                (
                    ("TP4", (2, 2, 2, 2)),
                    ("TEP4", (2, 2, 2, 2)),
                    ("DEP4", (0, 1, 2, 3)),
                    ("DEP4", (1, 1, 1, 1)),
                    ("TP4", (0, 0, 0, 0)),
                    ("DEP4", (0, 0, 0, 0)),
                )
            ):
                result, block_ref, exchange_ref = check_case(
                    device, strategy, args.rank, counts, args.output, case_id
                )
                require(
                    block_ref() is None and exchange_ref() is None,
                    "closed model workspaces require cyclic garbage collection",
                )
                checks.append(result)
            require(source_identity() == sources, "block sources changed")
            report = dict(
                passed=True,
                collectives_executed=True,
                checkpoint_used=False,
                rank=args.rank,
                device_uuid=identity,
                checks=checks,
                sources=sources,
                device_native=native_identity(device),
                collective_native=loaded_native_identity(),
            )
            with (args.output / f"rank-{args.rank}.json").open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        except BaseException as error:
            fatal(error)
    device.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", "--check-cpu", action="store_true")
    parser.add_argument("--output", type=Path, help="fresh four-rank report directory")
    parser.add_argument("--rank", type=int, choices=range(4))
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if args.cpu:
        print(json.dumps(check_cpu()), flush=True)
        return
    require(args.output is not None, "--output is required")
    require(
        np.isfinite(args.timeout) and args.timeout > 0, "invalid supervisor timeout"
    )
    if args.rank is None:
        supervise(
            args,
            module="development.model.deepseek_v41.check_block",
            identify=source_identity,
            scope="synthetic-exact-four-rank-block-only",
        )
    else:
        run_rank(args)


if __name__ == "__main__":
    main()
