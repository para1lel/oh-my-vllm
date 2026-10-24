"""Check exact synthetic TP4 attention composition, not full-model acceptance."""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from types import MappingProxyType

import numpy as np

from development.operators.performance_inputs import bf16
from development.operators.storage import decode

from .reference_quantization import activation_fp8

ROOT = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def source_identity():
    paths = {Path(__file__)}
    for name in (
        "development/operators/collective_peer.py",
        "development/operators/check_deepseek_exact.py",
        "development/operators/performance_inputs.py",
        "development/operators/storage.py",
        "development/serving/device.py",
        "development/serving/gpu.py",
        "development/model/deepseek_v41/reference_quantization.py",
        "development/model/deepseek_v41/quantization.py",
        "python/oh_my_vllm/models/deepseek_v41/attention.py",
        "python/oh_my_vllm/models/deepseek_v41/workspace.py",
        "python/oh_my_vllm/models/deepseek_v41/weights.py",
        "python/oh_my_vllm/models/deepseek_v41/checkpoint.py",
        "python/oh_my_vllm/models/deepseek_v41/placement.py",
    ):
        paths.add(ROOT / name)
    for family in (
        "linear",
        "projection",
        "quantization",
        "runtime",
        "sparse_attention",
        "transform",
    ):
        paths.update(
            path
            for path in (ROOT / "python/oh_my_vllm/operators" / family).rglob("*")
            if path.suffix in (".py", ".cu", ".cuh", ".cpp", ".hpp")
        )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def quarter_turns(positions):
    pairs = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)), np.float32)
    return np.repeat(pairs[np.asarray(positions) % 4, None, :], 32, axis=1)


def window_reference():
    coefficients = quarter_turns(np.arange(17))[:, 0]
    values = np.ones((17, 512), np.float32)
    values[:, 448::2] = (coefficients[:, 0] - coefficients[:, 1])[:, None]
    values[:, 449::2] = (coefficients[:, 1] + coefficients[:, 0])[:, None]
    return activation_fp8(values)


def query_reference(positions):
    coefficients = quarter_turns(positions)[:, 0]
    query = np.zeros((len(positions), 16, 512), np.float32)
    for head in range(16):
        query[:, head, 0] = 2 ** (head % 3)
        scale = 2 ** (head % 2)
        query[:, head, 448:450] = coefficients * scale
        query[:, head, 450:452] = -coefficients * scale
    return query


def expected_output(positions, selections, rank):
    """Integer KV sums, BF16 mean/inverse RoPE, then independent FP8 down input."""
    window = window_reference().dequantized.astype(np.int64)
    means = np.array(
        [window[ids, 448:450].sum(axis=0) / len(ids) for ids in selections],
        np.float32,
    )
    means = decode(bf16(means), "bfloat16").astype(np.float32)
    coefficients = quarter_turns(positions)[:, 0]
    inverse = means[:, 0] * coefficients[:, 0] + means[:, 1] * coefficients[:, 1]
    inverse = decode(bf16(inverse), "bfloat16").astype(np.float32)
    latent = np.zeros((len(positions), 2048), np.float32)
    for group in range(2):
        latent[:, group * 1024] = inverse * np.float32(2**group)
    down = activation_fp8(latent).dequantized
    output = np.empty((len(positions), 5120), np.float32)
    for group in range(2):
        output[:, group::2] = (down[:, group * 1024] * np.float32(2 ** (rank - group)))[
            :, None
        ]
    return bf16(output)


def expectations():
    query = query_reference(np.arange(17))
    scores = np.einsum("rhd,sd->rhs", query, window_reference().dequantized)
    constant = np.array([2 ** (head % 3) for head in range(16)], np.float32)
    require(
        np.array_equal(scores, np.broadcast_to(constant[None, :, None], scores.shape)),
        "nonzero query must preserve uniform attention over every selection",
    )
    cases = {
        "prefill": (np.arange(16), [list(range(row + 1)) for row in range(16)]),
        "decode": (np.array([16]), [list(range(17))]),
        "selected": (np.array([16]), [[1]]),
    }
    result = {}
    for name, (positions, selections) in cases.items():
        partial = [expected_output(positions, selections, rank) for rank in range(4)]
        decoded = [decode(value, "bfloat16") for value in partial]
        # Every subset sum is exact in BF16, independent of the reduction tree.
        for mask in range(1, 16):
            total = sum(decoded[rank] for rank in range(4) if mask & (1 << rank))
            require(
                np.array_equal(decode(bf16(total), "bfloat16"), total),
                f"{name}: reduction order is not exact",
            )
        result[name] = (partial, bf16(sum(decoded)))
    require(
        not np.array_equal(result["decode"][0][0], result["selected"][0][0]),
        "selection mutation must change the expected output",
    )
    return result


def owner_weights(device, rank):
    from oh_my_vllm.models.deepseek_v41.placement import RankPlan
    from oh_my_vllm.models.deepseek_v41.weights import RankWeights
    from oh_my_vllm.operators.projection.api import Weights
    from oh_my_vllm.operators.projection.workspace import prepare_weights

    plan = RankPlan(MappingProxyType({}), MappingProxyType({}), 0, 0, "TP4", rank)
    owner = RankWeights(device, plan, 1 << 30, 0)
    prefix = "layers.0.attn"

    def packed(name, raw):
        outputs, width = raw.shape
        scales = np.full((outputs // 32, width // 32), 127, np.uint8)
        owner.packed[f"{prefix}.{name}.weight"] = prepare_weights(
            device, Weights(device.upload(raw), device.upload(scales)), bits=8
        )

    raw = np.zeros((1280, 5120), np.uint8)
    raw[:, 0] = np.where(np.arange(1280) % 2, 0xB8, 0x38)
    packed("wq_a", raw)
    raw = np.zeros((16 * 512, 1280), np.uint8)
    for head in range(16):
        raw[head * 512, 0] = 0x38 + 8 * (head % 3)
        raw[head * 512 + 448, 2] = 0x38 + 8 * (head % 2)
        raw[head * 512 + 450, 2] = 0xB8 + 8 * (head % 2)
    packed("wq_b", raw)
    raw = np.zeros((512, 5120), np.uint8)
    raw[:, 0] = 0x38
    packed("wkv", raw)
    raw = np.zeros((5120, 2048), np.uint8)
    for group in range(2):
        raw[group::2, group * 1024] = 0x38 + 8 * (rank - group)
    packed("wo_b", raw)
    down = np.zeros((2048, 4096), np.float32)
    for group in range(2):
        down[group * 1024, 448] = 2**group
    for name, value, dtype in (
        ("q_norm.weight", bf16(np.ones(1280, np.float32)), "bfloat16"),
        ("kv_norm.weight", bf16(np.ones(512, np.float32)), "bfloat16"),
        ("wo_a.weight", bf16(down), "bfloat16"),
        ("attn_sink", np.full(16, -1000, np.float32), "float32"),
    ):
        owner.tensors[f"{prefix}.{name}"] = device.upload(value, dtype)
    return owner


def rendezvous(directory, rows, rank):
    from oh_my_vllm.operators.runtime.collectives import unique_id

    path = directory / f"nccl-{rows}.id"
    if rank == 0:
        temporary = path.with_suffix(".tmp")
        with temporary.open("xb") as output:
            output.write(unique_id())
        temporary.rename(path)
    end = time.monotonic() + 120
    while not path.exists():
        require(time.monotonic() < end, "NCCL rendezvous deadline exceeded")
        time.sleep(0.01)
    identity = path.read_bytes()
    require(len(identity) == 128, "invalid NCCL rendezvous identity")
    return identity


def fatal(error):
    # Never unwind CUDA capture or destroy buffers while NCCL may be pending.
    traceback.print_exception(error, file=sys.stderr)
    sys.stderr.flush()
    os._exit(32)


def check_case(device, owner, rank, rows, directory, expected, prefix):
    from oh_my_vllm.models.deepseek_v41.attention import prepare
    from oh_my_vllm.operators.runtime.collectives import Communicator
    from oh_my_vllm.operators.runtime.distributed import (
        CollectiveOperation,
        CollectivePlan,
    )

    from development.operators.check_deepseek_exact import native_identity, rejected
    from development.operators.collective_peer import Probe, wait_ready

    workspace = prepare(owner, "layers.0", rows=rows, slots=17)
    positions = np.arange(16) if rows == 16 else np.array([16])
    host = np.zeros((rows, 5120), np.float32)
    host[:, 0] = 1
    value = device.upload(bf16(host), "bfloat16")
    coefficients = device.upload(quarter_turns(positions))
    selections = [list(range(position + 1)) for position in positions]
    ids = np.full((rows, 17), -1, np.int32)
    for row, selected in enumerate(selections):
        ids[row, : len(selected)] = selected
    indices = device.upload(ids)
    source_window = window_reference()
    cache = np.full((17, 512), 127, np.uint8)
    scales = np.full((17, 16), 255, np.uint8)
    if rows == 1:
        require(prefix is not None, "decode requires a checked prefill prefix")
        cache[:16], scales[:16] = prefix
    window, window_scales = device.upload(cache), device.upload(scales)
    compressed = device.empty((0, 256), "uint8")
    compressed_scales = device.empty((0, 32), "uint8")
    staged = (
        device.view(window, (rows, 512), int(positions[0]) * 512),
        device.view(window_scales, (rows, 16), int(positions[0]) * 16),
    )
    rejected(
        lambda: workspace.attend(
            indices, window, window_scales, compressed, compressed_scales
        )
    )
    rejected(lambda: prepare(owner, "layers.0", rows=rows, slots=17, byte_limit=1))
    communicator = Communicator(4, rank, rendezvous(directory, rows, rank))
    wait_ready(communicator, timeout=120)
    plan = CollectivePlan(
        4, (CollectiveOperation("all_reduce", "bf16", (rows * 5120,)),)
    )
    probe = Probe(device, communicator, plan, rank)
    workspace_native = [native_identity(part) for part in workspace.parts]
    checks = []

    def operation(capture=False):
        try:
            encoded = workspace.encode(value, coefficients)
            device.module.copy_bytes(staged[0], encoded.window)
            device.module.copy_bytes(staged[1], encoded.window_scales)
            outputs = workspace.attend(
                indices, window, window_scales, compressed, compressed_scales
            )
            device.module.copy_bytes(probe.buffers[0][0], outputs.value)
            if capture:
                probe.run()
            else:
                probe.prepared[0].run()
                wait_ready(communicator)
        except BaseException as error:
            fatal(error)

    def check(label, case, invalid=False):
        flags = [
            device.download(status, np.uint8) for status in workspace.outputs.status
        ]
        require(
            len(flags) >= 5
            and all(
                np.all(flag == (1 if invalid and index == 3 else 0))
                for index, flag in enumerate(flags)
            ),
            f"{label}: attention/quantization status differs",
        )
        for tensor, wanted, label_suffix in (
            (workspace.outputs.value, expected[case][0][rank], "partial"),
            (
                device.view(probe.buffers[0][1], (rows, 5120)),
                expected[case][1],
                "all-reduce",
            ),
        ):
            actual = decode(device.download(tensor, np.uint16), "bfloat16")
            target = np.zeros(actual.shape) if invalid else decode(wanted, "bfloat16")
            require(np.array_equal(actual, target), f"{label}: {label_suffix} differs")
        latent = device.download(workspace.encoded.query_latent, np.uint16)
        signs = np.where(np.arange(1280) % 2, -1, 1).astype(np.float32)
        require(
            np.array_equal(latent, np.broadcast_to(bf16(signs), latent.shape)),
            f"{label}: query latent differs",
        )
        require(
            np.array_equal(
                decode(device.download(workspace.encoded.query, np.uint16), "bfloat16"),
                query_reference(positions),
            ),
            f"{label}: nonzero projected/rotated query differs",
        )
        for tensor, wanted in (
            (workspace.encoded.window, source_window.encoded[positions]),
            (workspace.encoded.window_scales, source_window.scales[positions]),
            (device.view(window, (16, 512)), source_window.encoded[:16]),
            (device.view(window_scales, (16, 16)), source_window.scales[:16]),
        ):
            require(
                np.array_equal(device.download(tensor, np.uint8), wanted),
                f"{label}: encoded KV or immutable prefix differs",
            )
        checks.append(label)

    case = "prefill" if rows == 16 else "decode"
    operation()
    check(f"eager-{case}", case)
    if rows == 1:
        for prepared in probe.prepared:
            prepared.prepare_capture()
        with workspace.graph_lease(), device.graph(lambda: operation(True)) as replay:

            def invoke(label, selected_case="decode", invalid=False):
                try:
                    replay()
                    communicator.record_completion()
                    wait_ready(communicator)
                    check(label, selected_case, invalid)
                except BaseException as error:
                    fatal(error)

            invoke("graph-decode")
            changed = ids.copy()
            changed[0, 0] = 17
            invalid_ids = device.upload(changed)
            restored_ids = device.upload(ids)
            selected_ids = np.full((1, 17), -1, np.int32)
            selected_ids[0, 0] = 1
            selected_ids = device.upload(selected_ids)
            for source, label, selected_case, invalid in (
                (invalid_ids, "graph-invalid-id", "decode", True),
                (restored_ids, "graph-recovery", "decode", False),
                (selected_ids, "graph-changed-selection", "selected", False),
                (restored_ids, "graph-restored-selection", "decode", False),
            ):
                device.module.copy_bytes(indices, source)
                invoke(label, selected_case, invalid)
            workspace.close()
            owner.close()
            invoke("graph-closed-owners")
        rejected(lambda: workspace.encode(value, coefficients))
        rejected(lambda: prepare(owner, "layers.0", rows=1, slots=17))
    saved_prefix = (
        device.download(device.view(window, (16, 512)), np.uint8),
        device.download(device.view(window_scales, (16, 16)), np.uint8),
    )
    probe.close()
    workspace.close()
    return (
        dict(
            rows=rows,
            checks=checks,
            workspace_bytes=workspace.allocated_bytes,
            native=workspace_native,
        ),
        saved_prefix,
    )


def run_rank(args):
    from oh_my_vllm.operators.runtime.collectives import local_device
    from oh_my_vllm.operators.runtime.device import Device

    from development.operators.check_deepseek_exact import caller_stream
    from development.operators.collective_peer import loaded_native_identity
    from development.serving.device import selected_uuid

    sources = source_identity()
    expected = expectations()
    identity = selected_uuid()
    require(local_device() == (1, identity), "rank must see exactly its assigned GPU")
    require(
        identity == json.loads((args.output / "devices.json").read_text())[args.rank],
        "rank UUID differs from supervisor order",
    )
    device = Device()
    with caller_stream(device):
        try:
            owner = owner_weights(device, args.rank)
            prefill, prefix = check_case(
                device, owner, args.rank, 16, args.output, expected, None
            )
            decode_case, _ = check_case(
                device, owner, args.rank, 1, args.output, expected, prefix
            )
            device.synchronize_compute()
            require(source_identity() == sources, "attention sources changed")
            report = dict(
                passed=True,
                scope="synthetic-exact-TP4-composition-only",
                rank=args.rank,
                device_uuid=identity,
                checks=[prefill, decode_case],
                sources=sources,
                device_native=dict(
                    source_id=device.source_id,
                    binary_sha256=hashlib.sha256(
                        Path(device.library).read_bytes()
                    ).hexdigest(),
                ),
                collective_native=loaded_native_identity(),
            )
            with (args.output / f"rank-{args.rank}.json").open("x") as output:
                json.dump(report, output, indent=2)
                output.write("\n")
        except BaseException as error:
            fatal(error)
    device.close()


def lease_descriptors(identities):
    expected = {f"/tmp/oh-my-vllm-{identity}.lock" for identity in identities}
    found = {}
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
            if target in expected and os.get_inheritable(int(entry.name)):
                found[target] = int(entry.name)
        except OSError:
            continue
    require(set(found) == expected, "four inherited GPU lease descriptors are required")
    return tuple(found.values())


def supervise(args):
    from development.serving.gpu import (
        Settings,
        candidate_devices,
        parse_uuids,
        read_inventory,
    )

    identities = parse_uuids(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    require(len(identities) == 4, "use the four-GPU wrapper with UUIDs")
    settings = Settings(4, identities, frozenset({0}))
    devices = candidate_devices(settings, read_inventory())
    require(len(devices) == 4, "four permitted B200 GPUs are required")
    descriptors = lease_descriptors(identities)
    sources = source_identity()
    expectations()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "devices.json").write_text(json.dumps(identities) + "\n")
    children, logs = [], []

    def interrupted(number, frame):
        raise InterruptedError(f"supervisor received signal {number}")

    previous = {
        number: signal.signal(number, interrupted)
        for number in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        for rank, identity in enumerate(identities):
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=identity,
                OH_MY_VLLM_GPU_COUNT="1",
                OH_MY_VLLM_GPU_UUID=identity,
                PYTHONDONTWRITEBYTECODE="1",
            )
            environment.pop("OH_MY_VLLM_GPU_UUIDS", None)
            log = (args.output / f"rank-{rank}.log").open("x")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-m",
                        "development.model.deepseek_v41.check_attention",
                        "--rank",
                        str(rank),
                        "--output",
                        str(args.output.resolve()),
                    ],
                    cwd=ROOT,
                    env=environment,
                    pass_fds=descriptors,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        end = time.monotonic() + args.timeout
        while True:
            statuses = [child.poll() for child in children]
            require(
                all(status in (None, 0) for status in statuses),
                f"rank exit: {statuses}",
            )
            if all(status == 0 for status in statuses):
                break
            require(time.monotonic() < end, "four-rank attention deadline exceeded")
            time.sleep(0.05)
        reports = [
            json.loads((args.output / f"rank-{rank}.json").read_text())
            for rank in range(4)
        ]
        require(
            all(
                report["passed"]
                and report["rank"] == rank
                and report["device_uuid"] == identities[rank]
                and report["sources"] == sources
                for rank, report in enumerate(reports)
            ),
            "rank reports disagree with supervisor",
        )
        require(source_identity() == sources, "attention sources changed")
        with (args.output / "report.json").open("x") as output:
            json.dump(
                dict(
                    passed=True,
                    scope="synthetic-exact-TP4-composition-only",
                    ranks=4,
                    devices=[
                        dict(index=device.index, uuid=device.uuid) for device in devices
                    ],
                    sources=sources,
                ),
                output,
                indent=2,
            )
            output.write("\n")
    finally:
        for number in previous:
            signal.signal(number, signal.SIG_IGN)
        for child in children:
            if child.poll() is None:
                with suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGKILL)
        for child in children:
            child.wait()
        for log in logs:
            log.close()
        for number, handler in previous.items():
            signal.signal(number, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rank", type=int, choices=range(4))
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--check-cpu", action="store_true")
    args = parser.parse_args()
    if args.check_cpu:
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU check must hide CUDA"
        )
        result = expectations()
        require(
            not any(name in sys.modules for name in ("torch", "tvm_ffi")),
            "CPU oracle imported a GPU runtime",
        )
        print(
            json.dumps(
                dict(
                    passed=True,
                    scope="CPU-oracle-only",
                    subset_sums=45,
                    expected_sha256={
                        name: hashlib.sha256(b"".join([*partial, total])).hexdigest()
                        for name, (partial, total) in result.items()
                    },
                )
            )
        )
        return
    require(args.output is not None, "--output must be a fresh directory")
    require(
        np.isfinite(args.timeout) and args.timeout > 0, "invalid supervisor timeout"
    )
    if args.rank is None:
        supervise(args)
    else:
        run_rank(args)


if __name__ == "__main__":
    main()
