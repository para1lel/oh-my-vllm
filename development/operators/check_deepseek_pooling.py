"""Check frozen pooling and synthetic source compressor composition."""

import argparse
import hashlib
import importlib.util
import json
import os
from contextlib import closing
from pathlib import Path
from types import MappingProxyType

import numpy as np

from development.model.deepseek_v41 import fixtures
from development.model.deepseek_v41.reference_quantization import compressed_kv_fp4
from development.serving.device import device_identity, selected_uuid
from tests.test_collectives import metadata_tensor

from .check_deepseek_exact import (
    caller_stream,
    execute_modes,
    native_identity,
    rejected,
    require,
)
from .check_deepseek_exact import source_identity as common_sources
from .numerics import compare
from .performance_inputs import bf16
from .storage import decode, equal_bits


def composition_reference():
    normalized = bf16(np.tile([1, 2], 256) / np.sqrt(2.5))
    require(
        np.array_equal(normalized[:2], [0x3F22, 0x3FA2]),
        "independent compressor RMS literals differ",
    )
    values = np.tile(decode(normalized, "bfloat16"), (4, 1)).astype(np.float32)
    coefficients = np.repeat(
        np.array(((1, 0), (0, 1), (-1, 0), (0, -1)), np.float32)[:, None, :],
        32,
        axis=1,
    )
    for row, (cosine, sine) in enumerate(coefficients[:, 0]):
        left, right = values[row, 448::2].copy(), values[row, 449::2].copy()
        values[row, 448::2] = left * cosine - right * sine
        values[row, 449::2] = left * sine + right * cosine
    quantized = compressed_kv_fp4(values)
    require(
        np.all(quantized.encoded[:, :224] == 0x75)
        and np.array_equal(quantized.encoded[:, 224], [0x75, 0x5F, 0xFD, 0xD7])
        and np.all(quantized.scales == 0x26),
        "independent compressor FP4 literals differ",
    )
    return normalized, coefficients, quantized


def source_identity():
    root = Path(__file__).resolve().parents[2]
    sources = common_sources()
    paths = {
        root / "tests/test_collectives.py",
        *(
            root / f"python/oh_my_vllm/models/deepseek_v41/{name}.py"
            for name in (
                "compressor",
                "workspace",
                "weights",
                "placement",
                "checkpoint",
            )
        ),
        *(
            root / f"development/model/deepseek_v41/{name}.py"
            for name in ("reference_quantization", "quantization")
        ),
    }
    for family in ("compression", "linear", "transform", "quantization", "runtime"):
        paths.update(
            path
            for path in (root / "python/oh_my_vllm/operators" / family).rglob("*")
            if path.suffix in (".py", ".cu", ".cuh", ".cpp", ".hpp")
        )
    for path in paths:
        sources[str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def check_case(device, case, limits):
    from oh_my_vllm.operators.compression.api import Inputs
    from oh_my_vllm.operators.compression.plan import make_plan
    from oh_my_vllm.operators.compression.workspace import prepare

    rows = case.manifest["case"]["shape"]["rows"]
    uploaded = {
        key: device.upload(value, "float32") for key, value in case.inputs.items()
    }
    inputs, workspace = Inputs(**uploaded), prepare(make_plan(rows=rows), device)
    result = {}

    def check():
        output = decode(device.download(workspace.outputs.value, np.uint16), "bfloat16")
        measured = compare({"value": output}, case, limits)
        require(measured.passed, f"ratio2 pooling differs: {measured.failures}")
        for key, original in case.inputs.items():
            require(
                equal_bits(device.download(uploaded[key], np.float32), original),
                "ratio2 pooling changed an input",
            )
        result["metrics"] = measured.metrics

    try:
        execute_modes(device, (workspace,), lambda: workspace.run(inputs), check)
        return {
            "case": case.manifest["case"]["id"],
            **result,
            "native": native_identity(workspace),
            "modes": ["eager", "graph"],
        }
    finally:
        try:
            device.synchronize()
        finally:
            workspace.close()


def check_contracts(device):
    from oh_my_vllm.operators.compression.plan import make_plan

    for rows in (True, -1, 385, 1.0):
        rejected(lambda rows=rows: make_plan(rows=rows))
    for rows in (0, 1):
        check_contract(device, rows)
    return {
        "invalid_arguments": True,
        "empty_eager_graph": True,
        "input_output_overlap": True,
        "graph_after_workspace_close": True,
        "closed_workspace": True,
    }


def check_contract(device, rows):
    from oh_my_vllm.operators.compression.api import Inputs
    from oh_my_vllm.operators.compression.plan import make_plan
    from oh_my_vllm.operators.compression.workspace import prepare

    workspace = prepare(make_plan(rows=rows), device)
    kv, gates = (device.zeros((rows, 2, 512), "float32") for _ in range(2))
    inputs = Inputs(kv, gates)
    try:
        if rows:
            for invalid in (
                Inputs(device.empty((rows, 2, 512), "bfloat16"), gates),
                Inputs(kv, device.empty((rows, 2, 512), "bfloat16")),
                Inputs(kv, device.empty((rows, 1, 512), "float32")),
                Inputs(device.empty((rows, 2, 256), "float32"), gates),
            ):
                rejected(lambda invalid=invalid: workspace.run(invalid))
            for value in (kv, gates):
                # Valid BF16 metadata isolates the overlap check.
                alias, guards = metadata_tensor(
                    rows * 512, pointer=value.data_ptr(), code=4, bits=16
                )
                alias = device.view(alias, (rows, 512))
                require(str(alias.dtype) == "bfloat16", "overlap alias dtype differs")
                rejected(
                    lambda alias=alias: workspace.module.pool_ratio2(kv, gates, alias)
                )
                del alias, guards
            with (
                workspace.graph_lease(),
                device.graph(lambda: workspace.run(inputs)) as replay,
            ):
                workspace.close()
                replay()
                require(
                    not device.download(workspace.outputs.value, np.uint16).any(),
                    "recorded pooling graph failed after workspace close",
                )
        else:
            execute_modes(
                device,
                (workspace,),
                lambda: workspace.run(inputs),
                lambda: require(
                    workspace.outputs.value.shape == (0, 512),
                    "empty ratio2 pooling changed shape",
                ),
            )
    finally:
        try:
            device.synchronize()
        finally:
            workspace.close()
    rejected(lambda: workspace.run(inputs))


def compressor_owner(device, layer, strategy):
    from oh_my_vllm.models.deepseek_v41.placement import RankPlan
    from oh_my_vllm.models.deepseek_v41.weights import RankWeights

    plan = RankPlan(MappingProxyType({}), MappingProxyType({}), 0, 0, strategy, 0)
    owner = RankWeights(device, plan, 1 << 30, 0)
    weights = np.zeros((512, 5120), np.float32)
    weights[:, 0] = np.tile([1, 2], 256)
    ratio2 = layer != "layers.20"
    owner.tensors[f"{layer}.attn.compressor.wkv.weight"] = device.upload(
        weights if ratio2 else bf16(weights), "float32" if ratio2 else "bfloat16"
    )
    if ratio2:
        owner.tensors[f"{layer}.attn.compressor.wgate.weight"] = device.zeros(
            weights.shape, "float32"
        )
    owner.tensors[f"{layer}.attn.compressor.norm.weight"] = device.upload(
        np.full(512, 0x3F80, np.uint16), "bfloat16"
    )
    return owner


def reject_alias(device, storage, shape, dtype, operation):
    alias, guards = metadata_tensor(
        int(np.prod(shape)),
        pointer=storage.data_ptr(),
        code=4 if dtype == "bfloat16" else 2,
        bits=16 if dtype == "bfloat16" else 32,
    )
    alias = device.view(alias, shape)
    try:
        rejected(lambda alias=alias: operation(alias))
    finally:
        # Drop every Tensor view before releasing its borrowed managed descriptor.
        del alias, guards


def compressor_contracts(device, owner, workspace, value, coefficients, pairs):
    from oh_my_vllm.models.deepseek_v41.compressor import prepare

    rows, complete, ratio = (
        workspace.plan.rows,
        workspace.plan.completed_rows,
        workspace.plan.ratio,
    )
    layer = next(iter(owner.tensors)).split(".attn.")[0]
    rejected(lambda: workspace.finish(*pairs))
    rejected(lambda: workspace.pack(coefficients))
    rejected(lambda: workspace.encode(device.empty((rows, 5120), "float32")))
    for bad in (
        dict(rows=True, completed_rows=0),
        dict(rows=385, completed_rows=0),
        dict(rows=1, completed_rows=2),
        dict(rows=1, completed_rows=-1),
        dict(rows=1, completed_rows=True),
    ):
        rejected(lambda bad=bad: prepare(owner, layer, **bad))
    if ratio == 1:
        rejected(lambda: prepare(owner, layer, rows=1, completed_rows=0))
    rejected(lambda: prepare(owner, "layers.3", rows=0, completed_rows=0))
    rejected(
        lambda: prepare(owner, layer, rows=rows, completed_rows=complete, byte_limit=1)
    )
    if rows:
        reject_alias(
            device,
            workspace.kv.scratch,
            (rows, 5120),
            "bfloat16",
            workspace.encode,
        )
    if complete:
        workspace.encode(value)
        reject_alias(
            device,
            workspace.outputs.value,
            (complete, 2, 512) if ratio == 2 else (complete, 512),
            "float32" if ratio == 2 else "bfloat16",
            lambda alias: (
                workspace.finish(alias, pairs[1])
                if ratio == 2
                else workspace.finish(alias)
            ),
        )
    workspace.encode(value)
    rejected(lambda: workspace.finish(pairs[0], None if ratio == 2 else value))
    if ratio == 2 and rows:
        invalid = np.zeros((rows, 5120), np.uint16)
        invalid[0, 0] = 0x7FC0
        packet = workspace.encode(device.upload(invalid, "bfloat16"))
        flags = device.download(packet.status[0], np.uint8)
        require(
            equal_bits(flags, np.r_[np.uint8(1), np.zeros(rows - 1, np.uint8)]),
            "compressor widening did not report its invalid input row",
        )
        packet = workspace.encode(value)
        require(
            not device.download(packet.status[0], np.uint8).any(),
            "compressor widening status did not recover",
        )


def check_compressor(device, name, layer, strategy, rows, complete, prior=False):
    from oh_my_vllm.models.deepseek_v41.compressor import prepare

    normalized, turns, quantized = composition_reference()
    owner = compressor_owner(device, layer, strategy)
    workspace = prepare(owner, layer, rows=rows, completed_rows=complete)
    amplitudes = np.resize(np.array([2, 4, 6], np.float32), rows)
    if prior:
        amplitudes[:] = 4
    host = np.zeros((rows, 5120), np.float32)
    host[:, 0] = amplitudes
    value = device.upload(bf16(host), "bfloat16")
    previous = host.copy()
    previous[:, 0] = 2
    previous = device.upload(bf16(previous), "bfloat16")
    rotations = np.arange(complete) % 4
    coefficients = device.upload(turns[rotations].copy())
    ratio = workspace.plan.ratio
    pairs = (
        (
            device.empty((complete, 2, 512), "float32"),
            device.empty((complete, 2, 512), "float32"),
        )
        if ratio == 2
        else (workspace.encoded.kv, None)
    )
    # Continuation stages a previous GPU projection before encode reuses its buffers.
    first, second = (
        tuple(
            tuple(device.view(field, (1, 512), offset * 512) for field in pairs)
            for offset in range(2)
        )
        if prior
        else ((), ())
    )
    native_before = [native_identity(part) for part in workspace.parts]
    linear_images = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in map(Path, workspace.kv.module.linear_images())
    }

    def operation():
        if prior:
            packet = workspace.encode(previous)
            for target, source in zip(first, packet[:2], strict=True):
                device.module.copy_bytes(target, source)
        packet = workspace.encode(value)
        if ratio == 2:
            if prior:
                for target, source in zip(second, packet[:2], strict=True):
                    device.module.copy_bytes(target, source)
            else:
                for target, source in zip(pairs, packet[:2], strict=True):
                    source = device.view(source, tuple(target.shape))
                    device.module.copy_bytes(target, source)
        workspace.finish(*pairs)
        workspace.pack(coefficients)

    def check():
        expected = amplitudes[:, None] * np.tile([1, 2], 256)
        raw = device.download(
            workspace.encoded.kv, np.float32 if ratio == 2 else np.uint16
        )
        require(
            equal_bits(
                raw, expected.astype(np.float32) if ratio == 2 else bf16(expected)
            ),
            f"{name}: compressor input projection differs",
        )
        if ratio == 2:
            require(
                not device.download(workspace.encoded.gates, np.float32).any(),
                f"{name}: zero gate projection differs",
            )
            pooled = (
                np.array([3], np.float32)
                if prior
                else amplitudes[: complete * 2].reshape(complete, 2).mean(axis=1)
            )
            require(
                equal_bits(
                    device.download(workspace.pool.outputs.value, np.uint16),
                    bf16(pooled[:, None] * np.tile([1, 2], 256)),
                ),
                f"{name}: assembled pair pooling differs",
            )
        require(
            equal_bits(
                device.download(workspace.outputs.value, np.uint16),
                np.tile(normalized, (complete, 1)),
            ),
            f"{name}: normalized pre-RoPE latent differs",
        )
        for tensor, expected in (
            (workspace.packed.encoded, quantized.encoded[rotations]),
            (workspace.packed.scales, quantized.scales[rotations]),
        ):
            require(
                equal_bits(device.download(tensor, np.uint8), expected),
                f"{name}: compressed RoPE/FP4 differs",
            )
        require(
            all(
                not device.download(status, np.uint8).any()
                for status in workspace.packed.status
            ),
            f"{name}: nonzero compressor status",
        )
        require(
            equal_bits(device.download(value, np.uint16), bf16(host)),
            f"{name}: compressor modified normalized input",
        )

    try:
        compressor_contracts(device, owner, workspace, value, coefficients, pairs)
        execute_modes(device, workspace.parts, operation, check)
        if complete:
            workspace.encode(value)
            workspace.finish(*pairs)
            reject_alias(
                device,
                workspace.rope.outputs.value,
                (complete, 32, 2),
                "float32",
                workspace.pack,
            )
        with workspace.graph_lease(), device.graph(operation) as replay:
            workspace.close()
            owner.close()
            replay()
            check()
        rejected(lambda: workspace.encode(value))
        rejected(lambda: prepare(owner, layer, rows=rows, completed_rows=complete))
        require(
            [native_identity(part) for part in workspace.parts] == native_before,
            f"{name}: compressor native identity changed",
        )
        require(
            all(
                hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
                for path, digest in linear_images.items()
            ),
            f"{name}: linear dependency changed",
        )
        return dict(
            case=name,
            layer=layer,
            strategy=strategy,
            rows=rows,
            completed_rows=complete,
            allocated_bytes=workspace.allocated_bytes,
            native=native_before,
            linear_images=linear_images,
            modes=["eager", "graph"],
            closed_graph_replay=True,
        )
    finally:
        device.synchronize_compute()
        workspace.close()
        owner.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-cpu", action="store_true")
    args = parser.parse_args()
    composition_reference()
    require(
        importlib.util.find_spec("oh_my_vllm.models.deepseek_v41.compressor")
        is not None,
        "model compressor composition is not implemented",
    )
    if args.check_cpu:
        require(
            os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU check must hide CUDA"
        )
        importlib.import_module("oh_my_vllm.models.deepseek_v41.compressor")
        print(
            json.dumps(
                dict(
                    passed=True,
                    acceptance=False,
                    gpu_executed=False,
                    scope="compressor-independent-literals-and-import-only",
                )
            )
        )
        return
    require(args.output is not None, "--output must be a fresh report path")
    require(
        importlib.util.find_spec("oh_my_vllm.operators.compression") is not None,
        "native ratio2 pooling boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    metadata, cases = fixtures.load_pooling()
    sources = source_identity()
    report = {
        "acceptance": False,
        "fixture_sha256": fixtures.POOLING_SHA256,
        "checkpoint_manifest_sha256": metadata["checkpoint_manifest_sha256"],
        "scope": (
            "Frozen FP32 ratio2 pooling and exact synthetic source compression, "
            "caller pair assembly, RMS, compressed RoPE and FP4; "
            "not checkpoint, full-model, performance or numerical acceptance."
        ),
        "device": device_identity(selected_uuid()),
        "sources": sources,
    }
    with closing(Device()) as device, caller_stream(device):
        report["device_native"] = native_identity(device)
        report["contracts"] = check_contracts(device)
        report["cases"] = [
            check_case(device, case, metadata["cases"][name]["limits"])
            for name, case in cases.items()
        ]
        report["composition"] = [
            check_compressor(device, *case)
            for case in (
                ("ratio2-even", "layers.2", "TP4", 2, 1),
                ("ratio2-continuation", "layers.8", "DEP4", 1, 1, True),
                ("ratio2-incomplete", "layers.14", "TEP4", 1, 0),
                ("ratio2-empty", "layers.2", "DEP4", 0, 0),
                ("ratio2-maximum", "layers.14", "TP4", 384, 192),
                ("ratio1", "layers.20", "TEP4", 2, 2),
            )
        ]
    require(source_identity() == sources, "pooling sources changed during validation")
    report["passed"] = True
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
