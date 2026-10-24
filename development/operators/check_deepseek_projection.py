"""Check encoded A8W8/A8W4 projection against complete frozen FP64 outputs."""

import argparse
import hashlib
import importlib.util
import json
from contextlib import nullcontext
from pathlib import Path
from types import MappingProxyType

import numpy as np

from development.model.deepseek_v41.fixtures import (
    CHECKPOINT_MANIFEST_SHA256,
    PROJECTION_SHA256,
    RANK_ENCODED_PROJECTION_SHA256,
    load_projection,
    load_rank_encoded_projection,
)
from development.model.deepseek_v41.identity import file_state
from development.model.deepseek_v41.layout import tensor_layout
from development.model.deepseek_v41.weights import Weights as CheckpointWeights
from development.serving.device import device_identity, selected_uuid

from .check_deepseek_exact import (
    caller_stream,
    execute_modes,
    native_identity,
    rejected,
    require,
)
from .check_deepseek_hc import source_identity as common_sources
from .fixtures import Fixture
from .numerics import compare
from .storage import decode, equal_bits


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    sources = common_sources()
    paths = list((repository / "python/oh_my_vllm/operators/projection").rglob("*.py"))
    paths.extend((repository / "python/oh_my_vllm/models/deepseek_v41").glob("*.py"))
    paths.append(repository / "tests/test_collectives.py")
    for path in paths:
        sources[str(path.relative_to(repository))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def weight_values(reader, case):
    definition = case.manifest["case"]
    if not definition["weights"]:
        return [case.inputs[name] for name in ("weight", "weight_scales")]
    options = definition["options"]
    values = [reader._raw(name) for name in definition["weights"]]
    axis, rank = options["shard_axis"], options["rank"]
    if axis is not None:
        selected = []
        for value in values:
            slices = [slice(None)] * 2
            width = value.shape[axis] // 4
            slices[axis] = slice(rank * width, (rank + 1) * width)
            selected.append(np.ascontiguousarray(value[tuple(slices)]))
        values = selected
    return values


def scale_offsets(rows, width):
    groups = ((width + 127) // 128) * 4
    n, g = np.indices((rows, width // 32))
    # Native scale atoms contain 128 rows and four groups, with K tiles first.
    return (
        (n // 128) * (groups // 4) * 512
        + (g // 4) * 512
        + (n % 32) * 16
        + ((n % 128) // 32) * 4
        + g % 4
    )


def packed_scales(scales, rows, width, bits):
    groups = ((width + 127) // 128) * 4
    padded_rows = ((rows + 127) // 128) * 128
    result = np.full(padded_rows * groups, 127, np.uint8)
    n, g = np.indices((rows, width // 32))
    result[scale_offsets(rows, width)] = scales[n if bits == 4 else n // 32, g]
    return result


def check_grouped_plans():
    from oh_my_vllm.operators.projection import plan

    require(hasattr(plan, "make_grouped_plan"), "grouped projection is not implemented")
    base = dict(capacity=2304, groups=384, outputs=5120, width=576)
    selected = plan.make_grouped_plan(**base)
    require(selected.padded_width == 640, "grouped K padding differs")
    for change in (
        dict(capacity=-1),
        dict(capacity=2305),
        dict(capacity=True),
        dict(groups=0),
        dict(groups=385),
        dict(groups=True),
        dict(outputs=33),
        dict(width=31),
    ):
        rejected(lambda change=change: plan.make_grouped_plan(**(base | change)))


def check_grouped_case(device, case, limits, values, groups, repeats):
    from oh_my_vllm.operators.projection.api import GroupedInputs, Weights
    from oh_my_vllm.operators.projection.grouped import prepare
    from oh_my_vllm.operators.projection.plan import make_grouped_plan
    from oh_my_vllm.operators.projection.workspace import prepare_weights

    shape = case.manifest["case"]["shape"]
    rows, width, outputs = (shape[key] for key in ("rows", "width", "outputs"))
    capacity = rows * repeats
    raw, scale = values[0].view(np.uint8), values[1]
    variants = (
        (raw, scale),
        (raw[::-1].copy(), scale[::-1].copy()),
        (raw ^ np.uint8(0x88), scale),
    )
    weights = [
        prepare_weights(device, Weights(device.upload(w), device.upload(s)), bits=4)
        for w, s in variants
    ]
    encoded = np.tile(case.inputs["encoded"], (repeats, 1))
    scales = np.tile(case.inputs["scales"], (repeats, 1))
    inputs = GroupedInputs(
        device.upload(encoded),
        device.upload(scales),
        device.empty((groups,), "int32"),
        device.empty((groups + 1,), "int32"),
    )
    workspace = prepare(
        make_grouped_plan(
            capacity=capacity, groups=groups, outputs=outputs, width=width
        ),
        device,
        [weights[g % 3] for g in range(groups)],
    )
    dense = np.zeros(groups, np.int32)
    for row in range(capacity):
        dense[row // 384] += 1
    uneven = np.zeros(groups, np.int32)
    for row in range(capacity):
        uneven[(row * 37) % groups] += 1
    partial = np.zeros(groups, np.int32)
    for row in range(rows * (repeats // 2)):
        partial[row // 384] += 1
    require(max(dense.max(), uneven.max()) <= 384, "fixture exceeds expert row bound")
    states = []
    for counts, invalid in (
        (dense, False),
        (uneven, False),
        (partial, False),
        (np.zeros(groups, np.int32), False),
        (dense, True),
        (uneven, False),
    ):
        offsets = np.concatenate(
            (np.zeros(1, np.int32), np.cumsum(counts, dtype=np.int32))
        )
        if invalid:
            offsets[-1] = capacity + 1
        states.append(
            (counts, offsets, invalid, device.upload(counts), device.upload(offsets))
        )
    originals = [
        (tensor, device.download(tensor, np.uint8))
        for weight in weights
        for tensor in (weight.encoded, weight.scales)
    ]

    def operation():
        return workspace.run(inputs)

    try:
        for mode in ("eager", "graph"):
            with (
                workspace.graph_lease(),
                device.graph(operation)
                if mode == "graph"
                else nullcontext(operation) as invoke,
            ):
                for counts, offsets, invalid, count_tensor, offset_tensor in states:
                    device.module.copy_bytes(inputs.counts, count_tensor)
                    device.module.copy_bytes(inputs.offsets, offset_tensor)
                    invoke()
                    actual = decode(
                        device.download(workspace.outputs.value, np.uint16), "bfloat16"
                    )
                    status = device.download(workspace.outputs.status, np.uint8)
                    require(
                        bool(status.any()) == invalid, "grouped metadata status differs"
                    )
                    if invalid or not counts.any():
                        require(
                            not actual.any(), "empty or invalid groups retained outputs"
                        )
                        continue
                    require(
                        not actual[offsets[-1] :].any(),
                        "unused grouped output retained data",
                    )
                    for group in range(groups):
                        first, last = offsets[group : group + 2]
                        if group % 3 == 1:
                            actual[first:last] = actual[first:last, ::-1]
                        elif group % 3 == 2:
                            actual[first:last] *= -1
                    for start in range(0, int(offsets[-1]), rows):
                        compared = compare(
                            {"value": actual[start : start + rows]}, case, limits
                        )
                        require(
                            compared.passed,
                            f"grouped projection differs: {compared.failures}",
                        )
        for tensor, original in originals:
            require(
                equal_bits(device.download(tensor, np.uint8), original),
                "grouped weight changed",
            )
        for tensor, original in ((inputs.encoded, encoded), (inputs.scales, scales)):
            require(
                equal_bits(device.download(tensor, np.uint8), original),
                "grouped input changed",
            )
        return dict(
            case=case.manifest["case"]["id"],
            groups=groups,
            capacity=capacity,
            modes=["eager", "graph"],
            dynamic_replay=True,
            native=native_identity(workspace),
        )
    finally:
        device.synchronize()
        workspace.close()


def check_grouped_contracts(device):
    from oh_my_vllm.operators.projection.api import GroupedInputs, Weights
    from oh_my_vllm.operators.projection.grouped import prepare
    from oh_my_vllm.operators.projection.plan import make_grouped_plan
    from oh_my_vllm.operators.projection.workspace import prepare_weights

    from tests.test_collectives import metadata_tensor

    weights = prepare_weights(
        device,
        Weights(
            device.zeros((128, 64), "uint8"),
            device.upload(np.full((128, 4), 127, np.uint8)),
        ),
        bits=4,
    )

    def check(capacity, groups):
        workspace = prepare(
            make_grouped_plan(capacity=capacity, groups=groups, outputs=128, width=128),
            device,
            [weights] * groups,
        )
        inputs = GroupedInputs(
            device.zeros((capacity, 128), "uint8"),
            device.upload(np.full((capacity, 4), 127, np.uint8)),
            device.zeros((groups,), "int32"),
            device.zeros((groups + 1,), "int32"),
        )
        try:
            with (
                workspace.graph_lease(),
                device.graph(lambda: workspace.run(inputs)) as replay,
            ):
                for bad in (None, "negative", "too_many", "overflow", None):
                    counts = np.zeros(groups, np.int32)
                    offsets = np.zeros(groups + 1, np.int32)
                    if bad == "negative":
                        counts[0] = -1
                    elif bad == "too_many":
                        counts[0] = 385
                        offsets[1:] = 385
                    elif bad == "overflow":
                        offsets[0], offsets[1] = (
                            np.iinfo(np.int32).min,
                            np.iinfo(np.int32).max,
                        )
                    uploaded = device.upload(counts), device.upload(offsets)
                    for target, source in zip(inputs[2:], uploaded, strict=True):
                        device.module.copy_bytes(target, source)
                    replay()
                    require(
                        np.all(
                            device.download(workspace.outputs.status, np.uint8)
                            == (bad is not None)
                        )
                        and not device.download(
                            workspace.outputs.value, np.uint16
                        ).any(),
                        "grouped metadata guard failed",
                    )
            if capacity:
                rejected(
                    lambda: workspace.run(
                        inputs._replace(encoded=workspace.padded_input)
                    )
                )
                rejected(
                    lambda: workspace.run(
                        inputs._replace(counts=device.empty((groups,), "float32"))
                    )
                )
                rejected(
                    lambda: workspace.run(
                        inputs._replace(offsets=device.empty((groups,), "int32"))
                    )
                )
                for options in (
                    [dict(stride=2), dict(device=1)] if groups > 1 else [dict(device=1)]
                ):
                    invalid, guards = metadata_tensor(
                        groups,
                        pointer=inputs.counts.data_ptr(),
                        code=0,
                        bits=32,
                        **options,
                    )
                    rejected(
                        lambda invalid=invalid: workspace.run(
                            inputs._replace(counts=invalid)
                        )
                    )
                    del invalid, guards
            with (
                workspace.graph_lease(),
                device.graph(lambda: workspace.run(inputs)) as replay,
            ):
                workspace.close()
                rejected(lambda: workspace.run(inputs))
                rejected(lambda: workspace.graph_lease().__enter__())
                replay()
                require(
                    not device.download(workspace.outputs.status, np.uint8).any(),
                    "closed workspace graph failed",
                )
        finally:
            device.synchronize()
            workspace.close()

    for capacity, groups in ((0, 1), (1, 1), (384, 32)):
        check(capacity, groups)
    return dict(
        empty=True,
        invalid_metadata=True,
        overlap=True,
        stride=True,
        device=True,
        closed_workspace_graph=True,
    )


def composed_values(device, workspace):
    quantized = workspace.quantization.outputs
    return dict(
        encoded=device.download(quantized.encoded, np.uint8),
        scales=device.download(quantized.scales, np.uint8),
        roundtrip=device.download(quantized.dequantized, np.uint16),
        status=device.download(quantized.status, np.uint8),
        value=decode(device.download(workspace.outputs.value, np.uint16), "bfloat16"),
    )


def check_composed_view(device, owner, name, master, view, invalid_recipe):
    from oh_my_vllm.models.deepseek_v41.linear import prepare

    rows = view["case"]["shape"]["rows"]
    fixture = Fixture(
        {"case": view["case"]},
        {"value": master.inputs["value"][:rows]},
        {},
        {key: value[:rows] for key, value in master.expected.items()},
    )
    host = np.array(fixture.inputs["value"], copy=True)
    value = device.upload(host, "bfloat16")
    workspace = prepare(owner, name, rows=rows)
    weights = [
        (tensor, device.download(tensor, np.uint8))
        for tensor in (workspace.weights.encoded, workspace.weights.scales)
    ]
    result = {}

    def check():
        actual = composed_values(device, workspace)
        compared = compare(actual, fixture, view["limits"])
        require(compared.passed, f"rank composition differs: {compared.failures}")
        for first, last, column in view["case"]["options"]["zero_columns"]:
            zero = actual["value"][first:last]
            require(
                np.max(np.abs(zero if column < 0 else zero[:, column])) <= 1e-6,
                "composed cancellation endpoint differs",
            )
        require(equal_bits(device.download(value, np.uint16), host), "input changed")
        for tensor, original in weights:
            require(
                equal_bits(device.download(tensor, np.uint8), original),
                "weight changed",
            )
        result["metrics"] = compared.metrics

    def execute(check):
        with workspace.graph_lease():
            execute_modes(
                device,
                (workspace.quantization, workspace.projection),
                lambda: workspace.run(value),
                check,
            )

    try:
        execute(check)
        if rows == 1:
            invalid = host.copy()
            invalid[invalid_recipe["row"], invalid_recipe["channel"]] = invalid_recipe[
                "input_word"
            ]
            invalid_tensor = device.upload(invalid, "bfloat16")
            device.module.copy_bytes(value, invalid_tensor)
            expected = {
                key: data.copy()
                for key, data in fixture.expected.items()
                if key != "value"
            }
            group = invalid_recipe["affected_group"]
            for field in ("encoded", "roundtrip"):
                expected[field][0, group * 32 : (group + 1) * 32] = 0
            expected["scales"][0, group] = 0
            expected["status"][0, group] = invalid_recipe["status"]

            def check_invalid():
                actual = composed_values(device, workspace)
                require(actual["status"].any(), "invalid composition was accepted")
                for field, target in expected.items():
                    require(
                        equal_bits(actual[field], target), f"invalid {field} differs"
                    )
                require(
                    equal_bits(device.download(value, np.uint16), invalid),
                    "invalid input changed",
                )

            execute(check_invalid)
            valid_tensor = device.upload(host, "bfloat16")
            device.module.copy_bytes(value, valid_tensor)
            execute(check)
            result["invalid_then_valid_eager_graph"] = True
        result.update(
            rows=rows,
            modes=["eager", "graph"],
            quantization_native=native_identity(workspace.quantization),
            projection_native=native_identity(workspace.projection),
        )
        return result
    finally:
        device.synchronize()
        workspace.close()


def check_composed_contracts(device, owner, name, master, view):
    from oh_my_vllm.models.deepseek_v41.linear import prepare

    workspace = prepare(owner, name, rows=1)
    host = np.array(master.inputs["value"][:1], copy=True)
    value = device.upload(host, "bfloat16")
    try:
        rejected(lambda: prepare(owner, "missing.weight", rows=1))
        rejected(lambda: prepare(owner, name, rows=385))
        for invalid in (
            device.empty((1, 5120), "float32"),
            device.empty((2, 5120), "bfloat16"),
            workspace.quantization.outputs.dequantized,
        ):
            rejected(lambda invalid=invalid: workspace.run(invalid))
        # Reject cross-stage input/output aliases before either kernel writes.
        alias = device.empty((1, 5120), "bfloat16")

        def overlapping(shape, dtype):
            if shape == (1, 1280) and dtype == "bfloat16":
                return device.view(alias, shape)
            return device.empty(shape, dtype)

        aliased = prepare(owner, name, rows=1, allocate=overlapping)
        try:
            rejected(lambda: aliased.run(alias))
        finally:
            aliased.close()
        empty = prepare(owner, name, rows=0)
        empty_input = device.empty((0, 5120), "bfloat16")
        try:
            with empty.graph_lease():
                execute_modes(
                    device,
                    (empty,),
                    lambda: empty.run(empty_input),
                    lambda: require(
                        empty.outputs.value.shape == (0, 1280)
                        and empty.outputs.status.shape == (0, 160),
                        "empty composition shape differs",
                    ),
                )
        finally:
            device.synchronize()
            empty.close()
        workspace.close()
        rejected(lambda: workspace.run(value))
        workspace = prepare(owner, name, rows=1)
        outputs = workspace.outputs
        with (
            workspace.graph_lease(),
            device.graph(lambda: workspace.run(value)) as replay,
        ):
            owner.close()
            rejected(lambda: workspace.run(value))
            workspace.close()
            rejected(lambda: prepare(owner, name, rows=1))
            for tensor in (*workspace.quantization.outputs, outputs.value):
                device.module.zero(tensor)
            replay()
            require(
                not device.download(outputs.status, np.uint8).any(),
                "closed-owner graph status differs",
            )
            compared = compare(
                composed_values(device, workspace),
                Fixture(
                    {"case": view["case"]},
                    {"value": host},
                    {},
                    {key: data[:1] for key, data in master.expected.items()},
                ),
                view["limits"],
            )
            require(compared.passed, f"closed-owner graph differs: {compared.failures}")
    finally:
        device.synchronize()
        workspace.close()
    return dict(empty_eager_graph=True, closed_owner_graph=True, invalid_contracts=True)


def check_rank_composition(device, root, manifest):
    from oh_my_vllm.models.deepseek_v41.checkpoint import DTYPE_BYTES, read_layout
    from oh_my_vllm.models.deepseek_v41.placement import (
        LAYOUT_SHA256,
        RankPlan,
        rank_plan,
    )
    from oh_my_vllm.models.deepseek_v41.weights import _load_plan

    metadata, cases = load_rank_encoded_projection()
    master = cases["wq_a"]
    states = {
        name: file_state((root / name).stat(follow_symlinks=False))
        for name in manifest["files"]
    }
    model, layout = read_layout(root, manifest["files"], layout_sha256=LAYOUT_SHA256)
    full = rank_plan(model, layout, "TP4", 0)
    names = master.manifest["case"]["weights"]
    selected = {name: full.selections[name] for name in names}
    for name, spec in selected.items():
        record = metadata["checkpoint_selections"][name]
        require(
            spec.dtype == record["header"]["dtype"]
            and spec.shape == tuple(record["shape"])
            and spec.start == (0, 0)
            and spec.stop == spec.shape
            and name not in full.promotions,
            "rank wq_a selection differs",
        )
    size = sum(
        int(np.prod(spec.shape)) * DTYPE_BYTES[spec.dtype] for spec in selected.values()
    )
    require(size == 6560000, "selected resident/source bytes differ")
    plan = RankPlan(
        MappingProxyType(selected), MappingProxyType({}), size, size, "TP4", 0
    )
    owner = _load_plan(root, manifest["files"], device, plan, limit=64 << 20)
    identity = owner.identity.copy()
    try:
        require(
            owner.persistent_bytes == owner.allocated_bytes == 6758400,
            "persistent arena differs",
        )
        require(owner.peak_bytes == 6764800, "packing overlap differs")
        weight = owner.packed[names[0]]
        encoded = device.download(weight.encoded, np.uint8)
        packed = device.download(weight.scales, np.uint8)
        scale = packed[scale_offsets(weight.outputs, weight.width)[::32]]
        for name, raw in zip(names, (encoded, scale), strict=True):
            require(
                hashlib.sha256(raw.tobytes()).hexdigest()
                == metadata["checkpoint_selections"][name]["raw_sha256"],
                "loaded encoded weight or scale differs from frozen original payload",
            )
        require(
            equal_bits(packed, packed_scales(scale, weight.outputs, weight.width, 8)),
            "loaded scale atoms differ",
        )
        result = dict(
            fixture_sha256=RANK_ENCODED_PROJECTION_SHA256,
            loader_identity=identity,
            persistent_bytes=owner.persistent_bytes,
            allocated_bytes=owner.allocated_bytes,
            peak_bytes=owner.peak_bytes,
            host_bound_bytes=owner.host_bound_bytes,
            checks=[
                check_composed_view(
                    device, owner, names[0], master, view, metadata["invalid_recipe"]
                )
                for view in metadata["views"].values()
            ],
        )
        result["contracts"] = check_composed_contracts(
            device, owner, names[0], master, metadata["views"]["1"]
        )
        return result
    finally:
        owner.close()
        for name in identity:
            require(
                file_state((root / name).stat(follow_symlinks=False)) == states[name],
                "checkpoint changed after loading",
            )


def check_case(device, case, limits, values):
    from oh_my_vllm.operators.projection.api import Inputs, Weights
    from oh_my_vllm.operators.projection.plan import make_plan
    from oh_my_vllm.operators.projection.workspace import prepare, prepare_weights

    shape = case.manifest["case"]["shape"]
    rows, width, outputs = (shape[key] for key in ("rows", "width", "outputs"))
    bits = 4 if case.manifest["case"]["options"]["weight_dtype"] == "I8" else 8
    host_weight, host_scales = values[0].view(np.uint8), values[1]
    raw = Weights(device.upload(host_weight), device.upload(host_scales))
    weights = prepare_weights(device, raw, bits=bits)
    plan = make_plan(rows=rows, outputs=outputs, width=width, bits=bits)
    workspace = prepare(plan, device, weights)
    inputs = Inputs(
        *(device.upload(case.inputs[name]) for name in ("encoded", "scales"))
    )
    result = {}
    try:
        wanted_scale = packed_scales(host_scales, outputs, width, bits)
        actual_scale = device.download(weights.scales, np.uint8)
        require(equal_bits(actual_scale, wanted_scale), "weight scale layout differs")
        padding = ((width + 127) // 128 * 128 - width) // 2 if bits == 4 else 0
        wanted_weight = np.pad(host_weight, ((0, 0), (0, padding)))
        require(
            equal_bits(device.download(weights.encoded, np.uint8), wanted_weight),
            "prepared encoded weights differ",
        )
        if not padding:
            require(weights.encoded is raw.encoded, "aligned weights were duplicated")
        originals = (
            (raw.encoded, host_weight),
            (raw.scales, host_scales),
            (weights.encoded, wanted_weight),
            (weights.scales, wanted_scale),
            (inputs.encoded, case.inputs["encoded"]),
            (inputs.scales, case.inputs["scales"]),
        )

        def check():
            actual = decode(
                device.download(workspace.outputs.value, np.uint16), "bfloat16"
            )
            compared = compare({"value": actual}, case, limits)
            require(compared.passed, f"projection differs: {compared.failures}")
            for first, last, column in case.manifest["case"]["options"]["zero_columns"]:
                zero = actual[first:last] if column < 0 else actual[first:last, column]
                require(np.max(np.abs(zero)) <= 1e-6, "cancellation endpoint differs")
            for tensor, value in originals:
                require(
                    equal_bits(device.download(tensor, value.dtype), value),
                    "projection changed its inputs or weights",
                )
            result.update(metrics=compared.metrics)

        execute_modes(device, (workspace,), lambda: workspace.run(inputs), check)
        result.update(
            case=case.manifest["case"]["id"],
            native=native_identity(workspace),
            modes=["eager", "graph"],
            prepared_bytes=wanted_weight.nbytes + wanted_scale.nbytes,
        )
        return result
    finally:
        device.synchronize()
        workspace.close()


def check_contracts(device):
    from oh_my_vllm.operators.projection.api import Inputs, Weights
    from oh_my_vllm.operators.projection.plan import make_plan
    from oh_my_vllm.operators.projection.workspace import prepare, prepare_weights

    base = dict(rows=1, outputs=32, width=32, bits=8)
    invalids = (
        dict(rows=-1),
        dict(rows=385),
        dict(rows=True),
        dict(outputs=33),
        dict(outputs=65536),
        dict(width=31),
        dict(width=32768),
        dict(bits=16),
        dict(bits=True),
    )
    for change in invalids:
        rejected(lambda change=change: make_plan(**(base | change)))

    def check_format(bits):
        raw = Weights(
            device.zeros((32, 32 * bits // 8), "uint8"),
            device.upload(np.full((32 if bits == 4 else 1, 1), 127, np.uint8)),
        )
        weights = prepare_weights(device, raw, bits=bits)
        rejected(
            lambda: prepare_weights(
                device, Weights(raw.encoded, device.empty((1,), "float32")), bits=bits
            )
        )
        rejected(
            lambda: prepare(
                make_plan(**(base | dict(width=64, bits=bits))), device, weights
            )
        )
        workspace = prepare(make_plan(**(base | dict(bits=bits))), device, weights)
        inputs = Inputs(
            device.zeros((1, 32), "uint8"),
            device.upload(np.full((1, 1), 127, np.uint8)),
        )
        try:
            overlap = device.empty((4096,), "uint8")
            rejected(
                lambda: workspace.module.pack_weights(
                    device.view(overlap, raw.encoded.shape),
                    raw.scales,
                    device.view(overlap, weights.encoded.shape, 16),
                    weights.scales,
                    bits,
                )
            )
            for invalid in (
                Inputs(device.empty((1, 32), "float32"), inputs.scales),
                Inputs(inputs.encoded, device.empty((1, 1), "int32")),
                Inputs(device.empty((2, 32), "uint8"), inputs.scales),
            ):
                rejected(lambda invalid=invalid: workspace.run(invalid))
            rejected(
                lambda: workspace.module.pack_weights(
                    *raw, weights.encoded, device.view(raw.encoded, (512,)), bits
                )
            )
            rejected(
                lambda: workspace.module.project(
                    inputs.encoded,
                    device.view(workspace.packed_input_scales, (1, 1)),
                    weights.encoded,
                    weights.scales,
                    workspace.outputs.value,
                    workspace.padded_input,
                    workspace.packed_input_scales,
                    workspace.scratch,
                    32,
                    bits,
                )
            )
        finally:
            device.synchronize()
            workspace.close()
        rejected(lambda: workspace.run(inputs))
        empty = prepare(make_plan(**(base | dict(rows=0, bits=bits))), device, weights)
        values = Inputs(device.empty((0, 32), "uint8"), device.empty((0, 1), "uint8"))
        try:
            execute_modes(
                device,
                (empty,),
                lambda: empty.run(values),
                lambda: require(
                    empty.outputs.value.shape[0] == 0, "empty rows changed"
                ),
            )
        finally:
            device.synchronize()
            empty.close()

    for bits in (4, 8):
        check_format(bits)
    return dict(
        invalid_plans=len(invalids),
        invalid_contracts=16,
        empty_eager_graph=True,
        closed_workspace=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grouped-only", action="store_true")
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.projection") is not None,
        "native projection boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    check_grouped_plans()
    sources = source_identity()
    raw = args.checkpoint_manifest.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CHECKPOINT_MANIFEST_SHA256,
        "checkpoint identity differs",
    )
    reader = CheckpointWeights(
        args.model,
        json.loads(raw),
        tensor_layout(json.loads((args.model / "config.json").read_text())),
    )
    metadata, cases = load_projection()
    identity = device_identity(selected_uuid())
    device = Device()
    checks, contracts, composition = [], None, None
    try:
        with caller_stream(device):
            if not args.grouped_only:
                checks = [
                    check_case(
                        device,
                        case,
                        metadata["cases"][name]["limits"],
                        weight_values(reader, case),
                    )
                    for name, case in cases.items()
                ]
                contracts = check_contracts(device)
                composition = check_rank_composition(
                    device, args.model, json.loads(raw)
                )
            grouped = [
                check_grouped_case(
                    device,
                    cases[name],
                    metadata["cases"][name]["limits"],
                    weight_values(reader, cases[name]),
                    groups,
                    repeats,
                )
                for name, groups, repeats in (
                    ("fp4_decode", 1, 1),
                    ("fp4_prefill", 96, 1),
                    ("fp4_verification", 384, 6),
                    ("fp4_draft", 128, 1),
                    ("fp4_decisions", 32, 1),
                )
            ]
            grouped_contracts = check_grouped_contracts(device)
        reader._verify_states()
        require(
            source_identity() == sources, "projection sources changed during execution"
        )
    finally:
        device.synchronize()
        device.close()
    with args.output.open("x") as output:
        json.dump(
            dict(
                passed=True,
                grouped_only=args.grouped_only,
                device=identity,
                fixture_sha256=PROJECTION_SHA256,
                checks=checks,
                contracts=contracts,
                rank_composition=composition,
                grouped=grouped,
                grouped_contracts=grouped_contracts,
                sources=sources,
            ),
            output,
            indent=2,
            allow_nan=False,
        )
        output.write("\n")


if __name__ == "__main__":
    main()
