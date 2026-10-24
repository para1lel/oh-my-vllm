"""Check fixed routing, exact expert dispatch and ordered FP32 combination."""

import argparse
import hashlib
import importlib.util
import json
from contextlib import ExitStack, nullcontext
from pathlib import Path

import numpy as np

from development.model.deepseek_v41.fixtures import (
    CHECKPOINT_MANIFEST_SHA256,
    EXACT_OPERATORS_SHA256,
    ROUTING_SHA256,
    load_exact_operators,
    load_routing,
    load_startup_promotion,
)
from development.model.deepseek_v41.layout import tensor_layout
from development.model.deepseek_v41.weights import Weights as CheckpointWeights
from development.serving.device import device_identity, selected_uuid
from tests.test_collectives import metadata_tensor

from .check_deepseek_exact import (
    caller_stream,
    execute_modes,
    native_identity,
    rejected,
    repeated_case,
    require,
)
from .check_deepseek_hc import source_identity as common_sources
from .check_deepseek_projection import prepared_group_values
from .fixtures import Fixture
from .numerics import compare
from .performance_inputs import bf16
from .storage import decode, equal_bits, storage_dtype

DISPATCH_POLICY_SHA256 = (
    "deeed6e23f6716a2668aa089d75c5b1ff89445c47600ba045831d64fae59b1ec"
)
DISPATCH_SUBSETS_SHA256 = (
    "0b492f3aba9e66d8b5fc55a19b3b5ada9c692269db3888ffda2eeb3a26817bd9"
)
DISPATCH_IDS = {
    384: [
        [383, 0, 192, 95, 288, 96],
        [96, 288, 0, 192, 383, 95],
        [191, 287, 1, 97, 193, 289],
        [95, 96, 191, 192, 287, 288],
    ],
    128: [[127, 0, 32], [64, 96, 31], [32, 95, 63], [0, 64, 127]],
}
DISPATCH_EDGES = {
    384: [
        1,
        8,
        14,
        3,
        11,
        18,
        5,
        6,
        19,
        15,
        12,
        20,
        2,
        9,
        21,
        16,
        13,
        22,
        4,
        7,
        23,
        17,
        0,
        10,
    ],
    128: [1, 9, 5, 2, 6, 8, 3, 10, 7, 4, 0, 11],
}
COMBINE_PATTERNS = (
    (0x4B80, 0x3F80, 0xCB80, 0x4000, 0x4080, 0xC080),
    (0xCB80, 0xBF80, 0x4B80, 0xC000, 0xC080, 0x4080),
    (0x0001, 0x0001, 0x8001, 0x0000, 0x0000, 0x0000),
    (0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000),
    (0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80, 0x3F80),
)
# Independent integer RN-even scalar table; full ownership, then the four quarters.
COMBINE_BITS = {
    384: (
        (
            "40000000 c0000000 00010000 00000000 40c00000",
            "40000000 c0000000 00010000 00000000 40c00000",
            "40000000 c0000000 00010000 00000000 40c00000",
            "40000000 c0000000 00010000 00000000 40c00000",
        ),
        (
            "4b800000 cb800000 00020000 00000000 40000000",
            "4b800000 cb800000 00020000 00000000 40000000",
            "4b800000 cb800000 00010000 00000000 3f800000",
            "4b800000 cb800000 00010000 00000000 3f800000",
        ),
        (
            "cb800000 4b800000 80010000 00000000 3f800000",
            "cb800000 4b800000 80010000 00000000 3f800000",
            "cb7fffff 4b7fffff 00000000 00000000 40000000",
            "cb7fffff 4b7fffff 00000000 00000000 40000000",
        ),
        (
            "40000000 c0000000 00000000 00000000 3f800000",
            "40000000 c0000000 00000000 00000000 3f800000",
            "40c00000 c0c00000 00000000 00000000 40000000",
            "40c00000 c0c00000 00000000 00000000 40000000",
        ),
        (
            "00000000 00000000 00000000 00000000 40000000",
            "00000000 00000000 00000000 00000000 40000000",
            "c0800000 40800000 00000000 00000000 3f800000",
            "c0800000 40800000 00000000 00000000 3f800000",
        ),
    ),
    128: (
        (
            "00000000 00000000 00010000 00000000 40400000",
            "00000000 00000000 00010000 00000000 40400000",
            "00000000 00000000 00010000 00000000 40400000",
            "00000000 00000000 00010000 00000000 40400000",
        ),
        (
            "4b800000 cb800000 00010000 00000000 3f800000",
            "4b800000 cb800000 00010000 00000000 3f800000",
            "00000000 00000000 00000000 00000000 00000000",
            "4b800000 cb800000 00010000 00000000 3f800000",
        ),
        (
            "3f800000 bf800000 00010000 00000000 3f800000",
            "00000000 00000000 00000000 00000000 00000000",
            "4b800000 cb800000 00020000 00000000 40000000",
            "00000000 00000000 00000000 00000000 00000000",
        ),
        (
            "00000000 00000000 00000000 00000000 00000000",
            "3f800000 bf800000 00010000 00000000 3f800000",
            "cb800000 4b800000 80010000 00000000 3f800000",
            "3f800000 bf800000 00010000 00000000 3f800000",
        ),
        (
            "cb800000 4b800000 80010000 00000000 3f800000",
            "cb800000 4b800000 80010000 00000000 3f800000",
            "00000000 00000000 00000000 00000000 00000000",
            "cb800000 4b800000 80010000 00000000 3f800000",
        ),
    ),
}


def source_identity():
    repository = Path(__file__).resolve().parents[2]
    sources = common_sources()
    for family in ("routing", "projection"):
        for path in (repository / "python/oh_my_vllm/operators" / family).rglob("*.py"):
            sources[str(path.relative_to(repository))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    path = repository / "tests/test_collectives.py"
    sources[str(path.relative_to(repository))] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    return sources


def check_dispatch_plans():
    from inspect import signature

    from oh_my_vllm.operators.routing import api, plan

    require(
        callable(getattr(plan, "make_dispatch_plan", None)),
        "expert dispatch is not implemented",
    )
    require(
        hasattr(api, "EncodedDispatchInputs")
        and "encoded" in signature(plan.make_dispatch_plan).parameters,
        "encoded expert dispatch is not implemented",
    )
    for rows in (0, 1, 128, 384):
        for experts in (128, 384):
            for count in (experts, experts // 4):
                value = plan.make_dispatch_plan(
                    rows=rows, experts=experts, expert_count=count
                )
                capacity = rows * (6 if experts == 384 else 3)
                require(
                    value.storage
                    == (
                        ((capacity, 5120), "bfloat16"),
                        ((capacity,), "float32"),
                        ((count,), "int32"),
                        ((count + 1,), "int32"),
                        ((rows, value.topk), "int32"),
                        ((rows,), "uint8"),
                    )
                    and value.combine_storage
                    == (((rows, 5120), "float32"), ((rows,), "uint8"))
                    and value.scratch == ((capacity,), "int32"),
                    "dispatch capacity or output schema differs",
                )
                encoded = plan.make_dispatch_plan(
                    rows=rows, experts=experts, expert_count=count, encoded=True
                )
                require(
                    encoded.storage
                    == (
                        ((capacity, 5120), "uint8"),
                        ((((capacity + 127) // 128 + count) * 128 * 160,), "uint8"),
                        *value.storage[1:],
                    )
                    and encoded.reuse_key != value.reuse_key,
                    "encoded dispatch storage or reuse key differs",
                )
    for options in (
        {"rows": -1},
        {"rows": 385},
        {"rows": True},
        {"experts": 256},
        {"expert_count": 0},
        {"expert_count": 48},
        {"expert_count": True},
        {"expert_start": -1},
        {"expert_start": True},
        {"expert_start": 384},
        {"expert_start": 1, "expert_count": 96},
        {"expert_start": 96, "expert_count": 384},
        {"encoded": 1},
    ):
        rejected(
            lambda options=options: plan.make_dispatch_plan(
                **(dict(rows=1, experts=384) | options)
            )
        )
    require(
        len(
            {
                plan.make_dispatch_plan(
                    rows=1, experts=384, expert_start=start, expert_count=96
                ).reuse_key
                for start in (0, 96, 192, 288)
            }
        )
        == 4,
        "dispatch reuse key lost expert ownership",
    )


def dispatch_sample(
    fixture, rows, experts, start, count, state="valid", quantized=None
):
    from oh_my_vllm.operators.routing.api import DispatchOutputs, EncodedDispatchOutputs

    topk = 6 if experts == 384 else 3
    capacity = rows * topk
    seeds = np.arange(rows) % (4 if rows <= 4 else 2 if topk == 6 else 1)
    ids = np.asarray(DISPATCH_IDS[experts], np.int32)[seeds].copy()
    routes = np.resize(
        np.array(
            [0x3E000000, 0x3E800000, 0x80000000, 0x3F000000, 0x3F400000, 0x3FC00000],
            np.uint32,
        ),
        (rows, topk),
    ).view(np.float32)
    values = (
        np.resize(fixture["bf16_words"], (rows, 5120))
        if quantized is None
        else quantized.expected["encoded"]
    )
    quantization_status = np.zeros((rows, 160), np.uint8)
    input_scales = None if quantized is None else quantized.expected["scales"]
    if state == "quantization":
        quantization_status[(0, 1, 2), (0, 159, 71)] = 1
        values, input_scales = values.copy(), input_scales.copy()
        values.reshape(rows, 160, 32)[quantization_status != 0] = 0
        input_scales[quantization_status != 0] = 0
    if state in ("invalid", "invalid_other"):
        ids[0, 0] = experts if state == "invalid" else -1
        duplicate = count if start == 0 and count < experts else 0
        ids[1, :2] = duplicate
        routes[2, 0] = np.nan if state == "invalid" else np.inf
    elif state == "empty":
        ids[:] = np.arange(topk) + (count if start == 0 else 0)
    valid = np.array(
        [
            np.all((row >= 0) & (row < experts))
            and len(set(row)) == topk
            and np.isfinite(route).all()
            for row, route in zip(ids, routes, strict=True)
        ],
        dtype=bool,
    )
    valid &= ~quantization_status.any(axis=1)
    edges = sorted(
        (int(ids[r, j]), r, j)
        for r in range(rows)
        for j in range(topk)
        if valid[r] and start <= ids[r, j] < start + count
    )
    packed = np.zeros((capacity, 5120), values.dtype)
    compact_scales = np.zeros((capacity, 160), np.uint8)
    coefficients = np.zeros(capacity, np.float32)
    counts = np.zeros(count, np.int32)
    source_map = np.full((rows, topk), -1, np.int32)
    down = np.full((capacity, 5120), 0x7FC0, np.uint16)
    patterns = np.asarray(COMBINE_PATTERNS, np.uint16)
    for position, (expert, row, slot) in enumerate(edges):
        packed[position] = values[row]
        if quantized is not None:
            compact_scales[position] = input_scales[row]
        coefficients[position] = routes[row, slot]
        counts[expert - start] += 1
        source_map[row, slot] = position
        ordinal = sorted(ids[row]).index(expert)
        down[position] = np.resize(patterns[:, ordinal], 5120)
    offsets = np.concatenate((np.zeros(1, np.int32), np.cumsum(counts, dtype=np.int32)))
    group = 0 if count == experts else 1 + start // count
    scalar = np.array(
        [
            [int(word, 16) for word in line.split()]
            for line in COMBINE_BITS[experts][group]
        ],
        np.uint32,
    )
    combined = np.tile(scalar[seeds], (1, 1024))
    combined[~valid] = 0
    if state == "empty":
        combined[:] = 0
    expected = DispatchOutputs(
        packed, coefficients, counts, offsets, source_map, (~valid).astype(np.uint8)
    )
    if quantized is not None:
        encoded, scales = prepared_group_values(packed, compact_scales, counts, offsets)
        return (
            (values, input_scales, quantization_status, ids, routes),
            EncodedDispatchOutputs(encoded, scales, *expected[1:]),
            down,
            combined,
        )
    return (values, ids, routes), expected, down, combined


def check_dispatch_literals():
    fixture = load_startup_promotion()
    quantized = dispatch_quantization(load_exact_operators()[1]["activation_fp8"], 4)
    for experts in (128, 384):
        _, expected, _, combined = dispatch_sample(fixture, 4, experts, 0, experts)
        order = np.argsort(expected.source_map.ravel()).tolist()
        require(order == DISPATCH_EDGES[experts], "literal packed edge order differs")
        require(
            expected.offsets[-1] == 4 * (6 if experts == 384 else 3)
            and np.all(combined[:, 2::5] == 0x10000),
            "literal capacity or subnormal anchor differs",
        )
        _, encoded, _, _ = dispatch_sample(
            fixture, 4, experts, 0, experts, quantized=quantized
        )
        topk = 6 if experts == 384 else 3
        require(
            equal_bits(encoded.source_map, expected.source_map)
            and equal_bits(
                encoded.encoded[: len(order)],
                quantized.expected["encoded"][
                    np.array(DISPATCH_EDGES[experts]) // topk
                ],
            ),
            "encoded dispatch differs from literal permutation",
        )


def dispatch_quantization(fixture, rows):
    expanded = repeated_case(fixture, rows, 5120)
    row, group = np.indices((rows, 160))
    # Keep complete frozen groups; make row-32/128 and scale-atom swaps observable.
    index = (row + row // 32 + row // 128 + group + group // 4) % 4

    def permute(value):
        return value.reshape(4, -1)[index].reshape(rows, 160 * (value.size // 4))

    return Fixture(
        expanded.manifest,
        {"values": permute(fixture.inputs["values"])},
        {},
        {name: permute(value) for name, value in fixture.expected.items()},
    )


def check_dispatch_case(device, fixture, rows, experts, start, count, quantized=None):
    from oh_my_vllm.operators.quantization.api import Inputs as QuantizationInputs
    from oh_my_vllm.operators.quantization.plan import make_plan
    from oh_my_vllm.operators.quantization.workspace import prepare
    from oh_my_vllm.operators.routing.api import DispatchInputs, EncodedDispatchInputs
    from oh_my_vllm.operators.routing.plan import make_dispatch_plan
    from oh_my_vllm.operators.routing.workspace import prepare_dispatch

    states = ["valid"]
    if rows == 4:
        states += ["invalid", "valid", "invalid_other", "valid"]
        if quantized is not None:
            states += ["quantization", "valid"]
        if count != experts:
            states += ["empty", "valid"]
    samples = [
        dispatch_sample(fixture, rows, experts, start, count, state, quantized)
        for state in states
    ]
    plan = make_dispatch_plan(
        rows=rows,
        experts=experts,
        expert_start=start,
        expert_count=count,
        encoded=quantized is not None,
    )
    workspace = prepare_dispatch(plan, device)
    quantizer = (
        prepare(make_plan(rows=rows, width=5120, mode="fp8_ue8m0"), device)
        if quantized is not None
        else None
    )
    source = device.empty((rows, 5120), "bfloat16")
    payload = (
        (source,)
        if quantizer is None
        else (
            quantizer.outputs.encoded,
            quantizer.outputs.scales,
            quantizer.outputs.status,
        )
    )
    inputs = (DispatchInputs if quantized is None else EncodedDispatchInputs)(
        *payload,
        device.empty((rows, plan.topk), "int32"),
        device.empty((rows, plan.topk), "float32"),
    )
    down = device.empty((rows * plan.topk, 5120), "bfloat16")
    source_hosts = []
    for sample in samples:
        value = sample[0][0] if quantized is None else bf16(quantized.inputs["values"])
        if quantized is not None:
            row, group = np.nonzero(sample[0][2])
            value[row, group * 32] = 0x7FC0
        source_hosts.append((value, *sample[0][-2:], sample[2]))
    upload_targets = source, inputs.expert_ids, inputs.coefficients, down
    uploads = [
        tuple(
            device.upload(value, dtype)
            for value, dtype in zip(
                host,
                ("bfloat16", "int32", "float32", "bfloat16"),
                strict=True,
            )
        )
        for host in source_hosts
    ]
    outputs = (
        *(quantizer.outputs if quantizer else ()),
        *workspace.outputs,
        *workspace.combined,
        workspace.source_rows,
    )
    sentinels = [
        device.upload(np.full(t.shape, 0x5A, storage_dtype(str(t.dtype))), str(t.dtype))
        for t in outputs
    ]

    def operation():
        if quantizer is not None:
            quantizer.run(QuantizationInputs(source))
        workspace.run(inputs)
        return workspace.combine(down)

    try:
        for mode in ("eager", "graph"):
            with ExitStack() as resources:
                for owner in (workspace, quantizer):
                    if owner is not None:
                        resources.enter_context(owner.graph_lease())
                invoke = resources.enter_context(
                    device.graph(operation)
                    if mode == "graph"
                    else nullcontext(operation)
                )
                for sample, host, upload in zip(
                    samples, source_hosts, uploads, strict=True
                ):
                    for target, uploaded in zip(upload_targets, upload, strict=True):
                        device.module.copy_bytes(target, uploaded)
                    source_rows = np.full(rows * plan.topk, -1, np.int32)
                    row_ids, slots = np.where(sample[1].source_map >= 0)
                    source_rows[sample[1].source_map[row_ids, slots]] = row_ids
                    quantization_expected = ()
                    if quantizer is not None:
                        roundtrip = quantized.expected["dequantized_bf16_bits"].copy()
                        roundtrip.reshape(rows, 160, 32)[sample[0][2] != 0] = 0
                        quantization_expected = (
                            *sample[0][:2],
                            roundtrip,
                            sample[0][2],
                        )
                    expected = (
                        *quantization_expected,
                        *sample[1],
                        sample[3],
                        sample[1].status,
                        source_rows,
                    )
                    for _ in range(2):
                        for output, sentinel in zip(outputs, sentinels, strict=True):
                            device.module.copy_bytes(output, sentinel)
                        invoke()
                        require(
                            all(
                                equal_bits(device.download(output, host.dtype), host)
                                for output, host in zip(outputs, expected, strict=True)
                            ),
                            "dispatch or ordered combine differs",
                        )
                        require(
                            all(
                                equal_bits(device.download(tensor, host.dtype), host)
                                for tensor, host in zip(
                                    upload_targets,
                                    host,
                                    strict=True,
                                )
                            ),
                            "dispatch/combine changed an input",
                        )
                if mode == "graph" and quantizer is not None:
                    quantizer.close()
                    workspace.close()
                    rejected(lambda: operation())
                    for output, sentinel in zip(outputs, sentinels, strict=True):
                        device.module.copy_bytes(output, sentinel)
                    invoke()
                    require(
                        all(
                            equal_bits(device.download(t, value.dtype), value)
                            for t, value in zip(outputs, expected, strict=True)
                        ),
                        "composed graph lost closed producer storage",
                    )
        return dict(
            rows=rows,
            experts=experts,
            expert_start=start,
            expert_count=count,
            states=states,
            exact=True,
            modes=["eager", "graph"],
            native=native_identity(workspace),
            encoded=quantized is not None,
            quantization_native=native_identity(quantizer) if quantizer else None,
            live_quantization=quantizer is not None,
        )
    finally:
        device.synchronize()
        workspace.close()
        if quantizer is not None:
            quantizer.close()


def check_dispatch_failures(device, fixture, encoded=False):
    from oh_my_vllm.operators.routing.api import DispatchInputs, EncodedDispatchInputs
    from oh_my_vllm.operators.routing.plan import make_dispatch_plan
    from oh_my_vllm.operators.routing.workspace import prepare_dispatch

    workspace = prepare_dispatch(
        make_dispatch_plan(rows=4, experts=128, encoded=encoded), device
    )
    quantized = (
        dispatch_quantization(load_exact_operators()[1]["activation_fp8"], 4)
        if encoded
        else None
    )
    host, expected, down_host, result = dispatch_sample(
        fixture, 4, 128, 0, 128, quantized=quantized
    )
    dtypes = (
        ("uint8", "uint8", "uint8", "int32", "float32")
        if encoded
        else ("bfloat16", "int32", "float32")
    )
    inputs = (EncodedDispatchInputs if encoded else DispatchInputs)(
        *(device.upload(v, dtype) for v, dtype in zip(host, dtypes, strict=True))
    )
    down = device.upload(down_host, "bfloat16")
    value_field, value_dtype = (
        ("encoded", "uint8") if encoded else ("values", "bfloat16")
    )
    dispatch_native = (
        workspace.module.dispatch_encoded if encoded else workspace.module.dispatch
    )
    try:
        for invalid in (
            inputs._replace(**{value_field: device.empty((4, 5120), "float32")}),
            inputs._replace(**{value_field: device.empty((4, 256), value_dtype)}),
            inputs._replace(expert_ids=device.empty((4, 3), "int64")),
            inputs._replace(coefficients=device.empty((4, 3), "bfloat16")),
            inputs._replace(
                **{
                    value_field: device.view(
                        device.empty((4 * 5120 + 1,), value_dtype), (4, 5120), 1
                    )
                }
            ),
        ):
            rejected(lambda invalid=invalid: workspace.run(invalid))
        if encoded:
            for field in ("scales", "status"):
                for invalid in (
                    device.empty((4, 159), "uint8"),
                    device.empty((4, 160), "float32"),
                ):
                    rejected(
                        lambda field=field, invalid=invalid: workspace.run(
                            inputs._replace(**{field: invalid})
                        )
                    )
            rejected(
                lambda: dispatch_native(
                    *inputs,
                    *workspace.outputs._replace(
                        packed_scales=device.empty((1,), "uint8")
                    ),
                    workspace.source_rows,
                    128,
                    0,
                    128,
                )
            )
            rejected(
                lambda: workspace.run(
                    DispatchInputs(
                        inputs.encoded, inputs.expert_ids, inputs.coefficients
                    )
                )
            )
        for output in (
            workspace.outputs._replace(source_map=inputs.expert_ids),
            workspace.outputs._replace(
                coefficients=device.view(inputs.coefficients, (12,))
            ),
            workspace.outputs._replace(
                counts=device.view(workspace.outputs.offsets, (128,))
            ),
            workspace.outputs._replace(offsets=device.empty((128,), "int32")),
        ):
            rejected(
                lambda output=output: dispatch_native(
                    *inputs, *output, workspace.source_rows, 128, 0, 128
                )
            )
        for scratch in (
            device.empty((11,), "int32"),
            device.view(inputs.expert_ids, (12,)),
            device.view(workspace.outputs.source_map, (12,)),
        ):
            rejected(
                lambda scratch=scratch: dispatch_native(
                    *inputs, *workspace.outputs, scratch, 128, 0, 128
                )
            )
        for options in (dict(stride=2), dict(device=1)):
            invalid, guard = metadata_tensor(
                4,
                pointer=workspace.outputs.status.data_ptr(),
                code=1,
                bits=8,
                **options,
            )
            rejected(
                lambda invalid=invalid: dispatch_native(
                    *inputs,
                    *workspace.outputs._replace(status=invalid),
                    workspace.source_rows,
                    128,
                    0,
                    128,
                )
            )
            rejected(
                lambda invalid=invalid: workspace.module.combine(
                    down, workspace.outputs.source_map, invalid, *workspace.combined
                )
            )
            del invalid, guard
        alias, guard = metadata_tensor(
            4 * 5120, pointer=down.data_ptr(), code=2, bits=32
        )
        alias = device.view(alias, (4, 5120))
        rejected(
            lambda alias=alias: workspace.module.combine(
                down,
                workspace.outputs.source_map,
                workspace.outputs.status,
                alias,
                workspace.combined.status,
            )
        )
        del alias, guard
        rejected(lambda: workspace.combine(device.empty((11, 5120), "bfloat16")))
        rejected(lambda: workspace.combine(device.empty((12, 5120), "float32")))

        workspace.run(inputs)
        maps, downs = [], []
        for kind in ("negative", "outside", "duplicate", "nan", "inf", "overflow"):
            changed_map, changed_down = expected.source_map.copy(), down_host.copy()
            if kind == "negative":
                changed_map[0, 0] = -2
            elif kind == "outside":
                changed_map[0, 0] = 12
            elif kind == "duplicate":
                changed_map[0, 1] = changed_map[0, 0]
            elif kind in ("nan", "inf"):
                changed_down[changed_map[0, 0], 0] = 0x7FC0 if kind == "nan" else 0x7F80
            else:
                changed_down[np.sort(changed_map[0])[:2], 0] = 0x7F7F
            maps.append(device.upload(changed_map))
            downs.append(device.upload(changed_down, "bfloat16"))
        clean_map, clean_down = (
            device.upload(expected.source_map),
            device.upload(down_host, "bfloat16"),
        )
        poison = [
            device.upload(
                np.full(t.shape, 0x5A, storage_dtype(str(t.dtype))), str(t.dtype)
            )
            for t in workspace.combined
        ]
        for mode in ("eager", "graph"):
            with (
                workspace.graph_lease(),
                (
                    device.graph(lambda: workspace.combine(down))
                    if mode == "graph"
                    else nullcontext(lambda: workspace.combine(down))
                ) as invoke,
            ):
                for changed_map, changed_down in zip(maps, downs, strict=True):
                    for bad in (False, True, False):
                        device.module.copy_bytes(
                            workspace.outputs.source_map,
                            changed_map if bad else clean_map,
                        )
                        device.module.copy_bytes(
                            down, changed_down if bad else clean_down
                        )
                        for target, source in zip(
                            workspace.combined, poison, strict=True
                        ):
                            device.module.copy_bytes(target, source)
                        invoke()
                        value, status = result.copy(), np.zeros(4, np.uint8)
                        if bad:
                            value[0], status[0] = 0, 1
                        require(
                            equal_bits(
                                device.download(workspace.combined.value, np.uint32),
                                value,
                            )
                            and equal_bits(
                                device.download(workspace.combined.status, np.uint8),
                                status,
                            ),
                            "invalid combine was not isolated or status was stale",
                        )
        with (
            workspace.graph_lease(),
            device.graph(
                lambda: (workspace.run(inputs), workspace.combine(down))
            ) as replay,
        ):
            workspace.close()
            rejected(lambda: workspace.run(inputs))
            rejected(lambda: workspace.combine(down))
            rejected(lambda: workspace.graph_lease().__enter__())
            device.module.zero(workspace.combined.value)
            replay()
            require(
                equal_bits(
                    device.download(workspace.combined.value, np.uint32), result
                ),
                "dispatch graph lost resources after workspace close",
            )
    finally:
        device.synchronize()
        workspace.close()
    return dict(
        encoded=encoded,
        invalid_arguments=True,
        invalid_combine_isolation=True,
        finite_invalid_finite=True,
        closed_owner_graph=True,
    )


def check_dispatch(device):
    check_dispatch_literals()
    fixture = load_startup_promotion()
    _, exact = load_exact_operators()
    configurations = [
        (4, experts, start, count)
        for experts in (128, 384)
        for start, count in [(0, experts)]
        + [(group * (experts // 4), experts // 4) for group in range(4)]
    ]
    configurations += [
        (0, 128, 0, 32),
        (0, 384, 0, 384),
        (1, 128, 0, 128),
        (1, 384, 0, 384),
        (128, 128, 0, 32),
        (384, 384, 0, 384),
    ]
    return dict(
        policy_sha256=DISPATCH_POLICY_SHA256,
        scalar_table_sha256=DISPATCH_SUBSETS_SHA256,
        exact_quantization_sha256=EXACT_OPERATORS_SHA256,
        cases=[
            check_dispatch_case(
                device,
                fixture,
                *case,
                dispatch_quantization(exact["activation_fp8"], case[0])
                if encoded
                else None,
            )
            for case in configurations
            for encoded in (False, True)
        ],
        contracts=[
            check_dispatch_failures(device, fixture, encoded)
            for encoded in (False, True)
        ],
    )


def check_contracts(device):
    from oh_my_vllm.operators.routing.api import Inputs, Weights
    from oh_my_vllm.operators.routing.plan import make_plan
    from oh_my_vllm.operators.routing.workspace import prepare

    weights = Weights(*(device.zeros((384,), "float32") for _ in range(2)))
    workspace = prepare(make_plan(rows=1, experts=384), device, weights)
    logits = device.zeros((1, 384), "float32")
    mask = device.zeros((1,), "uint8")
    try:
        for invalid in (
            Inputs(device.empty((1, 384), "float64"), mask),
            Inputs(device.empty((1, 128), "float32"), mask),
            Inputs(logits, device.empty((1,), "int32")),
            Inputs(logits, device.empty((2,), "uint8")),
        ):
            rejected(lambda invalid=invalid: workspace.run(invalid))
        rejected(
            lambda: workspace.module.route(
                logits,
                *weights,
                mask,
                workspace.outputs.expert_ids,
                device.view(logits, (1, 6)),
            )
        )
        rejected(
            lambda: workspace.module.route(
                logits,
                device.empty((384,), "float64"),
                weights.bias_vl,
                mask,
                *workspace.outputs,
            )
        )
    finally:
        device.synchronize()
        workspace.close()
    for rows, experts in ((-1, 384), (True, 384), (385, 384), (1, 256), (1, True)):
        rejected(
            lambda rows=rows, experts=experts: make_plan(rows=rows, experts=experts)
        )
    host_logits = np.zeros((4, 384), np.float32)
    host_logits[0, 17] = np.nan
    image_bias = np.zeros(384, np.float32)
    image_bias[37] = np.nan
    mixed = prepare(
        make_plan(rows=4, experts=384),
        device,
        Weights(device.zeros((384,), "float32"), device.upload(image_bias)),
    )
    mixed_inputs = Inputs(
        device.upload(host_logits),
        device.upload(np.array([0, 2, 1, 0], np.uint8)),
    )
    try:

        def check_invalid_rows():
            ids = device.download(mixed.outputs.expert_ids, np.int32)
            coefficients = device.download(mixed.outputs.coefficients, np.float32)
            require(np.all(ids[:3] == -1), "invalid routing row has expert IDs")
            require(np.isnan(coefficients[:3]).all(), "invalid routing row is finite")
            require(
                np.array_equal(ids[3], np.arange(6))
                and np.allclose(coefficients[3], 0.25, rtol=0, atol=1e-7),
                "invalid routing row affected a valid text row",
            )

        execute_modes(
            device, (mixed,), lambda: mixed.run(mixed_inputs), check_invalid_rows
        )
    finally:
        device.synchronize()
        mixed.close()
    for experts in (128, 384):
        weights = Weights(*(device.empty((experts,), "float32") for _ in range(2)))
        empty = prepare(make_plan(rows=0, experts=experts), device, weights)
        inputs = Inputs(
            device.empty((0, experts), "float32"), device.empty((0,), "uint8")
        )
        try:
            execute_modes(
                device,
                (empty,),
                lambda empty=empty, inputs=inputs: empty.run(inputs),
                lambda empty=empty: require(
                    all(tensor.shape[0] == 0 for tensor in empty.outputs),
                    "empty routing changed shape",
                ),
            )
        finally:
            device.synchronize()
            empty.close()
        rejected(lambda empty=empty, inputs=inputs: empty.run(inputs))
    return {
        "invalid_arguments": 11,
        "invalid_value_isolation": True,
        "empty_eager_graph": True,
        "closed_workspace": True,
    }


def check_case(device, case, limits, parameters):
    from oh_my_vllm.operators.routing.api import Inputs, Weights
    from oh_my_vllm.operators.routing.plan import make_plan
    from oh_my_vllm.operators.routing.workspace import prepare

    groups = case.manifest["case"]["options"].get("groups", {"": None})
    workspaces, inputs, originals, outputs = [], [], [], {}
    try:
        for group in groups:
            prefix = group + "_" if group else ""
            logits, mask = (
                case.inputs[prefix + field] for field in ("logits", "image_mask")
            )
            biases = (
                [case.inputs[prefix + field] for field in ("bias", "bias_vl")]
                if group
                else parameters
            )
            weights = Weights(*(device.upload(value) for value in biases))
            values = Inputs(device.upload(logits), device.upload(mask))
            workspace = prepare(
                make_plan(rows=len(logits), experts=logits.shape[1]), device, weights
            )
            workspaces.append(workspace)
            inputs.append(values)
            originals.extend(
                zip((*values, *weights), (logits, mask, *biases), strict=True)
            )
            outputs.update(
                (prefix + name, tensor)
                for name, tensor in workspace.outputs._asdict().items()
            )
        result = {}

        def operation():
            for workspace, values in zip(workspaces, inputs, strict=True):
                workspace.run(values)

        def check():
            actual = {
                name: decode(
                    device.download(tensor, storage_dtype(str(tensor.dtype))),
                    str(tensor.dtype),
                )
                for name, tensor in outputs.items()
            }
            compared = compare(actual, case, limits)
            require(compared.passed, f"routing differs: {compared.failures}")
            for tensor, value in originals:
                require(
                    equal_bits(device.download(tensor, value.dtype), value),
                    "routing changed its inputs or weights",
                )
            result.update(metrics=compared.metrics)

        execute_modes(device, workspaces, operation, check)
        result.update(
            case=case.manifest["case"]["id"],
            native=native_identity(workspaces[0]),
            modes=["eager", "graph"],
        )
        return result
    finally:
        device.synchronize()
        for workspace in workspaces:
            workspace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check_dispatch_plans()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.routing") is not None,
        "native routing boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    sources = source_identity()
    raw = args.checkpoint_manifest.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CHECKPOINT_MANIFEST_SHA256,
        "checkpoint identity differs",
    )
    config = json.loads((args.model / "config.json").read_text())
    reader = CheckpointWeights(args.model, json.loads(raw), tensor_layout(config))
    metadata, cases = load_routing()
    identity = device_identity(selected_uuid())
    device = Device()
    try:
        with caller_stream(device):
            checks = []
            for name, case in cases.items():
                parameters = [
                    reader.tensor(weight).astype(np.float32)
                    for weight in case.manifest["case"]["weights"][1:]
                ]
                checks.append(
                    check_case(
                        device, case, metadata["cases"][name]["limits"], parameters
                    )
                )
            contracts = check_contracts(device)
            dispatch = check_dispatch(device)
        reader._verify_states()
        require(
            source_identity() == sources, "routing sources changed during execution"
        )
    finally:
        device.synchronize()
        device.close()
    report = dict(
        passed=True,
        device=identity,
        fixture_sha256=ROUTING_SHA256,
        checks=checks,
        contracts=contracts,
        dispatch=dispatch,
        sources=sources,
    )
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
