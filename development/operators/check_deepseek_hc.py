"""Check frozen HC boundaries and fused MoE rounding for finite inputs."""

import argparse
import hashlib
import importlib.util
import json
from functools import partial
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from development.model.deepseek_v41.fixtures import (
    CHECKPOINT_MANIFEST_SHA256,
    HYPERCONNECTION_SHA256,
    load_hyperconnection,
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
    require,
)
from .check_deepseek_exact import (
    source_identity as common_sources,
)
from .fixtures import Fixture
from .numerics import compare
from .performance_inputs import bf16
from .storage import decode, equal_bits, storage_dtype


def source_identity():
    from oh_my_vllm.operators.runtime.native import source_files

    repository = Path(__file__).resolve().parents[2]
    sources = common_sources()
    paths = list(
        (repository / "python/oh_my_vllm/operators/hyperconnection").rglob("*.py")
    )
    paths.extend(
        repository / "development/model/deepseek_v41" / name
        for name in ("weights.py", "layout.py", "identity.py", "quantization.py")
    )
    paths.append(repository / "tests/test_collectives.py")
    for name in (
        "hyperconnection/cuda/hyperconnection.cu",
        "runtime/cuda/device.cu",
    ):
        paths.extend(source_files(repository / "python/oh_my_vllm/operators" / name))
    for path in paths:
        sources[str(path.relative_to(repository))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return sources


def check_contracts(device, parameters):
    from oh_my_vllm.operators.hyperconnection.api import Inputs, Weights
    from oh_my_vllm.operators.hyperconnection.plan import make_plan
    from oh_my_vllm.operators.hyperconnection.workspace import prepare

    weights = Weights(*(device.upload(value) for value in parameters))
    workspace = prepare(make_plan(rows=1), device, weights)
    residual = device.zeros((1, 4, 5120), "bfloat16")
    incoming = device.zeros((1, 4), "float32")
    branch = device.zeros((1, 5120), "bfloat16")
    routed = device.zeros((1, 5120), "float32")
    fused_rejections = 0
    try:
        for invalid in (
            Inputs(device.empty((1, 4, 5120), "float32"), incoming, branch),
            Inputs(residual, device.empty((1, 3), "float32"), branch),
            Inputs(residual, incoming, device.empty((1, 5120), "float32")),
        ):
            rejected(lambda invalid=invalid: workspace.run(invalid))
        rejected(
            lambda: workspace.module.hc_expand(
                residual,
                branch,
                workspace.storage.post,
                workspace.storage.comb,
                residual,
            )
        )
        rejected(lambda: workspace.collapse(branch, incoming))
        rejected(lambda: workspace.collapse(residual, branch))
        rejected(
            lambda: workspace.module.hc_collapse(
                residual, incoming, device.view(residual, (1, 5120))
            )
        )
        arguments = [
            device.view(device.zeros((20480,), dtype), shape)
            for shape, dtype in (
                ((1, 4, 5120), "bfloat16"),
                ((1, 5120), "float32"),
                ((1, 5120), "bfloat16"),
                ((1, 4), "float32"),
                ((1, 4, 4), "float32"),
                ((1, 4, 5120), "bfloat16"),
            )
        ]
        for shared_dtype in ("bfloat16", "float32"):
            arguments[2] = device.view(device.zeros((20480,), shared_dtype), (1, 5120))
            for index, value in enumerate(arguments):
                for invalid in (
                    device.empty(
                        value.shape,
                        "float32" if str(value.dtype) == "bfloat16" else "bfloat16",
                    ),
                    device.empty(
                        (value.shape[0] + 1, *value.shape[1:]), str(value.dtype)
                    ),
                ):
                    if index == 2 and invalid.shape == value.shape:
                        continue  # Both shared dtypes are valid.
                    invalid_arguments = arguments.copy()
                    invalid_arguments[index] = invalid
                    rejected(
                        lambda args=invalid_arguments: workspace.module.hc_expand_moe(
                            *args
                        )
                    )
                    fused_rejections += 1
            for left, right in combinations(range(6), 2):
                value = arguments[right]
                alias, guards = metadata_tensor(
                    int(np.prod(value.shape)),
                    pointer=arguments[left].data_ptr(),
                    code=4 if str(value.dtype) == "bfloat16" else 2,
                    bits=16 if str(value.dtype) == "bfloat16" else 32,
                )
                alias = device.view(alias, value.shape)
                invalid_arguments = arguments.copy()
                invalid_arguments[right] = alias
                rejected(
                    lambda args=invalid_arguments: workspace.module.hc_expand_moe(*args)
                )
                fused_rejections += 1
                del invalid_arguments, alias, guards
        for invalid_routed, invalid_shared in (
            (branch, branch),
            (routed, incoming),
            (routed, device.empty((1, 5120), "int32")),
        ):
            rejected(
                lambda routed=invalid_routed, shared=invalid_shared: (
                    workspace.finish_moe(residual, routed, shared)
                )
            )
            fused_rejections += 1
    finally:
        device.synchronize()
        workspace.close()
    for count in (-1, True, 385):
        rejected(lambda count=count: make_plan(rows=count))
    empty = prepare(make_plan(rows=0), device, weights)
    inputs = Inputs(
        device.empty((0, 4, 5120), "bfloat16"),
        device.empty((0, 4), "float32"),
        device.empty((0, 5120), "bfloat16"),
    )
    try:
        execute_modes(
            device,
            (empty,),
            lambda: empty.run(inputs),
            lambda: require(
                all(tensor.shape[0] == 0 for tensor in empty.outputs),
                "empty HC changed shape",
            ),
        )
        execute_modes(
            device,
            (empty,),
            lambda: empty.collapse(inputs.residual, inputs.incoming_pre),
            lambda: require(
                empty.outputs.collapsed.shape[0] == 0, "empty collapse changed shape"
            ),
        )
        empty_routed = device.empty((0, 5120), "float32")
        for dtype in ("bfloat16", "float32"):
            shared = device.empty((0, 5120), dtype)

            def operation(shared=shared):
                empty.begin(inputs.residual, inputs.incoming_pre)
                empty.finish_moe(inputs.residual, empty_routed, shared)

            execute_modes(
                device,
                (empty,),
                operation,
                lambda: require(
                    all(tensor.shape[0] == 0 for tensor in empty.outputs),
                    "empty fused HC changed shape",
                ),
            )
    finally:
        device.synchronize()
        empty.close()
    rejected(lambda: empty.run(inputs))
    rejected(lambda: empty.collapse(inputs.residual, inputs.incoming_pre))
    rejected(lambda: empty.finish_moe(inputs.residual, empty_routed, shared))
    return {
        "invalid_arguments": 10,
        "fused_invalid_arguments": fused_rejections,
        "fused_overlap_pairs_per_dtype": 15,
        "empty_eager_graph": True,
        "empty_shared_dtypes": ["bfloat16", "float32"],
        "closed_workspace": True,
    }


def check_case(device, case, limits, parameters, *, permute=False):
    from oh_my_vllm.operators.hyperconnection.api import Inputs, Weights
    from oh_my_vllm.operators.hyperconnection.plan import make_plan
    from oh_my_vllm.operators.hyperconnection.workspace import prepare

    inputs, expected = dict(case.inputs), dict(case.expected)
    parameters = [value.copy() for value in parameters]
    if permute:
        permutation = np.arange(5120) * 73 % 5120
        inputs["residual"] = inputs["residual"][:, :, permutation]
        inputs["branch"] = inputs["branch"][:, permutation]
        expected["collapsed"] = expected["collapsed"][:, permutation]
        expected["expanded"] = expected["expanded"][:, :, permutation]
        parameters[0] = (
            parameters[0].reshape(24, 4, 5120)[:, :, permutation].reshape(24, 20480)
        )
    fixture = Fixture(case.manifest, inputs, {}, expected)
    weights = Weights(*(device.upload(value) for value in parameters))
    values = Inputs(
        device.upload(inputs["residual"], "bfloat16"),
        device.upload(inputs["incoming_pre"]),
        device.upload(inputs["branch"], "bfloat16"),
    )
    workspace = prepare(make_plan(rows=len(inputs["residual"])), device, weights)
    originals = [
        (tensor, value)
        for tensor, value in zip(
            (*values, *weights),
            (inputs["residual"], inputs["incoming_pre"], inputs["branch"], *parameters),
            strict=True,
        )
    ]
    native = native_identity(workspace)
    legacy_expanded = None
    result = {}
    try:

        def operation():
            workspace.run(values)

        def check(*, standalone=False, label="metrics", fused=False):
            nonlocal legacy_expanded

            selected = (
                {"collapsed": workspace.outputs.collapsed}
                if standalone
                else workspace.outputs._asdict()
            )
            actual = {
                name: decode(
                    device.download(tensor, storage_dtype(str(tensor.dtype))),
                    str(tensor.dtype),
                )
                for name, tensor in selected.items()
            }
            selected_fixture = Fixture(
                fixture.manifest,
                inputs,
                {},
                {name: expected[name] for name in selected},
            )
            selected_limits = dict(
                limits, fields={name: limits["fields"][name] for name in selected}
            )
            compared = compare(actual, selected_fixture, selected_limits)
            require(compared.passed, f"HC boundary differs: {compared.failures}")
            for tensor, original in originals:
                require(
                    equal_bits(device.download(tensor, original.dtype), original),
                    "HC changed an input or weight tensor",
                )
            if not standalone:
                expanded = device.download(workspace.outputs.expanded, np.uint16)
                if fused:
                    require(
                        equal_bits(expanded, legacy_expanded),
                        f"{label}: fused expansion differs from legacy HC",
                    )
                elif legacy_expanded is None:
                    legacy_expanded = expanded
            result["collapse_metrics" if standalone else label] = compared.metrics

        execute_modes(
            device,
            (workspace,),
            lambda: workspace.collapse(values.residual, values.incoming_pre),
            lambda: check(standalone=True),
        )
        execute_modes(device, (workspace,), operation, check)
        routed_host = np.zeros(inputs["branch"].shape, np.float32)
        routed = device.upload(routed_host)
        originals.append((routed, routed_host))
        for dtype in ("bfloat16", "float32"):
            shared_host = (
                inputs["branch"]
                if dtype == "bfloat16"
                else decode(inputs["branch"], "bfloat16").astype(np.float32)
            )
            shared = device.upload(shared_host, dtype)
            originals.append((shared, shared_host))

            def fused_operation(shared=shared):
                workspace.begin(values.residual, values.incoming_pre)
                workspace.finish_moe(values.residual, routed, shared)

            execute_modes(
                device,
                (workspace,),
                fused_operation,
                lambda dtype=dtype: check(label=f"fused_{dtype}_metrics", fused=True),
            )
        require(native_identity(workspace) == native, "HC native image changed")
        result.update(
            case=case.manifest["case"]["id"],
            channel_permutation={"multiplier": 73, "modulus": 5120}
            if permute
            else None,
            native=native,
            fused_shared_dtypes=["bfloat16", "float32"],
            fused_matches_legacy_bits=True,
            modes=["eager", "graph"],
        )
    finally:
        device.synchronize()
        workspace.close()
    return result


def check_literals(device, parameters):
    from oh_my_vllm.operators.hyperconnection.api import Weights
    from oh_my_vllm.operators.hyperconnection.plan import make_plan
    from oh_my_vllm.operators.hyperconnection.workspace import prepare

    # Finite inputs and representable intermediates are required. Words are fixed
    # independently of the implementation; all HC arithmetic here is exact dyadic.
    cases = {
        "float32": [
            ("shared_even", -1.0, 1 + 2**-8, 1, 0.0, 0x0000),
            ("shared_odd", -1.0, 1 + 3 * 2**-8, 1, 0.0, 0x3C80),
            ("negative_zero", -0.0, -0.0, 1, -0.0, 0x8000),
        ],
        "bfloat16": [
            ("branch_even", 2**-8, 1, 1.5, -1.5, 0x0000),
            ("branch_odd", 3 * 2**-8, 1, 1.5, -1.5, 0x3CC0),
            ("fp32_add", 2**-8 + 2**-25, 1, 1.5, -1.5, 0x0000),
            ("positive_zero", -0.0, 0.0, 1, -0.0, 0x0000),
        ],
    }
    require(
        np.float32(cases["bfloat16"][2][1]).view(np.uint32) == 0x3B800040,
        "FP32 add discriminator differs",
    )
    results = []
    weights = Weights(*(device.upload(value) for value in parameters))
    for dtype, rows in cases.items():
        count = len(rows)
        workspace = prepare(make_plan(rows=count), device, weights)
        native = native_identity(workspace)
        routed_host = np.broadcast_to(
            np.array([row[1] for row in rows], np.float32)[:, None], (count, 5120)
        ).copy()
        shared_values = np.broadcast_to(
            np.array([row[2] for row in rows], np.float32)[:, None], (count, 5120)
        ).copy()
        shared_host = bf16(shared_values) if dtype == "bfloat16" else shared_values
        residual_host = bf16(
            np.broadcast_to(
                np.array([row[4] for row in rows], np.float32)[:, None, None],
                (count, 4, 5120),
            )
        )
        post = np.broadcast_to(
            np.array([row[3] for row in rows], np.float32)[:, None], (count, 4)
        ).copy()
        comb = np.full((count, 4, 4), 0.25, np.float32)
        expected = np.broadcast_to(
            np.array([row[5] for row in rows], np.uint16)[:, None, None],
            (count, 4, 5120),
        )
        residual = device.upload(residual_host, "bfloat16")
        routed = device.upload(routed_host)
        shared = device.upload(shared_host, dtype)
        device.module.copy_bytes(workspace.storage.post, device.upload(post))
        device.module.copy_bytes(workspace.storage.comb, device.upload(comb))
        originals = [
            (residual, residual_host),
            (routed, routed_host),
            (shared, shared_host),
            (workspace.storage.post, post),
            (workspace.storage.comb, comb),
        ]
        try:

            def check(
                workspace=workspace,
                rows=rows,
                expected=expected,
                dtype=dtype,
                originals=originals,
            ):
                actual = device.download(workspace.outputs.expanded, np.uint16)
                for index, row in enumerate(rows):
                    require(
                        equal_bits(actual[index], expected[index]),
                        f"fused HC literal {dtype}/{row[0]} differs",
                    )
                for tensor, original in originals:
                    require(
                        equal_bits(device.download(tensor, original.dtype), original),
                        "fused HC literal changed an input",
                    )

            # Only expanded is written: post/comb are staged, without begin().
            target = SimpleNamespace(
                outputs=(workspace.outputs.expanded,),
                graph_lease=workspace.graph_lease,
            )
            execute_modes(
                device,
                (target,),
                partial(workspace.finish_moe, residual, routed, shared),
                check,
            )
            require(native_identity(workspace) == native, "HC native image changed")
            results.append(
                dict(
                    shared_dtype=dtype,
                    cases=[
                        dict(id=row[0], output_u16=f"0x{row[5]:04x}") for row in rows
                    ],
                    modes=["eager", "graph"],
                    native=native,
                )
            )
        finally:
            device.synchronize()
            workspace.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        importlib.util.find_spec("oh_my_vllm.operators.hyperconnection") is not None,
        "native HC boundary is not implemented",
    )
    from oh_my_vllm.operators.runtime.device import Device

    sources = source_identity()
    raw = args.checkpoint_manifest.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == CHECKPOINT_MANIFEST_SHA256,
        "checkpoint identity differs",
    )
    manifest = json.loads(raw)
    config = json.loads((args.model / "config.json").read_text())
    weights = CheckpointWeights(args.model, manifest, tensor_layout(config))
    metadata, cases = load_hyperconnection()
    names = cases["decode"].manifest["case"]["weights"]
    parameters = [weights.tensor(name).astype(np.float32) for name in names]
    identity = device_identity(selected_uuid())
    device = Device()
    device_native = native_identity(device)
    try:
        with caller_stream(device):
            checks = [
                check_case(device, case, metadata["cases"][name]["limits"], parameters)
                for name, case in cases.items()
            ]
            checks.append(
                check_case(
                    device,
                    cases["decode"],
                    metadata["cases"]["decode"]["limits"],
                    parameters,
                    permute=True,
                )
            )
            contracts = check_contracts(device, parameters)
            literals = check_literals(device, parameters)
        weights._verify_states()
        require(
            source_identity() == sources, "HC checker sources changed during execution"
        )
        require(native_identity(device) == device_native, "Device native image changed")
    finally:
        device.synchronize()
        device.close()
    report = {
        "passed": True,
        "device": identity,
        "fixture_sha256": HYPERCONNECTION_SHA256,
        "checkpoint_manifest_sha256": CHECKPOINT_MANIFEST_SHA256,
        "device_native": device_native,
        "fused_precondition": "finite inputs and representable intermediates",
        "checks": checks,
        "contracts": contracts,
        "literals": literals,
        "sources": sources,
    }
    with args.output.open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    main()
