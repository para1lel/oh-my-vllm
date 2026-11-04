"""Check captured DeepSeek operands against their recorded native outputs."""

import argparse
import json
import statistics
from pathlib import Path

import torch
from oh_my_vllm.models.deepseek_v41 import kernel
from oh_my_vllm.operators.sparse_attention.plan import scratch_bytes
from safetensors.torch import load_file


def fp32_case(tensors, logical_rows, module=None):
    values, weight = tensors["input_0"], tensors["input_1"]
    expected = tensors["output_0"]
    output = torch.empty_like(expected)
    module = module or kernel._hyperconnection_module()
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=values.device)

    def run():
        module.fp32_project(values, weight, output, workspace, logical_rows)
        return {"output_0": output}

    flops = 2 * values.size(0) * weight.numel()
    memory = sum(x.numel() * x.element_size() for x in (values, weight, output))
    bound = max(flops / 75e12, memory / 7.7e12)
    return run, {"output_0": expected}, bound


def hc_case(tensors, logical_rows, module=None):
    residual = tensors["residual"]
    rows = residual.size(0)
    module = module or kernel._hyperconnection_module()
    names = ("normalized_projection", "post", "comb", "collapsed", "next_pre")
    output = {name: torch.empty_like(tensors[name]) for name in names}
    flat = torch.empty_like(tensors["flat_input"])
    statistics = torch.empty_like(tensors["statistics"])
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=residual.device)

    def run():
        module.hc_prepare_grouped(
            residual,
            tensors["incoming_pre"],
            tensors["weight"],
            tensors["scale"],
            tensors["base"],
            output["normalized_projection"],
            output["post"],
            output["comb"],
            output["collapsed"],
            output["next_pre"],
            flat,
            statistics,
            workspace,
            logical_rows or rows,
        )
        return output

    return run, {name: tensors[name] for name in names}, None


def sparse_case(tensors, _logical_rows, module=None):
    query, sink = tensors["input_0"], tensors["input_2"]
    indices = tensors["input_3"]
    window, compressed = tensors["kw_native_window"], tensors["kw_native_compressed"]
    batch, sequence, heads, _ = query.shape
    rows, slots = batch * sequence, indices.size(-1)
    offsets = torch.arange(batch, device=query.device, dtype=torch.int32).view(
        batch, 1, 1
    )
    ids = torch.where(
        indices < 0,
        -1,
        torch.where(
            indices < window.size(1),
            indices + offsets * window.size(1),
            indices + (batch - 1) * window.size(1) + offsets * compressed.size(1),
        ),
    ).reshape(rows, slots)
    scales = [
        torch.empty((x.size(0) * x.size(1), 0), dtype=torch.uint8, device=query.device)
        for x in (window, compressed)
    ]
    output = torch.empty_like(query)
    status = torch.empty((rows, heads // 16), dtype=torch.uint8, device=query.device)
    scratch = torch.empty(
        scratch_bytes(rows, heads, slots), dtype=torch.uint8, device=query.device
    )
    module = module or kernel._sparse_attention_module()

    def run():
        module.attend(
            query.reshape(rows, heads, 512),
            sink,
            ids,
            window.reshape(-1, 512),
            scales[0],
            compressed.reshape(-1, 512),
            scales[1],
            scratch,
            output.reshape(rows, heads, 512),
            status,
        )
        return {"output_0": output, "status": status}

    pairs = int((indices >= 0).sum().item())
    flops = 4 * pairs * query.size(2) * 512
    selected_rows = torch.unique(ids[ids >= 0]).numel()
    memory = 2 * query.numel() * query.element_size()
    memory += (
        indices.numel() * indices.element_size() + sink.numel() * sink.element_size()
    )
    memory += selected_rows * 512 * window.element_size()
    bound = max(flops / 2.25e15, memory / 7.7e12)
    return (
        run,
        {"output_0": tensors["output_0"], "status": torch.zeros_like(status)},
        bound,
    )


def check_bits(actual, expected):
    result = {}
    for name, reference in expected.items():
        value = actual[name]
        integer = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[
            reference.element_size()
        ]
        different = int((value.view(integer) != reference.view(integer)).sum().item())
        result[name] = different
    if any(result.values()):
        raise AssertionError(f"captured output bits differ: {result}")
    return result


def time_case(run, repeats):
    measurements = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1e-3)
    return statistics.median(measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument(
        "--operator",
        action="append",
        choices=("fp32_project", "hc_projection", "sparse_attn"),
    )
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--baseline-hc", type=Path)
    parser.add_argument("--baseline-sparse", type=Path)
    options = parser.parse_args()
    if options.repeats < 1:
        parser.error("repeats must be positive")
    root = options.captures.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    cases = {
        "fp32_project": fp32_case,
        "hc_projection": hc_case,
        "sparse_attn": sparse_case,
    }
    selected = set(options.operator or cases)
    baselines = {}
    from tvm_ffi import load_module

    for operator, library in (
        ("fp32_project", options.baseline_hc),
        ("hc_projection", options.baseline_hc),
        ("sparse_attn", options.baseline_sparse),
    ):
        if library is not None:
            baselines[operator] = load_module(str(library.resolve()))
    for name, case in manifest["samples"].items():
        operator = case["operator"]
        if operator not in selected:
            continue
        path = (root / case["file"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("capture path is outside the capture root")
        tensors = load_file(str(path), device="cuda")
        run, expected, bound = cases[operator](
            tensors, case.get("kwargs", {}).get("logical_rows", 0)
        )
        kernel.reset_sparse_status()
        different = check_bits(run(), expected)
        kernel.require_sparse_status()
        result = {
            "case": name,
            "source_commit": case["source_commit"],
            "different_bits": different,
        }
        if options.time:
            run()
            torch.cuda.synchronize()
            seconds = time_case(run, options.repeats)
            result["seconds"] = seconds
            if operator in baselines:
                baseline_run, baseline_expected, _ = cases[operator](
                    tensors,
                    case.get("kwargs", {}).get("logical_rows", 0),
                    baselines[operator],
                )
                check_bits(baseline_run(), baseline_expected)
                baseline_run()
                torch.cuda.synchronize()
                baseline_seconds = time_case(baseline_run, options.repeats)
                result["baseline_seconds"] = baseline_seconds
                result["speedup"] = baseline_seconds / seconds
                del baseline_run, baseline_expected
            if bound is not None:
                result["resource_seconds"] = bound
                result["resource_percent"] = 100 * bound / seconds
                result["resource_scope"] = (
                    "captured operands and v3 resource rates; not HTTP efficiency"
                )
        print(json.dumps(result), flush=True)
        del run, expected, tensors


if __name__ == "__main__":
    main()
