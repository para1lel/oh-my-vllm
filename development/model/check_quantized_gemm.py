"""Compare grouped FP4 outputs with an immutable native baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from tvm_ffi import load_module

SHAPES = {
    "tp4_up": (384, 576, 5120),
    "tp4_down": (384, 1280, 2304),
    "tep4_up": (96, 2304, 5120),
    "tep4_down": (96, 5120, 2304),
}
ROWS = (
    *range(1, 65),
    65,
    97,
    127,
    128,
    129,
    255,
    256,
    257,
    511,
    512,
    513,
    1023,
    1024,
    1025,
    4097,
    16385,
    32768,
)


def native(path):
    manifest = json.loads(path.read_text())
    with Path(manifest["library"]).open("rb") as library:
        binary_id = hashlib.file_digest(library, "sha256").hexdigest()
    if binary_id != manifest["binary_id"]:
        raise ValueError("native library differs from its manifest")
    return load_module(manifest["library"]), binary_id


def prepare_weights(module, outputs, width, device):
    generator = torch.Generator(device=device).manual_seed(7283)
    prepared = []
    for _ in range(4):
        weight = torch.randint(
            0,
            256,
            (outputs, width // 2),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        scales = torch.randint(
            121,
            130,
            (outputs, width // 32),
            dtype=torch.uint8,
            device=device,
            generator=generator,
        )
        packed = torch.empty(
            ((outputs + 127) // 128 * 128 * ((width + 127) // 128 * 4),),
            dtype=torch.uint8,
            device=device,
        )
        module.pack_weights(weight, scales, weight, packed, 4)
        prepared.append((weight, packed))
    return prepared


def pointer_arrays(weights, groups, device):
    return tuple(
        torch.tensor(
            [weights[i % 4][which].data_ptr() for i in range(groups)],
            dtype=torch.int64,
            device=device,
        )
        for which in (0, 1)
    )


def run(module, encoded, scales, pointers, counts, outputs, dtype):
    rows, width = encoded.shape
    max_rows = max(1, rows)
    output = torch.empty((rows, outputs), dtype=dtype, device=encoded.device)
    scratch = torch.empty(
        module.grouped_workspace_size(rows, counts.numel(), max_rows, outputs, width),
        dtype=torch.uint8,
        device=encoded.device,
    )
    module.grouped_project(
        encoded, scales, *pointers, counts, output, scratch, max_rows
    )
    return output


def compare_case(modules, shape, count_values, pointers, dtype):
    rows = sum(count_values)
    _, outputs, width = shape
    generator = torch.Generator(device="cuda").manual_seed(9181 + rows)
    encoded = torch.randint(
        0, 256, (rows, width), dtype=torch.uint8, device="cuda", generator=generator
    )
    encoded.masked_fill_((encoded & 127) == 127, 0)
    scales = torch.randint(
        120,
        130,
        (rows, width // 32),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    counts = torch.tensor(count_values, dtype=torch.int32, device="cuda")
    expected, actual = (
        run(module, encoded, scales, pointers, counts, outputs, dtype)
        for module in modules
    )
    mismatches = int((expected.view(torch.uint8) != actual.view(torch.uint8)).sum())
    return {
        "rows": rows,
        "max_expert_rows": max(count_values),
        "active_experts": sum(value > 0 for value in count_values),
        "dtype": str(dtype),
        "elements": actual.numel(),
        "different_bytes": mismatches,
    }


def check_shape(name, modules, row_counts, output):
    shape = SHAPES[name]
    groups, outputs, width = shape
    weights = prepare_weights(modules[0], outputs, width, "cuda")
    pointers = pointer_arrays(weights, groups, "cuda")
    distributions = [[0] * groups, [1] * groups, [17] * groups]
    for rows in row_counts:
        counts = [0] * groups
        counts[0], counts[2], counts[-1] = rows, rows // 7, max(1, rows // 31)
        distributions.append(counts)
    for counts in distributions:
        for dtype in (torch.float32, torch.bfloat16):
            record = {
                "shape": name,
                **compare_case(modules, shape, counts, pointers, dtype),
            }
            output.write(json.dumps(record) + "\n")
            output.flush()
            if record["different_bytes"]:
                raise AssertionError(record)
    print(
        json.dumps({"shape": name, "checks": len(distributions) * 2, "passed": True}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shape", choices=tuple(SHAPES), action="append")
    parser.add_argument("--rows", type=str)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("write numerical records outside the repository")
    modules, identities = zip(
        *(native(path) for path in (args.baseline, args.candidate)), strict=True
    )
    row_counts = (
        ROWS
        if args.rows is None
        else tuple(int(value) for value in args.rows.split(","))
    )
    if not row_counts or any(not 1 <= rows <= 32768 for rows in row_counts):
        raise ValueError("expert row counts must be between 1 and 32768")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(
            json.dumps(
                {
                    "baseline": identities[0],
                    "candidate": identities[1],
                    "scope": (
                        "Native arithmetic regression; not an independent frozen "
                        "model reference or a performance result."
                    ),
                }
            )
            + "\n"
        )
        for name in args.shape or SHAPES:
            check_shape(name, modules, row_counts, output)


if __name__ == "__main__":
    main()
