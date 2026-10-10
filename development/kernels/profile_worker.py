"""One complete formal operation inside a profiler-only NVTX range."""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--backend", required=True, choices=("cuda", "tilelang"))
    args = parser.parse_args()
    os.environ["OH_MY_VLLM_KERNEL_BACKEND"] = args.backend

    import torch

    from development.kernels.cases import cases
    from development.kernels.fixtures import fixture
    from development.kernels.reference import verify_reference

    verify_reference()
    case = next(c for c in cases() if c["id"] == args.case)
    torch.manual_seed(784)
    reference, candidate = fixture(case["configuration"])
    function = candidate if args.backend == "cuda" else reference
    for _ in range(10):
        output = function()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("profile-operation"):
        output = function()
        torch.cuda.synchronize()
    del output
    print(
        "PROFILE_WORKER_IDENTITY "
        + json.dumps({"pid": os.getpid(), "case": args.case, "backend": args.backend}),
        flush=True,
    )


if __name__ == "__main__":
    main()
