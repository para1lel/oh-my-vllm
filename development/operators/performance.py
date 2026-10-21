import ctypes
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from development.serving.device import device_identity, selected_uuid

from . import accounting, performance_inputs
from .cases import digest
from .fixtures import file_digest
from .run import PreparedCase, bind


def accounting_identity():
    from development.serving import accounting as http
    from development.serving import kernels

    return {
        "version": accounting.VERSION,
        "sources": {
            "operator": file_digest(accounting.__file__),
            "http": file_digest(http.__file__),
            "kernels": file_digest(kernels.__file__),
        },
        "rates": accounting.RATES,
        "cache_capacity": accounting.CACHE_CAPACITY,
        "cache_policy": "one 128MiB budget for serial memory excess; no writes",
    }


def timing_identity():
    from . import timing, timing_graph

    return {
        "version": timing.VERSION,
        "sources": {
            name: file_digest(module.__file__)
            for name, module in (("timing", timing), ("graph", timing_graph))
        },
        "graph": "recaptured-production-child; serial-repeats; external-event-nodes",
        "host": "original-production-invoke",
        "minimum_seconds": timing.TARGET_SECONDS,
    }


def device_spec():
    uuid = selected_uuid()
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cache = ctypes.c_int()
    if driver.cuDeviceGetAttribute(ctypes.byref(cache), 38, 0):
        raise RuntimeError("cannot read device cache size")
    spec = {**device_identity(uuid), "l2_bytes": cache.value}
    if "B200" not in spec["name"] or not 0 < cache.value <= accounting.CACHE_CAPACITY:
        raise ValueError("device differs from operator accounting assumptions")
    return spec


def compatible_spec(spec):
    return {name: value for name, value in spec.items() if name != "uuid"}


def snapshot(spec):
    rows = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    for row in rows:
        uuid, pid = (value.strip() for value in row.split(","))
        if uuid == spec["uuid"] and int(pid) != os.getpid():
            raise RuntimeError(
                f"concurrent process {pid} on {uuid} prevents operator timing"
            )
    current = device_spec()
    if current != spec:
        raise ValueError("device settings changed during measurement")
    fields = (
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "power.limit",
        "temperature.gpu",
        "pstate",
        "clocks_event_reasons.active",
    )
    values = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--id",
                spec["uuid"],
                "--query-gpu=" + ",".join(fields),
                "--format=csv,noheader",
            ],
            text=True,
        )
        .strip()
        .split(",")
    )
    return {
        "time": datetime.now(UTC).isoformat(),
        **dict(zip(fields, (value.strip() for value in values), strict=True)),
    }


def input_identity(fixture):
    contents, memo = {}, {}
    for group in ("inputs", "state"):
        for name, array in getattr(fixture, group).items():
            if id(array) not in memo:
                memo[id(array)] = hashlib.sha256(
                    memoryview(array).cast("B")
                ).hexdigest()
            contents[group + "." + name] = dict(
                sha256=memo[id(array)],
                shape=list(array.shape),
                dtype=fixture.dtype(group, name),
            )
    return digest(
        dict(
            case=fixture.manifest["case"],
            checkpoint=fixture.manifest["checkpoint"],
            writes=fixture.manifest["writes"],
            arrays=contents,
        )
    )


def free_bytes(device):
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    function = device.runtime.cudaMemGetInfo
    function.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t),
    ]
    function.restype = ctypes.c_int
    device.checked(function(ctypes.byref(free), ctypes.byref(total)))
    return free.value


def prepare_performance(case, device, model):
    from .weights import CaseWeights

    started = perf_counter()
    weights = CaseWeights(device, model, case)
    fixture = performance_inputs.make(case, weights)
    prepared = PreparedCase(case, device, fixture)
    try:
        prepared.weights = weights
        prepared.input_identity = input_identity(fixture)
        device.synchronize()
        before = free_bytes(device)
        bind(prepared)
        device.synchronize()
        prepared.setup["temporary_storage_bytes"] = max(0, before - free_bytes(device))
        prepared.setup["temporary_storage_scope"] = (
            "device memory difference for binding, warmup, and graph creation"
        )
        prepared.setup["reset_storage_bytes"] = sum(
            value.nbytes for value in fixture.state.values()
        )
        prepared.stage_resets()
        prepared.setup["preparation_seconds"] = perf_counter() - started
        return prepared
    except BaseException:
        prepared.close()
        raise


def input_provenance(fixture):
    return {
        "prototype_sha256": fixture.manifest["prototype_sha256"],
        "generator_sha256": file_digest(performance_inputs.__file__),
        "storage_sha256": file_digest(Path(__file__).with_name("storage.py")),
        "numpy": np.__version__,
    }


def external_directory(path):
    path = Path(path).resolve()
    repository = Path(__file__).resolve().parents[2]
    if path == repository or repository in path.parents:
        raise ValueError("keep performance reports outside the repository")
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_report(path, payload):
    with Path(path).open("x") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")
    return file_digest(path)
