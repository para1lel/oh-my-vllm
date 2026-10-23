"""Frozen local Engram lookup and shared registration checks; no timing gate."""

import ctypes
import fcntl
import hashlib
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from oh_my_vllm.operators.engram import api, lookup
from oh_my_vllm.operators.engram.shared import SharedWeightFile
from oh_my_vllm.operators.runtime.native import load

from development.model.deepseek_v41.fixtures import (
    ENGRAM_LOOKUP_SHA256,
    load_engram_lookup,
)

from .check_deepseek_exact import execute_modes, rejected, require
from .storage import equal_bits, storage_dtype

TABLE_OFFSET = 64
SCALE_OFFSET = TABLE_OFFSET + 16 * 256 + 32


def lookup_payload(arrays):
    return (
        bytes([0xA7]) * TABLE_OFFSET
        + arrays["table_e4m3"].tobytes()
        + bytes([0xA7]) * 32
        + arrays["scales_ue8m0"].tobytes()
    )


def locked_bytes():
    line = next(
        line
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmLck:")
    )
    return int(line.split()[1]) * 1024


def registration_capabilities(device):
    query = device.runtime.cudaDeviceGetAttribute
    query.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    capabilities = {}
    for name, attribute in (
        ("pageable_access", 88),
        ("host_register_supported", 99),
        ("host_page_tables", 100),
        ("readonly_supported", 113),
    ):
        value = ctypes.c_int()
        device.checked(query(ctypes.byref(value), attribute, 0))
        capabilities[name] = value.value
    return capabilities


def workspace_for(device, table, rows, heads=24):
    plan = lookup.make_plan(
        rows=rows,
        heads=heads,
        table_rows=16,
        table_offset=TABLE_OFFSET,
        scale_offset=SCALE_OFFSET,
    )
    return lookup.prepare(plan, device, table=table)


def lookup_case(device, table, arrays, rows, heads, *, invalid=False):
    indices = np.tile(arrays["indices"], ((rows + 7) // 8, 1))[:rows, :heads].copy()
    expected = np.tile(arrays["expected_bf16_bits"], ((rows + 7) // 8, 1, 1))[
        :rows, :heads
    ].copy()
    status = np.zeros(indices.shape, np.uint8)
    if invalid:
        indices[0, :6] = [-1, 16, 1, 2, 3, 4]
        status[np.isin(indices, [1, 2, 3, 4])] = 2
        status[(indices < 0) | (indices >= 16)] = 1
        expected[status != 0] = 0
    workspace = workspace_for(device, table, rows, heads)
    try:
        value = device.upload(indices)
        original = device.download(table, np.uint8)

        def operation():
            workspace.run(value)

        def check():
            output, errors = workspace.outputs
            require(
                equal_bits(device.download(output, np.uint16), expected),
                "Engram lookup BF16 bits differ",
            )
            require(
                equal_bits(device.download(errors, np.uint8), status),
                "Engram lookup status differs",
            )
            require(
                equal_bits(device.download(value, np.int64), indices)
                and equal_bits(device.download(table, np.uint8), original),
                "Engram lookup changed a read-only input",
            )

        execute_modes(device, (workspace,), operation, check)
        if rows == 8 and not invalid:
            rejected(lambda: workspace.run(device.empty((rows, heads), "int32")))
            rejected(lambda: workspace.run(device.empty((rows + 1, heads), "int64")))
            output, errors = workspace.outputs
            for destination, error, offset in (
                (device.empty((rows, heads, 256), "float32"), errors, TABLE_OFFSET),
                (output, device.view(table, (rows, heads)), TABLE_OFFSET),
                (output, errors, SCALE_OFFSET),
            ):
                rejected(
                    lambda destination=destination, error=error, offset=offset: (
                        workspace.module.engram_lookup(
                            table, value, destination, error, 16, offset, SCALE_OFFSET
                        )
                    )
                )
        return {
            "case": "engram_lookup_invalid" if invalid else "engram_lookup",
            "rows": rows,
            "heads": heads,
            "modes": ["eager", "graph"],
        }
    finally:
        try:
            device.synchronize()
        finally:
            workspace.close()


def registration_checks(device, module, shared, size, cap):
    before = module.engram_pinned_bytes()
    shared.seal()
    table = module.engram_register(shared.fileno(), size, cap)
    require(
        module.engram_pinned_bytes() == before + shared.mapped_bytes,
        "registration charge differs",
    )
    with SharedWeightFile(size, limit=cap) as unsealed:
        rejected(lambda: module.engram_register(unsealed.fileno(), size, cap * 2))
    rejected(lambda: module.engram_register(shared.fileno(), size, size))
    rejected(lambda: module.engram_register(shared.fileno(), size + cap, cap * 3))
    rejected(lambda: module.engram_register(shared.fileno(), size, cap))

    def during_capture():
        with device.graph(
            lambda: module.engram_register(shared.fileno(), size, cap * 2)
        ):
            pass

    rejected(during_capture)
    require(
        module.engram_pinned_bytes() == before + shared.mapped_bytes,
        "failed registration leaked its charge",
    )
    return table


def graph_lifetime(device, module, payload, arrays, cap):
    before = module.engram_pinned_bytes()
    with SharedWeightFile(len(payload), limit=cap) as shared:
        shared.append(payload)
        shared.seal()
        table = module.engram_register(shared.fileno(), len(payload), cap)
    workspace = workspace_for(device, table, 8)
    values = device.upload(arrays["indices"])
    outputs = workspace.outputs
    holder = [workspace]

    def operation():
        holder[0].run(values)

    with ExitStack() as resources:
        resources.callback(workspace.close)
        resources.enter_context(workspace.graph_lease())
        replay = resources.enter_context(device.graph(operation))
        workspace.close()
        holder.clear()
        del workspace, table
        require(
            module.engram_pinned_bytes() == before + cap,
            "graph lost its registered table",
        )
        for output in outputs:
            device.module.copy_bytes(
                output,
                device.upload(
                    np.full(output.shape, 0x5A, storage_dtype(str(output.dtype))),
                    str(output.dtype),
                ),
            )
        replay()
        require(
            equal_bits(
                device.download(outputs[0], np.uint16), arrays["expected_bf16_bits"]
            )
            and not np.any(device.download(outputs[1], np.uint8)),
            "graph replay lost closed workspace resources",
        )
    require(module.engram_pinned_bytes() == before, "graph release leaked registration")


def check_lookup(device, inherited_fd=None):
    metadata, arrays = load_engram_lookup()
    payload = lookup_payload(arrays)
    page = os.sysconf("SC_PAGE_SIZE")
    cap = (len(payload) + page - 1) // page * page
    module, source_id, library = load("engram", api.FUNCTIONS)
    initial = module.engram_pinned_bytes()
    require(initial == 0, "lookup checker requires no existing registered weights")
    capabilities = registration_capabilities(device)
    uses_mlock = bool(
        capabilities["host_page_tables"]
        or not (
            capabilities["host_register_supported"]
            and capabilities["readonly_supported"]
        )
    )
    locked_before = locked_bytes()
    info = None
    if inherited_fd is None:
        with SharedWeightFile(len(payload), limit=cap) as shared:
            shared.append(payload)
            table = registration_checks(device, module, shared, len(payload), cap)
    else:
        try:
            require(
                os.pread(inherited_fd, len(payload), 0) == payload,
                "inherited payload differs",
            )
            info = os.fstat(inherited_fd)
            required = (
                fcntl.F_SEAL_WRITE
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_SEAL
            )
            require(
                fcntl.fcntl(inherited_fd, fcntl.F_GET_SEALS) & required == required,
                "inherited file is not immutable",
            )
            table = module.engram_register(inherited_fd, len(payload), cap)
        finally:
            os.close(inherited_fd)
    locked_live = locked_bytes()
    require(module.engram_pinned_bytes() == cap, "live pinned charge differs")
    if uses_mlock:
        require(
            locked_live >= locked_before + cap, "shared mapping has no OS page locks"
        )
    records = []
    try:
        for rows, heads in ((8, 24), (0, 24), (1, 1), (384, 24)):
            records.append(lookup_case(device, table, arrays, rows, heads))
    finally:
        device.synchronize()
        del table
    require(
        module.engram_pinned_bytes() == initial, "lookup workspaces leaked registration"
    )
    if inherited_fd is None:
        invalid = {name: value.copy() for name, value in arrays.items()}
        invalid["table_e4m3"][1, 31] = 127
        invalid["scales_ue8m0"][2, 1] = 255
        invalid["table_e4m3"][3, 0] = 126
        invalid["scales_ue8m0"][3, 0] = 254
        invalid["table_e4m3"][4, 0] = 255
        bad_payload = lookup_payload(invalid)
        with SharedWeightFile(len(bad_payload), limit=cap) as shared:
            shared.append(bad_payload)
            shared.seal()
            table = module.engram_register(shared.fileno(), len(bad_payload), cap)
        try:
            records.append(lookup_case(device, table, arrays, 8, 24, invalid=True))
        finally:
            device.synchronize()
            del table
        graph_lifetime(device, module, payload, arrays, cap)
        require(
            module.engram_pinned_bytes() == initial,
            "lookup cleanup leaked registration",
        )
    require(locked_bytes() == locked_before, "lookup cleanup leaked OS page locks")
    return (
        {
            "source_id": source_id,
            "binary_sha256": hashlib.sha256(Path(library).read_bytes()).hexdigest(),
        },
        records,
        {
            "fixture_sha256": ENGRAM_LOOKUP_SHA256,
            "checkpoint_manifest_sha256": metadata["checkpoint_manifest_sha256"],
            "reference_sources": metadata["source_sha256"],
            "pinned_bytes_after": module.engram_pinned_bytes(),
            "pinned_bytes_live": cap,
            "vm_locked_bytes": [locked_before, locked_live, locked_bytes()],
            "registration_capabilities": capabilities,
            "uses_mlock": uses_mlock,
            "shared_inode": None if info is None else [info.st_dev, info.st_ino],
        },
    )
