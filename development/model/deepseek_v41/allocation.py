"""Checked allocation design over one existing memory plan, without a runtime."""

from .memory import DEFAULT_LIMITS, _keys, _rank_bytes, _size, _sum, _uint

MiB = 2**20
GiB = 2**30
PAGEABLE = (
    "control",
    "request_payloads",
    "media",
    "processes",
    "registration",
    "kernel_io",
)
UNRESOLVED = (
    "llguidance parser, compiled grammar, and deep_copy physical byte bounds",
    "request/response accounting versus physical copies and protocol objects",
    "decoder and media preparation physical peak within the supplied allowance",
    "native/library/NCCL/graph opaque allocations and enforcement before Ready",
    "native packed formats, exact chunk continuation, and allocation/lease enforcement",
)


def _aligned(value, alignment):
    return _size((_uint(value) + alignment - 1) // alignment, alignment)


def _arena(cap, children, *, lifetime="transaction", alignment=256):
    _uint(cap)
    if _uint(alignment) == 0 or alignment & (alignment - 1):
        raise ValueError("Alignment must be a positive power of two")
    buffers = {}
    for name, shape, element, count in children:
        if name in buffers:
            raise ValueError(f"Duplicate buffer: {name}")
        size = _size(_aligned(_size(*shape, element), alignment), count)
        buffers[name] = dict(
            shape=list(shape),
            element_bytes=element,
            count=count,
            bytes=size,
        )
    payload = _sum(b["bytes"] for b in buffers.values())
    if payload > cap:
        raise ValueError("Arena children exceed cap")
    return dict(
        cap_bytes=cap,
        payload_bytes=payload,
        spare_bytes=cap - payload,
        buffers=buffers,
        alignment_bytes=alignment,
        lifetime=lifetime,
        reuse="lifetime_ended_and_all_consumers_and_transfers_complete",
    )


def _cap(arena, floor):
    cap = max(arena["cap_bytes"], _uint(floor))
    return dict(arena, cap_bytes=cap, spare_bytes=cap - arena["payload_bytes"])


def _device_arenas(strategy):
    rows, edges = 384, 2304
    ih, qh = (32, 64) if strategy == "DEP4" else (8, 16)
    compute = [
        ("index_scores", (rows, 4096, ih + 1), 4, 1),
        ("index_query", (rows, ih, 128), 2, 1),
        ("index_key_tile", (4096, 128), 2, 1),
        ("index_head_weights", (rows, ih), 4, 1),
        ("chunk_window_shadow", (43, rows, 528), 1, 1),
        ("chunk_global_shadow", (1536, 356), 1, 1),
        ("index_merge", (rows, 4608, 16), 1, 2),
        ("candidate_merge", (rows, 2560, 16), 1, 2),
        ("sparse_selections", (rows, 8, 512), 4, 1),
        ("candidate_blocks", (rows, 2048), 4, 1),
        ("residual_pingpong", (rows, 4, 5120), 2, 2),
        ("hyperconnection_coefficients", (rows, 24), 4, 3),
        ("normalized_and_projected", (rows, 15360), 2, 4),
        ("query_and_output", (rows, qh, 512), 2, 2),
        ("attention_dequant", (16, 640, 512), 2, 2),
        ("attention_scores", (16, qh, 128), 4, 2),
        ("attention_statistics", (rows, qh, 2), 4, 1),
        ("compressor_values_and_gates", (rows, 512), 4, 2),
        ("expert_intermediates", (edges, 2304), 4, 2),
        ("expert_accumulator", (edges, 5120), 4, 1),
        ("routing_scores", (rows, 384), 4, 2),
        ("routing_permutation", (edges, 32), 1, 2),
        ("shared_expert", (rows, 2304), 4, 2),
        ("target_probabilities", (rows, 129280), 4, 1),
        ("sampling_sort_cdf", (rows, 129280), 8, 2),
        ("grammar_masks", (rows, 16160), 1, 2),
        ("draft_feature_gather", (rows, 15360), 2, 1),
        ("draft_base_hidden", (320, 5120), 2, 1),
        ("kernel_scratch", (GiB,), 1, 1),
    ]
    media = [
        ("patch_input", (9216, 588), 2, 1),
        ("vision_residuals", (9216, 1024), 2, 2),
        ("vision_qkv", (9216, 3, 1024), 2, 1),
        ("vision_mlp", (9216, 2816), 2, 2),
        ("attention_tiles", (16, 128, 128), 4, 2),
        ("attention_statistics", (9216, 16, 2), 4, 1),
        ("vision_rotary", (9216, 64), 4, 2),
        ("aligner_cells", (1024, 9216), 2, 1),
        ("aligner_intermediate", (1024, 5120), 2, 2),
        ("feature_banks", (1024, 5120), 2, 2),
        ("vision_scratch", (256 * MiB,), 1, 1),
    ]
    return {
        "compute": _arena(4 * GiB, compute),
        "staging": _arena(
            64 * MiB,
            [
                ("engram_packed", (384, 24, 256), 1, 2),
                ("engram_scales", (384, 24, 8), 1, 2),
                ("engram_bf16", (384, 24, 256), 2, 2),
                ("engram_ids", (384, 24), 8, 2),
                ("upload_descriptors", (128, 256), 1, 2),
            ],
        ),
        "communication": _arena(
            GiB,
            [
                ("expert_send_recv_output_return", (edges, 5120), 2, 4),
                ("route_headers", (edges, 32), 1, 4),
                ("expert_counts_offsets", (4, 385), 8, 4),
                ("dense_reduction", (rows, 5120), 4, 2),
                ("vocab_allgather", (rows, 129280), 4, 1),
                ("vocab_local", (rows, 32320), 4, 1),
                ("feature_transfer", (1024, 5120), 2, 2),
            ],
        ),
        "state_extra": _arena(
            512 * MiB,
            [
                ("cow_bundles", (64, 911360), 1, 1),
                ("prefix_snapshots", (64, 3 * MiB), 1, 1),
                ("block_descriptors", (65600, 128), 1, 1),
                ("block_maps", (64, 1024), 8, 1),
                ("transactions", (64, 4096), 1, 1),
                ("prefix_records", (128, 256), 1, 1),
                ("sparse_penalty_overlays", (64, 7, 4096), 1, 1),
                ("lease_descriptors", (4096, 256), 1, 1),
                ("plane_and_tensor_padding", (64 * MiB,), 1, 1),
            ],
            lifetime="lease",
        ),
        "graphs": _arena(
            2 * GiB,
            [
                ("fixed_metadata", (3, rows, 256), 1, 2),
                ("fixed_storage_allowance", (GiB - 589824,), 1, 1),
                ("opaque_capture_allowance", (GiB,), 1, 1),
            ],
            lifetime="graph",
        ),
        "native": _arena(
            6 * GiB,
            [
                ("context_allowance", (2 * GiB,), 1, 1),
                ("library_allowance", (GiB,), 1, 1),
                ("NCCL_allowance", (GiB,), 1, 1),
                ("allocator_allowance", (2 * GiB,), 1, 1),
            ],
            lifetime="process",
        ),
        "media": _arena(GiB, media, lifetime="media_job"),
        "media_other": _arena(
            64 * MiB,
            [media[-2]],
            lifetime="media_job",
        ),
        "load": _arena(
            128 * MiB,
            [
                ("source_tile", (64 * MiB,), 1, 1),
                ("conversion_tile", (64 * MiB,), 1, 1),
            ],
            lifetime="startup",
        ),
    }


def plan_allocation(
    plan,
    *,
    device_reserve_bytes,
    host_capacity_bytes,
    host_unavailable_bytes,
    host_reserve_bytes,
    pageable_reservations,
    request_state_max_bytes,
    response_max_bytes,
):
    """Check a fixed full-capacity design; caller supplies actual available budgets."""
    if (
        plan["schema_version"] != 1
        or plan["world_size"] != 4
        or plan["strategy"] not in ("TP4", "TEP4", "DEP4")
        or plan["limits"] != DEFAULT_LIMITS | {"verification_rows": 6}
    ):
        raise ValueError("Allocation v1 requires the full supported capacity")
    reserves = _rank_bytes(device_reserve_bytes)
    _keys(pageable_reservations, PAGEABLE)
    arenas = _device_arenas(plan["strategy"])
    ranks = []
    for rank, reserve in zip(plan["ranks"], reserves, strict=True):
        local = {
            k: v for k, v in arenas.items() if k not in ("media", "media_other", "load")
        }
        local["media"] = arenas["media" if rank["rank"] == 0 else "media_other"]
        padding = _sum(
            _aligned(t["rank_resident_bytes"][rank["rank"]], 256)
            - t["rank_resident_bytes"][rank["rank"]]
            for t in plan["tensors"].values()
        )
        if padding > 64 * MiB:
            raise ValueError("Weight rounding exceeds padding allowance")
        for key in ("graphs", "staging", "communication", "media", "chunks"):
            name = "compute" if key == "chunks" else key
            local[name] = _cap(local[name], rank["reservations"][key]["charged_bytes"])
        persistent = _sum((rank["weight_bytes"], rank["cache_bytes"]))
        runtime = _sum((persistent, _sum(a["cap_bytes"] for a in local.values())))
        startup = _sum((runtime, arenas["load"]["cap_bytes"]))
        available = _uint(rank["available_bytes"]) - reserve
        if startup > available:
            raise ValueError("Device startup budget exhausted")
        ranks.append(
            dict(
                rank=rank["rank"],
                persistent_bytes=persistent,
                runtime_bytes=runtime,
                startup_bytes=startup,
                available_bytes=available,
                headroom_bytes=available - startup,
                weight_alignment_bytes=padding,
                arena_caps_bytes={name: a["cap_bytes"] for name, a in local.items()},
            )
        )
    engram = _sum(
        plan["host"][name] for name in ("engram_table_bytes", "engram_scale_bytes")
    )
    pinned = {
        "engram": _arena(
            _aligned(engram, 2 * MiB),
            [("tables_and_scales", (engram,), 1, 1)],
            lifetime="service",
            alignment=2 * MiB,
        ),
        "kv_offload": _arena(
            256 * GiB,
            [("rank_images", (4, 64, GiB), 1, 1)],
            lifetime="lease",
            alignment=4096,
        ),
        "uploads": _arena(
            512 * MiB,
            [
                ("engram_gather", (4, 2, 384, 24, 264), 1, 1),
                ("engram_ids", (4, 2, 384, 24), 8, 1),
                ("live_cached_retired_metadata", (4, 2, 16 * MiB), 1, 2),
            ],
            alignment=4096,
        ),
        "transfers": _arena(
            GiB,
            [("transfer_banks", (4, 2, 2, 64 * MiB), 1, 1)],
            alignment=4096,
        ),
    }
    for name, key in (
        ("kv_offload", "kv_offload"),
        ("uploads", "uploads"),
        ("transfers", "transfer_leases"),
    ):
        pinned[name] = _cap(pinned[name], plan["host"]["reservations"][key])
    offload = _size(4, max(r["cache_bytes"] for r in plan["ranks"]) + 512 * MiB)
    if offload > pinned["kv_offload"]["cap_bytes"]:
        raise ValueError("Offload images exceed pinned cap")
    pageable = {
        name: _arena(cap, [], lifetime="service", alignment=4096)
        for name, cap in pageable_reservations.items()
    }
    pageable["control"] = _arena(
        pageable_reservations["control"],
        [
            ("token_history", (64, 1048576), 4, 1),
            ("prefix_token_blocks", (65600, 1024), 4, 1),
            ("block_catalog", (4, 65600, 128), 1, 1),
            ("block_maps", (4, 64, 1024), 8, 1),
            ("transactions_and_leases", (4, 64, 4096), 1, 5),
            ("pending_streams", (64, 8 * MiB), 1, 1),
        ],
        lifetime="lease",
        alignment=4096,
    )
    pageable["request_payloads"] = _arena(
        pageable_reservations["request_payloads"],
        [
            ("request_state_budget", (request_state_max_bytes,), 1, 1),
            ("stored_response_budget", (response_max_bytes,), 1, 1),
        ],
        lifetime="lease",
        alignment=4096,
    )
    startup_pinned = _arena(
        512 * MiB,
        [("read_banks", (4, 2, 64 * MiB), 1, 1)],
        lifetime="startup",
        alignment=4096,
    )
    startup_pageable = _arena(
        512 * MiB,
        [
            ("headers", (4, 64 * MiB), 1, 1),
            ("bounce", (256 * MiB,), 1, 1),
        ],
        lifetime="startup",
        alignment=4096,
    )
    pinned_runtime = _sum(a["cap_bytes"] for a in pinned.values())
    pageable_runtime = _sum(a["cap_bytes"] for a in pageable.values())
    pinned_startup = _sum((pinned_runtime, startup_pinned["cap_bytes"]))
    pageable_startup = _sum((pageable_runtime, startup_pageable["cap_bytes"]))
    runtime = _sum((pinned_runtime, pageable_runtime))
    startup = _sum((pinned_startup, pageable_startup))
    available = (
        _uint(host_capacity_bytes)
        - _uint(host_unavailable_bytes)
        - _uint(host_reserve_bytes)
    )
    if pinned_startup > _uint(plan["host"]["available_bytes"]):
        raise ValueError("Host pinned startup budget exhausted")
    if startup > available:
        raise ValueError("Host total startup budget exhausted")
    return dict(
        schema_version=1,
        strategy=plan["strategy"],
        rows=dict(target=384, predictor=320),
        routed_edges=dict(target=2304, draft=960),
        ranks=ranks,
        device_arenas=arenas,
        host=dict(
            pinned=pinned,
            pageable=pageable,
            startup_pinned=startup_pinned,
            startup_pageable=startup_pageable,
            pinned_runtime_bytes=pinned_runtime,
            pageable_runtime_bytes=pageable_runtime,
            pinned_startup_bytes=pinned_startup,
            pageable_startup_bytes=pageable_startup,
            runtime_bytes=runtime,
            startup_bytes=startup,
            available_bytes=available,
            headroom_bytes=available - startup,
        ),
        unresolved_bounds=list(UNRESOLVED),
        runtime_feasibility_verified=False,
    )
