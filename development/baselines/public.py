from .inputs import Prepared


class Unsupported(ValueError):
    pass


FA_NAMES = (
    "flashinfer_auto",
    "flashinfer_fa2",
    "flashinfer_fa3",
    "flashinfer_cudnn",
    "flashinfer_cute",
    "flashinfer_trtllm",
    "flashinfer_decode_fa2",
    "flashinfer_decode_cudnn",
    "flashinfer_decode_cute",
    "flashinfer_decode_trtllm",
    "flashinfer_decode_cuda",
    "flashinfer_cutlass_ragged",
)
PUBLIC_NAMES = (
    *FA_NAMES,
    "flashinfer_gdn_decode",
    "flashinfer_gdn_decode_auto",
    "flashinfer_gdn_decode_cake",
    "flashinfer_gdn_prefill",
    "flashinfer_gdn_prefill_cp",
    "flashinfer_gdn_prefill_auto",
    "flashinfer_gdn_prefill_cudnn",
    "flashinfer_gdn_prefill_cudnn_exact",
    "flashinfer_gdn_prefill_cake",
    "flashinfer_gdn_prefill_cake_cp",
    "flashinfer_gdn_prefill_cake_auto",
    "flashqla_gdn_prefill",
)


def prepare(name, case, inputs, mode):
    if name in FA_NAMES:
        if case.operator != "fa_attention":
            raise Unsupported("FA implementation cannot supply GDN")
        if name == "flashinfer_cutlass_ragged":
            raise Unsupported(
                "FlashInfer ragged CUTLASS declines square head dimension 256"
            )
        if name.startswith("flashinfer_decode_"):
            return prepare_fa_decode(name, case, inputs, mode)
        return prepare_fa(name, case, inputs, mode)
    if case.operator != "gdn_attention":
        raise Unsupported("GDN implementation cannot supply FA")
    if name.startswith("flashinfer_gdn_decode"):
        if not case.verify:
            raise Unsupported(
                "decode API is used only for at most five tokens per sequence"
            )
        return prepare_gdn_decode(name, case, inputs)
    if name.startswith("flashinfer_gdn_prefill") or name == "flashqla_gdn_prefill":
        return prepare_gdn_prefill(name, case, inputs)
    raise Unsupported(f"unknown public implementation: {name}")


def fa_cache(case, inputs, backend):
    import torch

    if backend not in {"cute-dsl", "trtllm-gen", "cudnn"}:
        return (
            inputs.cache,
            inputs.page_offsets,
            inputs.page_indices,
            inputs.last_page_lengths,
            784,
            lambda: None,
            "none; native 784-token paged KV",
        )
    original_indices = inputs.page_indices.cpu().tolist()
    first_page, last_page = min(original_indices), max(original_indices)
    source = (
        inputs.cache[first_page : last_page + 1]
        .view(-1, 2, 49, 16, 4, 256)
        .permute(0, 2, 1, 3, 4, 5)
    )
    packed = torch.empty(source.shape, device="cuda", dtype=torch.bfloat16)
    cache = packed.view(-1, 2, 16, 4, 256)
    original_offsets = inputs.page_offsets.cpu().tolist()
    indices, offsets = [], [0]
    for sequence, context in enumerate(case.contexts):
        pages = original_indices[
            original_offsets[sequence] : original_offsets[sequence + 1]
        ]
        expanded = [
            (page - first_page) * 49 + part for page in pages for part in range(49)
        ]
        indices.extend(expanded[: (context + 15) // 16])
        offsets.append(len(indices))

    def integers(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    return (
        cache,
        integers(offsets),
        integers(indices),
        integers([(context - 1) % 16 + 1 for context in case.contexts]),
        16,
        lambda: packed.copy_(source),
        "timed 784-to-16 page conversion without guard pages;"
        " immutable metadata planned outside timing",
    )


def prepare_fa(name, case, inputs, mode):
    import torch
    from flashinfer import BatchPrefillWithPagedKVCacheWrapper

    backend = {
        "flashinfer_auto": "auto",
        "flashinfer_fa2": "fa2",
        "flashinfer_fa3": "fa3",
        "flashinfer_cudnn": "cudnn",
        "flashinfer_cute": "cute-dsl",
        "flashinfer_trtllm": "trtllm-gen",
    }[name]
    if backend == "fa3":
        from flashinfer.utils import is_sm90a_supported

        if not is_sm90a_supported(torch.device("cuda")):
            raise Unsupported("FA3 requires an SM90a device; this device is SM100")
    if backend == "cute-dsl":
        raise Unsupported(
            "public CuTe prefill fixes a 128x128 tile; D256 uses 459776 bytes"
            " of shared memory, above this device's 232448-byte limit"
        )
    cache, offsets, indices, last_lengths, page_size, pack, adaptation = fa_cache(
        case, inputs, backend
    )
    workspace = torch.zeros(1024 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graph_buffers = (
        {
            "qo_indptr_buf": inputs.tensors["sequence_offsets"],
            "paged_kv_indptr_buf": offsets,
            "paged_kv_indices_buf": indices,
            "paged_kv_last_page_len_buf": last_lengths,
        }
        if mode == "graph"
        else {}
    )
    wrapper = BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend=backend,
        use_cuda_graph=mode == "graph",
        **graph_buffers,
    )
    wrapper.plan(
        inputs.tensors["sequence_offsets"],
        offsets,
        indices,
        last_lengths,
        24,
        4,
        256,
        page_size,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=256**-0.5,
    )
    output = torch.empty_like(inputs.q)

    def run():
        pack()
        wrapper.run(inputs.q, cache, out=output)

    return Prepared(
        run,
        lambda: None,
        {"output": output},
        {
            "backend": backend,
            "planning": "outside_timing; immutable shape metadata",
            "adaptation": adaptation,
            "precision": "BF16 Q/K/V/output; FP32 softmax; BF16 probabilities",
        },
        wrapper,
    )


def prepare_fa_decode(name, case, inputs, mode):
    import torch
    from flashinfer import BatchDecodeWithPagedKVCacheWrapper

    cuda_core = name == "flashinfer_decode_cuda"
    if cuda_core and any(count != 1 for count in case.queries):
        raise Unsupported("pinned CUDA-core decode API supports one query per request")
    if len(set(case.queries)) != 1:
        raise Unsupported("decode API requires a balanced query batch")
    backend = {
        "flashinfer_decode_cuda": "fa2",
        "flashinfer_decode_fa2": "fa2",
        "flashinfer_decode_cudnn": "cudnn",
        "flashinfer_decode_cute": "cute-dsl",
        "flashinfer_decode_trtllm": "trtllm-gen",
    }[name]
    cache, offsets, indices, last_lengths, page_size, pack, adaptation = fa_cache(
        case, inputs, backend
    )
    workspace = torch.zeros(1024 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    graph_buffers = (
        {
            "paged_kv_indptr_buffer": offsets,
            "paged_kv_indices_buffer": indices,
            "paged_kv_last_page_len_buffer": last_lengths,
        }
        if mode == "graph"
        else {}
    )
    wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout="NHD",
        backend=backend,
        use_tensor_cores=not cuda_core,
        use_cuda_graph=mode == "graph",
        **graph_buffers,
    )
    wrapper.plan(
        offsets,
        indices,
        last_lengths,
        24,
        4,
        256,
        page_size,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
        sm_scale=256**-0.5,
        q_len_per_req=case.queries[0],
    )
    output = torch.empty_like(inputs.q)
    zero_output = (
        backend == "cute-dsl" and wrapper._cute_dsl_wrapper._reduction == "atomic"
    )

    def run():
        pack()
        if zero_output:
            output.zero_()
        wrapper.run(inputs.q, cache, out=output)

    return Prepared(
        run,
        lambda: None,
        {"output": output},
        {
            "backend": backend,
            "api": "BatchDecodeWithPagedKVCacheWrapper",
            "tensor_cores": not cuda_core,
            "kernel_family": "CUDA core" if cuda_core else "Tensor Core",
            "planning": "outside_timing; immutable shape metadata",
            "adaptation": adaptation,
            "output_reset": "timed zero for atomic reduction"
            if zero_output
            else "none",
            "precision": (
                "BF16 Q/K/V/output; FP32 softmax; FP32 probabilities"
                if cuda_core
                else "BF16 Q/K/V/output; FP32 softmax; BF16 probabilities"
            ),
        },
        wrapper,
    )


def prepare_gdn_decode(name, case, inputs):
    import torch
    from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

    backend = {
        "flashinfer_gdn_decode": "flashinfer",
        "flashinfer_gdn_decode_auto": "auto",
        "flashinfer_gdn_decode_cake": "cake_gdn",
    }[name]
    pool = inputs.state_pool.clone()
    output = torch.empty((case.rows, 48, 128), device="cuda", dtype=torch.bfloat16)
    calls = []
    balanced = len(set(case.queries)) == 1
    sequences = (
        (range(case.batch),) if balanced else tuple((i,) for i in range(case.batch))
    )
    for group in sequences:
        first_sequence, last_sequence = group[0], group[-1]
        length = case.queries[first_sequence]
        count = len(group)
        first = inputs.sequence_offsets[first_sequence]
        last = inputs.sequence_offsets[last_sequence + 1]
        exported = [
            slot
            for slot, item in zip(
                inputs.checkpoint_slots, inputs.checkpoint_sequences, strict=True
            )
            if item in group
        ]
        read = inputs.read_slots[first_sequence : last_sequence + 1].clone()
        final = torch.tensor(
            [exported[(i + 1) * length - 1] for i in range(count)],
            device="cuda",
            dtype=torch.int32,
        )
        intermediate = pool[exported[0] : exported[-1] + 1].view(
            count, length, 48, 128, 128
        )
        calls.append((first, last, count, length, read, final, intermediate))

    def run():
        for first, last, count, length, read, final, intermediate in calls:
            gated_delta_rule_decode_pretranspose(
                inputs.q[first:last].view(count, length, 16, 128),
                inputs.k[first:last].view(count, length, 16, 128),
                inputs.v[first:last].view(count, length, 48, 128),
                None,
                inputs.A_log,
                inputs.a[first:last].view(count, length, 48),
                inputs.dt_bias,
                inputs.b[first:last].view(count, length, 48),
                output=output[first:last].view(count, length, 48, 128),
                initial_state=pool,
                initial_state_indices=read,
                output_state_indices=final,
                intermediate_states_buffer=intermediate if length > 1 else None,
                disable_state_update=length > 1,
                backend=backend,
            )

    return Prepared(
        run,
        lambda: pool.copy_(inputs.state_pool),
        {"output": output, "state_pool": pool},
        {
            "backend": backend,
            "adaptation": "sequence views; private BF16 pool checkpoints",
            "precision": (
                "BF16 persistent state; FP32 live carry and normalized recurrent Q/K"
            ),
            "read_slots_unchanged": True,
            "batch_policy": (
                "one call for balanced batches; one call per ragged sequence"
            ),
            "metadata_plan": (
                "read-index groups copied to aligned storage during preparation;"
                " fixed metadata adaptation outside the timed callback"
            ),
        },
        calls,
    )


def prepare_gdn_prefill(name, case, inputs):
    import torch

    pool = inputs.state_pool.clone()
    context_parallelism = {
        "flashinfer_gdn_prefill": False,
        "flashinfer_gdn_prefill_cp": True,
        "flashinfer_gdn_prefill_auto": "auto",
        "flashinfer_gdn_prefill_cudnn": False,
        "flashinfer_gdn_prefill_cudnn_exact": False,
        "flashinfer_gdn_prefill_cake": False,
        "flashinfer_gdn_prefill_cake_cp": True,
        "flashinfer_gdn_prefill_cake_auto": "auto",
        "flashqla_gdn_prefill": False,
    }[name]
    backend = (
        "cake_gdn" if "_cake" in name else "cudnn" if "_cudnn" in name else "flashinfer"
    )
    output = torch.empty((case.rows, 48, 128), device="cuda", dtype=torch.bfloat16)
    grouped = {}
    for sequence in range(case.batch):
        first = inputs.sequence_offsets[sequence]
        for row, slot, index in zip(
            inputs.checkpoint_rows,
            inputs.checkpoint_slots,
            inputs.checkpoint_sequences,
            strict=True,
        ):
            if index == sequence:
                local_first = first - inputs.sequence_offsets[sequence]
                key = (local_first, row + 1 - first)
                grouped.setdefault(key, []).append((sequence, first, row + 1, slot))
                first = row + 1
    segments = [
        (
            rows,
            torch.tensor([row[0] for row in rows], device="cuda", dtype=torch.int64),
            torch.tensor([row[3] for row in rows], device="cuda", dtype=torch.int64),
        )
        for _, rows in sorted(grouped.items())
    ]
    if name.startswith("flashinfer_gdn_prefill"):
        from flashinfer.gdn_prefill import chunk_gated_delta_rule

        segment_offsets = {
            (len(rows), rows[0][2] - rows[0][1]): torch.arange(
                len(rows) + 1, dtype=torch.int32, device="cuda"
            )
            * (rows[0][2] - rows[0][1])
            for rows, _, _ in segments
        }
    else:
        from flash_qla import chunk_gated_delta_rule

        segment_offsets = {}

    cake_plans = {}
    cudnn_plans = (
        cudnn_gdn_plans(segments, name.endswith("_exact")) if backend == "cudnn" else []
    )

    def run():
        g = -inputs.A_log.exp() * torch.nn.functional.softplus(
            inputs.a.float() + inputs.dt_bias
        )
        beta = torch.sigmoid(inputs.b.float())
        carries = inputs.state_pool.index_select(0, inputs.read_slots).float()
        for segment_index, (rows, sequence_ids, slots) in enumerate(segments):
            count, length = len(rows), rows[0][2] - rows[0][1]
            packed = [
                value[rows[0][1] : rows[0][2]].unsqueeze(0)
                if count == 1
                else torch.stack([value[first:last] for _, first, last, _ in rows])
                for value in (inputs.q, inputs.k, inputs.v, g, beta)
            ]
            initial = (
                carries
                if count == case.batch
                else carries.index_select(0, sequence_ids)
            )
            if name == "flashinfer_gdn_prefill_cake_cp":
                result, carry = run_cake_cp(
                    cake_plans,
                    segment_index,
                    packed,
                    initial,
                    segment_offsets[(count, length)],
                    length,
                )
            elif backend == "cudnn":
                from flashinfer.cudnn import cudnn_chunk_gated_delta_rule

                result, carry = cudnn_chunk_gated_delta_rule(
                    *(tensor.flatten(0, 1) for tensor in packed),
                    initial_state=initial,
                    output_final_state=True,
                    output_state=torch.empty_like(initial),
                    cu_seqlens=segment_offsets[(count, length)],
                    use_qk_l2norm_in_kernel=True,
                    batch_invariant=cudnn_plans[segment_index]["batch_invariant"],
                    gate_domain="log",
                )
            elif name.startswith("flashinfer_gdn_prefill"):
                result, carry = chunk_gated_delta_rule(
                    packed[0].flatten(0, 1),
                    packed[1].flatten(0, 1),
                    packed[2].flatten(0, 1),
                    packed[3].flatten(0, 1),
                    packed[4].flatten(0, 1),
                    initial_state=initial,
                    output_final_state=True,
                    output_state=torch.empty_like(initial),
                    cu_seqlens=segment_offsets[(count, length)],
                    use_qk_l2norm_in_kernel=True,
                    use_cp=context_parallelism,
                    backend=backend,
                    max_seqlen=length,
                    gate_domain="log",
                )
            else:
                result, carry = chunk_gated_delta_rule(
                    *packed,
                    initial_state=initial,
                    output_final_state=True,
                    state_v_first=True,
                    auto_cp=False,
                    use_qk_l2norm_in_kernel=True,
                )
            result = result.reshape(count, length, 48, 128)
            for item, (_, first, last, _) in enumerate(rows):
                output[first:last].copy_(result[item])
            pool.index_copy_(0, slots, carry.to(torch.bfloat16))
            if count == case.batch:
                carries = carry
            else:
                carries.index_copy_(0, sequence_ids, carry)

    return Prepared(
        run,
        lambda: pool.copy_(inputs.state_pool),
        {"output": output, "state_pool": pool},
        {
            "backend": name,
            "adaptation": (
                "timed Q/K normalization and BF16 stores; timed FP32 gates;"
                " timed source-slot gather and BF16-to-FP32 incoming state;"
                " segments at each 784 boundary or recurrent private row;"
                " BF16 checkpoint exports; FP32 carry continues between segments"
            ),
            "precision": "BF16 Q/K/V/output; FP32 live carry; BF16 checkpoints",
            "auto_cp": False,
            "context_parallelism": context_parallelism,
            "cudnn_plans": cudnn_plans,
            "prepared_plans": (
                "one prepare_gdn_cp_prefill(_capture_graph=False) plan per"
                " checkpoint segment; launch_with_bindings inside timing"
                if name == "flashinfer_gdn_prefill_cake_cp"
                else "public high-level wrapper"
            ),
            "history_evidence": (
                "Cake SM100 BF16 CP routes cp_prefill_bf16/"
                "cp_prefill_generic_bf16 set source_final_block_cg0/cg1=0,"
                " so terminal pruning guards are disabled;"
                " state_fixup kernels load initial_state and scan every chunk;"
                " non-CP Cake routes must match the frozen head geometry;"
                "FlashInfer blackwell/gated_delta_net_cp.py loads gInitialState"
                " and scans cutlass.range(start, num_chunks) with no decay cutoff;"
                " FlashQLA auto_cp=False bypasses cp_preprocess_fwd partitions;"
                " cuDNN uses default only for exact piece-chain or value-split"
                " geometry; other default cuDNN plans are unsupported;"
                " cudnn_exact sets batch_invariant=True to disable warmup cuts"
            ),
            "batch_policy": "batch sequences with the same segment start and length",
            "history": "complete; nonzero incoming state; no gate-decay truncation",
            "read_slots_unchanged": True,
        },
        segments,
    )


def cudnn_gdn_plans(segments, always_exact):
    import torch
    from cudnn.linear_attention.frost.common.piece_chain import (
        choose_pieces,
        is_dv_split,
    )
    from cudnn.linear_attention.frost.kernel.gdn_prefill_config import CFG

    result = []
    for rows, sequence_ids, _ in segments:
        count, length = len(rows), rows[0][2] - rows[0][1]
        geometry = {
            "num_seqs": count,
            "heads_out": 48,
            "num_sm": torch.cuda.get_device_properties(
                sequence_ids.device
            ).multi_processor_count,
            "total_tokens": count * length,
            "b_t": CFG.B_T,
            "expand_num": 1,
        }
        pieces, unit = choose_pieces(
            **geometry, cadence_tokens=0, batch_invariant=False
        )
        value_split = is_dv_split(**geometry, dim_v=128, unit_chunks=unit)
        if not always_exact and not (pieces > 0 or value_split):
            raise Unsupported(
                "default cuDNN GDN selects decay-warmup split-K, which can omit"
                " incoming state and earlier history; use cudnn_exact"
            )
        invariant = always_exact
        result.append(
            {
                **geometry,
                "length": length,
                "batch_invariant": invariant,
                "default_pieces": pieces,
                "default_value_split": value_split,
                "engine": "exact batch-invariant"
                if invariant
                else "exact piece chain"
                if pieces > 0
                else "exact value split",
            }
        )
    return result


def run_cake_cp(plans, index, packed, initial, offsets, length):
    import torch
    from flashinfer.gdn_kernels.blackwell.cake_gdn_cp_backend import (
        prepare_gdn_cp_prefill,
    )
    from flashinfer.gdn_kernels.qk_l2norm import normalize_qk

    q, k = normalize_qk(packed[0].flatten(0, 1), packed[1].flatten(0, 1))
    v = packed[2].flatten(0, 1)
    alpha = packed[3].flatten(0, 1).exp()
    beta = packed[4].flatten(0, 1)
    if index not in plans:
        plans[index] = prepare_gdn_cp_prefill(
            q,
            k,
            v,
            alpha,
            beta,
            offsets,
            initial,
            output=torch.empty_like(v),
            output_state=torch.empty_like(initial),
            max_seqlen=length,
            scale=128**-0.5,
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            _capture_graph=False,
        )
    else:
        plan = plans[index]
        plan.launch_with_bindings(
            q=q,
            k=k,
            v=v,
            alpha=alpha,
            beta=beta,
            initial_state=initial,
            output=plan.output,
            output_state=plan.final_state,
            state_checkpoints=None,
            cu_seqlens=offsets,
            state_indices=None,
            checkpoint_cu_starts=None,
        )
    return plans[index].output, plans[index].final_state
