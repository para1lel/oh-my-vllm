import hashlib
import json
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROMPT = (
    760,
    3841,
    13477,
    37550,
    33075,
    888,
    279,
    15217,
    5388,
    13,
    79091,
    3069,
    279,
    12515,
    369,
    6105,
)


def runtime_identity():
    import oh_my_vllm
    from oh_my_vllm.operators.runtime.native import (
        cache_directory,
        file_sha256,
        source_fingerprint,
    )

    root = Path(oh_my_vllm.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    native = {}
    names = ["device", "ffn", "fa_fp8", "gdn_radix", "sample", "mtp_fp8", "vision"]
    for name in names:
        manifest = json.loads((cache_directory() / f"{name}.json").read_text())
        if manifest["source_fingerprint"] != source_fingerprint(name):
            raise ValueError("native module differs from its current source")
        if file_sha256(manifest["library"]) != manifest["binary_id"]:
            raise ValueError("native binary differs from its manifest")
        native[name] = {key: manifest[key] for key in ("source_id", "binary_id")}
    return dict(source=digest.hexdigest(), native=native)


def capture_config(context, tokens, mtp=False):
    from oh_my_vllm.models.qwen38.cache_budget import (
        CHECKPOINT_TOKENS,
        PAGES_PER_CHECKPOINT,
    )

    checkpoints = context // CHECKPOINT_TOKENS + 1
    return dict(
        num_gpu_blocks=PAGES_PER_CHECKPOINT * checkpoints,
        mamba_blocks=checkpoints,
        max_model_len=262144,
        max_num_seqs=1,
        max_num_batched_tokens=tokens,
        num_speculative_tokens=4 if mtp else 0,
    )


def capture_exports(cache, start, end, resume, private, verify):
    from oh_my_vllm.models.qwen38.cache_budget import CHECKPOINT_TOKENS

    if verify:
        available = [entry for entry in private if entry["slot"] != resume["slot"]]
        exports = [
            dict(end=position, **entry, role="private_endpoint")
            for position, entry in zip(
                range(start + 1, end + 1), available, strict=False
            )
        ]
        if len(exports) != end - start:
            raise ValueError("capture verification exceeds its private checkpoint pool")
        return exports
    ends = list(range(CHECKPOINT_TOKENS, end + 1, CHECKPOINT_TOKENS))
    exports = [
        dict(
            end=position,
            slot=index + 1,
            version=cache.versions.get(index + 1, 0) + 1,
            role="periodic",
        )
        for index, position in enumerate(ends)
    ]
    if not ends or ends[-1] != end:
        slot = len(ends) + 1
        exports.append(
            dict(
                end=end,
                slot=slot,
                version=cache.versions.get(slot, 0) + 1,
                role="candidate_tail",
            )
        )
    return exports


def capture_row(cache, request, tokens, start, request_id, verify=False):
    from oh_my_vllm.models.qwen38.cache_budget import LAYOUT_ID, PAGE_TOKENS

    tokens = list(tokens)
    end, layout = start + len(tokens), cache.layout
    resume = (
        request.endpoints[start].wire() if start else dict(end=0, slot=0, version=0)
    )
    private = [
        dict(slot=slot, version=cache.versions.get(slot, 0) + 1)
        for slot in range(layout.private_start, layout.scratch_start)
    ]
    mtp_end = cache.mtp_contexts.get(resume["slot"])
    return dict(
        request_id=request_id,
        lease_id=1,
        purpose="advance",
        action="forward",
        sampling=dict(kind="verify", drafts=len(tokens) - 1)
        if verify and len(tokens) > 1
        else dict(kind="target"),
        token_ids=tokens,
        forward_start=start,
        forward_end=end,
        accepted_end=end,
        endpoint=end,
        model_state=dict(
            kind="qwen_radix_v1",
            layout_fingerprint=LAYOUT_ID,
            checkpoint_hit_end=start,
            kv_contiguous_end=request.kv_valid_end,
            resume=resume,
            private_checkpoints=private,
            page_table=[
                dict(
                    page_index=index,
                    slot=index + 1,
                    version=1,
                    valid_tokens=max(
                        0, min(PAGE_TOKENS, request.kv_valid_end - index * PAGE_TOKENS)
                    ),
                    access="write",
                )
                for index in range((end + 4 + PAGE_TOKENS - 1) // PAGE_TOKENS)
            ],
            kv_spans=[dict(start=start, end=end, mode="write")],
            page_copies=[],
            exports=capture_exports(cache, start, end, resume, private, verify),
            endpoint_source=dict(kind="forward"),
            mtp_endpoint=dict(resume, end=mtp_end)
            if start and mtp_end == start - 1
            else None,
        ),
    )


def run(model_path, *, decode_token=None, mode="decode"):
    from oh_my_vllm.models.qwen38.cache import Cache
    from oh_my_vllm.models.qwen38.model import Model
    from oh_my_vllm.models.qwen38.weights import Weights
    from oh_my_vllm.operators.output.plan import make_plan
    from oh_my_vllm.operators.output.workspace import prepare
    from oh_my_vllm.operators.runtime.device import Device
    from oh_my_vllm.worker.execution import execution_policy

    identity = runtime_identity()
    device = Device()
    weights = Weights(device, model_path, False)
    cache = Cache(device, capture_config(len(PROMPT) + 1, len(PROMPT)))
    model = Model(
        device, weights, cache, False, SimpleNamespace(rows=lambda _: (None, None))
    )
    sample = prepare(make_plan(rows=1), device, (weights.sample,))
    parameters = (
        device.zeros((1, 2, 248320), "int32"),
        device.upload(np.full((1, 248320 // 8), 255, np.uint8)),
        device.upload(np.array([[0, -1, 1, 0, 0, 0, 1]], np.float32)),
        device.upload(np.zeros((1, 2), np.uint64)),
    )
    request = SimpleNamespace(
        state_position=0,
        state_slot=None,
        kv_valid_end=0,
        endpoints={},
        media=None,
        position=lambda pos: (pos, pos, pos),
    )
    row = capture_row(cache, request, PROMPT, 0, "logits")

    try:
        batch = cache.batch([row], [request], False)
        prefill = _forward_logits(
            model, device, weights, sample, parameters, batch, False
        )
        cache.commit(request, batch, 0, None)
        if decode_token is None:
            decode_token = int(np.argmax(prefill))
        outputs = {"prefill": prefill}
        if mode == "decode":
            row = capture_row(
                cache, request, [decode_token], len(PROMPT), "logits", True
            )
            outputs["decode"] = _forward_logits(
                model,
                device,
                weights,
                sample,
                parameters,
                cache.batch([row], [request], True),
                True,
            )
        checkpoint = weights.checkpoint.verify_identity()
        if runtime_identity() != identity:
            raise ValueError("runtime changed during logits capture")
        metadata = dict(
            runtime=identity,
            cache_layout=cache.layout.wire(),
            execution=execution_policy(),
            checkpoint=checkpoint,
            case=dict(tokens=list(PROMPT), decode_token=decode_token, iterations=1),
        )
        return metadata, outputs
    finally:
        model.close()
        device.synchronize()


def _forward_logits(model, device, weights, sample, parameters, batch, graph):
    from oh_my_vllm.operators.output import api as output

    copies = [
        item for row in batch.scheduled for item in row["model_state"]["page_copies"]
    ]
    model.cache.validate_pages(batch.plans, copies)
    model.cache.copy_prefix(copies)
    result, _, graph = model.forward(batch, graph)
    model.cache.complete_exports(batch)
    value, residual = (
        device.view(tensor, (1, 5120), (batch.rows - 1) * 5120) for tensor in result
    )

    def project():
        output.sample(
            weights.sample,
            sample,
            output.Inputs(value, residual),
            output.Settings(*parameters[:3], greedy=True),
            parameters[3],
        )

    with ExitStack() as resources:
        project()
        if graph:
            device.synchronize()
            replay = resources.enter_context(device.graph(project))
            replay()
        logits = sample.outputs.logits
        bits = device.download(logits, np.uint16).numpy().reshape(-1)
    value = (bits.astype(np.uint32) << 16).view(np.float32)
    if value.shape != (248320,) or not np.isfinite(value).all():
        raise ValueError("model returned invalid endpoint logits")
    return value
