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
    names = ["device", "ffn", "fa", "gdn", "sample", "mtp", "vision"]
    for name in names:
        manifest = json.loads((cache_directory() / f"{name}.json").read_text())
        if manifest["source_fingerprint"] != source_fingerprint(name):
            raise ValueError("native module differs from its current source")
        if file_sha256(manifest["library"]) != manifest["binary_id"]:
            raise ValueError("native binary differs from its manifest")
        native[name] = {key: manifest[key] for key in ("source_id", "binary_id")}
    return dict(source=digest.hexdigest(), native=native)


def run(model_path, *, decode_token=None, mode="decode"):
    from oh_my_vllm.models.qwen38.cache import Cache
    from oh_my_vllm.models.qwen38.model import Model
    from oh_my_vllm.models.qwen38.weights import Weights
    from oh_my_vllm.operators.output import api as output
    from oh_my_vllm.operators.output.plan import make_plan
    from oh_my_vllm.operators.output.workspace import prepare
    from oh_my_vllm.operators.runtime.device import Device
    from oh_my_vllm.worker.execution import execution_policy

    identity = runtime_identity()
    device = Device()
    weights = Weights(device, model_path, False)
    cache = Cache(
        device,
        dict(
            num_gpu_blocks=2,
            mamba_blocks=2,
            max_num_seqs=1,
            max_num_batched_tokens=16,
            num_speculative_tokens=0,
        ),
    )
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
        media=None,
        position=lambda pos: (pos, pos, pos),
    )
    row = dict(
        request_id="logits",
        num_computed_tokens=0,
        token_ids=list(PROMPT),
        fa_block_table=[1],
        mamba_block_table=[1],
    )

    def forward(batch, graph):
        result = model.forward(batch, graph)
        value, residual = (
            device.view(tensor, (1, 5120), (batch.rows - 1) * 5120)
            for tensor in result[0]
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
            bits = device.download(logits, np.uint16).reshape(-1)
        value = (bits.astype(np.uint32) << 16).view(np.float32)
        if value.shape != (248320,) or not np.isfinite(value).all():
            raise ValueError("model returned invalid endpoint logits")
        return value

    try:
        batch = cache.batch([row], [request], False)
        prefill = forward(batch, False)
        cache.commit(request, batch, 0, None)
        if decode_token is None:
            decode_token = int(np.argmax(prefill))
        outputs = {"prefill": prefill}
        if mode == "decode":
            row.update(num_computed_tokens=len(PROMPT), token_ids=[decode_token])
            outputs["decode"] = forward(cache.batch([row], [request], True), True)
        checkpoint = weights.checkpoint.verify_identity()
        if runtime_identity() != identity:
            raise ValueError("runtime changed during logits capture")
        metadata = dict(
            runtime=identity,
            execution=execution_policy(),
            checkpoint=checkpoint,
            case=dict(tokens=list(PROMPT), decode_token=decode_token, iterations=1),
        )
        return metadata, outputs
    finally:
        model.close()
        device.synchronize()
