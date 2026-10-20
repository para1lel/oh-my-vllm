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
    from oh_my_vllm.kernels.native import cache_directory, file_sha256

    root = Path(oh_my_vllm.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".cu", ".cuh"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    native = {}
    names = ["device", "ffn", "fa", "gdn", "sample"]
    for name in names:
        manifest = json.loads((cache_directory() / f"{name}.json").read_text())
        if file_sha256(manifest["library"]) != manifest["binary_id"]:
            raise ValueError("native binary differs from its manifest")
        native[name] = {key: manifest[key] for key in ("source_id", "binary_id")}
    return dict(source=digest.hexdigest(), native=native)


def run(model_path, *, decode_token=None, mode="decode"):
    from oh_my_vllm.kernels.sample import Sample
    from oh_my_vllm.worker.cache import Cache
    from oh_my_vllm.worker.device import Device
    from oh_my_vllm.worker.execution import execution_policy
    from oh_my_vllm.worker.model import Model
    from oh_my_vllm.worker.weights import Weights

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
    sample = Sample(weights.sample, 1, device.empty)
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
            sample(value, residual, *parameters, greedy=True)

        with ExitStack() as resources:
            project()
            if graph:
                device.synchronize()
                replay = resources.enter_context(device.graph(project))
                replay()
            logits = sample.buffers[1]
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
