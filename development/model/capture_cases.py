import hashlib
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .capture import runtime_identity


def harness_identity():
    root = Path(__file__).resolve().parents[2]
    sources = [
        root / "development/model" / name
        for name in (
            "__main__.py",
            "capture.py",
            "capture_cases.py",
            "cases.py",
            "compare.py",
        )
    ]
    sources += [
        root / "development/operators" / name
        for name in ("cases.py", "fixtures.py", "storage.py")
    ]
    digest = hashlib.sha256()
    for path in sorted(sources):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return dict(source=digest.hexdigest(), numpy=np.__version__)


def download_logits(device, tensor):
    bits = device.download(tensor, np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def media_processor(device, weights, case, arrays):
    from oh_my_vllm.models.qwen38.media import Media, MediaProcessor, rope_positions

    items = [
        Media(
            arrays[media.name + "_pixels"].copy(),
            media.grid,
            [] if media.frames == 1 else [0.0, 1.0],
            list(media.spans),
        )
        for media in case.media
    ]
    markers = dict(
        image_pad=248056, video_pad=248057, vision_start=248053, vision_end=248054
    )
    tokenizer = SimpleNamespace(
        tokenizer=SimpleNamespace(token_to_id=lambda name: markers[name[2:-2]])
    )
    processor = MediaProcessor(device, weights, tokenizer)
    positions, delta = rope_positions(case.prefix_length, items)
    if not np.array_equal(positions, arrays["positions"]):
        raise ValueError("media position construction differs from the fixed fixture")
    for item in items:
        processor.encode(item)
    return processor, items, delta


class Capture:
    def __init__(self, resources, model_path, case, fixture):
        from oh_my_vllm.models.qwen38.cache import Cache
        from oh_my_vllm.models.qwen38.model import Model
        from oh_my_vllm.models.qwen38.speculation import Speculation
        from oh_my_vllm.models.qwen38.weights import Weights
        from oh_my_vllm.operators.runtime.device import Device

        self.resources, self.case, self.fixture = resources, case, fixture
        self.device = Device()
        self.weights = Weights(self.device, model_path, "mtp_first" in case.outputs)
        pages = (case.prefix_length + 5 + 783) // 784
        self.cache = Cache(
            self.device,
            dict(
                num_gpu_blocks=pages + 1,
                mamba_blocks=pages + 1,
                max_num_seqs=1,
                max_num_batched_tokens=case.prefix_length,
                num_speculative_tokens=4 if "mtp_first" in case.outputs else 0,
            ),
        )
        media, items, delta = media_processor(
            self.device, self.weights, case, fixture.inputs
        )
        coordinates = fixture.inputs["positions"]

        def position(index):
            if index < case.prefix_length:
                return tuple(coordinates[:, index])
            return (index + delta,) * 3

        self.request = SimpleNamespace(
            state_position=0,
            state_slot=None,
            mtp_position=0,
            history=fixture.inputs["tokens"].tolist(),
            media=items,
            position=position,
            finish=False,
            generated=0,
            limit=32,
        )
        self.model = Model(
            self.device, self.weights, self.cache, "mtp_first" in case.outputs, media
        )
        resources.callback(self.model.close)
        self.speculation = Speculation(
            self.device,
            self.weights,
            self.cache,
            media,
            SimpleNamespace(record=lambda *args, **kwargs: None),
        )
        resources.callback(self.speculation.close)
        self.row = dict(
            request_id=case.id,
            num_computed_tokens=0,
            token_ids=fixture.inputs["tokens"].tolist(),
            fa_block_table=list(range(1, pages + 1)),
            mamba_block_table=list(range(1, pages + 1)),
        )

    def batch(self, tokens, start, verify):
        row = {**self.row, "token_ids": list(tokens), "num_computed_tokens": start}
        return self.cache.batch([row], [self.request], verify)

    def forward(self, batch, *, graph, all_rows=False):
        from oh_my_vllm.operators.output import api as output
        from oh_my_vllm.operators.output.plan import make_plan
        from oh_my_vllm.operators.output.workspace import prepare

        d = self.device
        (value, residual), hidden, graph = self.model.forward(batch, graph)
        count = batch.rows if all_rows else 1
        if not all_rows:
            value, residual = (
                d.view(tensor, (1, 5120), (batch.rows - 1) * 5120)
                for tensor in (value, residual)
            )
        sample = prepare(make_plan(rows=count), d, (self.weights.sample,))

        def project():
            sample.project(self.weights.sample, output.Inputs(value, residual))

        project()
        if graph:
            d.synchronize()
            with d.graph(project) as replay:
                replay()
        logits = download_logits(d, sample.outputs.logits)
        return logits if all_rows else logits[0], hidden

    def draft(self, endpoint, table, *, graph):
        from oh_my_vllm.models.qwen38.speculation import Draft

        d, request = self.device, self.request
        hidden = endpoint[0]
        outputs = {}
        for stage, name in enumerate(("mtp_first", "mtp_next")):
            if name not in self.case.outputs:
                continue
            operation = Draft(d, self.weights, 1, stage == 0)
            self.resources.callback(operation.close)
            operation.copy(operation.hidden, hidden)
            position = request.state_position + stage
            token = (
                request.history[-1]
                if stage == 0
                else int(self.fixture.inputs["draft_token"][0])
            )
            arrays = (
                np.asarray(request.position(position), np.int64)[:, None],
                np.asarray(
                    [
                        self.row["fa_block_table"][position // 784] * 784
                        + position % 784
                    ],
                    np.int32,
                ),
                table,
                np.array([0, 1], np.int32),
                np.array([position + 1], np.int32),
                np.array([token], np.int64),
            )
            operation.metadata_inputs.update(arrays)
            operation.cache = self.cache.fa[-1]
            operation.execute(graph)
            outputs[name] = download_logits(
                d, operation.sampling.operation.outputs.logits
            )[0]
            hidden = operation.sampling.operation.outputs.hidden
        return outputs

    def run(self):
        case, arrays, request = self.case, self.fixture.inputs, self.request
        prefix = self.batch(arrays["tokens"], 0, False)
        prefill, hidden = self.forward(prefix, graph=False)
        self.cache.commit(request, prefix, 0, None)
        continuation = arrays["continuation"].tolist()
        if case.restore_checkpoint:
            self.speculation.context(prefix, hidden, False, {})
            request.state_slot = None
            request.state_position = 0
            batch = self.batch(arrays["verification"], case.prefix_length, True)
            if batch.schedule.read_slots != (1,):
                raise ValueError("verification did not restore the logical checkpoint")
            verified, hidden = self.forward(batch, graph=True, all_rows=True)
            self.cache.commit(request, batch, 0, case.accepted_rows - 1)
            request.history += arrays["verification"][: case.accepted_rows].tolist()
            request.history += continuation
            endpoints = self.speculation.context(batch, hidden, True, {0: True})
            outputs = {
                "verified": verified,
                **self.draft(endpoints[0], batch.metadata[2], graph=True),
            }
            endpoint, _ = self.forward(
                self.batch(continuation, request.state_position, True), graph=True
            )
            outputs["restored"] = endpoint
        else:
            request.history += continuation
            outputs = dict(prefill=prefill)
            if "mtp_first" in case.outputs:
                endpoints = self.speculation.context(prefix, hidden, False, {0: True})
                outputs.update(
                    self.draft(endpoints[0], prefix.metadata[2], graph=False)
                )
            endpoint, _ = self.forward(
                self.batch(continuation, request.state_position, True), graph=True
            )
            outputs["decode"] = endpoint
        return outputs


def run(model_path, case, fixture):
    from oh_my_vllm.worker.execution import execution_policy

    identity = runtime_identity()
    harness = harness_identity()
    with ExitStack() as resources:
        capture = Capture(resources, model_path, case, fixture)
        outputs = capture.run()
        checkpoint = capture.weights.checkpoint.verify_identity()
        capture.device.synchronize()
        if runtime_identity() != identity:
            raise ValueError("runtime changed during full-model case capture")
        if harness_identity() != harness:
            raise ValueError("capture harness changed during full-model case capture")
        if checkpoint != fixture.manifest["checkpoint"]:
            raise ValueError("checkpoint differs from the frozen full-model fixture")
        metadata = dict(
            runtime=identity,
            harness=harness,
            execution=execution_policy(),
            checkpoint=checkpoint,
            case=case.definition(),
            case_sha256=case.identity,
            fixture_sha256=case.fixture_sha256,
        )
    return metadata, outputs
