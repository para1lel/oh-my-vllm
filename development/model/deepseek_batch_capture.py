import hashlib


def cpu(value, torch):
    from oh_my_vllm.models.deepseek_v41.packed_cache import PackedPage

    if isinstance(value, PackedPage):
        return cpu(vars(value), torch)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu(item, torch) for item in value]
    return value


def differences(expected, actual, torch, name="endpoint"):
    if isinstance(expected, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            return [dict(name=name, reason="tensor metadata differs")]
        if not torch.equal(
            expected.contiguous().view(torch.uint8),
            actual.contiguous().view(torch.uint8),
        ):
            return [
                dict(
                    name=name,
                    count=int((expected != actual).sum()),
                    size=expected.numel(),
                    max_abs=float(
                        (expected.float() - actual.float()).abs().nan_to_num().max()
                    ),
                )
            ]
        return []
    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return [dict(name=name, reason="keys differ")]
        return [
            item
            for key in expected
            for item in differences(
                expected[key], actual[key], torch, name + "/" + str(key)
            )
        ]
    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return [
                dict(
                    name=name,
                    reason="length differs",
                    expected=len(expected),
                    actual=len(actual),
                )
            ]
        return [
            item
            for index, (left, right) in enumerate(zip(expected, actual, strict=True))
            for item in differences(left, right, torch, name + "/" + str(index))
        ]
    return (
        []
        if expected == actual
        else [dict(name=name, expected=expected, actual=actual)]
    )


def fingerprint(value, torch):
    from oh_my_vllm.models.deepseek_v41.packed_cache import PackedPage

    if isinstance(value, PackedPage):
        return fingerprint(vars(value), torch)
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu()
        return dict(
            shape=list(value.shape),
            dtype=str(value.dtype),
            sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
        )
    if isinstance(value, dict):
        return {key: fingerprint(item, torch) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(item, torch) for item in value]
    return value


def cache_state(worker, torch, *, digest=False):
    with torch.inference_mode():
        return _read_cache_state(worker, torch, digest=digest)


def _read_cache_state(worker, torch, *, digest=False):
    snapshot = fingerprint if digest else cpu
    result = {}
    cache = worker.cache
    for request_id, request in worker.requests.items():
        slot, end = request.cache_slot, request.state_position
        if slot is None:
            continue
        for kind, buffers in (("window", cache._window), ("mtp", cache._mtp_window)):
            for index, value in enumerate(buffers):
                result[f"{request_id}/{kind}/{index}"] = snapshot(value[slot], torch)
        for kind, buffers in (
            ("compressed", cache._compressed),
            ("index", cache._index),
        ):
            for index, (value, ratio) in enumerate(buffers):
                result[f"{request_id}/{kind}/{index}"] = snapshot(
                    value[slot, : end // ratio], torch
                )
        for index, value in enumerate(cache._engram):
            result[f"{request_id}/engram/{index}"] = snapshot(value[slot, :end], torch)
        for index, (kv, score) in enumerate(cache._tails):
            result[f"{request_id}/tail/{index}"] = snapshot(kv[slot], torch)
            result[f"{request_id}/score/{index}"] = snapshot(score[slot], torch)
    for kind, pages in (("fa", cache.fa_pages), ("mamba", cache.mamba_pages)):
        for identity, page in pages.items():
            result[f"pages/{kind}/{identity}"] = snapshot(vars(page), torch)
    return result


def row(request_id, tokens, position, *, blocks=2):
    first = (request_id - 1) * blocks + 1
    table = list(range(first, first + blocks))
    return dict(
        request_id=request_id,
        token_ids=tokens,
        num_computed_tokens=position,
        fa_block_table=list(table),
        mamba_block_table=list(table),
    )


def target_signature(position, length, cycle, window, ratios, bucket):
    return dict(
        length=length,
        bucket=bucket(position + length),
        residue=position % cycle,
        window_prefix=min(window, position) if length > 1 else window,
        window_block=position // window,
        compressed_groups={str(ratio): position // ratio for ratio in ratios},
    )


class TargetCapture:
    def __init__(self, worker, torch, batch):
        self.worker, self.torch, self.batch = worker, torch, batch
        self.targets = {}
        self.calls = []

    def record(
        self, logits, hidden, positions, slots, execution, graphs_before, *, length
    ):
        owners = {
            request.cache_slot: key for key, request in self.worker.requests.items()
        }
        for index, slot in enumerate(slots):
            key = owners[slot]
            self.targets.setdefault(key, []).append(
                cpu(
                    dict(
                        logits=logits[index],
                        hidden=None if hidden is None else hidden[index],
                    ),
                    self.torch,
                )
            )
        ratios = {
            layer.attn.compress_ratio
            for layer in self.worker.model.layers
            if layer.attn.compress_ratio > 1
        }
        self.calls.append(
            dict(
                requests=[owners[slot] for slot in slots],
                slots=list(slots),
                positions=list(positions),
                batch_size=len(slots),
                length=length,
                signatures=[
                    target_signature(
                        value,
                        length,
                        self.worker.decode_cycle,
                        self.worker.args.window_size,
                        ratios,
                        self.batch.context_token_bucket,
                    )
                    for value in positions
                ],
                execution=execution,
                captured=len(self.worker.decode_graphs) - graphs_before,
                graphs=len(self.worker.decode_graphs),
            )
        )

    def __enter__(self):
        self.original_target = self.worker._target
        self.original_group = self.worker._target_batch_group
        self.original_batch = self.batch.target_decode_batch

        def target(tokens, position, slot, **options):
            before = len(self.worker.decode_graphs)
            result = self.original_target(tokens, position, slot, **options)
            self.record(
                result[1],
                result[2],
                (position,),
                (slot,),
                self.worker._target_execution,
                before,
                length=len(tokens),
            )
            return result

        def group(tokens, position, slots, **options):
            before = len(self.worker.decode_graphs)
            result = self.original_group(tokens, position, slots, **options)
            self.record(
                result[1],
                result[2],
                (position,) * len(tokens),
                slots.tolist(),
                self.worker._target_execution,
                before,
                length=len(tokens[0]),
            )
            return result

        def batched(worker, tokens, positions, slots, **options):
            before = len(worker.decode_graphs)
            result = self.original_batch(worker, tokens, positions, slots, **options)
            self.record(
                result.logits,
                result.hidden,
                positions,
                slots,
                result.execution,
                before,
                length=len(tokens[0]),
            )
            return result

        self.worker._target = target
        self.worker._target_batch_group = group
        self.batch.target_decode_batch = batched
        return self

    def __exit__(self, *_error):
        self.worker._target = self.original_target
        self.worker._target_batch_group = self.original_group
        self.batch.target_decode_batch = self.original_batch


def request_state(worker):
    return {
        key: dict(
            history=list(request.history),
            computed=request.state_position,
            finish=request.finish,
        )
        for key, request in worker.requests.items()
    }
