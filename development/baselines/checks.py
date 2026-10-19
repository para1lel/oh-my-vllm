import numpy as np

from development.kernels.reference import attention, bf16, numerical_error


def cpu(value):
    return value.detach().float().cpu().numpy()


def gdn_reference(inputs):
    case = inputs.case
    arrays = {
        name: cpu(inputs.tensors[name])
        for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias")
    }
    output = np.empty((case.rows, 48, 128), dtype=np.float32)
    states = {}
    checkpoint_rows = set(inputs.checkpoint_rows)
    initial = cpu(inputs.state_pool)
    read_slots = inputs.read_slots.cpu().tolist()
    for sequence, length in enumerate(case.queries):
        first = inputs.sequence_offsets[sequence]
        last = first + length
        q, k = (arrays[name][first:last].astype(np.float64) for name in ("q", "k"))
        q /= np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + 1e-6)
        k /= np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + 1e-6)
        if not case.verify:
            q, k = bf16(q).astype(np.float64), bf16(k).astype(np.float64)
        q, k = (np.repeat(value, 3, axis=1) for value in (q, k))
        decay = -np.exp(arrays["A_log"].astype(np.float64)) * np.logaddexp(
            0, arrays["a"][first:last].astype(np.float64) + arrays["dt_bias"]
        )
        beta = 1 / (1 + np.exp(-arrays["b"][first:last].astype(np.float64)))
        state = initial[read_slots[sequence]].astype(np.float64)
        for token in range(length):
            state *= np.exp(decay[token])[:, None, None]
            recall = np.einsum("hvk,hk->hv", state, k[token])
            delta = beta[token, :, None] * (arrays["v"][first + token] - recall)
            state += delta[:, :, None] * k[token, :, None, :]
            output[first + token] = bf16(
                np.einsum("hvk,hk->hv", state, q[token]) / np.sqrt(128)
            )
            if first + token in checkpoint_rows:
                states[first + token] = bf16(state)
    return {
        "output": output,
        "checkpoints": np.stack([states[row] for row in inputs.checkpoint_rows]),
    }


def fa_reference(inputs):
    case = inputs.case
    q = cpu(inputs.q)
    cache = cpu(inputs.cache)
    pages = inputs.page_indices.cpu().tolist()
    offsets = inputs.page_offsets.cpu().tolist()
    outputs = []
    for sequence, (queries, context) in enumerate(
        zip(case.queries, case.contexts, strict=True)
    ):
        physical = pages[offsets[sequence] : offsets[sequence + 1]]
        kv = cache[physical].transpose(1, 0, 2, 3, 4).reshape(2, -1, 4, 256)
        first = inputs.sequence_offsets[sequence]
        outputs.append(
            attention(q[first : first + queries], kv[0, :context], kv[1, :context])
        )
    return {"output": np.concatenate(outputs)}


def check_outputs(prepared, inputs, expected):
    result = {
        "output": numerical_error(cpu(prepared.outputs["output"]), expected["output"]),
    }
    if inputs.case.operator == "gdn_attention":
        pool = prepared.outputs["state_pool"]
        checkpoints = pool[list(inputs.checkpoint_slots)]
        result["checkpoints"] = numerical_error(
            cpu(checkpoints), expected["checkpoints"], state=True
        )
        mask = np.ones(pool.shape[0], dtype=bool)
        mask[list(inputs.checkpoint_slots)] = False
        result["unchanged_state"] = bool(
            np.array_equal(cpu(pool)[mask], cpu(inputs.state_pool)[mask])
        )
    result["passed"] = all(
        item["passed"] if isinstance(item, dict) else item for item in result.values()
    )
    return result


def snapshot(prepared, inputs):
    expected = {"output": cpu(prepared.outputs["output"])}
    if inputs.case.operator == "gdn_attention":
        expected["checkpoints"] = cpu(
            prepared.outputs["state_pool"][list(inputs.checkpoint_slots)]
        )
    return expected


def independent_reference(inputs):
    return (
        fa_reference(inputs)
        if inputs.case.operator == "fa_attention"
        else gdn_reference(inputs)
    )
