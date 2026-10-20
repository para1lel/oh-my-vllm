from oh_my_vllm.kernels.fa import FA


def cache(prepared, prefix=""):
    return tuple(
        prepared.state[prefix + name] for name in ("kv", "fp8_kv", "kv_scales")
    )


def arguments(prepared, prefix="", state_prefix=""):
    inputs = prepared.inputs
    return (
        inputs[prefix + "positions"],
        cache(prepared, state_prefix),
        *(
            inputs[prefix + name]
            for name in ("slots", "tables", "sequence_rows", "lengths")
        ),
    )


def bind(prepared):
    case, weights, inputs = prepared.case, prepared.weights.bindings, prepared.inputs
    operation = prepared.own(
        FA(
            weights[0],
            case.shape["rows"],
            prepared.device.empty,
            max_queries=case.shape["queries"],
            max_context=case.shape["context"],
            min_context=case.shape["context"] - case.shape["queries"] + 1,
        )
    )
    prepared.outputs.update(
        zip(("value", "residual"), operation.buffers[:2], strict=True)
    )
    calls = [
        arguments(prepared, state_prefix=f"layer{layer}." if len(weights) > 1 else "")
        for layer in case.weights
    ]

    def run():
        for weight, metadata in zip(weights, calls, strict=True):
            operation.weights = weight
            operation(inputs["value"], inputs["residual"], *metadata)

    prepared.operation = run
    return run
