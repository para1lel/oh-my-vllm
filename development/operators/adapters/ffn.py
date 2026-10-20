from oh_my_vllm.kernels.ffn import FFN


def bind(prepared):
    weights, inputs = prepared.weights.bindings, prepared.inputs
    operation = prepared.own(
        FFN(weights[0], prepared.case.shape["rows"], prepared.device.empty)
    )
    prepared.outputs.update(
        zip(("value", "residual"), operation.buffers[:2], strict=True)
    )

    def run():
        for weight in weights:
            operation.weights = weight
            operation(inputs["value"], inputs["residual"])

    prepared.operation = run
    return run
