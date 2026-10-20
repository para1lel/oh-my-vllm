from oh_my_vllm.kernels.sample import Sample

from .sampling import observe_cdf


def bind(prepared):
    inputs, case, device = prepared.inputs, prepared.case, prepared.device
    if case.boundary == "output.hidden":
        hidden = device.empty((case.shape["rows"], 5120), "bfloat16")
        prepared.outputs["hidden"] = hidden

        def run():
            device.module.final_hidden(
                inputs["value"],
                inputs["residual"],
                prepared.weights.norm,
                hidden,
                prepared.state["hidden_cache"],
                inputs["slots"],
            )

        prepared.operation = run
        return run
    operation = prepared.own(
        Sample(prepared.weights.sample, case.shape["rows"], device.empty)
    )
    prepared.outputs.update(
        logits=operation.buffers[1], probabilities=operation.buffers[6]
    )

    def project():
        operation.project(inputs["value"], inputs["residual"])

    if case.boundary == "output.distribution":

        def run():
            project()
            operation.distribution(
                inputs["history"],
                inputs["mask"],
                inputs["parameters"],
                greedy=case.options["greedy"],
            )
    else:
        prepared.observers["selection_cdf"] = lambda: observe_cdf(prepared, operation)
        prepared.outputs.update(
            tokens=operation.buffers[7], selected_probability=operation.buffers[8]
        )

        def run():
            project()
            operation.draw(
                inputs["history"],
                inputs["mask"],
                inputs["parameters"],
                prepared.state["rng"],
                greedy=case.options["greedy"],
            )

    prepared.operation = run
    return project
