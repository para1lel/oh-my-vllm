import numpy as np
from oh_my_vllm.kernels.gdn import GDN, Schedule


def bind(prepared):
    case, weights, inputs = prepared.case, prepared.weights.bindings, prepared.inputs
    host = prepared.fixture.inputs
    schedule = Schedule(
        *(
            tuple(host[name].tolist())
            for name in (
                "sequence_rows",
                "read_slots",
                "write_slots",
                "checkpoint_rows",
                "checkpoint_slots",
                "positions",
            )
        ),
        slots=next(iter(prepared.fixture.state.values())).shape[0],
        verify=case.options["verify"],
    )
    operation = prepared.own(
        GDN(
            weights[0],
            schedule,
            prepared.device.empty,
            lambda values: prepared.device.upload(np.asarray(values, dtype=np.int32)),
        )
    )
    prepared.outputs.update(
        zip(("value", "residual"), operation.buffers[:2], strict=True)
    )
    states = []
    for layer in case.weights:
        prefix = f"layer{layer}." if len(weights) > 1 else ""
        states.append(
            tuple(
                prepared.state[prefix + name] for name in ("recurrent", "convolution")
            )
        )

    def run():
        for weight, state in zip(weights, states, strict=True):
            operation.weights = weight
            operation(inputs["value"], inputs["residual"], *state, inputs.get("tokens"))

    prepared.operation = run
    return run
