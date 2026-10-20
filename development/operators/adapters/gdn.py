import numpy as np
from oh_my_vllm.operators.gdn import api as gdn
from oh_my_vllm.operators.gdn.api import Schedule
from oh_my_vllm.operators.gdn.plan import make_plan
from oh_my_vllm.operators.gdn.workspace import prepare


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
    operation = prepared.own(prepare(make_plan(schedule), prepared.device, weights))
    prepared.outputs.update(operation.outputs._asdict())
    batch = gdn.BatchView(
        *(
            prepared.device.upload(np.asarray(values, dtype=np.int32))
            for values in (
                schedule.sequence_rows,
                schedule.read_slots,
                schedule.validate(),
            )
        )
    )
    value = gdn.Inputs(inputs["value"], inputs["residual"], inputs.get("tokens"))
    states = []
    for layer in case.weights:
        prefix = f"layer{layer}." if len(weights) > 1 else ""
        states.append(
            gdn.StateView(
                *(
                    prepared.state[prefix + name]
                    for name in ("recurrent", "convolution")
                )
            )
        )

    def run():
        for weight, state in zip(weights, states, strict=True):
            gdn.run(weight, operation, value, batch, state)

    prepared.operation = run
    return run
