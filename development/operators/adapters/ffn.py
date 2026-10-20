from oh_my_vllm.operators.ffn import api as ffn
from oh_my_vllm.operators.ffn.plan import make_plan
from oh_my_vllm.operators.ffn.workspace import prepare


def bind(prepared):
    weights, inputs = prepared.weights.bindings, prepared.inputs
    operation = prepared.own(
        prepare(make_plan(rows=prepared.case.shape["rows"]), prepared.device, weights)
    )
    prepared.outputs.update(operation.outputs._asdict())
    value = ffn.Inputs(inputs["value"], inputs["residual"])

    def run():
        for weight in weights:
            ffn.run(weight, operation, value)

    prepared.operation = run
    return run
