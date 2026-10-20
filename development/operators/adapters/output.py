from oh_my_vllm.operators.output import api as output
from oh_my_vllm.operators.output.plan import make_plan
from oh_my_vllm.operators.output.workspace import prepare

from .sampling import observe_cdf


def bind(prepared):
    inputs, case, device = prepared.inputs, prepared.case, prepared.device
    hidden = case.boundary == "output.hidden"
    weights = (
        output.Weights(prepared.weights.norm) if hidden else prepared.weights.sample
    )
    operation = prepared.own(
        prepare(
            make_plan(rows=case.shape["rows"], hidden_only=hidden), device, (weights,)
        )
    )
    value = output.Inputs(inputs["value"], inputs["residual"])
    if hidden:
        prepared.outputs["hidden"] = operation.outputs.hidden
        state = output.HiddenState(prepared.state["hidden_cache"], inputs["slots"])

        def run():
            output.hidden(weights, operation, value, state)

        prepared.operation = run
        return run
    prepared.outputs.update(
        logits=operation.outputs.logits, probabilities=operation.outputs.probabilities
    )
    settings = output.Settings(
        inputs["history"], inputs["mask"], inputs["parameters"], case.options["greedy"]
    )

    def project():
        operation.project(weights, value)

    if case.boundary == "output.distribution":

        def run():
            output.distribution(weights, operation, value, settings)
    else:
        prepared.observers["selection_cdf"] = lambda: observe_cdf(prepared, operation)
        prepared.outputs.update(
            tokens=operation.outputs.tokens,
            selected_probability=operation.outputs.selected_probability,
        )

        def run():
            output.sample(weights, operation, value, settings, prepared.state["rng"])

    prepared.operation = run
    return project
