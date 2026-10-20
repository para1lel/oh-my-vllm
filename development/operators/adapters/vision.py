from oh_my_vllm.operators.vision import api as vision
from oh_my_vllm.operators.vision.plan import make_plan
from oh_my_vllm.operators.vision.workspace import prepare


def bind(prepared):
    case, shape = prepared.case, prepared.case.shape
    variant = {
        "vision.input": "input",
        "vision.block": "default",
        "vision.merge": "merge",
    }[case.boundary]
    weights = tuple(
        vision.layer_weights(prepared.weights.visual, layer) for layer in case.weights
    )
    plan = make_plan(
        rows=shape["rows"],
        variant=variant,
        grid=(shape["height"], shape["width"]),
        frames=shape["frames"],
    )
    operation = prepared.own(prepare(plan, prepared.device, weights))
    prepared.outputs["hidden"] = operation.outputs.hidden
    value = vision.Inputs(prepared.inputs["pixels" if variant == "input" else "hidden"])

    def run():
        for weight in weights:
            vision.run(weight, operation, value)

    prepared.operation = run
    return run
