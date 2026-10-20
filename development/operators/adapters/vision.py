from oh_my_vllm.kernels.vision import VisionBlock


def bind(prepared):
    case, shape = prepared.case, prepared.case.shape
    variant = {
        "vision.input": "input",
        "vision.block": "default",
        "vision.merge": "merge",
    }[case.boundary]
    operation = prepared.own(
        VisionBlock(
            prepared.weights.visual,
            shape["rows"],
            shape["height"] * shape["width"],
            prepared.device.empty,
            variant=variant,
            grid=(shape["height"], shape["width"]),
            frames=shape["frames"],
            layer=case.weights[0],
        )
    )
    prepared.outputs["hidden"] = operation.output

    def run():
        for layer in case.weights:
            operation.layer = layer
            operation(prepared.inputs["pixels" if variant == "input" else "hidden"])

    prepared.operation = run
    return run
