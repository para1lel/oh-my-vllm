import numpy as np
from oh_my_vllm.kernels.mtp import MTPContext
from oh_my_vllm.kernels.sample import MTPAccept
from oh_my_vllm.worker.draft_chain import DraftChain
from oh_my_vllm.worker.speculation import Draft

from .fa import arguments, cache
from .sampling import observe_cdf


def context(prepared):
    weights, case, inputs = prepared.weights, prepared.case, prepared.inputs
    operation = prepared.own(
        MTPContext(
            weights.mtp_input,
            weights.attention_variant(weights.mtp_attention, "kv_only"),
            case.shape["rows"],
            prepared.device.empty,
            max_queries=case.shape["queries"],
            batch=case.shape["batch"],
        )
    )
    prepared.own(operation.attention)
    prepared.outputs.update(
        zip(("value", "residual"), operation.endpoints, strict=True)
    )
    input_arguments = tuple(
        inputs[name]
        for name in (
            "hidden",
            "tokens",
            "hidden_rows",
            "boundary_hidden",
            "boundary_rows",
        )
    )
    attention_arguments = arguments(prepared)

    def run():
        operation(input_arguments, attention_arguments)

    prepared.operation = run
    return run


def draft(prepared):
    case, inputs = prepared.case, prepared.inputs
    first = case.options["variant"] == "query_only"
    operation = prepared.own(
        Draft(prepared.device, prepared.weights, case.shape["batch"], first)
    )
    prepared.own(operation.attention)
    prepared.own(operation.ffn)
    operation.hidden = inputs["hidden"]
    operation.cache = cache(prepared)
    operation.metadata = tuple(
        inputs[name]
        for name in ("positions", "slots", "tables", "sequence_rows", "lengths")
    )
    operation.tokens = inputs["tokens"]
    bind_sampling(operation, prepared)
    buffers = operation.sampling.operation.buffers
    prepared.observers["selection_cdf"] = lambda: observe_cdf(
        prepared, operation.sampling.operation
    )
    prepared.outputs.update(
        hidden=buffers[0],
        logits=buffers[1],
        tokens=buffers[7],
        selected_probability=buffers[8],
        probabilities=buffers[6],
    )

    def run():
        operation.run()
        operation.sampling.draw()

    prepared.operation = run
    return operation.run


def bind_sampling(stage, prepared):
    sampler = stage.sampling
    sampler.history = prepared.inputs["history"]
    sampler.mask = prepared.inputs["mask"]
    sampler.parameters = prepared.inputs["parameters"]
    sampler.rng = prepared.state["rng"]
    sampler.greedy = prepared.case.boundary == "mtp.chain" or prepared.case.options.get(
        "greedy", False
    )


def chain(prepared):
    case, inputs, device = prepared.case, prepared.inputs, prepared.device
    stages = []
    for index in range(4):
        stage = prepared.own(
            Draft(device, prepared.weights, case.shape["batch"], index == 0)
        )
        prepared.own(stage.attention)
        prepared.own(stage.ffn)
        stages.append(stage)
    operation = prepared.own(DraftChain(device, stages, case.shape["batch"]))
    for index, stage in enumerate(stages):
        stage.metadata = tuple(
            inputs[f"stage{index}." + name]
            for name in ("positions", "slots", "tables", "sequence_rows", "lengths")
        )
        stage.cache = cache(prepared)
        bind_sampling(stage, prepared)
    stages[0].hidden = inputs["hidden"]
    stages[0].tokens = inputs["tokens"]
    prepared.outputs["tokens"] = operation.tokens
    for name, index in (("hidden", 0), ("logits", 1), ("probabilities", 6)):
        prepared.observers[name] = lambda index=index: np.stack(
            [
                prepared.download(stage.sampling.operation.buffers[index])
                for stage in stages
            ]
        )
    prepared.operation = operation.run
    return None


def accept(prepared):
    inputs, shape = prepared.inputs, prepared.case.shape
    operation = prepared.own(MTPAccept(shape["batch"], prepared.device.empty))
    prepared.outputs.update(
        counts=operation.buffers[1], candidate_slots=operation.buffers[2]
    )
    prepared.observers["tokens"] = lambda: prepared.download(operation.buffers[0])[
        :, :5
    ]
    capacity = int(prepared.fixture.inputs["capacity"].item())

    def run():
        operation(
            inputs["target"],
            inputs["proposal"],
            inputs["drafts"],
            prepared.state["rng"],
            inputs["private_slots"],
            capacity,
        )

    prepared.operation = run
    return None


def bind(prepared):
    return {
        "mtp.context": context,
        "mtp.draft": draft,
        "mtp.chain": chain,
        "mtp.accept": accept,
    }[prepared.case.boundary](prepared)
