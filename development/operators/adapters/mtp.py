import numpy as np
from oh_my_vllm.models.qwen38.draft_chain import DraftChain
from oh_my_vllm.models.qwen38.speculation import Draft
from oh_my_vllm.operators.mtp import api as mtp
from oh_my_vllm.operators.mtp.plan import accept_plan, context_plan
from oh_my_vllm.operators.mtp.workspace import prepare_accept, prepare_context

from .fa import cache, views
from .sampling import observe_cdf


def context(prepared):
    weights, case, inputs = prepared.weights, prepared.case, prepared.inputs
    binding = mtp.Weights(
        weights.mtp_input, weights.attention_variant(weights.mtp_attention, "kv_only")
    )
    operation = prepared.own(
        prepare_context(
            context_plan(
                rows=case.shape["rows"],
                batch=case.shape["batch"],
                table_width=inputs["tables"].shape[1],
                max_queries=case.shape["queries"],
            ),
            prepared.device,
            (binding,),
        )
    )
    prepared.outputs.update(operation.outputs._asdict())
    value = mtp.InputView(
        *(
            inputs[name]
            for name in (
                "hidden",
                "tokens",
                "hidden_rows",
                "boundary_hidden",
                "boundary_rows",
            )
        )
    )
    arguments = mtp.Inputs(value, *views(prepared))

    def run():
        mtp.context(binding, operation, arguments)

    prepared.operation = run
    return run


def draft(prepared):
    case, inputs = prepared.case, prepared.inputs
    first = case.options["variant"] == "query_only"
    operation = prepared.own(
        Draft(
            prepared.device,
            prepared.weights,
            case.shape["batch"],
            first,
            table_width=inputs["tables"].shape[1],
        )
    )
    operation.hidden = inputs["hidden"]
    operation.cache = cache(prepared)
    operation.metadata = tuple(
        inputs[name]
        for name in ("positions", "slots", "tables", "sequence_rows", "lengths")
    )
    operation.tokens = inputs["tokens"]
    bind_sampling(operation, prepared)
    outputs = operation.sampling.operation.outputs
    prepared.observers["selection_cdf"] = lambda: observe_cdf(
        prepared, operation.sampling.operation
    )
    prepared.outputs.update(
        hidden=outputs.hidden,
        logits=outputs.logits,
        tokens=outputs.tokens,
        selected_probability=outputs.selected_probability,
        probabilities=outputs.probabilities,
    )

    def run():
        operation.run()
        operation.sampling.draw()

    prepared.operation = run
    return operation.run


def bind_sampling(stage, prepared):
    sampler = stage.sampling
    sampler.inputs.compact = False
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
            Draft(
                device,
                prepared.weights,
                case.shape["batch"],
                index == 0,
                table_width=inputs[f"stage{index}.tables"].shape[1],
            )
        )
        stages.append(stage)
    operation = prepared.own(DraftChain(device, stages, case.shape["batch"], True))
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
    for name in ("hidden", "logits", "probabilities"):
        prepared.observers[name] = lambda name=name: np.stack(
            [
                prepared.download(getattr(stage.sampling.operation.outputs, name))
                for stage in stages
            ]
        )
    prepared.operation = operation.run
    return None


def accept(prepared):
    inputs, shape = prepared.inputs, prepared.case.shape
    operation = prepared.own(
        prepare_accept(accept_plan(batch=shape["batch"]), prepared.device)
    )
    prepared.outputs.update(
        counts=operation.outputs.counts,
        candidate_slots=operation.outputs.candidate_slots,
    )
    prepared.observers["tokens"] = lambda: prepared.download(operation.outputs.tokens)[
        :, :5
    ]
    capacity = int(prepared.fixture.inputs["capacity"].item())

    arguments = mtp.AcceptInputs(
        inputs["target"],
        inputs["proposal"],
        inputs["drafts"],
        prepared.state["rng"],
        inputs["private_slots"],
        capacity,
    )

    def run():
        mtp.accept(operation, arguments)

    prepared.operation = run
    return None


def bind(prepared):
    return {
        "mtp.context": context,
        "mtp.draft": draft,
        "mtp.chain": chain,
        "mtp.accept": accept,
    }[prepared.case.boundary](prepared)
