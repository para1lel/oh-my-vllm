from oh_my_vllm.operators.fa import api as fa
from oh_my_vllm.operators.fa.plan import make_plan
from oh_my_vllm.operators.fa.workspace import prepare


def cache(prepared, prefix=""):
    return (prepared.state[prefix + "kv"],)


def arguments(prepared, prefix="", state_prefix=""):
    inputs = prepared.inputs
    return (
        inputs[prefix + "positions"],
        cache(prepared, state_prefix),
        *(
            inputs[prefix + name]
            for name in ("slots", "tables", "sequence_rows", "lengths")
        ),
    )


def views(prepared, prefix="", state_prefix=""):
    positions, kv, slots, tables, starts, lengths = arguments(
        prepared, prefix, state_prefix
    )
    return fa.BatchView(positions, starts, lengths), fa.CacheView(kv, slots, tables)


def bind(prepared):
    case, weights, inputs = prepared.case, prepared.weights.bindings, prepared.inputs
    plan = make_plan(
        rows=case.shape["rows"],
        batch=case.shape["batch"],
        table_width=inputs["tables"].shape[1],
        max_queries=case.shape["queries"],
        max_context=case.shape["context"],
        min_context=case.shape["context"] - case.shape["queries"] + 1,
    )
    operation = prepared.own(prepare(plan, prepared.device, weights))
    prepared.outputs.update(operation.outputs._asdict())
    value = fa.Inputs(inputs["value"], inputs["residual"])
    calls = [
        views(prepared, state_prefix=f"layer{layer}." if len(weights) > 1 else "")
        for layer in case.weights
    ]

    def run():
        for weight, (batch, state) in zip(weights, calls, strict=True):
            fa.run(weight, operation, value, batch, state)

    prepared.operation = run
    return run
