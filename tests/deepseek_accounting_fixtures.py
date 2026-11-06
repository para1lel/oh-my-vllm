from dataclasses import replace

from development.serving import deepseek_accounting as acct

MODEL = replace(
    acct.DEEPSEEK_V41_MODEL,
    num_layers=2,
    compress_ratios=(1, 1),
    kv_source_layers=(0,),
    index_source_layers=(0, 1),
    candidate_source_layer=0,
    candidate_positions=8,
    index_topk=4,
    engram_layers=(0,),
)


def case(strategy="tp4", **values):
    return dict(
        mode="dspark5",
        strategy=strategy,
        batch=4,
        input_tokens=20,
        output_tokens=8,
        **values,
    )


def trace(strategy="tp4", **values):
    value = case(strategy)
    value.update(values)
    return acct.ideal_trace(value, model=MODEL)


def binding(**overrides):
    values = trace()["binding"]
    values.update(overrides)
    return acct.build_binding(**values)


def decode_trace(accepted, proposed):
    value = trace(batch=1, output_tokens=accepted + 2)
    for record in value["calls"]:
        if record["phase"] == "decode":
            record["request_rows"][0]["proposed"] = proposed
    value["binding"]["proposed"] = proposed
    return value
