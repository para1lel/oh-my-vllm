import os

from oh_my_vllm.models.deepseek_v41.worker_limits import (
    DEFAULT_PREFILL_ROWS,
    prefill_row_limit,
)


def workload_cache_requirements(cases):
    requirements = {}
    for case in cases:
        strategy = case["strategy"]
        if strategy not in ("tp4", "tep4", "dep4"):
            raise ValueError("unknown cache placement strategy")
        requests = case["batch"]
        total = case["input_tokens"] + case["output_tokens"]
        if type(requests) is not int or requests < 1 or total < 1:
            raise ValueError("cache workload needs positive requests and context")
        local = (requests + 3) // 4 if strategy == "dep4" else requests
        drafts = 5 if case["mode"] == "dspark5" else 0
        token_pages = (total + drafts + 127) // 128 + 1
        current = requirements.setdefault(
            strategy,
            {
                "block_tokens": 128,
                "placement": "request_owner" if strategy == "dep4" else "replicated",
                "resident_requests": 0,
                "maximum_total_tokens": 0,
                "blocks": 1,
                "mamba_blocks": 1,
            },
        )
        current["resident_requests"] = max(current["resident_requests"], local)
        current["maximum_total_tokens"] = max(current["maximum_total_tokens"], total)
        current["blocks"] = max(current["blocks"], 1 + local * token_pages)
        current["mamba_blocks"] = max(current["mamba_blocks"], 1 + local * (2 + drafts))
    return requirements


def effective_cache_configuration(arguments, strategy=None):
    requirements = getattr(arguments, "cache_requirements", {})
    selected = (
        [requirements[strategy]]
        if strategy in requirements
        else list(requirements.values())
    )
    minimum_blocks = max((row["blocks"] for row in selected), default=2)
    minimum_states = max((row["mamba_blocks"] for row in selected), default=2)
    blocks = arguments.blocks
    if blocks is None:
        value = os.environ.get("OH_MY_VLLM_NUM_GPU_BLOCKS")
        blocks = max(2048, minimum_blocks) if value is None else int(value)
    states = arguments.mamba_blocks
    if states is None:
        states = max(128, minimum_states)
    for name, value, minimum, maximum in (
        ("blocks", blocks, minimum_blocks, 2**32 - 3),
        ("mamba_blocks", states, minimum_states, 65536),
    ):
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(
                f"{name}={value} cannot cover the selected resident workload; "
                f"need {minimum}..{maximum}, including the null page and private state"
            )
    return {"blocks": blocks, "mamba_blocks": states}


def effective_service_configuration(arguments):
    return {
        "max_num_batched_tokens": getattr(arguments, "max_num_batched_tokens", 32768),
        "max_num_seqs": getattr(arguments, "max_num_seqs", 64),
        "prefill_rows": prefill_row_limit(
            os.environ.get("OH_MY_VLLM_DEEPSEEK_PREFILL_ROWS", DEFAULT_PREFILL_ROWS)
        ),
    }


def service_config(arguments, mode, strategy):
    drafts = 5 if mode == "dspark5" else 0
    service_configuration = effective_service_configuration(arguments)
    return {
        "model_path": str(arguments.model),
        "strategy": strategy,
        "mode": mode,
        "drafts": drafts,
        "cache_context": arguments.cache_context,
        **service_configuration,
    }
