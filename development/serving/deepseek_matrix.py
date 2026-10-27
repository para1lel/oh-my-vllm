DS_PARK_STRATEGIES = ("tp4", "tep4", "dep4")
DS_PARK_BATCHES = (4, 16, 64)
DS_PARK_INPUT_TOKENS = (400000, 800000)
OUTPUT_TOKENS = (64, 8192)


def _case(mode, strategy, batch, input_tokens, output_tokens):
    return {
        "mode": mode,
        "strategy": strategy,
        "batch": batch,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


HTTP_CASES = (
    *(
        _case("dspark5", strategy, batch, input_tokens, output_tokens)
        for strategy in DS_PARK_STRATEGIES
        for batch in DS_PARK_BATCHES
        for input_tokens in DS_PARK_INPUT_TOKENS
        for output_tokens in OUTPUT_TOKENS
    ),
    _case("ordinary", "tp4", 4, 400000, 64),
    _case("ordinary", "tp4", 4, 400000, 8192),
)

AGENTX_CASES = (
    {"concurrency": 4, "strategy": "tp4"},
    {"concurrency": 16, "strategy": "tep4"},
    {"concurrency": 64, "strategy": "dep4"},
)


def case_id(case):
    if case["mode"] == "ordinary":
        return ":".join(
            str(case[key]) for key in ("mode", "batch", "input_tokens", "output_tokens")
        )
    return ":".join(
        str(case[key])
        for key in (
            "mode",
            "strategy",
            "batch",
            "input_tokens",
            "output_tokens",
        )
    )
