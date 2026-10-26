def execution_policy():
    return {"prefill": "adaptive", "decode_only": "cuda_graph", "mixed": "per_fragment"}
