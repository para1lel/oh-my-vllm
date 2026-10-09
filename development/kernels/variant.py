"""Fail-closed CUDA host-dispatch checks for formal model-shape fixtures."""

REQUIRED_FAST = frozenset(
    (
        "norm",
        "add_norm",
        "gated_norm",
        "qk",
        "recurrent",
        "convolution",
        "gated_norm_fp8_linear",
    )
)


def verify_fast_dispatch(operation, before, after):
    """Require one fast host dispatch and no generic one during verification."""
    if operation not in REQUIRED_FAST:
        return None
    fast = after[operation]["fast"] - before[operation]["fast"]
    generic = after[operation]["generic"] - before[operation]["generic"]
    if (fast, generic) != (1, 0):
        raise AssertionError(
            f"{operation} CUDA fixture dispatched fast={fast}, generic={generic}; "
            "expected exactly one fast and no generic launch"
        )
    return {"operation": operation, "fast": fast, "generic": generic, "passed": True}
