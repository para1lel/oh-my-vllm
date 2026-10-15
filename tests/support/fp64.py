"""CPU input copies and Double comparisons for independent numerical equations."""

import torch


def cpu(value):
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu()
        return values.clone() if value.device.type == "cpu" else values
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(item) for item in value)
    if isinstance(value, dict):
        return {key: cpu(item) for key, item in value.items()}
    return value


def close(actual, expected, *, atol, rtol, equal_nan=False, check_dtype=True):
    assert actual.shape == expected.shape
    if check_dtype:
        assert actual.dtype == expected.dtype
    torch.testing.assert_close(
        actual.detach().cpu().double(),
        expected.detach().cpu().double(),
        atol=atol,
        rtol=rtol,
        equal_nan=equal_nan,
    )


def specials(actual, expected):
    """Check IEEE classes and signed zeros independently of numeric tolerance."""
    actual, expected = cpu(actual), cpu(expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    for classify in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(classify(actual), classify(expected))
    zero = expected == 0
    assert torch.equal(actual == 0, zero)
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))
