"""Correctness gate for whole-operation CUDA versus frozen TileLang fixtures."""

import torch

# Match the existing independent operator tests. State copies have exact BF16
# values; GDN recurrent state retains the model-path BF16/FP32 tolerance.
_TOLERANCES = {
    "gates": (2e-6, 2e-6),
    "qk": (0.002, 0.004),
    "quant": (0, 0),
    "silu_quant": (0, 0),
    "append": (0, 0),
    "dspark_append": (0, 0),
}


def _tolerance(operation, path):
    if operation == "prepare_attention" and path == "written_value":
        return 0, 0
    if operation in ("add_norm", "add_norm_fp8_linear") and path == "return[0]":
        return 0, 0
    if operation == "convolution" and path.startswith("written_state"):
        return 0, 0
    if operation == "gates" and path == "return[1]":
        return 1e-7, 1e-6
    if operation in ("quant", "silu_quant") and path.endswith("[1]"):
        return 1e-15, 1e-6
    return _TOLERANCES.get(operation, (0.03, 0.03))


def _compare(operation, path, expected, actual):
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(
                f"{operation}/{path}: shape or dtype mismatch: "
                f"{expected.shape}/{expected.dtype} vs {actual.shape}/{actual.dtype}"
            )
        if (
            operation in ("quant", "silu_quant")
            and path == "return[1]"
            and expected.stride() != actual.stride()
        ):
            raise AssertionError(f"{operation}/{path}: scale stride mismatch")
        if operation == "recurrent" and path == "written_state":
            error = actual.float() - expected.float()
            if not torch.isfinite(error).all():
                raise AssertionError(f"{operation}/{path}: nonfinite state error")
            axes = tuple(range(1, error.ndim))
            reference = expected.float()
            nrmse = error.square().mean(axes).sqrt() / reference.square().mean(
                axes
            ).sqrt().clamp_min(1e-10)
            relative_max = error.abs().amax(axes) / reference.abs().amax(
                axes
            ).clamp_min(1e-10)
            failed = (nrmse > 0.01) | (relative_max > 0.02)
            if failed.any():
                slot = int(failed.nonzero()[0, 0])
                raise AssertionError(
                    f"{operation}/{path}[{slot}]: NRMSE={nrmse[slot].item():.5g}, "
                    f"relative_max={relative_max[slot].item():.5g}"
                )
            return 1
        atol, rtol = _tolerance(operation, path)
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=atol,
            rtol=rtol,
            msg=lambda message: f"{operation}/{path}: {message}",
        )
        return 1
    if isinstance(expected, (tuple, list)) and type(expected) is type(actual):
        if len(expected) != len(actual):
            raise AssertionError(f"{operation}/{path}: output length mismatch")
        return sum(
            _compare(operation, f"{path}[{index}]", old, new)
            for index, (old, new) in enumerate(zip(expected, actual, strict=True))
        )
    if expected is None and actual is None:
        return 0
    raise AssertionError(f"{operation}/{path}: output type mismatch")


def verify(operation, reference, candidate, witnesses=()):
    """Run both backends once and compare returns plus written cache/state slots.

    This runs outside CUDA Graph timing. The passed callables are the exact
    whole-operation functions used by the formal collector; witnesses expose
    only written regions, never uninitialized pool bytes.
    """
    checked = _compare(operation, "return", reference(), candidate())
    names = []
    for name, old, new in witnesses:
        checked += _compare(operation, name, old(), new())
        names.append(name)
    if checked == 0:
        raise AssertionError(f"{operation}: no output or side effect was compared")
    return dict(passed=True, compared_tensors=checked, witnessed_writes=names)
