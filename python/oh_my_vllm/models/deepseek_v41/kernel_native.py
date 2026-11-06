from ...operators.hyperconnection.api import FUNCTIONS as HYPERCONNECTION_FUNCTIONS
from ...operators.transform.api import FUNCTIONS as TRANSFORM_FUNCTIONS

_PROJECTION_MODULE = None


_ROUTING_MODULE = None


_ACTIVATION_MODULE = None


_HYPERCONNECTION_MODULE = None


_TRANSFORM_MODULE = None


NATIVE_ROW_LIMIT = 384


def _native_available(x):
    return x.is_cuda


def _projection_module():
    global _PROJECTION_MODULE
    if _PROJECTION_MODULE is None:
        from ...operators.projection.api import FUNCTIONS
        from ...operators.runtime.native import load

        _PROJECTION_MODULE, _, _ = load("projection", FUNCTIONS)
    return _PROJECTION_MODULE


def _routing_module():
    global _ROUTING_MODULE
    if _ROUTING_MODULE is None:
        from ...operators.routing.api import FUNCTIONS
        from ...operators.runtime.native import load

        _ROUTING_MODULE, _, _ = load("routing", FUNCTIONS)
    return _ROUTING_MODULE


def _activation_module():
    global _ACTIVATION_MODULE
    if _ACTIVATION_MODULE is None:
        from ...operators.runtime.native import load

        _ACTIVATION_MODULE, _, _ = load("activation", ("swiglu",))
    return _ACTIVATION_MODULE


def _hyperconnection_module():
    global _HYPERCONNECTION_MODULE
    if _HYPERCONNECTION_MODULE is None:
        from ...operators.runtime.native import load

        _HYPERCONNECTION_MODULE, _, _ = load(
            "hyperconnection", HYPERCONNECTION_FUNCTIONS
        )
    return _HYPERCONNECTION_MODULE


def _transform_module():
    global _TRANSFORM_MODULE
    if _TRANSFORM_MODULE is None:
        from ...operators.runtime.native import load

        _TRANSFORM_MODULE, _, _ = load("transform", TRANSFORM_FUNCTIONS)
    return _TRANSFORM_MODULE
