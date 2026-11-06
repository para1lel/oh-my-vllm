from ..runtime.native import load
from .api import FUNCTIONS
from .layout import validate_cache


def copy_prefix(codes, scales, source, destination, valid_tokens):
    validate_cache(codes, scales)
    if (
        any(type(x) is not int for x in (source, destination, valid_tokens))
        or source == destination
        or not 0 < source < codes.shape[0]
        or not 0 < destination < codes.shape[0]
        or not 1 <= valid_tokens <= 127
    ):
        raise ValueError(
            "FP8 page copy requires different valid pages and 1..127 tokens"
        )
    load("fa_fp8", FUNCTIONS)[0].fa_fp8_copy_pages(
        codes, scales, source, destination, valid_tokens
    )
