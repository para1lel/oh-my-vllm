from dataclasses import dataclass

import torch

from .packed_cache import PackedPage


def _table(value):
    if not isinstance(value, list) or any(
        type(block) is not int or not 0 <= block <= 0xFFFFFFFF for block in value
    ):
        raise ValueError("invalid DeepSeek cache table")
    return value


@dataclass
class _FaPage:
    compressed: list[PackedPage]
    index: list[PackedPage]
    engram: list[torch.Tensor]
    valid_tokens: int = 0


@dataclass
class _MambaPage:
    window: list[torch.Tensor]
    mtp_window: list[torch.Tensor]
    engram: list[torch.Tensor]
    tails: list[tuple[torch.Tensor, torch.Tensor]]
    valid_tokens: int = 0
    endpoint: int = 0
