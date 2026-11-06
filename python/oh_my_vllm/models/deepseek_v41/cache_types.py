from dataclasses import dataclass

import torch

from .packed_cache import PackedPage


def _blocks(value):
    if not isinstance(value, list) or any(
        type(block) is not int or not 0 < block <= 0xFFFFFFFF for block in value
    ):
        raise ValueError("invalid DeepSeek cache block list")
    if len(value) != len(set(value)):
        raise ValueError("DeepSeek cache blocks must be unique")
    return value


def _table(value):
    if not isinstance(value, list) or any(
        type(block) is not int or not 0 <= block <= 0xFFFFFFFF for block in value
    ):
        raise ValueError("invalid DeepSeek cache table")
    return value


def _operation(operation):
    for name in ("transfer_id", "snapshot_id"):
        if (
            type(operation.get(name)) is not int
            or not 0 <= operation[name] <= 0xFFFFFFFFFFFFFFFF
        ):
            raise ValueError(f"invalid DeepSeek cache {name}")
    if operation.get("request_id") is not None and (
        type(operation["request_id"]) is not int
        or not 0 <= operation["request_id"] <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("invalid DeepSeek cache request_id")
    if operation.get("kind") not in ("store", "load", "drop"):
        raise ValueError("invalid DeepSeek cache operation")
    _blocks(operation.get("fa_blocks"))
    _blocks(operation.get("mamba_blocks"))
    if operation["kind"] == "drop" and (
        operation["fa_blocks"] or operation["mamba_blocks"]
    ):
        raise ValueError("DeepSeek cache drop cannot contain blocks")


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


@dataclass
class _Snapshot:
    fa: list[_FaPage]
    mamba: list[_MambaPage]
    bytes: int
    request_id: int | None
    layout_id: str
    layout_version: int
