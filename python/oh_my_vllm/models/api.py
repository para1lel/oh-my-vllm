from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class ModelAdapter(Protocol):
    def ready(self) -> dict: ...

    def handle(self, message: dict) -> dict | None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class AdapterSpec:
    key: str
    architecture: str
    model_type: str
    capabilities: frozenset[str]
    validate_config: Callable[[dict], None]
    build: Callable[[dict, bytes], ModelAdapter]
