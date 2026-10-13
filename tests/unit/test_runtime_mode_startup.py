"""Private init maps all retained startup modes without positional field drift."""

import pytest
from oh_my_vllm.worker import zmq_bridge


@pytest.mark.parametrize(
    "mode,count,draft,threshold",
    [("none", 0, None, 0.2), ("mtp", 4, None, 0.2), ("dspark", 7, "draft", 0.05)],
)
def test_init_keeps_startup_mode_capacity_and_threshold(
    monkeypatch, mode, count, draft, threshold
):
    calls = []

    class Worker:
        def __init__(self, config):
            self.config = config

        def init_device(self):
            calls.append("init")

        def load_model(self):
            calls.append("load")

        def initialize_cache(self):
            calls.append("cache")

    monkeypatch.setattr(zmq_bridge, "OhMyVllmWorker", Worker)
    monkeypatch.setattr(zmq_bridge, "ExecutionTrace", lambda worker: object())
    worker = zmq_bridge._handle_init(
        {
            "model_path": "target",
            "num_gpu_blocks": 4200,
            "mamba_blocks": 128,
            "max_model_len": 262144,
            "num_speculative_tokens": count,
            "speculative_mode": mode,
            "draft_model_path": draft,
            "dspark_confidence_threshold": threshold,
        }
    )
    assert calls == ["init", "load", "cache"]
    assert worker.config.model == "target"
    assert worker.config.speculative_mode == mode
    assert worker.config.speculative_tokens == count
    assert worker.config.max_model_len == 262144
    assert worker.config.num_gpu_blocks == 4200
    assert worker.config.mamba_blocks == 128
    assert worker.config.draft_model == draft
    assert worker.config.dspark_confidence_threshold == threshold
