"""Proxy cancellation must wake CPU preflight before it joins handler threads."""

import json
import threading
import time
import urllib.request
from contextlib import suppress

import pytest

from benchmarks.agentic_soak_proxy import ModelSlots, disk_proxy


def test_proxy_failure_cancels_blocked_preflight_before_join(tmp_path):
    entered, release = threading.Event(), threading.Event()
    client = None

    def blocked(body, path):
        entered.set()
        assert release.wait(5)
        raise InterruptedError("test preflight cancelled")

    def request(url):
        body = json.dumps({"stream": True}).encode()
        with suppress(Exception):
            urllib.request.urlopen(url + "/chat/completions", data=body, timeout=5)

    # An incorrect implementation must fail promptly instead of hanging pytest.
    timer = threading.Timer(2, release.set)
    timer.start()
    started = time.monotonic()
    try:
        with (
            pytest.raises(RuntimeError, match="cancel this run"),
            disk_proxy(
                "http://127.0.0.1:1/v1",
                tmp_path / "wire",
                ModelSlots(),
                preflight=blocked,
                cancel_preflight=release.set,
            ) as (url, _),
        ):
            client = threading.Thread(target=request, args=(url,))
            client.start()
            assert entered.wait(1)
            raise RuntimeError("cancel this run")
        assert time.monotonic() - started < 1.5
    finally:
        release.set()
        timer.cancel()
        if client:
            client.join(5)
            assert not client.is_alive()
