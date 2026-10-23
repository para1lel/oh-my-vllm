import unittest
import weakref
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from .cases import load_cases
from .fixtures import Fixture
from .run import PreparedCase
from .storage import equal_bits


class Device:
    def __init__(self, events, fail=False):
        self.events, self.fail = events, fail
        self.module = self

    def empty(self, shape, dtype):
        return np.empty(shape, dtype)

    def upload(self, value, dtype=None):
        return value.copy()

    def copy_bytes(self, destination, source):
        destination[...] = np.from_dlpack(source)

    def download(self, value, dtype):
        return value.astype(dtype).copy()

    def synchronize(self):
        self.events.append("synchronize")

    @contextmanager
    def graph(self, operation):
        with operation.__self__.graph_lease():
            self.events.append("record")
            if self.fail:
                raise RuntimeError("record failed")
            try:
                yield operation
            finally:
                self.events.append("destroy")


class Operation:
    def __init__(self, value, events):
        self.value, self.events = value, events

    def __call__(self):
        self.value += 1

    @contextmanager
    def graph_lease(self):
        self.events.append("lease")
        try:
            yield
        finally:
            self.events.append("release")

    def close(self):
        self.events.append("close")


class Lifecycle(unittest.TestCase):
    def test_projection_preparation_keeps_owners_and_original_failure(self):
        from oh_my_vllm.operators.projection import workspace
        from oh_my_vllm.operators.projection.api import Weights

        for mode in ("pack", "sync", "both"):
            with self.subTest(mode=mode):
                allocations, retained = [], []

                def allocate(shape, dtype, allocations=allocations):
                    value = np.empty(shape, dtype)
                    allocations.append(weakref.ref(value))
                    return value

                def pack(*args, mode=mode):
                    if mode != "sync":
                        raise RuntimeError("packing failed")

                def synchronize(allocations=allocations, mode=mode):
                    self.assertTrue(all(value() is not None for value in allocations))
                    if mode != "pack":
                        raise RuntimeError("completion failed")

                device = SimpleNamespace(
                    empty=allocate,
                    synchronize_compute=synchronize,
                    runtime=SimpleNamespace(cudaStreamIsCapturing=lambda *args: 0),
                    checked=lambda status: None,
                )
                weights = Weights(
                    np.zeros((32, 32), np.uint8), np.zeros((32, 2), np.uint8)
                )
                with (
                    patch.object(
                        workspace,
                        "load",
                        return_value=(SimpleNamespace(pack_weights=pack), None, None),
                    ),
                    patch.object(
                        workspace, "_FAILED_PREPARATIONS", retained, create=True
                    ),
                    self.assertRaisesRegex(
                        RuntimeError, "completion" if mode == "sync" else "packing"
                    ),
                ):
                    workspace.prepare_weights(device, weights, bits=4)
                self.assertEqual(len(retained), int(mode != "pack"))
                self.assertEqual(
                    [value() is not None for value in allocations],
                    [mode != "pack"] * 2,
                )

    def test_reset_keeps_addresses_and_recording_failure_releases_owners(self):
        self.assertFalse(equal_bits(np.array([0.0]), np.array([-0.0])))
        self.assertTrue(equal_bits(np.array([np.nan]), np.array([np.nan])))
        case = next(c for c in load_cases() if c.id == "output.sample.numerics.greedy")
        fixture = Fixture(
            {"tensors": {"state.rng": {"tensor_dtype": "uint64"}}, "writes": {}},
            {},
            {"rng": np.array([[17, 0]], np.uint64)},
            {},
        )
        for fail in (False, True):
            events = []
            prepared = PreparedCase(case, Device(events, fail), fixture)
            value = prepared.state["rng"]
            op = prepared.own(Operation(value, events))
            prepared.operation = op
            if fail:
                with self.assertRaisesRegex(RuntimeError, "record failed"):
                    prepared.start()
                self.assertEqual(events[-3:], ["release", "synchronize", "close"])
            else:
                prepared.start()
                prepared.stage_resets()
                self.assertIsNot(prepared.restores[0][1], value)
                np.testing.assert_array_equal(value, fixture.state["rng"])
                prepared.invoke()
                np.testing.assert_array_equal(value, [[18, 1]])
                prepared.reset()
                self.assertIs(prepared.state["rng"], value)
                np.testing.assert_array_equal(value, fixture.state["rng"])
                prepared.close()
                self.assertEqual(
                    events[-4:], ["destroy", "release", "synchronize", "close"]
                )
            prepared.close()
            self.assertEqual(events.count("close"), 1)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                prepared.invoke()
