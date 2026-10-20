import unittest
from contextlib import contextmanager

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
