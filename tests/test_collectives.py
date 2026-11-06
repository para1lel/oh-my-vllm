import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from tests.collectives_fixtures import CAPTURE_PENDING_CHECK as CAPTURE_PENDING_CHECK
from tests.collectives_fixtures import (
    CollectivesFixture,
    metadata_tensor,
    run_native_capture_check,
)


class Collectives(CollectivesFixture, unittest.TestCase):
    def test_invalid_bootstrap_cannot_load_native(self):
        api = self.module()
        with patch.object(api, "load", side_effect=AssertionError("native load")):
            for world, rank, unique_id in (
                (0, 0, bytes(128)),
                (2**31, 0, bytes(128)),
                (2, -1, bytes(128)),
                (2, 2, bytes(128)),
                (True, 0, bytes(128)),
                (2, False, bytes(128)),
                (2, 0, bytes(127)),
                (2, 0, [0] * 128),
            ):
                with (
                    self.subTest(world=world, rank=rank),
                    self.assertRaises((ValueError, TypeError)),
                ):
                    api.Communicator(world, rank, unique_id)

    def test_invalid_routes_cannot_reach_transport(self):
        api = self.module()
        for kind, peer in (
            ("broadcast", 0),
            ("send", -1),
            ("recv", True),
            ("send", 2**31),
        ):
            with (
                self.subTest(kind=kind, peer=peer),
                self.assertRaises((ValueError, TypeError)),
            ):
                api.Route(kind, peer, object())

    @unittest.skipUnless(
        os.environ.get("OH_MY_VLLM_TEST_COLLECTIVES_NATIVE") == "1",
        "requires native headers and compiler; does not use a GPU",
    )
    def test_capture_pending_keeps_owner_and_allows_host_only_progress(self):
        with TemporaryDirectory() as directory:
            result = run_native_capture_check(directory)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(
        os.environ.get("OH_MY_VLLM_TEST_COLLECTIVES_NATIVE") == "1",
        "requires the optional compiled module; does not use a GPU",
    )
    def test_staging_covers_exact_extents_views_routes_and_overflow(self):
        import tvm_ffi
        from oh_my_vllm.operators.runtime.device import FUNCTIONS as DEVICE_FUNCTIONS

        api = self.module()
        native = api.load("collectives", api.FUNCTIONS)[0]
        device = api.load("device", DEVICE_FUNCTIONS)[0]
        fixtures = []

        tensor, layout = self._staging_functions(fixtures, native, tvm_ffi)

        for code, bits in ((2, 32), (4, 16), (2, 16), (0, 32), (0, 64), (1, 8)):
            width = bits // 8
            for count in (0, 1, 3, 16 // width, 17):
                size = count * width
                padded = (size + 15) // 16 * 16
                for offset in (0, width):
                    with self.subTest(code=code, bits=bits, count=count, offset=offset):
                        source = tensor(count + bool(offset), 0x1000, code, bits)
                        destination = tensor(count + bool(offset), 0x2000, code, bits)
                        if offset:
                            source = device.view(source, tvm_ffi.Shape((count,)), 1)
                            destination = device.view(
                                destination, tvm_ffi.Shape((count,)), 1
                            )
                        expected = padded if size and (offset or size % 16) else 0
                        self.assertEqual(
                            layout("all_reduce", [source, destination]),
                            [expected, expected],
                        )
                        self.assertEqual(
                            layout("routing", [source, destination], (0, 1), (0, 0)),
                            [expected, expected],
                        )
            self.assertEqual(
                layout(
                    "all_gather",
                    [tensor(3, 0x1000, code, bits), tensor(6, 0x2000, code, bits)],
                ),
                [
                    (3 * width + 15) // 16 * 16,
                    0 if 6 * width % 16 == 0 else (6 * width + 15) // 16 * 16,
                ],
            )
            self.assertEqual(
                layout(
                    "reduce_scatter",
                    [tensor(6, 0x1000, code, bits), tensor(3, 0x2000, code, bits)],
                ),
                [
                    0 if 6 * width % 16 == 0 else (6 * width + 15) // 16 * 16,
                    (3 * width + 15) // 16 * 16,
                ],
            )
        for pointer, expected in ((0x1000, 0), (0x1001, 2**63 - 16)):
            self.assertEqual(
                layout("routing", [tensor(2**63 - 16, pointer, 1, 8)], (0,), (1,)),
                [expected],
            )
        for count, code, bits in (
            (2**63 - 15, 1, 8),
            (2**63 - 1, 1, 8),
            ((2**63 - 1) // 2, 4, 16),
        ):
            with (
                self.subTest(count=count, bits=bits),
                self.assertRaisesRegex(RuntimeError, "round"),
            ):
                layout("routing", [tensor(count, 0x1000, code, bits)], (0,), (1,))

    @unittest.skipUnless(
        os.environ.get("OH_MY_VLLM_TEST_COLLECTIVES_NATIVE") == "1",
        "requires the optional compiled module; does not use a GPU",
    )
    def test_native_rejects_invalid_inputs_before_cuda(self):
        import numpy as np
        import tvm_ffi

        api = self.module()
        native = api.load("collectives", api.FUNCTIONS)[0]
        self._check_native_bootstrap_errors(native, tvm_ffi, np)
        fixtures = []

        tensor, check = self._native_check_functions(fixtures, native, tvm_ffi)

        for code, bits in ((2, 32), (4, 16), (2, 16), (0, 32), (0, 64), (1, 8)):
            check(
                "all_reduce",
                [
                    tensor(code=code, bits=bits),
                    tensor(pointer=0x2000, code=code, bits=bits),
                ],
            )
        check("all_gather", [tensor(), tensor(8, 0x2000)])
        check("reduce_scatter", [tensor(8), tensor(pointer=0x2000)])
        check("routing", [tensor(0), tensor(0)], (0, 1), (0, 0))
        for kind, buffers, directions, peers, message in (
            ("all_reduce", [tensor(), tensor(pointer=0x1004)], (), (), "overlap"),
            ("all_reduce", [tensor(), tensor(3, 0x2000)], (), (), "counts"),
            ("all_gather", [tensor(), tensor(4, 0x2000)], (), (), "counts"),
            ("reduce_scatter", [tensor(), tensor(4, 0x2000)], (), (), "counts"),
            (
                "all_reduce",
                [tensor(device=1), tensor(pointer=0x2000)],
                (),
                (),
                "cuda:0",
            ),
            ("all_reduce", [tensor(stride=2), tensor(pointer=0x2000)], (), (), "flat"),
            ("all_reduce", [tensor(bits=64), tensor(pointer=0x2000)], (), (), "dtype"),
            (
                "all_reduce",
                [tensor(), tensor(pointer=0x2000, bits=16)],
                (),
                (),
                "dtypes",
            ),
            (
                "all_reduce",
                [tensor(pointer=0x1001), tensor(pointer=0x2000)],
                (),
                (),
                "bounds",
            ),
            ("routing", [tensor(), tensor(3, 0x2000)], (0, 1), (0, 0), "counts"),
            (
                "routing",
                [tensor(), tensor(pointer=0x2000)],
                (0, 0),
                (1, 1),
                "duplicate",
            ),
        ):
            with (
                self.subTest(kind=kind, message=message),
                self.assertRaisesRegex((RuntimeError, ValueError), message),
            ):
                check(kind, buffers, directions, peers)
        header, runtime = api.versions()
        self.assertGreater(header, 0)
        self.assertEqual(header, runtime)

    def _staging_functions(self, fixtures, native, tvm_ffi):
        def tensor(count, pointer, code, bits):
            value, keepalive = metadata_tensor(
                count, pointer=pointer, code=code, bits=bits
            )
            fixtures.append(keepalive)
            return value

        def layout(kind, buffers, directions=(), peers=()):
            result = native.collective_check(
                2,
                0,
                kind,
                tvm_ffi.convert(buffers),
                tvm_ffi.Shape(directions),
                tvm_ffi.Shape(peers),
            )
            self.assertIsNotNone(result, "native checks must return staging capacities")
            return list(result)

        return tensor, layout

    def _native_check_functions(self, fixtures, native, tvm_ffi):
        def tensor(count=4, pointer=0x1000, **kwargs):
            value, keepalive = metadata_tensor(count, pointer=pointer, **kwargs)
            fixtures.append(keepalive)
            return value

        def check(kind, buffers, directions=(), peers=()):
            native.collective_check(
                2,
                0,
                kind,
                tvm_ffi.convert(buffers),
                tvm_ffi.Shape(directions),
                tvm_ffi.Shape(peers),
            )

        return tensor, check

    def _check_native_bootstrap_errors(self, native, tvm_ffi, np):
        for world, rank, unique_id in (
            (0, 0, bytes(128)),
            (2, 2, bytes(128)),
            (2, 0, bytes(127)),
        ):
            with self.subTest(world=world, rank=rank), self.assertRaises(RuntimeError):
                native.collective_init(world, rank, unique_id)
        cpu = tvm_ffi.from_dlpack(np.zeros(4, dtype=np.float32))
        for kind, tensors, directions, peers in (
            ("all_reduce", [cpu, cpu], [], []),
            ("unknown", [], [], []),
            ("routing", [cpu], [0], [2]),
            ("routing", [], [0], []),
        ):
            with self.subTest(kind=kind), self.assertRaises(RuntimeError):
                native.collective_check(
                    2,
                    0,
                    kind,
                    tvm_ffi.convert(tensors),
                    tvm_ffi.Shape(directions),
                    tvm_ffi.Shape(peers),
                )


if __name__ == "__main__":
    unittest.main()
