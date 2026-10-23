import copy
import hashlib
import json
import os
import unittest
import weakref
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MappingProxyType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from . import identity, weights
from .test_identity import write_shard
from .test_layout import checkpoint_config


def reference_fixture(root, extra=None, config=None):
    arrays = {
        "embed.weight": (
            "BF16",
            np.array([[0x3F80, 0xC000], [0, 0x3F00]], dtype="<u2"),
        ),
        "layers.1.engram.embed.weight": (
            "F8_E4M3",
            np.array([[0x38] * 32, [0x3C] * 32], dtype=np.uint8),
        ),
        "layers.1.engram.embed.scale": (
            "F8_E8M0",
            np.array([[128], [127]], dtype=np.uint8),
        ),
        "layers.0.ffn.experts.0.w1.weight": (
            "I8",
            np.full((2, 16), -98, dtype=np.int8),
        ),
        "layers.0.ffn.experts.0.w1.scale": (
            "F8_E8M0",
            np.array([[127], [128]], dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.weight": (
            "F8_E4M3",
            np.full((33, 64), 0x38, dtype=np.uint8),
        ),
        "layers.0.attn.wq_a.scale": (
            "F8_E8M0",
            np.array([[127, 128], [129, 130]], dtype=np.uint8),
        ),
    }
    arrays.update(extra or {})
    layout, header, payload = {}, {}, bytearray()
    for name, (dtype, array) in arrays.items():
        layout[name] = {"dtype": dtype, "shape": list(array.shape)}
        start = len(payload)
        payload.extend(array.tobytes())
        header[name] = dict(layout[name], data_offsets=[start, len(payload)])
    (root / "config.json").write_text(json.dumps(config or {}))
    (root / "tokenizer.json").write_text("{}")
    (root / "tokenizer_config.json").write_text("{}")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(payload)},
                "weight_map": dict.fromkeys(layout, "model.safetensors"),
            }
        )
    )
    write_shard(root / "model.safetensors", header, payload)
    return layout, identity.scan_checkpoint(root, layout)


def shared_fixture(root):
    config = checkpoint_config()
    config["text_config"]["engram_num_embeddings"] = [3, 5]
    extra = {
        "layers.1.engram.embed.weight": ("F8_E4M3", np.full((3, 256), 0x38, np.uint8)),
        "layers.1.engram.embed.scale": ("F8_E8M0", np.full((3, 8), 128, np.uint8)),
        "layers.14.engram.embed.weight": ("F8_E4M3", np.full((5, 256), 0xBC, np.uint8)),
        "layers.14.engram.embed.scale": ("F8_E8M0", np.full((5, 8), 127, np.uint8)),
    }
    _, manifest = reference_fixture(root, extra, config)
    return manifest["files"]


class DeferredWeightDevice:
    def __init__(self):
        self.allocations, self.pending = [], []
        self.fail_copy = self.fail_sync = False
        self.copies = 0
        self.runtime = SimpleNamespace(cudaStreamIsCapturing=lambda stream, status: 0)
        self.module = SimpleNamespace(copy_bytes=self.copy_bytes)

    def checked(self, status):
        if status:
            raise RuntimeError("device status")

    def empty(self, shape, dtype):
        value = np.empty(shape, np.uint16 if dtype == "bfloat16" else dtype)
        self.allocations.append(weakref.ref(value))
        return value

    def view(self, value, shape, offset=0):
        return value.reshape(-1)[offset : offset + int(np.prod(shape))].reshape(shape)

    def copy_bytes(self, destination, source):
        self.copies += 1
        self.pending.append(
            (weakref.ref(destination), weakref.ref(source), source.tobytes())
        )
        if self.fail_copy:
            raise RuntimeError("original upload failure")

    def synchronize_compute(self):
        if self.fail_sync:
            raise RuntimeError("completion failure")
        for destination, source, expected in self.pending:
            if source() is None or destination() is None:
                raise AssertionError("upload owner released before completion")
            if source().tobytes() != expected:
                raise AssertionError("host tile reused before completion")
            destination().view(np.uint8).reshape(-1)[:] = np.frombuffer(
                expected, np.uint8
            )
        self.pending.clear()

    @property
    def live_bytes(self):
        return sum(
            reference().nbytes
            for reference in self.allocations
            if reference() is not None
        )


class SharedEngramWeights(unittest.TestCase):
    def test_load_keeps_encoded_ranges_and_seals_one_bounded_store(self):
        from oh_my_vllm.models.deepseek_v41.engram import load_engram

        with TemporaryDirectory() as directory:
            root = Path(directory)
            files = shared_fixture(root)
            with load_engram(root, files, limit=4096, read_bytes=17) as loaded:
                self.assertTrue(loaded.shared.sealed)
                self.assertEqual(loaded.shared.size, 2112)
                self.assertEqual(loaded.shared.mapped_bytes, 4096)
                self.assertEqual(set(loaded.tables), {1, 14})
                for layer, rows, weight, scale in (
                    (1, 3, 0x38, 128),
                    (14, 5, 0xBC, 127),
                ):
                    table = loaded.tables[layer]
                    self.assertEqual(table.rows, rows)
                    self.assertEqual(
                        os.pread(
                            loaded.shared.fileno(), rows * 256, table.table_offset
                        ),
                        bytes([weight]) * (rows * 256),
                    )
                    self.assertEqual(
                        os.pread(loaded.shared.fileno(), rows * 8, table.scale_offset),
                        bytes([scale]) * (rows * 8),
                    )
                with self.assertRaises(OSError):
                    os.pwrite(loaded.shared.fileno(), b"x", 0)
                descriptor = os.dup(loaded.shared.fileno())
            try:
                self.assertEqual(os.fstat(descriptor).st_size, 4096)
                with self.assertRaises(RuntimeError):
                    loaded.shared.fileno()
            finally:
                os.close(descriptor)

    def test_rejects_unverified_changed_and_invalid_checkpoint_bytes(self):
        from oh_my_vllm.models.deepseek_v41.engram import load_engram

        for mode in ("digest", "payload", "truncated", "shape", "index", "link", "cap"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                files = shared_fixture(root)
                shard = root / "model.safetensors"
                if mode == "digest":
                    files[shard.name]["sha256"] = None
                elif mode == "payload":
                    with shard.open("r+b") as source:
                        source.seek(-1, 2)
                        source.write(b"\xff")
                elif mode == "truncated":
                    shard.write_bytes(shard.read_bytes()[:-1])
                elif mode in ("shape", "index"):
                    name = (
                        "config.json"
                        if mode == "shape"
                        else "model.safetensors.index.json"
                    )
                    path = root / name
                    data = json.loads(path.read_text())
                    if mode == "shape":
                        data["text_config"]["engram_num_embeddings"][0] = 4
                    else:
                        data["weight_map"]["layers.1.engram.embed.scale"] = "../outside"
                    path.write_text(json.dumps(data))
                    files[name] = {
                        "size": path.stat().st_size,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                elif mode == "link":
                    shard.rename(root / "renamed")
                    shard.symlink_to("renamed")
                with self.assertRaises((ValueError, OSError)):
                    load_engram(root, files, limit=2112 if mode == "cap" else 4096)

    def test_interrupted_or_changed_load_closes_partial_shared_storage(self):
        from oh_my_vllm.models.deepseek_v41.engram import load_engram
        from oh_my_vllm.operators.engram.shared import SharedWeightFile

        append = SharedWeightFile.append
        for mode in ("interrupt", "change"):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                files = shared_fixture(root)
                owners = []

                def inject(owner, data, owners=owners, mode=mode, root=root):
                    result = append(owner, data)
                    if not owners:
                        owners.append(owner)
                        if mode == "interrupt":
                            raise KeyboardInterrupt
                        path = root / "config.json"
                        path.write_bytes(path.read_bytes() + b" ")
                    return result

                with (
                    patch.object(SharedWeightFile, "append", new=inject),
                    self.assertRaises(
                        KeyboardInterrupt if mode == "interrupt" else ValueError
                    ),
                ):
                    load_engram(root, files, limit=4096, read_bytes=17)
                self.assertEqual(len(owners), 1)
                self.assertIsNone(owners[0].fd)


class EncodedCheckpointWeights(unittest.TestCase):
    def test_rank_assembly_keeps_async_tiles_and_exact_promotions(self):
        from oh_my_vllm.models.deepseek_v41 import weights as runtime
        from oh_my_vllm.models.deepseek_v41.checkpoint import TensorSlice
        from oh_my_vllm.models.deepseek_v41.placement import RankPlan

        self.assertTrue(callable(getattr(runtime, "_load_plan", None)))
        with TemporaryDirectory() as directory:
            root, device = Path(directory), DeferredWeightDevice()
            raw = np.full((35, 32), 0x38, np.uint8)
            raw[::2] = 0xB8
            extra = {
                "head.weight": (
                    "BF16",
                    np.array([[0, 0xC000], [0, 0x3F00], [0, 0x4040]], "<u2"),
                ),
                "norm.weight": ("BF16", np.array([0x3F80, 0x8000, 0, 0x3F00], "<u2")),
                "float": ("F32", np.array([0.0, -0.0, 1.25], "<f4")),
                "layers.0.attn.wo_a.weight": ("F8_E4M3", raw),
                "layers.0.attn.wo_a.scale": (
                    "F8_E8M0",
                    np.array([[127], [128]], np.uint8),
                ),
            }
            _, manifest = reference_fixture(root, extra)
            selections = {
                name: TensorSlice(kind, value.shape, (0,) * value.ndim, value.shape)
                for name, (kind, value) in extra.items()
            }
            selections["head.weight"] = TensorSlice("BF16", (3, 2), (0, 1), (3, 2))
            selections["layers.0.attn.wo_a.weight"] = TensorSlice(
                "F8_E4M3", (35, 32), (31, 0), (35, 32)
            )
            plan = RankPlan(
                MappingProxyType(selections),
                MappingProxyType(
                    {"head.weight": "F32", "layers.0.attn.wo_a.weight": "BF16"}
                ),
                288,
                156,
                "TP4",
                3,
            )
            with (
                patch("tvm_ffi.from_dlpack", new=lambda value: value),
                runtime._load_plan(
                    root, manifest["files"], device, plan, limit=1024, tile_bytes=2048
                ) as loaded,
            ):
                np.testing.assert_array_equal(
                    loaded.tensors["head.weight"], [[-2.0], [0.5], [3.0]]
                )
                np.testing.assert_array_equal(
                    loaded.tensors["norm.weight"], extra["norm.weight"][1]
                )
                self.assertEqual(
                    loaded.tensors["float"].tobytes(), extra["float"][1].tobytes()
                )
                expected = np.repeat(
                    np.array([0x3F80, 0xC000, 0x4000, 0xC000], np.uint16)[:, None],
                    32,
                    axis=1,
                )
                np.testing.assert_array_equal(
                    loaded.tensors["layers.0.attn.wo_a.weight"], expected
                )
                self.assertEqual(
                    set(loaded.tensors), set(extra) - {"layers.0.attn.wo_a.scale"}
                )
                self.assertFalse(loaded.packed)
                self.assertEqual(
                    (loaded.persistent_bytes, loaded.peak_bytes), (288, 1024)
                )
                self.assertEqual(loaded.allocated_bytes, 1024)
                self.assertEqual(len(device.allocations), 2)
                self.assertEqual(
                    loaded.identity["model.safetensors"],
                    manifest["files"]["model.safetensors"]["sha256"],
                )
                self.assertGreaterEqual(device.copies, 5)
            self.assertEqual(device.live_bytes, 0)
            self.assertFalse(device.pending)
            count = len(device.allocations)
            with self.assertRaisesRegex(ValueError, "budget"):
                runtime._load_plan(
                    root, manifest["files"], device, plan, limit=1023, tile_bytes=2048
                )
            self.assertEqual(len(device.allocations), count)
            with self.assertRaises(ValueError):
                runtime.load_rank(
                    root, manifest["files"], device, strategy="TP4", rank=3, limit=288
                )
            self.assertEqual(len(device.allocations), count)

    def test_rank_assembly_charges_packing_overlap_and_retains_failed_transfers(self):
        from oh_my_vllm.models.deepseek_v41 import weights as runtime
        from oh_my_vllm.models.deepseek_v41.checkpoint import TensorSlice
        from oh_my_vllm.models.deepseek_v41.placement import RankPlan
        from oh_my_vllm.operators.projection import workspace

        self.assertTrue(callable(getattr(runtime, "_load_plan", None)))

        def pack(device, weights, *, bits, allocate):
            self.assertEqual(weights.encoded.dtype, np.uint8)
            code = int(weights.encoded.view(np.int8)[0, 0])
            self.assertIn(code, (-98, -37))
            np.testing.assert_array_equal(weights.encoded.view(np.int8), code)
            encoded = allocate((32, 64), "uint8")
            scales = allocate((512,), "uint8")
            encoded[:, :32] = weights.encoded
            encoded[:, 32:] = 0
            scales[:] = 127
            device.synchronize_compute()
            if mode == "finish":
                path = root / "config.json"
                path.write_bytes(path.read_bytes() + b" ")
            return workspace.PackedWeights(encoded, scales, 64, bits, 32)

        for mode in (
            "success",
            "body",
            "overlap",
            "final",
            "copy",
            "completion",
            "sync",
            "finish",
            "nonfinite",
        ):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root, device, retained = Path(directory), DeferredWeightDevice(), []
                extra = {
                    "packed.weight": ("I8", np.full((32, 32), -98, np.int8)),
                    "packed.scale": ("F8_E8M0", np.full((32, 2), 127, np.uint8)),
                }
                if mode == "success":
                    extra.update(
                        {
                            "zpacked.weight": ("I8", np.full((32, 32), -37, np.int8)),
                            "zpacked.scale": (
                                "F8_E8M0",
                                np.full((32, 2), 127, np.uint8),
                            ),
                        }
                    )
                if mode == "nonfinite":
                    extra = {"head.weight": ("BF16", np.array([0x7F80], "<u2"))}
                _, manifest = reference_fixture(root, extra)
                selections = {
                    name: TensorSlice(kind, value.shape, (0,) * value.ndim, value.shape)
                    for name, (kind, value) in extra.items()
                }
                plan = RankPlan(
                    selections,
                    {"head.weight": "F32"} if mode == "nonfinite" else {},
                    4 if mode == "nonfinite" else 2176 if mode == "success" else 1088,
                    2 if mode == "nonfinite" else 2176 if mode == "success" else 1088,
                    "TP4",
                    0,
                )
                device.fail_copy = mode in ("copy", "completion")
                device.fail_sync = mode in ("completion", "sync")
                limit = (
                    3839
                    if mode == "overlap"
                    else 2559
                    if mode == "final"
                    else 6400
                    if mode == "success"
                    else 3840
                )
                with (
                    patch("tvm_ffi.from_dlpack", new=lambda value: value),
                    patch.object(workspace, "prepare_weights", new=pack),
                    patch.object(runtime, "_FAILED_LOADS", retained),
                ):
                    if mode == "success":
                        with runtime._load_plan(
                            root,
                            manifest["files"],
                            device,
                            plan,
                            limit=limit,
                            tile_bytes=2048,
                        ) as loaded:
                            self.assertEqual(
                                set(loaded.packed), {"packed.weight", "zpacked.weight"}
                            )
                            self.assertEqual(
                                (loaded.persistent_bytes, loaded.peak_bytes),
                                (5120, 6400),
                            )
                            self.assertEqual(device.live_bytes, 5120)
                            self.assertEqual(loaded.allocated_bytes, 5120)
                            self.assertEqual(len(device.allocations), 2)
                            for name, code in (
                                ("packed.weight", -98),
                                ("zpacked.weight", -37),
                            ):
                                np.testing.assert_array_equal(
                                    loaded.packed[name].encoded[:, :32].view(np.int8),
                                    code,
                                )
                    elif mode == "body":
                        with (
                            self.assertRaisesRegex(ValueError, "body failure"),
                            runtime._load_plan(
                                root,
                                manifest["files"],
                                device,
                                plan,
                                limit=limit,
                                tile_bytes=2048,
                            ),
                        ):
                            device.fail_sync = True
                            raise ValueError("body failure")
                    else:
                        expected = (
                            "original upload failure"
                            if device.fail_copy
                            else "finite"
                            if mode == "nonfinite"
                            else "completion failure"
                            if mode == "sync"
                            else "changed"
                            if mode == "finish"
                            else "budget"
                        )
                        with self.assertRaisesRegex(
                            (ValueError, RuntimeError), expected
                        ):
                            runtime._load_plan(
                                root,
                                manifest["files"],
                                device,
                                plan,
                                limit=limit,
                                tile_bytes=2048,
                            )
                    if mode in ("completion", "sync", "body"):
                        self.assertEqual(len(retained), 1)
                        self.assertGreater(device.live_bytes, 0)
                        self.assertTrue(
                            all(
                                destination() is not None and source() is not None
                                for destination, source, _ in device.pending
                            )
                        )
                    else:
                        self.assertFalse(retained)
                        self.assertFalse(device.pending)
                        self.assertEqual(device.live_bytes, 0)

    def test_startup_promotions_preserve_exact_bits_and_reject_nonfinite_values(self):
        import importlib.util

        from .fixtures import load_startup_promotion

        module = "oh_my_vllm.models.deepseek_v41.weights"
        self.assertIsNotNone(importlib.util.find_spec(module))
        from oh_my_vllm.models.deepseek_v41.weights import promote_bf16, promote_fp8

        fixture = load_startup_promotion()
        widened = promote_bf16(fixture["bf16_words"])
        self.assertEqual(widened.dtype, np.float32)
        np.testing.assert_array_equal(
            widened.view(np.uint32), fixture["bf16_expected_f32_bits"]
        )
        for word in fixture["bf16_invalid_words"]:
            with self.assertRaises(ValueError):
                promote_bf16(np.array([word], np.uint16))

        codes = fixture["fp8_codes"]
        overflow = fixture["fp8_overflow_mask"].astype(bool)
        raw = np.repeat(np.broadcast_to(codes[:, None], overflow.shape), 32, axis=1)
        raw[np.repeat(overflow, 32, axis=1)] = 0
        scales = np.tile(fixture["scale_codes"], (8, 1))
        expected = fixture["fp8_expected_bf16_bits"].copy()
        expected[overflow] = 0
        np.testing.assert_array_equal(
            promote_fp8(raw, scales), np.repeat(expected, 32, axis=1)
        )
        for row, column in np.argwhere(overflow):
            with self.assertRaises(ValueError):
                promote_fp8(
                    np.array([[codes[row]]], np.uint8),
                    np.array([[fixture["scale_codes"][column]]], np.uint8),
                )
        for code, scale in ((127, 127), (255, 127), (0, 255), (128, 255)):
            with self.assertRaises(ValueError):
                promote_fp8(np.array([[code]], np.uint8), np.array([[scale]], np.uint8))
        for prefix in ("original", "synthetic"):
            raw, scales = fixture[prefix + "_weight"], fixture[prefix + "_scale"]
            expected = fixture[prefix + "_expected_bf16_bits"]
            np.testing.assert_array_equal(promote_fp8(raw, scales), expected)
            np.testing.assert_array_equal(
                promote_fp8(raw[31:], scales, row_offset=31), expected[31:]
            )

    def test_startup_inventory_checks_complete_layout_before_rank_selection(self):
        from oh_my_vllm.models.deepseek_v41 import checkpoint

        self.assertTrue(callable(getattr(checkpoint, "read_layout", None)))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            expected, manifest = reference_fixture(root, config={"name": "fixture"})
            config, actual = checkpoint.read_layout(
                root,
                manifest["files"],
                layout_sha256=manifest["layout_sha256"],
            )
            self.assertEqual(config, {"name": "fixture"})
            self.assertEqual(actual, expected)
            with self.assertRaisesRegex(ValueError, "layout"):
                checkpoint.read_layout(root, manifest["files"], layout_sha256="0" * 64)
            path = root / "config.json"
            path.write_text('{"name": "changed"}')
            with self.assertRaises(ValueError):
                checkpoint.read_layout(
                    root,
                    manifest["files"],
                    layout_sha256=manifest["layout_sha256"],
                )

    def test_column_slices_and_vectors_keep_raw_bits_with_bounded_reads(self):
        from oh_my_vllm.models.deepseek_v41.checkpoint import EncodedReader, TensorSlice

        with TemporaryDirectory() as directory:
            root = Path(directory)
            extra = {
                "columns": ("F8_E4M3", np.arange(32, dtype=np.uint8).reshape(4, 8)),
                "packed": ("I8", np.array([[-128, -1, 0, 127]], np.int8)),
                "bits": ("BF16", np.array([0, 0x8000, 0x7FC1, 0x3F80], "<u2")),
                "float": ("F32", np.array([1.25, -0.0, -2.5], "<f4")),
                "scale": ("F8_E8M0", np.array([0, 127, 255], np.uint8)),
            }
            _, manifest = reference_fixture(root, extra)
            index_path = root / "model.safetensors.index.json"
            index = json.loads(index_path.read_text())
            index["weight_map"]["nonowned"] = "absent.safetensors"
            index_path.write_text(json.dumps(index))
            manifest["files"][index_path.name] = {
                "size": index_path.stat().st_size,
                "sha256": hashlib.sha256(index_path.read_bytes()).hexdigest(),
            }
            selections = {
                "columns": TensorSlice("F8_E4M3", (4, 8), (1, 2), (4, 4)),
                "packed": TensorSlice("I8", (1, 4), (0, 0), (1, 4)),
                "bits": TensorSlice("BF16", (4,), (1,), (4,)),
                "float": TensorSlice("F32", (3,), (0,), (3,)),
                "scale": TensorSlice("F8_E8M0", (3,), (0,), (3,)),
            }
            with EncodedReader(
                root, manifest["files"], MappingProxyType(selections), read_bytes=17
            ) as reader:
                output = np.full((4, 2), 90, np.uint8)
                self.assertEqual(reader.read_into("columns", 0, output), 3)
                np.testing.assert_array_equal(
                    output, [[10, 11], [18, 19], [26, 27], [90, 90]]
                )
                self.assertEqual(reader.read_into("columns", 1, output[:2]), 2)
                np.testing.assert_array_equal(output[:2], [[18, 19], [26, 27]])
                self.assertEqual(reader.read_into("columns", 3, output), 0)
                for name, wanted in (
                    ("packed", np.array([[-128, -1, 0, 127]], np.int8)),
                    ("bits", np.array([0x8000, 0x7FC1, 0x3F80], "<u2")),
                    ("float", np.array([1.25, -0.0, -2.5], "<f4")),
                    ("scale", np.array([0, 127, 255], np.uint8)),
                ):
                    actual = np.empty_like(wanted)
                    self.assertEqual(reader.read_into(name, 0, actual), len(wanted))
                    self.assertEqual(actual.tobytes(), wanted.tobytes())
                verified = reader.finish()
                self.assertEqual(
                    set(verified),
                    {"config.json", index_path.name, "model.safetensors"},
                )
                for name, digest in verified.items():
                    self.assertEqual(digest, manifest["files"][name]["sha256"])
            with self.assertRaises(RuntimeError):
                reader.finish()

    def test_rejects_invalid_selections_destinations_and_tile_limits(self):
        from oh_my_vllm.models.deepseek_v41.checkpoint import EncodedReader, TensorSlice

        for args in (
            ("F16", (2,), (0,), (2,)),
            ("BF16", (True,), (0,), (1,)),
            ("BF16", (2,), (-1,), (1,)),
            ("BF16", (2,), (0,), (3,)),
            ("BF16", (2, 2), (0,), (2,)),
            ("BF16", (2, 2, 2), (0, 0, 0), (2, 2, 2)),
        ):
            with self.subTest(args=args), self.assertRaises(ValueError):
                TensorSlice(*args)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest = reference_fixture(root)
            selection = {"embed.weight": TensorSlice("BF16", (2, 2), (0, 1), (2, 2))}
            for cap in (0, True, (64 << 20) + 1):
                with self.assertRaises(ValueError):
                    EncodedReader(root, manifest["files"], selection, read_bytes=cap)
            with EncodedReader(
                root, manifest["files"], selection, read_bytes=3
            ) as reader:
                readonly = np.zeros((2, 1), "<u2")
                readonly.flags.writeable = False
                for destination in (
                    np.empty((2, 1), np.uint8),
                    np.empty((2, 3), "<u2"),
                    np.empty((2, 2), "<u2")[:, :1],
                    readonly,
                    np.empty((2,), "<u2"),
                ):
                    with self.assertRaises(ValueError):
                        reader.read_into("embed.weight", 0, destination)
                for row in (-1, 3, True):
                    with self.assertRaises(ValueError):
                        reader.read_into("embed.weight", row, np.empty((1, 1), "<u2"))
                with self.assertRaises(ValueError):
                    reader.read_into("nonowned", 0, np.empty((1, 1), "<u2"))
                raw = np.empty((2, 1), "<u2")
                self.assertEqual(reader.read_into("embed.weight", 0, raw), 2)
                np.testing.assert_array_equal(raw, [[0xC000], [0x3F00]])
            for kind, shape in (("BF16", (4, 2)), ("F32", (2, 2))):
                wrong = {"embed.weight": TensorSlice(kind, shape, (0, 0), (2, 2))}
                with (
                    EncodedReader(root, manifest["files"], wrong) as reader,
                    self.assertRaises(ValueError),
                ):
                    reader.read_into(
                        "embed.weight",
                        0,
                        np.empty((2, 2), "<f4" if kind == "F32" else "<u2"),
                    )

    def test_verifies_unselected_payload_before_copy_and_rechecks_controls(self):
        from oh_my_vllm.models.deepseek_v41.checkpoint import EncodedReader, TensorSlice

        for mode in (
            "digest",
            "config.json",
            "model.safetensors.index.json",
            "model.safetensors",
        ):
            with self.subTest(mode=mode), TemporaryDirectory() as directory:
                root = Path(directory)
                _, manifest = reference_fixture(root)
                selection = {
                    "embed.weight": TensorSlice("BF16", (2, 2), (0, 0), (2, 2))
                }
                if mode == "digest":
                    with (root / "model.safetensors").open("r+b") as stream:
                        stream.seek(-1, 2)
                        stream.write(b"\xff")
                with EncodedReader(
                    root, manifest["files"], selection, read_bytes=17
                ) as reader:
                    output = np.full((2, 2), 0x5A5A, "<u2")
                    if mode == "digest":
                        with self.assertRaisesRegex(ValueError, "SHA256"):
                            reader.read_into("embed.weight", 0, output)
                        self.assertTrue(np.all(output == 0x5A5A))
                        with self.assertRaisesRegex(ValueError, "verification"):
                            reader.finish()
                    else:
                        reader.read_into("embed.weight", 0, output)
                        path = root / mode
                        path.write_bytes(path.read_bytes() + b" ")
                        with self.assertRaisesRegex(ValueError, "changed"):
                            reader.read_into("embed.weight", 0, output)
                        with self.assertRaisesRegex(ValueError, "changed"):
                            reader.finish()


class ReferenceWeights(unittest.TestCase):
    def test_source_tiles_keep_column_shards_fp4_bytes_and_engram_tail(self):
        from . import source_weights

        with TemporaryDirectory() as directory:
            root, config = Path(directory), checkpoint_config()
            column = np.arange(512, dtype=np.uint16).astype(np.uint8).reshape(4, 128)
            packed = np.arange(-16, 16, dtype=np.int8).reshape(2, 16)
            extra = {
                "embed.weight": ("BF16", np.zeros((8, 2), dtype="<u2")),
                "layers.0.attn.wo_b.weight": ("F8_E4M3", column),
                "layers.0.attn.wo_b.scale": ("F8_E8M0", np.full((1, 4), 127, np.uint8)),
                "layers.0.ffn.experts.288.w1.weight": ("I8", packed),
                "layers.0.ffn.experts.288.w1.scale": (
                    "F8_E8M0",
                    np.full((2, 1), 127, np.uint8),
                ),
                "layers.1.engram.embed.weight": (
                    "F8_E4M3",
                    np.full((7, 32), 0x38, np.uint8),
                ),
                "layers.1.engram.embed.scale": (
                    "F8_E8M0",
                    np.full((7, 1), 129, np.uint8),
                ),
            }
            layout, manifest = reference_fixture(root, extra, config)
            with patch.object(source_weights, "tensor_layout", return_value=layout):
                reader = source_weights.SourceWeights(
                    root, manifest, config, 3, read_bytes=64
                )
            expected = {
                "layers.0.attn.wo_b.weight": column[:, 96:],
                "layers.0.ffn.experts.288.w1.weight": packed,
                "layers.1.engram.embed.weight": np.array(
                    [[0x38] * 32, [0] * 32], np.uint8
                ),
                "layers.1.engram.embed.scale": np.array([[129], [127]], np.uint8),
            }
            self.assertNotIn("layers.0.ffn.experts.0.w1.weight", reader.parameters)
            for name, wanted in expected.items():
                tiles = list(reader.tiles(name))
                self.assertTrue(all(tile.nbytes <= 64 for _, tile in tiles))
                np.testing.assert_array_equal(
                    np.concatenate([tile for _, tile in tiles]), wanted
                )

    def test_source_wo_a_uses_global_scale_rows_and_validates_destinations(self):
        from . import source_weights

        with TemporaryDirectory() as directory:
            root, config = Path(directory), checkpoint_config()
            extra = {
                "embed.weight": ("BF16", np.zeros((8, 2), dtype="<u2")),
                "layers.0.attn.wo_a.weight": (
                    "F8_E4M3",
                    np.full((128, 64), 0x38, np.uint8),
                ),
                "layers.0.attn.wo_a.scale": (
                    "F8_E8M0",
                    np.arange(127, 135, dtype=np.uint8).reshape(4, 2),
                ),
            }
            layout, manifest = reference_fixture(root, extra, config)
            with patch.object(source_weights, "tensor_layout", return_value=layout):
                reader = source_weights.SourceWeights(
                    root, manifest, config, 1, read_bytes=64
                )
            name = "layers.0.attn.wo_a.weight"
            tiles = list(reader.tiles(name))
            self.assertNotIn(name.replace("weight", "scale"), reader.parameters)
            np.testing.assert_array_equal(
                np.concatenate([tile for _, tile in tiles]),
                np.array([[0x4080] * 32 + [0x4100] * 32] * 32, dtype="<u2"),
            )
            self.assertEqual([row for row, _ in tiles], list(range(32)))
            with self.assertRaisesRegex(ValueError, "parameters"):
                reader.load_into(SimpleNamespace(named_parameters=lambda: []))
            destinations = {
                key: SimpleNamespace(
                    shape=spec["shape"],
                    dtype="torch." + source_weights.TORCH_DTYPES[spec["dtype"]],
                )
                for key, spec in reader.parameters.items()
            }
            model = SimpleNamespace(named_parameters=lambda: destinations.items())
            for field, invalid in (("shape", (1,)), ("dtype", "torch.float64")):
                original = getattr(destinations[name], field)
                setattr(destinations[name], field, invalid)
                with self.assertRaisesRegex(ValueError, "shape/dtype"):
                    reader.load_into(model)
                setattr(destinations[name], field, original)
            with (root / "model.safetensors").open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "changed"):
                next(reader.tiles(name))

    def test_packed_and_dense_scales_survive_row_selection(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            layout, manifest = reference_fixture(root)
            reader = weights.Weights(root, manifest, layout)
            # Packed 0x9e stores -4 in the low nibble and -0.5 in the high nibble.
            expert = "layers.0.ffn.experts.0.w1.weight"
            expected = np.array([[-4.0, -0.5] * 16, [-8.0, -1.0] * 16])
            np.testing.assert_array_equal(reader.tensor(expert), expected)
            np.testing.assert_array_equal(
                reader.tensor(expert, rows=[1, 0, 1]), expected[[1, 0, 1]]
            )
            dense = "layers.0.attn.wq_a.weight"
            expected = np.array(
                [[1.0] * 32 + [2.0] * 32] * 32 + [[4.0] * 32 + [8.0] * 32]
            )
            np.testing.assert_array_equal(reader.tensor(dense), expected)
            np.testing.assert_array_equal(
                reader.tensor(dense, rows=[32, 0, 31]), expected[[32, 0, 31]]
            )

    def test_row_lookup_decodes_only_requested_engram_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            layout, manifest = reference_fixture(root)
            reader = weights.Weights(root, manifest, layout)
            np.testing.assert_array_equal(
                reader.tensor("embed.weight"), [[1.0, -2.0], [0.0, 0.5]]
            )
            values = reader.tensor("layers.1.engram.embed.weight", rows=[1, 0, 1])
            self.assertEqual(values.dtype, np.float64)
            np.testing.assert_array_equal(values, [[1.5] * 32, [2.0] * 32, [1.5] * 32])
            values[0, 0] = 999
            self.assertEqual(
                reader.tensor("layers.1.engram.embed.weight", rows=[1])[0, 0], 1.5
            )
            with self.assertRaisesRegex(ValueError, "rows"):
                reader.tensor("layers.1.engram.embed.weight")
            for invalid in ([-1], [2], [False], [1.0], 1):
                with self.subTest(rows=invalid), self.assertRaises(ValueError):
                    reader.tensor("layers.1.engram.embed.weight", rows=invalid)

    def test_rejects_unverified_or_changed_weight_identity(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            layout, manifest = reference_fixture(root)
            metadata = copy.deepcopy(manifest)
            metadata["verification"] = "metadata-only"
            with self.assertRaisesRegex(ValueError, "full"):
                weights.Weights(root, metadata, layout)
            for filename in ("config.json", "model.safetensors"):
                partial = copy.deepcopy(manifest)
                del partial["files"][filename]
                with self.subTest(filename=filename), self.assertRaises(ValueError):
                    weights.Weights(root, partial, layout)
            reader = weights.Weights(root, manifest, layout)
            reader.tensor("embed.weight")
            with (root / "model.safetensors").open("r+b") as stream:
                stream.seek(-1, 2)
                stream.write(b"\x01")
            with self.assertRaisesRegex(ValueError, "changed"):
                reader.tensor("embed.weight")
            fresh = weights.Weights(root, manifest, layout)
            with self.assertRaisesRegex(ValueError, "identity|digest"):
                fresh.tensor("embed.weight")
