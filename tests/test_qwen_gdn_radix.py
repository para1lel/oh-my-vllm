import ctypes
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from oh_my_vllm.operators.gdn_radix import api
from oh_my_vllm.operators.gdn_radix.metadata import make_layout
from oh_my_vllm.operators.gdn_radix.plan import make_plan
from oh_my_vllm.operators.gdn_radix.workspace import prepare

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "python/oh_my_vllm/operators/gdn_radix/cuda/metadata_layout.cuh"
HOST_BRIDGE = f'''
#include "{HEADER}"
extern "C" int layout(const int *writes, const int *starts, int batch, int capacity,
                       int *metadata, int *offsets) {{
  int count = gdn_radix_metadata::layout_chunks(writes, starts, starts[batch], batch);
  if (count < 0 || count > capacity)
    return -1;
  int base = 0;
  offsets[0] = 0;
  for (int sequence = 0; sequence < batch; ++sequence) {{
    int first = starts[sequence], last = starts[sequence + 1];
    if (!gdn_radix_metadata::build_sequence(
            writes, first, last, metadata, base, capacity))
      return -1;
    base += gdn_radix_metadata::chunk_count(writes, first, last);
    offsets[sequence + 1] = base;
  }}
  return base;
}}
'''


def schedule(counts, checkpoints=(), *, verify=False, positions=None):
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    writes = tuple(range(1, starts[-1] + 1)) if verify else (-1,) * starts[-1]
    return api.Schedule(
        tuple(starts),
        (0,) * len(counts),
        writes,
        tuple(checkpoints),
        tuple(range(1, len(checkpoints) + 1)),
        positions or (0,) * len(counts),
        max(2, starts[-1] + 1),
        verify,
    )


class GdnRadix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="qwen-gdn-radix-")
        library = Path(cls.temporary.name) / "metadata.so"
        subprocess.run(
            [
                "c++",
                "-std=c++17",
                "-O2",
                "-shared",
                "-fPIC",
                "-x",
                "c++",
                "-",
                "-o",
                str(library),
            ],
            input=HOST_BRIDGE,
            text=True,
            check=True,
            capture_output=True,
        )
        cls.native = ctypes.CDLL(str(library))
        pointer = ctypes.POINTER(ctypes.c_int)
        cls.native.layout.argtypes = [
            pointer,
            pointer,
            ctypes.c_int,
            ctypes.c_int,
            pointer,
            pointer,
        ]
        cls.native.layout.restype = ctypes.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def native_layout(self, item, capacity=None):
        layout = make_layout(item)
        capacity = layout.chunks if capacity is None else capacity
        integers = ctypes.c_int
        writes = (integers * len(item.write_slots))(*item.validate())
        starts = (integers * len(item.sequence_rows))(*item.sequence_rows)
        metadata = (integers * (capacity * 4 + 4))(*([-999] * (capacity * 4 + 4)))
        offsets = (integers * len(item.sequence_rows))()
        count = self.native.layout(
            writes, starts, len(item.read_slots), capacity, metadata, offsets
        )
        self.assertEqual(tuple(metadata[capacity * 4 :]), (-999,) * 4)
        if count < 0:
            self.assertEqual(tuple(metadata), (-999,) * (capacity * 4 + 4))
        return (
            count,
            tuple(tuple(metadata[i : i + 4]) for i in range(0, max(count, 0) * 4, 4)),
            tuple(offsets),
        )

    def test_exports_cross_8192_without_early_checkpoint(self):
        item = schedule((8321,), (8191, 8320))
        count, metadata, offsets = self.native_layout(item)
        self.assertEqual(count, 131)
        self.assertEqual(offsets, (0, 131))
        self.assertEqual(metadata[0], (0, 64, -1, 131))
        self.assertEqual(metadata[127], (8128, 8192, 1, 0))
        self.assertEqual(metadata[-1], (8320, 8321, 2, 0))
        self.assertEqual(metadata, make_layout(item).metadata)

    def test_private_chunk_tail_restarts_64_row_math(self):
        item = schedule((150, 0, 3), (6, 72, 149, 152))
        expected = (
            (0, 7, 1, 5),
            (7, 71, -1, 0),
            (71, 73, 2, 0),
            (73, 137, -1, 0),
            (137, 150, 3, 0),
            (150, 153, 4, 1),
        )
        self.assertEqual(make_layout(item).metadata, expected)
        self.assertEqual(self.native_layout(item), (6, expected, (0, 5, 5, 6)))
        self.assertEqual(self.native_layout(item, 5)[0], -1)

    def test_every_verify_row_is_an_independent_endpoint(self):
        item = schedule((5, 0, 1), verify=True)
        self.assertEqual(item.validate(), (1, 2, 3, 4, 5, 6))
        self.assertEqual(self.native_layout(item)[0], 6)
        self.assertEqual(make_plan(item).chunks, 0)
        with self.assertRaises(ValueError):
            make_plan(schedule((6,), verify=True))

    def test_zero_rows_need_no_export_or_native_run(self):
        item = schedule((0, 0), positions=(8192, 8301))
        self.assertEqual(self.native_layout(item), (0, (), (0, 0, 0)))
        plan = make_plan(item)
        module = SimpleNamespace(gdn_radix=lambda *args: self.fail("zero-row GDN ran"))
        device = SimpleNamespace(empty=lambda shape, dtype: (shape, dtype))
        with patch(
            "oh_my_vllm.operators.gdn_radix.workspace.load",
            return_value=(module, "id", "lib"),
        ):
            workspace = prepare(plan, device, ())
        self.assertIs(workspace.run(None, None, None, None), workspace.outputs)
        workspace.close()
        with self.assertRaises(RuntimeError):
            workspace.run(None, None, None, None)

    def test_export_validation_rejects_alias_and_invalid_metadata(self):
        item = schedule((129,), (128,))
        bad = (
            replace(item, checkpoint_rows=(129,)),
            replace(item, checkpoint_rows=(128, 128), checkpoint_slots=(1, 2)),
            replace(item, checkpoint_slots=(0,)),
            replace(item, write_slots=(-2, *item.write_slots[1:])),
            replace(item, positions=(262144,)),
            replace(item, sequence_rows=(0, 130)),
            replace(item, checkpoint_slots=(item.slots,)),
            replace(item, checkpoint_rows=(1, 128), checkpoint_slots=(1, 1)),
        )
        for invalid in bad:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                invalid.validate()

    def test_sparse_exports_charge_all_intermediate_snapshots(self):
        item = schedule((32768,), (8191, 16383, 24575, 32767))
        plan = make_plan(item)
        self.assertEqual(plan.chunks, 512)
        self.assertEqual(plan.storage[18], ((512, 48, 128, 128), "bfloat16"))
        self.assertEqual(plan.chunk_scratch_bytes, 1824530688)
        self.assertGreater(plan.scratch_bytes, plan.chunk_scratch_bytes)
        dense = make_plan(schedule((129,), tuple(range(129))))
        self.assertEqual(dense.chunks, 129)
        self.assertNotEqual(
            dense.reuse_key, make_plan(schedule((129,), (128,))).reuse_key
        )

    def test_run_reuses_storage_and_new_symbol(self):
        plan = make_plan(schedule((129,), (128,)))
        allocations, calls = [], []
        module = SimpleNamespace(gdn_radix=lambda *args: calls.append(args))
        device = SimpleNamespace(
            empty=lambda *args: allocations.append(args) or object()
        )
        weights = api.Weights(*(object() for _ in range(10)))
        with patch(
            "oh_my_vllm.operators.gdn_radix.workspace.load",
            return_value=(module, "id", "lib"),
        ) as load:
            workspace = prepare(plan, device, (weights,))
        load.assert_called_once_with("gdn_radix", ("gdn_radix",))
        allocated = len(allocations)
        inputs = api.Inputs(object(), object())
        batch = api.BatchView(object(), object(), object())
        state = api.StateView(object(), object())
        with workspace.graph_lease():
            workspace.run(weights, inputs, batch, state)
            workspace.run(weights, inputs, batch, state)
        self.assertEqual(len(allocations), allocated)
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[0][-2], workspace.storage.slot_owners)
        self.assertEqual(calls[0][-1], 129)


if __name__ == "__main__":
    unittest.main()
