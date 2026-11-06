import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tests.offload_worker_fixtures import Pool as Pool
from tests.offload_worker_fixtures import Tensor as Tensor
from tests.offload_worker_fixtures import Transfers as Transfers
from tests.offload_worker_fixtures import exchange, operation, worker_fixture


class OffloadWorker(unittest.TestCase):
    def test_request_restores_selected_endpoint_and_preserves_cpu_state(self):
        worker, request, engine, pool = worker_fixture()
        cpu, decoder_state, endpoint, blocks, checkpoints, proposals, embedding = (
            self._offload_snapshot(worker, request)
        )
        self.assertEqual(
            worker.offload.stats()["host_cache_capacity_bytes"], pool.limit
        )
        with patch("tvm_ffi.from_dlpack", side_effect=lambda value: value):
            size = self._check_offload_store(worker, request, engine, embedding)
            for array in (
                *worker.cache.state,
                *worker.cache.convolution,
                worker.cache.hidden,
            ):
                array[:] = -10
            for (array,) in worker.cache.fa:
                array[:] = -10
            worker.cache.private_slots(8, None)
            used = pool.used_bytes
            self.assertEqual(
                exchange(worker, operation(2, "load", fa=(5, 6), mamba=(4,)))[
                    "completed"
                ][0]["status"],
                "deferred",
            )
            self.assertEqual(pool.used_bytes, used)
            self.assertFalse(engine.jobs)
            self.assertIsNone(request.state_slot)
            worker.cache.release(8)
            result = exchange(worker, operation(3, "load", fa=(5, 6), mamba=(4,)))
            self.assertEqual(result["completed"], [])
            self.assertIsNone(request.state_slot)
            self.assertIsNone(request.proposals)
            engine.completed = [3]
            self.assertEqual(exchange(worker)["completed"][0]["status"], "complete")
            self.assertEqual(request.state_slot, 8)
            for value, expected in zip(worker.cache.fa, blocks, strict=True):
                np.testing.assert_array_equal(value[0][5:7], expected)
            for array, expected, checkpoint in zip(
                (*worker.cache.state, *worker.cache.convolution, worker.cache.hidden),
                endpoint,
                checkpoints,
                strict=True,
            ):
                np.testing.assert_array_equal(array[8], expected)
                np.testing.assert_array_equal(array[4], checkpoint)
            np.testing.assert_array_equal(request.proposals, proposals)
            for index, row in enumerate(request.proposal_rows):
                np.testing.assert_array_equal(row, proposals[index : index + 1])
            self.assertEqual(request.media[0].embedding.device.type, "cpu")
            self.assertTrue(torch.equal(request.media[0].embedding, embedding))
            for key, value in cpu.items():
                self.assertIs(getattr(request, key), value, key)
            self.assertEqual(request.decoder.getstate(), decoder_state)
            worker.offload.check_execute([7])
            exchange(worker, operation(4, "drop", fa=(), mamba=()))
            self.assertEqual(pool.used_bytes, 0)
            stats = worker.offload.stats()
            self.assertEqual(stats["host_cache_snapshots"], 0)
            self.assertEqual(stats["host_cache_pending_transfers"], 0)
            self.assertEqual(stats["host_cache_completed_transfers"], 2)
            self.assertEqual(stats["host_cache_load_bytes"], size)
            self.assertEqual(stats["host_cache_used_bytes"], 0)
            self.assertGreater(stats["host_cache_allocated_bytes"], 0)
            worker.offload.close()
            self.assertTrue(pool.closed)

    def test_cancellation_keeps_copy_owners_and_private_slots_until_completion(self):
        for kind in ("store", "load", "drop_before_abort"):
            with (
                self.subTest(kind=kind),
                patch("tvm_ffi.from_dlpack", side_effect=lambda value: value),
            ):
                worker, request, engine, pool = worker_fixture()
                embedding = request.media[0].embedding
                exchange(worker, operation(1, "store"))
                if kind == "drop_before_abort":
                    engine.completed = [1]
                    exchange(worker)
                    exchange(
                        worker,
                        operation(2, "drop", request_id=None, fa=(), mamba=()),
                    )
                    with self.assertRaisesRegex(ValueError, "resident"):
                        worker.offload.check_execute([7])
                    worker.release(7)
                    self.assertNotIn(7, worker.requests)
                    self.assertNotIn(7, worker.offload.suspended)
                    self.assertEqual(pool.used_bytes, 0)
                    worker.offload.close()
                    continue
                if kind == "load":
                    engine.completed = [1]
                    exchange(worker)
                    exchange(worker, operation(2, "load"))
                worker.release(7)
                self.assertNotIn(7, worker.requests)
                self.assertEqual(worker.cache.indices, {7: 0})
                self.assertGreater(pool.used_bytes, 0)
                self.assertFalse(worker.cache.free_indices)
                with self.assertRaisesRegex(ValueError, "owned"):
                    worker.handle(
                        dict(type="register", request_id=7, prompt_token_ids=[1])
                    )
                engine.completed = [1 if kind == "store" else 2]
                result = exchange(worker)
                self.assertEqual(result["completed"][0]["status"], "complete")
                self.assertEqual(worker.cache.indices, {})
                self.assertIsNone(request.state_slot)
                self.assertIsNone(request.proposals)
                self.assertEqual(request.media[0].embedding.device.type, "cpu")
                self.assertTrue(torch.equal(request.media[0].embedding, embedding))
                exchange(worker, operation(3, "drop", fa=(), mamba=()))
                self.assertNotIn(7, worker.offload.suspended)
                self.assertEqual(pool.used_bytes, 0)
                worker.offload.close()

    def test_prefix_capacity_deferral_validation_and_shutdown(self):
        with patch("tvm_ffi.from_dlpack", side_effect=lambda value: value):
            worker, _, engine, pool = worker_fixture()
            empty = exchange(
                worker,
                operation(
                    1001, "store", snapshot_id=1001, request_id=None, fa=(), mamba=()
                ),
            )["completed"][0]
            self.assertEqual(
                empty, {"transfer_id": 1001, "status": "complete", "bytes": 0}
            )
            self.assertEqual(
                exchange(
                    worker,
                    operation(
                        1002, "drop", snapshot_id=1001, request_id=None, fa=(), mamba=()
                    ),
                )["completed"][0]["status"],
                "complete",
            )
            store = operation(1, "store", request_id=None)
            pool.blocked, engine.idle = True, False
            self.assertEqual(
                exchange(worker, store)["completed"][0]["status"], "deferred"
            )
            self.assertEqual(pool.used_bytes, 0)
            engine.idle = True
            self.assertEqual(
                exchange(worker, {**store, "transfer_id": 2})["completed"][0]["status"],
                "capacity",
            )
            pool.blocked = False
            exchange(worker, {**store, "transfer_id": 3})
            with self.assertRaisesRegex(ValueError, "pending|transfer"):
                exchange(worker, operation(4, "drop", request_id=None, fa=(), mamba=()))
            engine.completed = [3]
            exchange(worker)
            for bad in (
                operation(5, "load", request_id=None, fa=(5,)),
                operation(6, "load", request_id=None, fa=(5, 5)),
                operation(7, "load", request_id=None, mamba=(0,)),
            ):
                with self.assertRaises(ValueError):
                    exchange(worker, bad)
            exchange(
                worker, operation(8, "load", request_id=None, fa=(5, 6), mamba=(4,))
            )
            worker.offload.close()
            self.assertTrue(engine.closed)
            self.assertTrue(pool.closed)

    def test_host_budget_caps_cgroup_ancestors_and_explicit_limit(self):
        from oh_my_vllm.worker.offload_budget import (
            cgroup_remaining,
            host_budget,
            process_memory_remaining,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "parent" / "worker"
            leaf.mkdir(parents=True)
            for path, limit, used in (
                (root, "max", 10),
                (leaf.parent, "1000", 600),
                (leaf, "2000", 200),
            ):
                (path / "memory.max").write_text(limit)
                (path / "memory.current").write_text(str(used))
            self.assertEqual(cgroup_remaining(root, leaf), 400)
            read_text = Path.read_text
            files = {
                "/proc/self/cgroup": "0::/parent/worker\n",
                "/proc/self/mountinfo": f"1 0 0:1 / {root} rw - cgroup2 cgroup rw\n",
            }
            with patch.object(Path, "read_text", autospec=True) as read:
                read.side_effect = lambda path: files.get(str(path)) or read_text(path)
                self.assertEqual(process_memory_remaining(), 400)
                for path, limit, used in ((leaf.parent, 300, 200), (leaf, 1000, 200)):
                    (path / "memory.limit_in_bytes").write_text(str(limit))
                    (path / "memory.usage_in_bytes").write_text(str(used))
                files["/proc/self/cgroup"] = "5:memory:/parent/worker\n"
                files["/proc/self/mountinfo"] = (
                    f"1 0 0:1 /parent {leaf.parent} rw - cgroup cgroup rw,memory\n"
                )
                self.assertEqual(process_memory_remaining(), 100)
        with (
            patch(
                "oh_my_vllm.worker.offload_budget.available_memory", return_value=8000
            ),
            patch(
                "oh_my_vllm.worker.offload_budget.process_memory_remaining",
                return_value=4000,
            ),
        ):
            self.assertEqual(host_budget(None, 20000), 1000)
            self.assertEqual(host_budget(500, 20000), 500)
            self.assertEqual(host_budget(20000, 20000), 1000)
            self.assertEqual(host_budget(None, 200), 200)
            self.assertEqual(host_budget(0, 20000), 0)

    def test_media_cpu_torch_embedding_stays_host_and_reencodes(self):
        from oh_my_vllm.models.qwen38.media import MediaProcessor

        class Device:
            def __init__(self):
                self.copies = []
                self.downloads = 0

            def download(self, value, dtype):
                self.downloads += 1
                return value.view(torch.uint16)

            def scratch(self, shape, dtype):
                self.scratches = getattr(self, "scratches", [])
                self.scratches.append((shape, dtype))
                return torch.empty(shape, dtype=torch.bfloat16)

            def copy(self, destination, source):
                self.copies.append((destination, source))

            def synchronize_compute(self):
                pass

        device = Device()
        processor = MediaProcessor.__new__(MediaProcessor)
        processor.device = device
        embedding = torch.tensor(
            np.arange(3 * 5, dtype=np.uint16).reshape(3, 5),
            dtype=torch.uint16,
        )
        item = SimpleNamespace(embedding=embedding, pixels=None)
        processor.offload(SimpleNamespace(media=[item]))
        self.assertIs(item.embedding, embedding)
        self.assertEqual(device.downloads, 0)
        value = processor.encode(item)
        self.assertIs(item.embedding, value)
        self.assertIs(device.copies[0][1], embedding)
        self.assertEqual(value.dtype, torch.bfloat16)

    def test_graph_warmup_and_metadata_use_compute_and_reusable_uploads(self):
        from oh_my_vllm.models.qwen38.model import Workspace
        from oh_my_vllm.models.qwen38.sampling import Sampling
        from oh_my_vllm.models.qwen38.speculation_ops import GraphOperation

        events, upload_events, native_copies, uploads, device = (
            self._graph_upload_device()
        )
        workspace = Workspace.__new__(Workspace)
        workspace.device, workspace.replay = device, None
        workspace.runs = 0
        workspace.warmups = [lambda: events.append("warmup")]
        workspace.ffn = workspace.gdn = workspace.fa = workspace.hidden_operation = None
        workspace.resources, workspace.output = ExitStack(), object()
        workspace.forward = lambda: events.append("target")
        workspace.run(True)
        workspace.run(True)
        workspace.resources.close()
        operation = GraphOperation(device)
        source = np.arange(4, dtype=np.int32)
        destination = np.zeros_like(source)
        operation.copy(destination, source)
        self.assertFalse(native_copies, "NumPy metadata bypassed reusable uploads")
        np.testing.assert_array_equal(destination, source)
        operation.copy(destination, source + 1)
        self.assertEqual(upload_events, ["create", "copy", "copy"])
        gpu_source = torch.empty(4, dtype=torch.int32, device="meta")
        operation.copy(destination, gpu_source)
        self.assertEqual(len(native_copies), 1)
        self.assertIs(native_copies[0][1], gpu_source)
        operation.run = lambda: events.append("draft")
        operation.execute(True)
        operation.execute(True)
        operation.close()
        self.assertEqual(upload_events, ["create", "copy", "copy", "close"])
        sampler = Sampling.__new__(Sampling)
        sampler.device, sampler.uploads = device, uploads()
        sampler.indices, sampler.last_indices = np.zeros(2, np.int32), None
        sampler.value = sampler.residual = np.empty((2, 3), np.float32)
        sampler.execute = lambda *_: None
        sampler.select_from_logits = lambda *_args, **_options: (None, None)
        sampler.select(None, None, [3, 1], [], False)
        np.testing.assert_array_equal(sampler.indices, [3, 1])
        sampler.select(None, None, [3, 1], [], False)
        sampler.select(None, None, [1, 3], [], False)
        np.testing.assert_array_equal(sampler.indices, [1, 3])
        self.assertEqual(upload_events[-3:], ["create", "copy", "copy"])
        self.assertEqual(len(native_copies), 1, "sampling indices bypassed uploads")
        sampler.uploads.close()
        self.assertEqual(
            events,
            [
                "warmup",
                "wait_compute",
                "target",
                "target",
                "draft",
                "wait_compute",
                "draft",
                "draft",
            ],
        )

    def _check_offload_store(self, worker, request, engine, embedding):
        result = exchange(worker, operation(1, "store"))
        self.assertEqual(result["type"], "cache_result")
        self.assertEqual(result["rpc_id"], 99)
        self.assertEqual(result["completed"], [])
        stats = worker.handle(dict(type="stats", rpc_id=98))["stats"]
        self.assertEqual(stats["host_cache_snapshots"], 1)
        self.assertEqual(stats["host_cache_pending_transfers"], 1)
        self.assertEqual(stats["host_cache_completed_transfers"], 0)
        self.assertGreater(stats["host_cache_used_bytes"], 0)
        self.assertGreaterEqual(
            stats["host_cache_allocated_bytes"], stats["host_cache_used_bytes"]
        )
        self.assertEqual(worker.cache.indices, {7: 0})
        self.assertEqual(request.state_slot, 11)
        with self.assertRaisesRegex(ValueError, "resident|transfer|offload"):
            worker.offload.check_execute([7])
        engine.completed = [1]
        completed = exchange(worker)["completed"]
        self.assertEqual(completed[0]["status"], "complete")
        self.assertGreater(completed[0]["bytes"], 0)
        size = completed[0]["bytes"]
        self.assertEqual(worker.offload.stats()["host_cache_store_bytes"], size)
        self.assertEqual(worker.cache.indices, {})
        self.assertIsNone(request.state_slot)
        self.assertIsNone(request.proposals)
        self.assertEqual(request.media[0].embedding.device.type, "cpu")
        self.assertTrue(torch.equal(request.media[0].embedding, embedding))
        return size

    def _offload_snapshot(self, worker, request):
        cpu = {
            key: value
            for key, value in vars(request).items()
            if key not in ("state_slot", "proposals", "proposal_rows", "media")
        }
        decoder_state = request.decoder.getstate()
        endpoint = [
            array[11].copy()
            for array in (
                *worker.cache.state,
                *worker.cache.convolution,
                worker.cache.hidden,
            )
        ]
        blocks = [array[1:3].copy() for (array,) in worker.cache.fa]
        checkpoints = [
            array[3].copy()
            for array in (
                *worker.cache.state,
                *worker.cache.convolution,
                worker.cache.hidden,
            )
        ]
        proposals = request.proposals.copy()
        embedding = request.media[0].embedding.clone()
        return cpu, decoder_state, endpoint, blocks, checkpoints, proposals, embedding

    def _graph_upload_device(self):
        events, upload_events, native_copies = [], [], []

        def uploads():
            def copy(destination, source):
                upload_events.append("copy")
                np.copyto(destination, source)

            upload_events.append("create")
            return SimpleNamespace(
                copy=copy, close=lambda: upload_events.append("close")
            )

        def global_wait():
            self.fail("graph warmup waited for unrelated transfers")

        device = SimpleNamespace(
            synchronize=global_wait,
            synchronize_compute=lambda: events.append("wait_compute"),
            graph=lambda operation: nullcontext(operation),
            uploads=uploads,
            copy=lambda destination, source: native_copies.append(
                (destination, source)
            ),
            module=SimpleNamespace(
                copy_bytes=lambda destination, source: native_copies.append(
                    (destination, source)
                ),
                gather_rows=lambda *_: None,
            ),
        )
        return events, upload_events, native_copies, uploads, device
