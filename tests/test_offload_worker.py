import unittest
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace

import numpy as np
import torch


class SharedWorker(unittest.TestCase):
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
