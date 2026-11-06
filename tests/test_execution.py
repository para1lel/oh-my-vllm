import ctypes
import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from oh_my_vllm.models.qwen38 import speculation_ops

from tests.execution_capacity import ExecutionCapacity as ExecutionCapacity
from tests.execution_policy import ExecutionPolicy as ExecutionPolicy
from tests.execution_range_uploads import ExecutionRangeUploads as ExecutionRangeUploads
from tests.execution_uploads import ExecutionUploads as ExecutionUploads


class Execution(unittest.TestCase):
    def test_transfer_keeps_torch_owners_until_completion(self):
        from oh_my_vllm.operators.runtime.transfers import TransferEngine

        source = torch.arange(4, dtype=torch.float32)
        destination = torch.zeros(4, dtype=torch.float32)
        calls = []
        device = SimpleNamespace(
            runtime=SimpleNamespace(
                cudaEventCreateWithFlags=lambda *_: 0,
                cudaEventRecord=lambda *_: 0,
                cudaStreamWaitEvent=lambda *_: 0,
                cudaEventQuery=lambda *_: 0,
                cudaStreamQuery=lambda *_: 0,
                cudaStreamSynchronize=lambda *_: 0,
            ),
            checked=lambda status: self.assertEqual(status, 0),
            transfer_engines=set(),
            destroy_handles=lambda handles: None,
        )
        engine = TransferEngine.__new__(TransferEngine)
        engine.device = device
        engine.streams = {"h2d": ctypes.c_void_p(11)}
        engine.pending, engine.events, engine.closed = {}, [], False
        with (
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch_adapter.current_stream",
                return_value=7,
            ),
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch.ops.oh_my_vllm.native_copy_batch",
                side_effect=lambda inputs, outputs, stream: calls.append(
                    (inputs, outputs, stream)
                ),
            ),
        ):
            engine.submit(1, (destination,), (source,), "h2d")
            source_ref, destination_ref = weakref.ref(source), weakref.ref(destination)
            del source, destination
            gc.collect()
            self.assertIsNotNone(source_ref())
            self.assertIsNotNone(destination_ref())
            self.assertEqual(calls[0][2], 11)
            engine.close()
            calls.clear()
        gc.collect()
        self.assertIsNone(source_ref())
        self.assertIsNone(destination_ref())

    def test_metadata_uses_torch_host_arenas_and_upload_sources(self):
        from oh_my_vllm.worker.metadata import Metadata

        copied = []

        class Uploads:
            def __init__(self):
                self.prepared = []

            def prepare(self, destination, source):
                snapshot = torch.zeros_like(source)
                self.prepared.append((destination, snapshot))
                return snapshot

            def update(self, writer, *args):
                writer(*args)
                for destination, source in self.prepared:
                    copied.append((destination, source))
                    destination.copy_(source)

            def close(self):
                pass

        device = SimpleNamespace(
            uploads=lambda: Uploads(),
            scratch=lambda shape, dtype: torch.zeros(
                shape, dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype
            ),
            empty=lambda shape, dtype: torch.empty(shape, dtype=getattr(torch, dtype)),
            view=lambda value, shape, offset=0: value.reshape(-1)[
                offset : offset + int(torch.tensor(shape).prod())
            ].reshape(shape),
        )
        values = (
            np.array([1, 2], dtype=np.int32),
            torch.tensor([3.0], requires_grad=True),
            torch.tensor([4], dtype=torch.int64),
            torch.empty((0, 3), dtype=torch.int32),
            torch.tensor(5, dtype=torch.uint8),
            torch.tensor([1.25, 2.5], dtype=torch.bfloat16),
            torch.tensor([1 + 2j], dtype=torch.complex64).conj(),
        )
        for single_upload, uploads in ((False, 6), (True, 1)):
            with self.subTest(single_upload=single_upload):
                self._check_metadata_updates(
                    Metadata, copied, device, single_upload, uploads, values
                )
        self._check_acceptance_metadata(copied, device)
        self.assertTrue(all(isinstance(source, torch.Tensor) for _, source in copied))

    def test_qwen_metadata_upload_selection_at_all_three_call_sites(self):
        from oh_my_vllm.models.qwen38 import model, speculation

        device, weights = self._metadata_device_and_weights()
        for count in (1, 2, 4):
            for verify, graph in ((False, True), (True, True), (True, False)):
                with self.subTest(batch=count, verify=verify, graph=graph):
                    batch = self._metadata_batch(count, verify)
                    cache = SimpleNamespace(parallel=count, hidden=None, fa=(None,))
                    operation = Mock(outputs=(torch.zeros(count, 5120),) * 2)
                    with (
                        patch.object(model, "prepare_fa", return_value=operation),
                        patch.object(model, "prepare_gdn", return_value=operation),
                        patch.object(model, "prepare_ffn", return_value=operation),
                        patch.object(
                            speculation_ops, "prepare_context", return_value=operation
                        ),
                        patch.object(
                            speculation_ops, "prepare_draft", return_value=operation
                        ),
                        patch.object(speculation_ops, "Sampling"),
                        patch.object(speculation.Context, "execute"),
                    ):
                        workspace = model.Workspace(
                            device, weights, cache, batch, (None,) * 4
                        )
                        owner = speculation.Speculation(
                            device,
                            weights,
                            cache,
                            SimpleNamespace(rows=lambda _: (None, None)),
                            Mock(),
                        )
                        draft = speculation.Draft(
                            device, weights, count, True, table_width=1
                        )
                        try:
                            owner.context(batch, torch.zeros(count, 5120), graph, {})
                            context = (
                                next(iter(owner.context_decode.values()))
                                if graph and verify
                                else owner.context_prefill
                            )
                            for metadata, enabled in (
                                (workspace.metadata_inputs, verify and count >= 2),
                                (
                                    context.metadata_inputs,
                                    not (graph and verify) or count >= 2,
                                ),
                                (draft.metadata_inputs, count >= 2),
                            ):
                                self.assertIs(metadata.single_upload, enabled)
                                self.assertEqual(
                                    set(metadata.device_arenas),
                                    {torch.uint8}
                                    if enabled
                                    else {torch.int32, torch.int64},
                                )
                        finally:
                            workspace.close()
                            owner.close()
                            draft.close()

    def _check_acceptance_metadata(self, copied, device):
        for count in (4, 2, 1):
            copied.clear()
            values = (
                np.arange(count * 5, dtype=np.int32),
                np.arange(count * 5, dtype=np.int64).reshape(count, 5),
                np.arange(count * 5, dtype=np.int32).reshape(count, 5) + 17,
                np.full((count, 2), 2**64 - 1, dtype=np.uint64),
            )
            from oh_my_vllm.models.qwen38.speculation import Acceptance

            with patch("oh_my_vllm.models.qwen38.speculation_ops.prepare_accept"):
                acceptance = Acceptance(device, count, values)
            metadata = acceptance.metadata_inputs
            self.assertEqual(len(copied), 1)
            for actual, expected in zip(
                (
                    acceptance.gather,
                    acceptance.drafts,
                    acceptance.slots,
                    acceptance.rng,
                ),
                values,
                strict=True,
            ):
                np.testing.assert_array_equal(actual.numpy(), expected)
            for actual, expected in zip(metadata.arrays, values, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected)
            bindings = tuple(value.data_ptr() for value in metadata.arrays)
            for value in values:
                value.fill(7)
            for actual in metadata.arrays:
                self.assertFalse(torch.all(actual == 7))
            metadata.update(values)
            self.assertEqual(len(copied), 2)
            self.assertEqual(
                bindings, tuple(value.data_ptr() for value in metadata.arrays)
            )
            for actual, expected in zip(metadata.arrays, values, strict=True):
                np.testing.assert_array_equal(actual.numpy(), expected)
            metadata.close()

    def _check_metadata_updates(
        self, Metadata, copied, device, single_upload, uploads, values
    ):
        copied.clear()
        metadata = Metadata(device, values, single_upload=single_upload)
        self.assertIs(metadata.single_upload, single_upload)
        self.assertEqual(len(copied), uploads)
        self.assertTrue(
            all(
                isinstance(value, torch.Tensor)
                for value in metadata.host_arenas.values()
            )
        )
        for actual, expected in zip(metadata.arrays, values, strict=True):
            torch.testing.assert_close(actual, torch.as_tensor(expected))
            self.assertEqual(actual.storage_offset() * actual.element_size() % 16, 0)
        if single_upload:
            self.assertEqual(set(metadata.device_arenas), {torch.uint8})
            self.assertEqual(metadata.device_arenas[torch.uint8].numel(), 96)
        updated = tuple(torch.as_tensor(value) + 7 for value in values)
        metadata.update(updated)
        for actual, expected in zip(metadata.arrays, updated, strict=True):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(len(copied), 2 * uploads)
        with self.assertRaisesRegex(ValueError, "shape or dtype changed"):
            metadata.update((torch.zeros(3, dtype=torch.int32), *values[1:]))
        self.assertEqual(len(copied), 2 * uploads)
        alternate = tuple(
            value
            if value.dtype == torch.bfloat16 or value.requires_grad
            else value.numpy()
            for value in updated
        )
        metadata.update(alternate)
        for actual, expected in zip(metadata.arrays, updated, strict=True):
            torch.testing.assert_close(actual, expected)
        self.assertEqual(len(copied), 3 * uploads)
        with self.assertRaisesRegex(ValueError, "contiguous CPU arrays"):
            metadata.update((np.arange(4, dtype=np.int32)[::2], *alternate[1:]))
        with self.assertRaisesRegex(ValueError, "shape or dtype changed"):
            metadata.update((np.zeros(2, dtype=np.int64), *alternate[1:]))
        self.assertEqual(len(copied), 3 * uploads)
        for actual, expected in zip(metadata.arrays, updated, strict=True):
            torch.testing.assert_close(actual, expected)
        before = {
            dtype: value.detach().clone()
            for dtype, value in metadata.host_arenas.items()
        }
        with self.assertRaisesRegex(ValueError, "shape or dtype changed"):
            metadata.update(
                (
                    np.array([91, 92], np.int32),
                    *alternate[1:-1],
                    np.zeros(2, np.complex64),
                )
            )
        for dtype, value in metadata.host_arenas.items():
            torch.testing.assert_close(value, before[dtype])
        self.assertEqual(len(copied), 3 * uploads)
        conjugated = torch.tensor([3 + 4j], dtype=torch.complex64).conj()
        metadata.update((*alternate[:-1], conjugated))
        torch.testing.assert_close(metadata.arrays[-1], conjugated)
        self.assertEqual(len(copied), 4 * uploads)
        metadata.close()

    def _metadata_batch(self, count, verify):
        batch = SimpleNamespace(
            rows=count,
            requests=[
                SimpleNamespace(
                    state_position=2,
                    mtp_position=0,
                    history=[0, 1],
                    position=lambda value: (value,) * 3,
                )
                for _ in range(count)
            ],
            scheduled=[
                dict(
                    request_id=index,
                    num_computed_tokens=0,
                    fa_block_table=[0],
                )
                for index in range(count)
            ],
            schedule=SimpleNamespace(
                verify=verify,
                sequence_rows=tuple(range(count + 1)),
                read_slots=(0,) * count,
            ),
            metadata=(
                np.zeros((3, count), np.int64),
                np.zeros(count, np.int32),
                np.zeros((count, 1), np.int32),
                np.arange(count + 1, dtype=np.int32),
                np.ones(count, np.int32),
            ),
            tokens=np.ones(count, np.int64),
            writes=list(range(count)),
        )
        return batch

    def _metadata_device_and_weights(self):
        def scratch(shape, dtype):
            return torch.zeros(
                shape, dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype
            )

        device = SimpleNamespace(
            uploads=lambda: SimpleNamespace(
                copy=lambda *_: None,
                prepare=lambda destination, source: source,
                update=lambda writer, *args: writer(*args),
                close=lambda: None,
            ),
            scratch=scratch,
            empty=scratch,
            zeros=scratch,
            zero=lambda value: value.zero_(),
            upload=lambda value: value.clone(),
            view=lambda value, shape, offset=0: value.reshape(-1)[
                offset : offset + int(np.prod(shape))
            ].reshape(shape),
        )
        weights = SimpleNamespace(
            layers=(),
            mtp_input=None,
            mtp_layer=SimpleNamespace(attention=None, feed_forward=None),
            mtp_sample=None,
            attention_variant=lambda *_: None,
        )
        return device, weights


if __name__ == "__main__":
    unittest.main()
