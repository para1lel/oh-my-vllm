import ctypes
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch


class PinnedTransfers(unittest.TestCase):
    def test_incomplete_cuda_event_is_not_a_transfer_failure(self):
        from oh_my_vllm.operators.runtime.transfers import TransferEngine

        def checked(status):
            if status:
                raise RuntimeError(f"CUDA failure {status}")

        engine = TransferEngine.__new__(TransferEngine)
        engine.device = SimpleNamespace(checked=checked)
        self.assertFalse(engine.ready(600))
        self.assertTrue(engine.ready(0))
        with self.assertRaisesRegex(RuntimeError, "700"):
            engine.ready(700)

    def test_draft_d2h_retains_pending_owners_and_reuses_events(self):
        from oh_my_vllm.operators.runtime.transfers import TransferEngine

        device, events = self._device()
        source, destination = torch.tensor([9]), torch.tensor([0])
        with (
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch_adapter.current_stream",
                return_value=19,
            ),
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch.ops.oh_my_vllm.native_copy_batch"
            ) as copy,
        ):
            engine = TransferEngine(device)
            engine.submit(7, (destination,), (source,), "d2h")
            self.assertIs(engine.pending[7].sources[0], source)
            self.assertIs(engine.pending[7].destinations[0], destination)
            self.assertEqual(copy.call_args.args[2], engine.streams["d2h"].value)
            device.runtime.cudaEventQuery.side_effect = [600, 0]
            self.assertEqual(engine.poll(), [])
            self.assertIn(7, engine.pending)
            with self.assertRaises(ValueError):
                engine.submit(7, (destination,), (source,), "d2h")
            self.assertEqual(engine.poll(), [7])
            allocated = events[0]
            engine.submit(8, (destination,), (source,), "d2h")
            self.assertEqual(events[0], allocated)
            engine.close()
        self.assertFalse(engine.pending)
        self.assertFalse(device.transfer_engines)
        self.assertEqual(device.runtime.cudaStreamSynchronize.call_count, 2)
        self.assertEqual(len(device.destroy_handles.call_args.args[0]), 4)

    def test_failed_batch_joins_submitted_copies_before_owner_release(self):
        from oh_my_vllm.operators.runtime.transfers import TransferEngine

        device, _ = self._device()
        calls = []
        device.runtime.cudaStreamSynchronize.side_effect = lambda stream: (
            calls.append("joined") or 0
        )
        with (
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch_adapter.current_stream",
                return_value=19,
            ),
            patch(
                "oh_my_vllm.operators.runtime.transfers.torch.ops.oh_my_vllm.native_copy_batch",
                side_effect=RuntimeError("copy failed"),
            ),
        ):
            engine = TransferEngine(device)
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                engine.submit(7, (torch.zeros(1),), (torch.ones(1),), "d2h")
            self.assertEqual(calls, ["joined"])
            self.assertEqual(engine.pending, {})
            self.assertEqual(len(engine.events), 2)
            engine.close()

    def test_transfer_rejects_invalid_direction_and_tensor_lists(self):
        from oh_my_vllm.operators.runtime.transfers import TransferEngine

        device, _ = self._device()
        engine = TransferEngine(device)
        value = torch.zeros(1)
        for destinations, sources, direction in (
            ((), (), "d2h"),
            ((value,), (), "d2h"),
            ((value,), (value,), "invalid"),
        ):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                engine.submit(7, destinations, sources, direction)
        self.assertEqual(engine.pending, {})
        engine.close()
        with self.assertRaises(ValueError):
            engine.submit(7, (value,), (value,), "d2h")

    def _device(self):
        handles = [0]

        def allocate(pointer, flags):
            handles[0] += 1
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p))[0] = handles[0]
            return 0

        def checked(status):
            if status:
                raise RuntimeError(f"CUDA failure {status}")

        runtime = SimpleNamespace(
            cudaStreamCreateWithFlags=allocate,
            cudaEventCreateWithFlags=allocate,
            cudaEventRecord=Mock(return_value=0),
            cudaStreamWaitEvent=Mock(return_value=0),
            cudaEventQuery=Mock(return_value=0),
            cudaStreamQuery=Mock(return_value=0),
            cudaStreamSynchronize=Mock(return_value=0),
        )
        return SimpleNamespace(
            runtime=runtime,
            checked=checked,
            transfer_engines=set(),
            destroy_handles=Mock(),
        ), handles


if __name__ == "__main__":
    unittest.main()
