"""CPU-only validation guard tests for KRN-04, KRN-05, PY-02, MNT-02.

These tests verify wrapper-level input validation without requiring a GPU.
KRN-01 (grouped decode group_size limit) is tested in
test_independent_decode_attention.py.
PY-01 (FA page sharing) is tested in test_batch_plan.py.
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestKRN04FP8Int32Overflow(unittest.TestCase):
    """KRN-04: fp8.quantize and elementwise.silu_mul reject int32-overflow shapes."""

    def test_fp8_quantize_overflow_rejected(self):
        """rows * width >= 2**31 must raise ValueError (no GPU needed)."""
        from oh_my_vllm.kernels import fp8

        # rows * width == 2**31, which would overflow a signed int32 offset
        # Use shape values that pass earlier checks (numel > 0, width % 128 == 0)
        # but fail the overflow check. We mock is_cuda / is_contiguous so no
        # actual GPU tensor is needed.
        t = MagicMock(spec=torch.Tensor)
        t.dtype = torch.bfloat16
        t.ndim = 2
        t.numel.return_value = 2**31  # non-zero, passes the numel==0 guard
        t.shape = (2**24, 128)  # rows * width = 2**24 * 128 = 2**31
        t.is_contiguous.return_value = True
        t.is_cuda = True

        with self.assertRaisesRegex(ValueError, "int32 flat offset"):
            fp8.quantize(t)

    def test_fp8_quantize_silu_overflow_rejected(self):
        """rows * width * 2 >= 2**31 with silu_gate=True must raise ValueError."""
        from oh_my_vllm.kernels import fp8

        t = MagicMock(spec=torch.Tensor)
        t.dtype = torch.bfloat16
        t.ndim = 2
        t.numel.return_value = 2**31
        # width must be divisible by 256 for silu_gate, rows * (width//2) >= 2**31/2
        # rows * width * 2 >= 2**31  =>  rows * width >= 2**30
        t.shape = (2**23, 256)  # rows * width * 2 = 2**23 * 256 * 2 = 2**32 > 2**31
        t.is_contiguous.return_value = True
        t.is_cuda = True

        with self.assertRaisesRegex(ValueError, "int32 flat offset"):
            fp8.quantize(t, silu_gate=True)

    def test_silu_mul_overflow_rejected(self):
        """rows * half_width * 2 >= 2**31 must raise ValueError."""
        from oh_my_vllm.kernels.elementwise import silu_mul

        t = MagicMock(spec=torch.Tensor)
        t.dtype = torch.bfloat16
        t.ndim = 2
        # shape[1] must be even; rows * shape[1] // 2 * 2 >= 2**31
        t.shape = (2**24, 256)  # rows * width * 2 = 2**24 * 128 * 2 = 2**32 > 2**31
        t.is_contiguous.return_value = True
        t.is_cuda = True
        t.numel.return_value = 2**24 * 256

        with self.assertRaisesRegex(ValueError, "int32 flat offset"):
            silu_mul(t)

    @pytest.mark.gpu
    def test_silu_mul_within_range_passes_shape_check(self):
        """Small tensors must not trigger the overflow guard."""
        from oh_my_vllm.kernels.elementwise import silu_mul

        # Use a real CUDA tensor only if GPU available, else skip
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        t = torch.zeros(1, 256, dtype=torch.bfloat16, device="cuda")
        # Should not raise (will call kernel, result shape is correct)
        out = silu_mul(t)
        self.assertEqual(out.shape, (1, 128))


class TestKRN05NormalizeQKValidation(unittest.TestCase):
    """KRN-05: gdn.normalize_qk validates dtype, shape, and strides."""

    def _make_mock(self, dtype=torch.bfloat16, shape=(4, 8, 128), strides=None):
        t = MagicMock(spec=torch.Tensor)
        t.dtype = dtype
        t.shape = torch.Size(shape)
        t.ndim = len(shape)
        if strides is None:
            # contiguous strides
            s = [1]
            for d in reversed(shape[1:]):
                s.append(s[-1] * d)
            strides = list(reversed(s))
        t.stride = MagicMock(
            side_effect=lambda i: strides[i] if i >= 0 else strides[len(shape) + i]
        )
        t.device = torch.device("cuda")
        t.is_cuda = True
        return t

    def test_fp32_input_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock(dtype=torch.float32)
        k = self._make_mock(dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            normalize_qk(q, k)

    def test_fp16_input_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock(dtype=torch.float16)
        k = self._make_mock(dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            normalize_qk(q, k)

    def test_shape_mismatch_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock(shape=(4, 8, 128))
        k = self._make_mock(shape=(4, 8, 128))
        # Change k shape to mismatch
        k.shape = torch.Size((4, 16, 128))
        with self.assertRaisesRegex(ValueError, "shapes must match"):
            normalize_qk(q, k)

    def test_head_dim_64_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock(shape=(4, 8, 64), strides=[512, 64, 1])
        k = self._make_mock(shape=(4, 8, 64), strides=[512, 64, 1])
        with self.assertRaisesRegex(ValueError, "head_dim=128"):
            normalize_qk(q, k)

    def test_non_unit_last_stride_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        # stride(-1) = 2 instead of 1
        q = self._make_mock(shape=(4, 8, 128), strides=[2048, 256, 2])
        k = self._make_mock(shape=(4, 8, 128), strides=[2048, 256, 2])
        with self.assertRaisesRegex(ValueError, "unit stride"):
            normalize_qk(q, k)

    def test_head_stride_not_128_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        # stride(-2) = 256 instead of 128
        q = self._make_mock(shape=(4, 8, 128), strides=[2048, 256, 1])
        k = self._make_mock(shape=(4, 8, 128), strides=[2048, 256, 1])
        with self.assertRaisesRegex(ValueError, "head_stride=128"):
            normalize_qk(q, k)

    def test_two_dimensional_input_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock(shape=(8, 128))
        k = self._make_mock(shape=(8, 128))
        with self.assertRaisesRegex(ValueError, r"\[tokens, heads, 128\]"):
            normalize_qk(q, k)

    def test_empty_input_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock(shape=(0, 8, 128))
        k = self._make_mock(shape=(0, 8, 128))
        with self.assertRaisesRegex(ValueError, "nonempty"):
            normalize_qk(q, k)

    def test_overlapping_token_rows_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock(strides=[512, 128, 1])
        k = self._make_mock(strides=[512, 128, 1])
        with self.assertRaisesRegex(ValueError, "non-overlapping"):
            normalize_qk(q, k)

    def test_cpu_and_cross_device_inputs_rejected(self):
        from oh_my_vllm.kernels.gdn import normalize_qk

        q = self._make_mock()
        k = self._make_mock()
        q.is_cuda = False
        q.device = torch.device("cpu")
        with self.assertRaisesRegex(ValueError, "same CUDA device"):
            normalize_qk(q, k)
        q.is_cuda = True
        q.device = torch.device("cuda:0")
        k.device = torch.device("cuda:1")
        with self.assertRaisesRegex(ValueError, "same CUDA device"):
            normalize_qk(q, k)


class TestPY02FP8ProjectionAlignment(unittest.TestCase):
    """PY-02: Checkpoint.linear rejects FP8 projections not divisible by 128."""

    def _make_checkpoint_with_weights(self, *row_counts):
        """Return a mock Checkpoint whose tensor() and files attributes simulate
        FP8 weights with the given row counts."""
        from oh_my_vllm.models.qwen import Checkpoint

        cp = object.__new__(Checkpoint)
        names = [f"layer_{i}" for i in range(len(row_counts))]
        cp.files = {}
        for name, _rows in zip(names, row_counts, strict=True):
            cp.files[f"{name}.weight"] = "shard.safetensors"
            cp.files[f"{name}.weight_scale_inv"] = "shard.safetensors"

        tensors = {}
        for name, rows in zip(names, row_counts, strict=True):
            tensors[f"{name}.weight"] = torch.zeros(
                rows, 128, dtype=torch.float8_e4m3fn
            )
            tensors[f"{name}.weight_scale_inv"] = torch.zeros(
                rows // 128 if rows % 128 == 0 else 1, 1
            )

        def fake_tensor(key):
            if key in tensors:
                return tensors[key]
            raise KeyError(key)

        cp.tensor = fake_tensor
        return cp, names

    def test_128_aligned_rows_accepted(self):
        cp, names = self._make_checkpoint_with_weights(256, 512)
        # Should not raise
        linear = cp.linear(*names)
        self.assertIsNotNone(linear)

    def test_48_rows_rejected(self):
        """48 rows is the Qwen in_proj_a/b shape that triggers this bug."""
        cp, names = self._make_checkpoint_with_weights(48)
        with self.assertRaisesRegex(ValueError, "128"):
            cp.linear(*names)

    def test_96_rows_rejected(self):
        cp, names = self._make_checkpoint_with_weights(96)
        with self.assertRaisesRegex(ValueError, "128"):
            cp.linear(*names)

    def test_mixed_aligned_and_misaligned_rejected(self):
        cp, names = self._make_checkpoint_with_weights(256, 48)
        with self.assertRaisesRegex(ValueError, "128"):
            cp.linear(*names)


class TestMNT02EmptyTensorGuards(unittest.TestCase):
    """MNT-02: silu_mul and delta_gates return empty tensors for 0-row input."""

    @pytest.mark.gpu
    def test_silu_mul_empty_returns_without_launch(self):
        """0-row silu_mul should return an empty tensor, not launch a kernel."""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")
        from oh_my_vllm.kernels.elementwise import silu_mul

        packed = torch.empty(0, 256, dtype=torch.bfloat16, device="cuda")
        out = silu_mul(packed)
        self.assertEqual(out.shape, (0, 128))
        self.assertEqual(out.dtype, torch.bfloat16)

    def test_silu_mul_empty_cpu_mock(self):
        """Verify empty early-return path via mock (no GPU required)."""
        from oh_my_vllm.kernels.elementwise import silu_mul

        t = MagicMock(spec=torch.Tensor)
        t.dtype = torch.bfloat16
        t.ndim = 2
        t.shape = torch.Size([0, 256])
        t.is_contiguous.return_value = True
        t.is_cuda = True
        t.numel.return_value = 0

        # The function should return early before reaching the kernel call.
        # We patch _silu_mul to verify it's never called.
        with (
            patch("oh_my_vllm.kernels.elementwise._silu_mul") as mock_kernel,
            patch("torch.empty") as mock_empty,
        ):
            mock_empty.return_value = MagicMock()
            silu_mul(t)
            mock_kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
