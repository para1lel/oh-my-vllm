import unittest

import torch
from oh_my_vllm.models.deepseek_v41 import kernel


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeepSeekQuantizationMemory(unittest.TestCase):
    def test_encoded_quantization_scratch_does_not_grow_with_history(self):
        for rows in (4096, 8192):
            for quantize, block, dtype in (
                (kernel.fp4_act_quant, 16, torch.float8_e4m3fn),
                (kernel.act_quant, 32, torch.float8_e8m0fnu),
            ):
                with self.subTest(rows=rows, quantize=quantize.__name__):
                    values = torch.zeros(
                        (2, rows, 512), device="cuda", dtype=torch.bfloat16
                    )
                    torch.cuda.synchronize()
                    baseline = torch.cuda.memory_allocated()
                    torch.cuda.reset_peak_memory_stats()

                    codes, scales = quantize(values, block, scale_dtype=dtype)
                    torch.cuda.synchronize()

                    output_bytes = codes.numel() + scales.numel()
                    scratch = (
                        torch.cuda.max_memory_allocated() - baseline - output_bytes
                    )
                    self.assertLessEqual(scratch, 1 << 20)
                    del values, codes, scales
