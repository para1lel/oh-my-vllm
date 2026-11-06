import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, kernel_native

from tests.deepseek_operators_fixtures import rms_rounding_case


class DeepSeekHyperconnectionContracts(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_rms_keeps_source_reduction_and_mean_rounding(self):
        for rows in (1, 2, 33, 125, 126, 385):
            for epsilon in (1e-20, 1e-6):
                with self.subTest(rows=rows, epsilon=epsilon):
                    values, weight, expected = rms_rounding_case(rows)
                    if epsilon == 1e-6:
                        expected[:, 4864] = 0.09033203125
                    actual = kernel.rms(values, weight, epsilon)
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_rms_graph_replays_on_the_caller_stream(self):
        module = kernel_native._transform_module()
        for rows in (1, 33):
            with self.subTest(rows=rows):
                values, weight, expected = rms_rounding_case(rows)
                output = torch.empty_like(values)
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        module.rms(values, weight, output, 1e-20)
                    for factor in (1, 2):
                        with torch.cuda.stream(stream):
                            weight.fill_(0.0301513671875 * factor)
                            output.fill_(-7)
                            graph.replay()
                            observed = output.clone()
                            finished = torch.cuda.Event()
                            finished.record()
                        finished.synchronize()
                        torch.testing.assert_close(
                            observed, expected * factor, atol=0, rtol=0
                        )
                finally:
                    stream.synchronize()
                    graph.reset()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_rms_rejects_metadata_and_aliases_before_writes(self):
        module = kernel_native._transform_module()
        values, weight, _ = rms_rounding_case(2)
        original = values.clone()
        output = torch.full_like(values, -7)
        strided = torch.full((2, 10240), -7, dtype=torch.bfloat16, device="cuda")[
            :, ::2
        ]
        invalid = (
            (values.float(), weight, output, 1e-20),
            (values, weight.float(), output, 1e-20),
            (values, weight[:-1], output, 1e-20),
            (values, weight.cpu(), output, 1e-20),
            (values, weight, strided, 1e-20),
            (values, weight, values, 1e-20),
            (values, weight, output, 1e-5),
        )
        for arguments in invalid:
            with self.subTest(shapes=[getattr(v, "shape", v) for v in arguments]):
                with self.assertRaisesRegex(Exception, "Check failed"):
                    module.rms(*arguments)
                torch.testing.assert_close(
                    output, torch.full_like(output, -7), atol=0, rtol=0
                )
                torch.testing.assert_close(
                    strided, torch.full_like(strided, -7), atol=0, rtol=0
                )
                torch.testing.assert_close(values, original, atol=0, rtol=0)
        module.rms(values[:0], weight, output[:0], 1e-20)
        torch.testing.assert_close(output, torch.full_like(output, -7), atol=0, rtol=0)

    def test_hc_vendor_identity_uses_tensor_runtime_libraries(self):
        from oh_my_vllm.operators.runtime import native

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [
                Path("nvidia/cu13/lib/libcublas.so.13"),
                Path("nvidia/cu13/lib/libcublasLt.so.13"),
                Path("nvidia/cu13/include/cublas_api.h"),
                Path("nvidia/cu13/include/cublas_v2.h"),
                Path("nvidia/cu13/include/cublasLt.h"),
            ]
            for path in files:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_bytes(path.name.encode())
            package = Mock(files=files, version="test")
            package.locate_file.side_effect = lambda path: root / path
            with patch.object(
                native.importlib.metadata, "distribution", return_value=package
            ):
                vendor = native.hyperconnection_vendor_dependencies()
                expected_paths = [str((root / path).resolve()) for path in files[:2]]
                self.assertEqual(
                    [item["path"] for item in vendor["images"]], expected_paths
                )
                self.assertEqual(
                    vendor["include_directory"], str((root / files[2].parent).resolve())
                )
                module = Mock()
                module.hc_images.return_value = expected_paths
                native.verify_hyperconnection_vendor_images(module, vendor)
                (root / files[0]).write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "vendor image differs"):
                    native.verify_hyperconnection_vendor_images(module, vendor)
                package.files = files[:-1]
                with self.assertRaisesRegex(RuntimeError, "cuBLAS"):
                    native.hyperconnection_vendor_dependencies()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_hc_sinkhorn_uses_released_row_and_column_sum_order(self):
        residual = torch.ones((1, 1, 4, 5120), dtype=torch.bfloat16, device="cuda")
        base = torch.tensor(
            [0.0] * 8
            + [
                -1.9,
                2.3,
                -4.1,
                0.7,
                3.1,
                -0.2,
                1.7,
                -2.6,
                0.3,
                -3.7,
                1.1,
                2.8,
                -0.6,
                3.9,
                -2.2,
                0.4,
            ],
            dtype=torch.float32,
            device="cuda",
        )
        actual = kernel.hc_prepare(
            residual,
            torch.ones((1, 1, 4), dtype=torch.float32, device="cuda"),
            torch.zeros((24, 20480), dtype=torch.float32, device="cuda"),
            torch.ones(3, dtype=torch.float32, device="cuda"),
            base,
        )[3]
        expected = torch.tensor(
            [
                0.11736833304166794,
                0.35925719141960144,
                0.03100590966641903,
                0.49236568808555603,
                0.6288173794746399,
                0.0010646817972883582,
                0.3694639801979065,
                0.0006558680324815214,
                0.09901463240385056,
                8.331682329298928e-5,
                0.5250361561775208,
                0.37586623430252075,
                0.154798686504364,
                0.6395938992500305,
                0.0744929388165474,
                0.13111117482185364,
            ],
            dtype=torch.float32,
            device="cuda",
        ).view(1, 1, 4, 4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hc_collapse_rounds_products_before_the_sum(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device):
                residual = torch.zeros(
                    (1, 1, 4, 5120), dtype=torch.bfloat16, device=device
                )
                residual[0, 0, :, 0] = torch.tensor(
                    [
                        0.01153564453125,
                        -0.01239013671875,
                        0.00099945068359375,
                        -0.19921875,
                    ],
                    device=device,
                )
                incoming = torch.tensor(
                    [
                        [
                            [
                                0.9483255743980408,
                                0.13579025864601135,
                                0.0016392555553466082,
                                0.04646666347980499,
                            ]
                        ]
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                actual = kernel.hc_prepare(
                    residual,
                    incoming,
                    torch.zeros((24, 20480), dtype=torch.float32, device=device),
                    torch.ones(3, dtype=torch.float32, device=device),
                    torch.zeros(24, dtype=torch.float32, device=device),
                )[0]
                expected = torch.zeros_like(actual)
                expected[..., 0] = 1.6987323760986328e-6
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_hc_expand_adds_the_branch_after_the_residual_sum(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device):
                residual = torch.zeros(
                    (1, 1, 4, 5120), dtype=torch.bfloat16, device=device
                )
                residual[0, 0, :, 0] = torch.tensor(
                    [-0.59375, -0.034912109375, -0.326171875, -0.09423828125],
                    device=device,
                )
                branch = torch.zeros((1, 1, 5120), dtype=torch.bfloat16, device=device)
                branch[..., 0] = 0.6484375
                post = torch.zeros((1, 1, 4), dtype=torch.float32, device=device)
                post[..., 1] = 5.7516052038408816e-5
                comb = torch.zeros((1, 1, 4, 4), dtype=torch.float32, device=device)
                comb[0, 0, :, 1] = torch.tensor(
                    [
                        0.0002151463704649359,
                        0.8902646899223328,
                        0.07320763170719147,
                        0.0363115631043911,
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                actual = kernel.hc_expand(residual, branch, post, comb)
                expected = torch.zeros_like(actual)
                expected[..., 1, 0] = -0.05859375
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
