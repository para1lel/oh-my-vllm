import gc
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel
from oh_my_vllm.operators.hyperconnection import workspace as hc_workspace
from oh_my_vllm.operators.hyperconnection.api import (
    BLAS_WORKSPACE_BYTES,
    FUNCTIONS,
    Inputs,
    Weights,
)
from oh_my_vllm.operators.hyperconnection.plan import make_plan
from oh_my_vllm.operators.runtime.native import load


class HyperconnectionStorage(unittest.TestCase):
    def test_workspace_owns_and_passes_all_three_scratch_buffers(self):
        module = Mock()
        device = SimpleNamespace(
            empty=lambda shape, dtype: SimpleNamespace(shape=shape, dtype=dtype)
        )
        weights = Weights(*(object() for _ in range(3)))
        with patch.object(
            hc_workspace, "load", return_value=(module, "source", "library")
        ) as native_load:
            workspace = hc_workspace.prepare(make_plan(rows=2), device, weights)
        try:
            native_load.assert_called_once_with("hyperconnection", FUNCTIONS)
            storage = workspace.storage
            self.assertEqual(len(storage), 9)
            self.assertEqual(storage.flat.shape, (2, 20480))
            self.assertEqual(storage.statistics.shape, (2, 2))
            self.assertEqual(storage.blas.shape, (BLAS_WORKSPACE_BYTES,))
            self.assertEqual(storage.flat.dtype, "float32")
            self.assertEqual(storage.statistics.dtype, "float32")
            self.assertEqual(storage.blas.dtype, "uint8")
            module.hc_warmup.assert_called_once_with(storage.collapsed)
            inputs = Inputs(*(object() for _ in range(3)))
            with workspace.graph_lease():
                outputs = workspace.run(inputs)
                arguments = module.hc_prepare.call_args.args
                self.assertEqual(len(arguments), 13)
                for actual, owner in zip(arguments[-3:], storage[-3:], strict=True):
                    self.assertIs(actual, owner)
                self.assertIs(outputs.collapsed, storage.collapsed)
                self.assertIs(outputs.expanded, storage.expanded)
                self.assertIs(outputs.next_pre, storage.next_pre)
                module.hc_expand.assert_called_once_with(
                    inputs.residual,
                    inputs.branch,
                    storage.post,
                    storage.comb,
                    storage.expanded,
                )
        finally:
            workspace.close()


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class HyperconnectionWorkspaceCuda(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module, _, _ = load("hyperconnection", FUNCTIONS)

    def make_case(self, rows):
        residual = torch.full((rows, 4, 5120), 2, dtype=torch.bfloat16, device="cuda")
        incoming = torch.zeros((rows, 4), device="cuda")
        incoming[:, 0] = 1
        branch = torch.ones((rows, 5120), dtype=torch.bfloat16, device="cuda")
        weights = Weights(
            torch.zeros((24, 20480), device="cuda"),
            torch.ones(3, device="cuda"),
            torch.zeros(24, device="cuda"),
        )
        workspace = kernel.make_hc_workspace(residual.device, max_rows=max(1, rows))
        return residual, incoming, branch, weights, workspace

    def prepare_arguments(self, case):
        residual, incoming, _, weights, workspace = case
        rows = residual.size(0)
        return (
            residual,
            incoming,
            *weights,
            workspace["mixed"][:rows],
            workspace["post"][:rows],
            workspace["comb"][:rows],
            workspace["collapsed"][:rows],
            workspace["next_pre_a"][:rows],
            workspace["flat"][:rows],
            workspace["statistics"][:rows],
            workspace["blas"][: self.module.workspace_size(rows)],
        )

    def fill_sentinels(self, workspace):
        for tensor in workspace.values():
            tensor.fill_(127 if tensor.dtype == torch.uint8 else -7)

    def assert_sentinels(self, workspace):
        for name, tensor in workspace.items():
            with self.subTest(buffer=name):
                expected = 127 if tensor.dtype == torch.uint8 else -7
                self.assertTrue(torch.equal(tensor, torch.full_like(tensor, expected)))

    def test_prepared_pipeline_replays_on_nondefault_caller_stream(self):
        for rows in (1, 33):
            with self.subTest(rows=rows):
                proxy = Mock(wraps=self.module)
                with patch.object(
                    kernel, "_hyperconnection_module", return_value=proxy
                ):
                    case = self.make_case(rows)
                    residual, _, _, _, workspace = case
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())

                    def run(case=case):
                        residual, incoming, branch, weights, workspace = case
                        collapsed, _, post, comb = kernel.hc_prepare(
                            residual, incoming, *weights, workspace=workspace
                        )
                        expanded = kernel.hc_expand(
                            residual, branch, post, comb, workspace=workspace
                        )
                        return collapsed, expanded

                    with torch.cuda.stream(stream):
                        run()
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    try:
                        with torch.cuda.graph(graph, stream=stream):
                            collapsed, expanded = run()
                        recorded = (
                            proxy.hc_prepare.call_args.args,
                            proxy.hc_expand.call_args.args,
                        )
                        self.assertEqual(len(recorded[0]), 13)
                        self.assertEqual(len(recorded[1]), 5)
                        for argument, name in zip(
                            recorded[0][-3:],
                            ("flat", "statistics", "blas"),
                            strict=True,
                        ):
                            self.assertEqual(
                                argument.data_ptr(), workspace[name].data_ptr()
                            )
                        self.assertEqual(workspace["blas"].data_ptr() % 256, 0)
                        # Only the workspace and input owners retain graph storage.
                        del recorded, argument
                        proxy.reset_mock()
                        gc.collect()
                        for value in (2, 4):
                            with torch.cuda.stream(stream):
                                residual.fill_(value)
                                collapsed.fill_(-7)
                                expanded.fill_(-7)
                                workspace["mixed"].fill_(-7)
                                workspace["statistics"].fill_(-7)
                                graph.replay()
                                observed = (
                                    collapsed.clone(),
                                    expanded.clone(),
                                    workspace["statistics"].clone(),
                                    workspace["mixed"].clone(),
                                )
                                finished = torch.cuda.Event()
                                finished.record()
                            finished.synchronize()
                            torch.testing.assert_close(
                                observed[0],
                                torch.full_like(observed[0], value),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                observed[1],
                                torch.full_like(observed[1], value + 1),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                observed[2].cpu(),
                                torch.tensor([value * value, 1 / value]).expand(
                                    rows, 2
                                ),
                                rtol=0,
                                atol=0,
                            )
                            torch.testing.assert_close(
                                observed[3],
                                torch.zeros_like(observed[3]),
                                rtol=0,
                                atol=0,
                            )
                    finally:
                        stream.synchronize()
                        graph.reset()

    def test_cold_thread_rejects_capture_before_writes_and_allows_recovery(self):
        case = self.make_case(1)
        workspace = case[-1]
        arguments = self.prepare_arguments(case)
        ready = torch.cuda.Event()
        ready.record()

        def check_capture():
            stream = torch.cuda.Stream()
            stream.wait_event(ready)
            with torch.cuda.stream(stream):
                self.fill_sentinels(workspace)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            failures = (
                (
                    lambda: self.module.hc_prepare(*arguments),
                    "prepare HC before Graph capture",
                ),
                (
                    lambda: self.module.hc_warmup(workspace["collapsed"]),
                    "prepare HC outside Graph capture",
                ),
            )
            try:
                for operation, message in failures:
                    with (
                        warnings.catch_warnings(),
                        self.assertRaisesRegex(Exception, message),
                        torch.cuda.graph(graph, stream=stream),
                    ):
                        warnings.filterwarnings(
                            "ignore",
                            message=r"The CUDA Graph is empty\.",
                            category=UserWarning,
                        )
                        operation()
                    with torch.cuda.stream(stream):
                        self.assertFalse(torch.cuda.is_current_stream_capturing())
                    stream.synchronize()
                    self.assert_sentinels(workspace)
                    graph.reset()
                with torch.cuda.stream(stream):
                    self.module.hc_warmup(workspace["collapsed"])
                with torch.cuda.graph(graph, stream=stream):
                    self.module.hc_prepare(*arguments)
                with torch.cuda.stream(stream):
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    workspace["collapsed"],
                    torch.full_like(workspace["collapsed"], 2),
                    rtol=0,
                    atol=0,
                )
            finally:
                stream.synchronize()
                graph.reset()

        with ThreadPoolExecutor(max_workers=1) as threads:
            threads.submit(check_capture).result()

    def test_native_prepare_rejects_invalid_scratch_before_writes(self):
        case = self.make_case(2)
        arguments = self.prepare_arguments(case)
        workspace = case[-1]
        raw = torch.empty(BLAS_WORKSPACE_BYTES + 256, dtype=torch.uint8, device="cuda")
        offset = (16 - raw.data_ptr()) % 256
        misaligned = raw[offset : offset + BLAS_WORKSPACE_BYTES]
        self.assertEqual(misaligned.data_ptr() % 256, 16)
        strided_statistics = torch.empty((2, 4), device="cuda")[:, ::2]
        invalid = (
            ("flat dtype", 10, workspace["flat"].to(torch.bfloat16)),
            ("flat shape", 10, workspace["flat"][:, :-1]),
            ("statistics shape", 11, workspace["statistics"][:, :1]),
            ("statistics stride", 11, strided_statistics),
            ("statistics device", 11, torch.empty((2, 2), device="cpu")),
            ("blas dtype", 12, workspace["blas"].view(torch.float32)),
            ("insufficient blas", 12, workspace["blas"][:-256]),
            ("blas alignment", 12, misaligned),
            ("flat aliases weights", 10, case[3].projection[:2]),
            (
                "statistics aliases mixed",
                11,
                workspace["mixed"].flatten()[:4].view(2, 2),
            ),
            ("collapsed aliases residual", 8, case[0].flatten()[:10240].view(2, 5120)),
            (
                "flat aliases blas",
                10,
                workspace["blas"].view(torch.float32)[:40960].view(2, 20480),
            ),
        )
        self.fill_sentinels(workspace)
        for name, index, replacement in invalid:
            with self.subTest(case=name):
                malformed = list(arguments)
                malformed[index] = replacement
                with self.assertRaisesRegex(Exception, "Check failed"):
                    self.module.hc_prepare(*malformed)
                torch.cuda.current_stream().synchronize()
                self.assert_sentinels(workspace)

    def test_empty_rows_do_not_write_or_launch(self):
        case = self.make_case(0)
        workspace = case[-1]
        self.fill_sentinels(workspace)
        self.assertEqual(self.module.workspace_size(0), 0)
        self.module.hc_prepare(*self.prepare_arguments(case))
        self.module.hc_expand(
            case[0],
            case[2],
            workspace["post"][:0],
            workspace["comb"][:0],
            workspace["expanded_a"][:0],
        )
        torch.cuda.current_stream().synchronize()
        self.assert_sentinels(workspace)


if __name__ == "__main__":
    unittest.main()
