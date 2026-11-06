import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.operators.fa_fp8.api import BatchView, CacheView
from oh_my_vllm.operators.mtp_fp8.api import (
    AcceptInputs,
    ContextInputs,
    Inputs,
    InputView,
    InputWeights,
    Weights,
)
from oh_my_vllm.operators.mtp_fp8.metadata import make_context_metadata
from oh_my_vllm.operators.mtp_fp8.plan import accept_plan, context_plan, draft_plan
from oh_my_vllm.operators.mtp_fp8.workspace import (
    prepare_accept,
    prepare_context,
    prepare_draft,
)
from oh_my_vllm.operators.runtime.storage import storage_bytes


class Metadata(unittest.TestCase):
    def test_shift_and_accepted_endpoint_are_independent_of_proposal_tail(self):
        metadata = make_context_metadata(
            token_positions=((8192, 8193, 8194), (129, 130)),
            target_positions=((8192, 8193), (128, 129)),
            boundary_positions=(8191, 127),
            boundary_slots=(7, 9),
            endpoint_positions=(8193, 129),
        )
        self.assertEqual(metadata.sequence_rows, (0, 3, 5))
        self.assertEqual(metadata.hidden_rows, (-1, 0, 1, 2, 3))
        self.assertEqual(metadata.boundary_rows, (7, 7, 7, 9, 9))
        self.assertEqual(metadata.endpoint_rows, (1, 3))

    def test_head_only_uses_saved_previous_hidden_without_target_rows(self):
        metadata = make_context_metadata(
            token_positions=((8320,),),
            target_positions=((),),
            boundary_positions=(8319,),
            boundary_slots=(12,),
            endpoint_positions=(8320,),
        )
        self.assertEqual(metadata.hidden_rows, (-1,))
        self.assertEqual(metadata.endpoint_rows, (0,))

    def test_missing_shift_source_and_invalid_endpoint_fail(self):
        values = dict(
            token_positions=((8192, 8193),),
            target_positions=((8192,),),
            boundary_positions=(8191,),
            boundary_slots=(1,),
            endpoint_positions=(8192,),
        )
        for change in (
            {"boundary_positions": (8190,)},
            {"endpoint_positions": (8194,)},
            {"token_positions": ((0, 1),)},
            {"token_positions": ((8192, 8194),)},
            {"boundary_slots": (-1,)},
            {"target_positions": ()},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                make_context_metadata(**(values | change))


class Plans(unittest.TestCase):
    def test_page128_width_and_variants_use_new_fa(self):
        context = context_plan(rows=5)
        first, later = draft_plan(rows=2, first=True), draft_plan(rows=2)
        self.assertEqual(context.attention.table_width, 2048)
        self.assertEqual(context.attention.variant, 1)
        self.assertEqual(first.attention.variant, 2)
        self.assertEqual(later.attention.variant, 0)
        self.assertEqual(first.storage, ())
        self.assertEqual(first.mtp_scratch_bytes, 0)
        self.assertEqual(later.mtp_scratch_bytes, 245760 + 256)
        self.assertNotEqual(first.reuse_key, later.reuse_key)

    def test_exact_mtp_tensor_and_metadata_scratch(self):
        small = context_plan(rows=5, batch=1)
        large = context_plan(rows=33, batch=1)
        self.assertEqual(small.mtp_scratch_bytes, 122880 * 5 + 20480 + 256)
        self.assertEqual(large.mtp_scratch_bytes, 40960 * 33 + 20480 + 512)
        self.assertEqual(storage_bytes(small.metadata_storage), 10240)
        self.assertEqual(context_plan(rows=2, max_context=129).attention.table_width, 2)

    def test_invalid_dimensions_and_incomplete_table_fail(self):
        for arguments in (
            {"rows": 0},
            {"rows": 2, "batch": 3},
            {"rows": 1, "max_context": 262145},
            {"rows": 1, "max_context": 129, "table_width": 1},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                context_plan(**arguments)
        for rows in (0, 5):
            with self.assertRaises(ValueError):
                draft_plan(rows=rows)


class NativeBoundary:
    def __init__(self):
        self.prepared = []

    def mtp_fp8_prepare_input(self, weight, rows):
        self.prepared.append((weight, rows))

    def mtp_fp8_input(self, hidden, tokens, *arguments):
        hidden_rows, boundary, boundary_rows = arguments[4:7]
        _packed, output, zero, _partials = arguments[-4:]
        for row, source in enumerate(hidden_rows):
            output[row].copy_(
                hidden[source] if source >= 0 else boundary[boundary_rows[row]]
            )
        zero.zero_()

    def mtp_fp8_endpoints(self, value, zero, starts, endpoints, output, output_zero):
        for index, endpoint in enumerate(endpoints):
            if not starts[index] <= endpoint < starts[index + 1]:
                raise ValueError("invalid endpoint")
            output[index].copy_(value[endpoint])
            output_zero[index].copy_(zero[endpoint])


class Operation:
    source_id, library = "cpu-boundary", "cpu-boundary.so"

    def __init__(self):
        self.outputs, self.inputs, self.closed = object(), [], False

    def graph_lease(self):
        return nullcontext()

    def close(self):
        self.closed = True

    def run(self, weights, inputs, *rest):
        self.inputs.append((inputs, rest))
        return inputs

    def project(self, weights, inputs):
        self.inputs.append((inputs, ()))


def weights():
    return Weights(
        InputWeights(*(object() for _ in range(4))), object(), object(), object()
    )


def inputs(endpoint=1):
    value = InputView(
        torch.stack((torch.full((5120,), 20), torch.full((5120,), 30))).bfloat16(),
        torch.tensor([101, 102, 103]),
        torch.tensor([-1, 0, 1], dtype=torch.int32),
        torch.full((1, 5120), 10, dtype=torch.bfloat16),
        torch.zeros(3, dtype=torch.int32),
    )
    batch = BatchView(
        torch.tensor([[8192, 8193, 8194]] * 3),
        torch.tensor([0, 3], dtype=torch.int32),
        torch.tensor([8193, 8194, 8195], dtype=torch.int32),
    )
    state = CacheView(
        torch.zeros((3, 2, 128, 4, 256), dtype=torch.uint8),
        torch.ones((3, 2, 128, 4), dtype=torch.float32),
        torch.tensor([128, 129, 130], dtype=torch.int32),
        torch.ones((1, 2048), dtype=torch.int32),
    )
    return ContextInputs(
        value, batch, state, torch.tensor([endpoint], dtype=torch.int32)
    )


class Workspaces(unittest.TestCase):
    def setUp(self):
        self.native, self.attention, self.ffn, self.output = (
            NativeBoundary(),
            Operation(),
            Operation(),
            Operation(),
        )
        self.allocations = []
        self.device = SimpleNamespace(empty=self.allocate)
        self.weights = weights()
        self.patches = (
            patch(
                "oh_my_vllm.operators.mtp_fp8.input.load",
                return_value=(self.native, "mtp", "mtp.so"),
            ),
            patch(
                "oh_my_vllm.operators.mtp_fp8.workspace.prepare_attention",
                return_value=self.attention,
            ),
            patch(
                "oh_my_vllm.operators.mtp_fp8.workspace.prepare_ffn",
                return_value=self.ffn,
            ),
            patch(
                "oh_my_vllm.operators.mtp_fp8.workspace.prepare_output",
                return_value=self.output,
            ),
        )
        for replacement in self.patches:
            replacement.start()
            self.addCleanup(replacement.stop)

    def allocate(self, shape, dtype):
        self.allocations.append((shape, dtype))
        return torch.zeros(shape, dtype=getattr(torch, dtype))

    def test_context_reads_selected_accepted_row_and_retains_fp8_owner(self):
        workspace = prepare_context(context_plan(rows=3), self.device, (self.weights,))
        binding = inputs()
        allocated = len(self.allocations)
        for _ in range(2):
            with workspace.graph_lease():
                result = workspace.run(self.weights, binding)
            self.assertTrue(torch.all(result.value == 20))
            self.assertTrue(torch.all(result.residual == 0))
        self.assertEqual(len(self.allocations), allocated)
        self.assertEqual(len(self.native.prepared), 1)
        self.assertIs(workspace.bindings.state.codes, binding.state.codes)
        self.assertIs(workspace.bindings.state.scales, binding.state.scales)
        self.assertIs(self.attention.inputs[-1][1][1], binding.state)
        workspace.close()
        with self.assertRaises(RuntimeError):
            workspace.run(self.weights, binding)

    def test_first_draft_requires_context_and_later_draft_requires_shifted_input(self):
        first = prepare_draft(
            draft_plan(rows=1, first=True), self.device, (self.weights,)
        )
        binding = inputs()
        with self.assertRaises(ValueError):
            first.project(self.weights, Inputs(None, binding.batch, binding.state))
        endpoint = SimpleNamespace(value=object(), residual=object())
        first.project(
            self.weights, Inputs(None, binding.batch, binding.state, endpoint)
        )
        self.assertIs(self.attention.inputs[-1][0], endpoint)
        self.assertEqual(self.native.prepared, [])
        later = prepare_draft(draft_plan(rows=1), self.device, (self.weights,))
        with self.assertRaises(ValueError):
            later.project(self.weights, Inputs(None, binding.batch, binding.state))
        first.close()
        later.close()

    def test_acceptance_preserves_metadata_and_allocates_only_during_prepare(self):
        module = SimpleNamespace(mtp_accept=Mock())
        with patch(
            "oh_my_vllm.operators.mtp_fp8.workspace.load",
            return_value=(module, "accept-source", "mtp-fp8.so"),
        ):
            workspace = prepare_accept(accept_plan(batch=2), self.device)
        binding = AcceptInputs(*(object() for _ in range(5)), capacity=512)
        allocated = len(self.allocations)
        with workspace.graph_lease():
            result = workspace.run(binding)
        self.assertIs(result, workspace.outputs)
        module.mtp_accept.assert_called_once_with(
            binding.target,
            binding.proposal,
            binding.drafts,
            binding.rng,
            binding.private_slots,
            *result,
            *workspace.scratch,
            512,
        )
        self.assertEqual(len(self.allocations), allocated)
        self.assertEqual(accept_plan(batch=2).mtp_scratch_bytes, 8960)
        workspace.close()
        with self.assertRaises(RuntimeError):
            workspace.run(binding)
        self.assertEqual(module.mtp_accept.call_count, 1)

    def test_partial_media_binding_fails_before_native_launch(self):
        from oh_my_vllm.operators.mtp_fp8.input import InputWorkspace

        workspace = InputWorkspace(
            3, context_plan(rows=3).storage[:5], (self.weights.input,), self.allocate
        )
        binding = inputs().input
        invalid = InputView(
            binding.hidden,
            binding.tokens,
            binding.hidden_rows,
            binding.boundary_hidden,
            binding.boundary_rows,
            media=torch.zeros((1, 5120)),
        )
        with self.assertRaises(ValueError):
            workspace.run(self.weights.input, invalid)
