import unittest
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from oh_my_vllm.models.deepseek_v41 import kernel, model
from oh_my_vllm.models.deepseek_v41.decode import DecodeBindings, DecodeGraph


def fp4_literal(rows, width=64):
    encoded = torch.zeros((rows, width), dtype=torch.float32, device="cuda")
    encoded[:, 8] = -1
    encoded[:, 32] = 64
    encoded[:, 48] = 32
    encoded[:, 63] = 3
    encoded = encoded.to(torch.float8_e4m3fn)
    raw = torch.zeros((128, width // 2), dtype=torch.uint8, device="cuda")
    raw[:, 4], raw[:, 16], raw[:, 24], raw[:, 31] = 2, 6, 4, 32
    row_shifts = torch.arange(rows, device="cuda") % 3 - 1
    column_shifts = torch.arange(128, device="cuda") % 5 - 2
    input_scales = (
        (
            torch.tensor([113, 127] + [127] * (width // 32 - 2), device="cuda")
            + row_shifts[:, None]
        )
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    weight_scales = (
        (
            torch.tensor([126, 128] + [127] * (width // 32 - 2), device="cuda")
            + column_shifts[:, None]
        )
        .to(torch.uint8)
        .view(torch.float8_e8m0fnu)
    )
    factors = torch.exp2((row_shifts[:, None] + column_shifts[None, :]).float())
    return (
        encoded,
        input_scales,
        raw.view(torch.float4_e2m1fn_x2),
        weight_scales,
        factors,
    )


def grouped_literal(counts=(1, 0, 33, 2), max_rows=33, dtype=torch.bfloat16, width=64):
    rows = sum(counts)
    encoded, scales, weight, weight_scales, factors = fp4_literal(rows, width)
    weights = [
        (
            weight.clone(),
            (weight_scales.view(torch.uint8) + shift).view(weight_scales.dtype),
        )
        for shift in (-1, 0, 1, 2)
    ]
    packed = [kernel._packed_weight(w, s, 4) for w, s in weights]
    pointers = kernel._grouped_pointer_arrays(packed, encoded.device)
    module = kernel._projection_module()
    return SimpleNamespace(
        encoded=encoded.view(torch.uint8),
        scales=scales.view(torch.uint8),
        pointers=pointers,
        counts=torch.tensor(counts, dtype=torch.int32, device="cuda"),
        output=torch.full((rows, 128), -7, dtype=dtype, device="cuda"),
        scratch=torch.empty(
            module.grouped_workspace_size(rows, 4, max_rows, 128, width),
            dtype=torch.uint8,
            device="cuda",
        ),
        factors=factors,
        max_rows=max_rows,
        weights=weights,
        packed=packed,
    )


def run_grouped(module, case, *, output=None):
    module.grouped_project(
        case.encoded,
        case.scales,
        *case.pointers,
        case.counts,
        case.output if output is None else output,
        case.scratch,
        case.max_rows,
    )


def prepared_moe_literal(device, strategy="TP4", experts=4):
    args = model.ModelArgs(
        dim=32,
        moe_inter_dim=64,
        n_routed_experts=experts,
        n_activated_experts=2,
        expert_dtype="fp4",
        parallel_strategy=strategy,
    )
    moe = model.MoE(0, args).to(device)

    class ZeroShared(torch.nn.Module):
        def forward(self, value):
            return torch.zeros_like(value)

    moe.shared_experts = ZeroShared()
    with torch.no_grad():
        for expert in moe.experts:
            if expert is not None:
                for layer in (expert.w1, expert.w2, expert.w3):
                    raw = layer.weight.view(torch.uint8)
                    raw.zero_()
                    raw[0, 0] = 2
                    layer.scale.fill_(1)
    values = torch.zeros((1, 32), dtype=torch.bfloat16, device=device)
    values[:, 0] = 1
    routing = (torch.ones((1, 2), device=device), torch.tensor([[0, 1]], device=device))
    return moe, values, routing


def pack_weight_bytes(raw_weight, raw_scales, packed_weight, packed_scales, bits):
    if packed_weight.data_ptr() != raw_weight.data_ptr():
        packed_weight.zero_()
        packed_weight[:, : raw_weight.size(1)].copy_(raw_weight)
    packed_scales.zero_()
    packed_scales[: raw_scales.numel()].copy_(raw_scales.flatten())


class _RoutedDecodeModel(torch.nn.Module):
    def __init__(self, moe, values, routing):
        super().__init__()
        layer = torch.nn.Module()
        layer.ffn = moe
        layer.engram = None
        layer.attn = SimpleNamespace(compress_ratio=0, window_size=128)
        self.layers = torch.nn.ModuleList([layer])
        self.values, self.routing = values, routing

    def forward(self, _tokens, **_options):
        value = self.layers[0].ffn._forward_rows(self.values, routing=self.routing)
        return None, value, None


class GroupedProjectionPrecision(unittest.TestCase):
    def test_decode_graph_rejects_rebound_expert_bindings_without_synchronization(self):
        for strategy in ("TP4", "TEP4"):
            for segmented in (False, True):
                for change in (
                    "expert",
                    "projection",
                    "weight",
                    "scale",
                    "load",
                    "apply",
                ):
                    with (
                        self.subTest(
                            strategy=strategy, segmented=segmented, change=change
                        ),
                        patch.object(model, "world_size", 1),
                    ):
                        moe, values, routing = prepared_moe_literal("cpu", strategy)
                        layer = SimpleNamespace(
                            ffn=moe,
                            engram=None,
                            attn=SimpleNamespace(compress_ratio=0, window_size=128),
                        )
                        target = Mock()
                        target._state_export = None
                        target._physical_cache = None
                        target.layers = [layer]
                        target.modules.return_value = (moe,)

                        def forward(
                            *_args, moe=moe, values=values, routing=routing, **_kwargs
                        ):
                            return (
                                None,
                                moe._forward_rows(values, routing=routing),
                                None,
                            )

                        target.side_effect = forward
                        bindings = DecodeBindings(1, (0,), "cpu")
                        bindings.bind([1], 132)
                        stream, native_graph = Mock(), Mock()
                        with (
                            patch("torch.cuda.Stream", return_value=stream),
                            patch("torch.cuda.CUDAGraph", return_value=native_graph),
                            patch("torch.cuda.current_stream", return_value=stream),
                            patch(
                                "torch.cuda.stream",
                                side_effect=lambda *_: nullcontext(),
                            ),
                            patch(
                                "torch.cuda.graph",
                                side_effect=lambda *_args, **_kwargs: nullcontext(),
                            ),
                            patch("torch.cuda.synchronize") as synchronize,
                            patch(
                                "oh_my_vllm.models.deepseek_v41.decode.require_sparse_status"
                            ),
                        ):
                            graph = DecodeGraph(
                                target, bindings, 132, segmented=segmented
                            )
                            graph.capture()
                            synchronize.reset_mock()
                            graph.replay()
                            native_graph.replay.assert_called_once()
                            synchronize.assert_not_called()
                            native_graph.replay.reset_mock()
                            expert = moe.experts[0]
                            if change == "expert":
                                moe.experts[0] = model.Expert(
                                    32, 64, dtype=torch.float4_e2m1fn_x2
                                )
                            elif change == "projection":
                                expert.w2 = model.Linear(
                                    64, 32, dtype=torch.float4_e2m1fn_x2
                                )
                            elif change in ("weight", "scale"):
                                previous = getattr(expert.w2, change)
                                replacement = (
                                    previous.view(torch.uint8)
                                    .clone()
                                    .view(previous.dtype)
                                )
                                setattr(
                                    expert.w2,
                                    change,
                                    torch.nn.Parameter(
                                        replacement, requires_grad=False
                                    ),
                                )
                            elif change == "load":
                                moe.load_state_dict(moe.state_dict(), assign=True)
                            else:
                                expert.w2._apply(lambda value: value)
                            with self.assertRaisesRegex(
                                RuntimeError, "bindings.*changed"
                            ):
                                graph.replay()
                            native_graph.replay.assert_not_called()
                            synchronize.assert_not_called()
                            # Repreparation cannot validate the old graph.
                            moe.prepare_routed_experts()
                            with self.assertRaisesRegex(
                                RuntimeError, "bindings.*changed"
                            ):
                                graph.replay()
                            native_graph.replay.assert_not_called()
                            synchronize.assert_not_called()
                            replacement_graph = DecodeGraph(
                                target, bindings, 132, segmented=segmented
                            )
                            replacement_graph.capture()
                            synchronize.reset_mock()
                            replacement_graph.replay()
                            native_graph.replay.assert_called_once()
                            synchronize.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_decode_graph_rebind_reject_and_recapture_match_eager_bits(self):
        with patch.object(model, "world_size", 1):
            moe, values, routing = prepared_moe_literal("cuda")
            target = _RoutedDecodeModel(moe, values, routing)
            bindings = DecodeBindings(1, (0,), "cuda")
            bindings.bind([1], 132)
            with torch.inference_mode():
                expected = target(bindings.token_ids)[1].clone()
                graph = DecodeGraph(target, bindings, 132).capture()
                graph.replay()
                self.assertTrue(
                    torch.equal(
                        graph.logits.view(torch.uint8), expected.view(torch.uint8)
                    )
                )
                for expert in moe.experts:
                    weight = expert.w2.weight.view(torch.uint8).clone()
                    weight[0, 0] = 4
                    expert.w2.weight = torch.nn.Parameter(
                        weight.view(torch.float4_e2m1fn_x2), requires_grad=False
                    )
                with self.assertRaisesRegex(RuntimeError, "bindings.*changed"):
                    graph.replay()
                expected = target(bindings.token_ids)[1].clone()
                wanted = torch.zeros_like(values)
                wanted[:, 0] = 3
                self.assertTrue(torch.equal(expected, wanted))
                with self.assertRaisesRegex(RuntimeError, "bindings.*changed"):
                    graph.replay()
                replacement = DecodeGraph(target, bindings, 132).capture()
                replacement.replay()
                self.assertTrue(
                    torch.equal(
                        replacement.logits.view(torch.uint8), expected.view(torch.uint8)
                    )
                )

    def test_packed_projection_refreshes_after_in_place_state_load(self):
        for inference in (False, True):
            with (
                self.subTest(inference=inference),
                torch.inference_mode(inference),
                patch.dict(kernel._PACKED_WEIGHTS, {}, clear=True),
                patch.object(
                    kernel,
                    "_projection_module",
                    return_value=SimpleNamespace(pack_weights=pack_weight_bytes),
                ),
            ):
                layer = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
                with torch.no_grad():
                    layer.weight.view(torch.uint8).fill_(18)
                    layer.scale.view(torch.uint8).fill_(127)
                alias_weight = layer.weight[8:24]
                alias_scale = layer.scale[8:24]
                self.assertNotEqual(alias_weight.data_ptr(), layer.weight.data_ptr())
                self.assertNotEqual(alias_scale.data_ptr(), layer.scale.data_ptr())
                alias_first = kernel._packed_weight(alias_weight, alias_scale, 4)
                first = kernel._packed_weight(layer.weight, layer.scale, 4)
                self.assertIs(
                    first, kernel._packed_weight(layer.weight, layer.scale, 4)
                )
                addresses = (layer.weight.data_ptr(), layer.scale.data_ptr())
                state = {
                    name: value.view(torch.uint8).clone().view(value.dtype)
                    for name, value in layer.state_dict().items()
                }
                state["weight"].view(torch.uint8).fill_(52)
                state["scale"].view(torch.uint8).fill_(129)
                layer.load_state_dict(state, assign=False)
                self.assertEqual(
                    addresses, (layer.weight.data_ptr(), layer.scale.data_ptr())
                )
                refreshed = kernel._packed_weight(layer.weight, layer.scale, 4)
                self.assertTrue(
                    torch.equal(refreshed[0][:, :32], state["weight"].view(torch.uint8))
                )
                self.assertTrue(
                    torch.equal(
                        refreshed[1][:64], state["scale"].view(torch.uint8).flatten()
                    )
                )
                self.assertEqual(int(first[0][0, 0]), 18)
                self.assertEqual(int(first[1][0]), 127)
                alias_refreshed = kernel._packed_weight(alias_weight, alias_scale, 4)
                self.assertTrue(
                    torch.equal(
                        alias_refreshed[1][:32],
                        state["scale"][8:24].view(torch.uint8).flatten(),
                    )
                )
                self.assertEqual(int(alias_first[1][0]), 127)

    def test_packed_projection_tracks_direct_shared_view_mutations(self):
        with (
            patch.dict(kernel._PACKED_WEIGHTS, {}, clear=True),
            patch.object(
                kernel,
                "_projection_module",
                return_value=SimpleNamespace(pack_weights=pack_weight_bytes),
            ),
            torch.no_grad(),
        ):
            layer = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
            layer.weight.view(torch.uint8).fill_(18)
            layer.scale.view(torch.uint8).fill_(127)
            first = kernel._packed_weight(layer.weight, layer.scale, 4)
            layer.scale.detach().view(torch.uint8).fill_(129)
            updated_scale = kernel._packed_weight(layer.weight, layer.scale, 4)
            self.assertEqual(int(updated_scale[1][0]), 129)
            self.assertEqual(int(first[1][0]), 127)
            layer.weight.detach().view(torch.uint8).fill_(52)
            updated_weight = kernel._packed_weight(layer.weight, layer.scale, 4)
            self.assertTrue(torch.all(updated_weight[0][:, :32] == 52))
            self.assertTrue(torch.all(updated_weight[0][:, 32:] == 0))
            self.assertIs(
                updated_weight, kernel._packed_weight(layer.weight, layer.scale, 4)
            )
            self.assertEqual(len(kernel._PACKED_WEIGHTS), 1)

    def test_prepared_experts_refresh_after_rebinding_and_state_assignment(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device), patch.object(model, "world_size", 1):
                moe, values, routing = prepared_moe_literal(device)

                def check(expected, moe=moe, values=values, routing=routing):
                    actual = moe._forward_rows(values, routing=routing)
                    want = torch.zeros_like(values)
                    want[:, 0] = expected
                    torch.testing.assert_close(actual, want, rtol=0, atol=0)

                check(1.5)
                for expert in moe.experts:
                    expert.w2 = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2).to(
                        device
                    )
                    with torch.no_grad():
                        expert.w2.weight.view(torch.uint8).zero_()
                        expert.w2.weight.view(torch.uint8)[0, 0] = 4
                        expert.w2.scale.fill_(1)
                self.assertIsNone(moe._routed_state.weights)
                check(3)

                for expert in moe.experts:
                    weight = expert.w2.weight.detach().clone()
                    weight.view(torch.uint8).zero_()
                    weight.view(torch.uint8)[0, 0] = 6
                    expert.w2.weight = torch.nn.Parameter(weight, requires_grad=False)
                self.assertIsNone(moe._routed_state.weights)
                check(6)
                self.assertIs(
                    moe._routed_state.weights.down[0][0], moe.experts[0].w2.weight
                )

                for expert in moe.experts:
                    scale = expert.w2.scale.detach().clone()
                    expert.w2.scale = torch.nn.Parameter(scale, requires_grad=False)
                self.assertIsNone(moe._routed_state.weights)
                check(6)
                self.assertIs(
                    moe._routed_state.weights.down[0][1], moe.experts[0].w2.scale
                )

                state = {
                    name: value.view(torch.uint8).clone().view(value.dtype)
                    for name, value in moe.state_dict().items()
                }
                for name, value in state.items():
                    if name.endswith("w2.weight"):
                        value.view(torch.uint8)[0, 0] = 6
                moe.load_state_dict(state, assign=True)
                check(6)
                moe._apply(
                    lambda value: (
                        kernel.fp4_values(value).to(torch.bfloat16)
                        if value.dtype == torch.float4_e2m1fn_x2
                        else value
                    )
                )
                check(5.84375)

    def test_non_fp4_prepare_recovers_grouped_after_child_replacement(self):
        with patch.object(model, "world_size", 1):
            args = model.ModelArgs(
                dim=32,
                moe_inter_dim=64,
                n_routed_experts=4,
                n_activated_experts=2,
                expert_dtype="bf16",
                parallel_strategy="TEP4",
            )
            moe = model.MoE(0, args).to("cpu")

            class ZeroShared(torch.nn.Module):
                def forward(self, value):
                    return torch.zeros_like(value)

            moe.shared_experts = ZeroShared()
            moe.prepare_routed_experts()
            self.assertIs(moe._routed_state.weights, False)
            self.assertTrue(moe._routed_state.registered)
            for expert in moe.experts:
                for name, in_features, out_features in (
                    ("w1", 32, 64),
                    ("w3", 32, 64),
                    ("w2", 64, 32),
                ):
                    layer = model.Linear(
                        in_features,
                        out_features,
                        dtype=torch.float4_e2m1fn_x2,
                    )
                    with torch.no_grad():
                        layer.weight.view(torch.uint8).zero_()
                        layer.weight.view(torch.uint8)[0, 0] = 2
                        layer.scale.fill_(1)
                    setattr(expert, name, layer)
            self.assertIsNone(moe._routed_state.weights)
            values = torch.zeros((1, 32), dtype=torch.bfloat16)
            values[:, 0] = 1
            routing = (
                torch.ones((1, 2)),
                torch.tensor([[0, 1]]),
            )
            actual = moe._forward_rows(values, routing=routing)
            want = torch.zeros_like(values)
            want[:, 0] = 1.5
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
            self.assertIsNotNone(moe._routed_state.weights)

    def test_child_binding_changes_and_live_clamp_refresh_routed_execution(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            for change, expected in (
                ("load", 3),
                ("convert", 1.4609375),
                ("weight", 3),
                ("scale", 3),
                ("clamp", 0.3125),
            ):
                with (
                    self.subTest(device=device, change=change),
                    patch.object(model, "world_size", 1),
                ):
                    moe, values, routing = prepared_moe_literal(device)
                    moe._forward_rows(values, routing=routing)
                    for expert in moe.experts:
                        if change == "load":
                            state = {
                                name: value.view(torch.uint8).clone().view(value.dtype)
                                for name, value in expert.w2.state_dict().items()
                            }
                            state["weight"].view(torch.uint8)[0, 0] = 4
                            expert.w2.load_state_dict(state, assign=True)
                        elif change == "convert":
                            expert.w2._apply(
                                fn=lambda value: (
                                    kernel.fp4_values(value).to(torch.bfloat16)
                                    if value.dtype == torch.float4_e2m1fn_x2
                                    else value
                                ),
                                recurse=True,
                            )
                        elif change == "weight":
                            weight = expert.w2.weight.detach().clone()
                            weight.view(torch.uint8).zero_()
                            weight.view(torch.uint8)[0, 0] = 4
                            expert.w2.weight = torch.nn.Parameter(
                                weight, requires_grad=False
                            )
                        elif change == "scale":
                            scale = expert.w2.scale.detach().clone().fill_(2)
                            expert.w2.scale = torch.nn.Parameter(
                                scale, requires_grad=False
                            )
                        else:
                            expert.swiglu_limit = 0.5
                    if change != "clamp":
                        self.assertIsNone(moe._routed_state.weights)
                    actual = moe._forward_rows(values, routing=routing)
                    want = torch.zeros_like(values)
                    want[:, 0] = expected
                    torch.testing.assert_close(actual, want, rtol=0, atol=0)
                    for expert in moe.experts:
                        expert.swiglu_limit = 0
                        expert.w2 = model.Linear(
                            64, 32, dtype=torch.float4_e2m1fn_x2
                        ).to(device)
                        with torch.no_grad():
                            expert.w2.weight.view(torch.uint8).zero_()
                            expert.w2.weight.view(torch.uint8)[0, 0] = 4
                            expert.w2.scale.fill_(1)
                    moe.prepare_routed_experts()
                    grouped = moe._grouped_routed(
                        values.repeat(2, 1),
                        torch.zeros(2, dtype=torch.long, device=device),
                        torch.ones(2, device=device),
                        [1, 1, 0, 0],
                        torch.tensor([1, 1, 0, 0], device=device),
                    )
                    self.assertIsNotNone(grouped)
                    want = values.repeat(2, 1).zero_()
                    want[:, 0] = 1.5
                    torch.testing.assert_close(grouped[1], want, rtol=0, atol=0)

    def test_prepared_expert_parallel_ranges_keep_zero_row_steps(self):
        devices = [("cpu", "cpu")]
        if torch.cuda.is_available():
            devices.extend((("cuda", "cpu"), ("cuda", "cuda")))
        for device, default_device in devices:
            for strategy in ("TEP4", "DEP4"):
                for rank in (0, 1):
                    with (
                        self.subTest(
                            device=device,
                            default_device=default_device,
                            strategy=strategy,
                            rank=rank,
                        ),
                        torch.device(default_device),
                        patch.object(
                            model, "world_size", 4 if strategy == "TEP4" else 1
                        ),
                        patch.object(model, "expert_world_size", 4),
                        patch.object(model, "rank", rank),
                    ):
                        moe, values, _ = prepared_moe_literal(device, strategy, 8)
                        if rank:
                            with torch.no_grad():
                                for expert in moe.experts[2:4]:
                                    expert.w2.weight.view(torch.uint8)[0, 0] = 6
                        with torch.no_grad():
                            second = moe.experts[moe.experts_start_idx + 1]
                            second.w2.weight.view(torch.uint8)[0, 0] = 4
                        empty = moe._grouped_routed(
                            values[:0],
                            torch.empty(0, dtype=torch.long, device=device),
                            torch.empty(0, device=device),
                            [0] * 8,
                            torch.zeros(8, dtype=torch.int32, device=device),
                        )
                        self.assertIsNone(empty)
                        self.assertIsNone(moe._routed_state.pointers)
                        pointers = None
                        first_value = 3 if rank else 0.75
                        for counts, rows, expected in (
                            ([1, 0, 1, 0, 0, 0, 0, 0], [rank], first_value),
                            (
                                [0, 0, 2, 0, 0, 0, 0, 0],
                                [0, 1] if rank else [],
                                first_value,
                            ),
                            ([0, 1, 0, 1, 0, 0, 0, 0], [rank], 1.5),
                            (
                                [2, 0, 0, 0, 0, 0, 0, 0],
                                [] if rank else [0, 1],
                                first_value,
                            ),
                            ([1, 0, 1, 0, 0, 0, 0, 0], [rank], first_value),
                        ):
                            grouped = moe._grouped_routed(
                                values.repeat(2, 1),
                                torch.arange(2, device=device),
                                torch.ones(2, device=device),
                                counts,
                                torch.tensor(counts, device=device),
                            )
                            if not rows:
                                self.assertIsNone(grouped)
                            else:
                                torch.testing.assert_close(
                                    grouped[0], torch.tensor(rows, device=device)
                                )
                                want = torch.zeros(
                                    (len(rows), 32), dtype=torch.bfloat16, device=device
                                )
                                want[:, 0] = expected
                                torch.testing.assert_close(
                                    grouped[1], want, rtol=0, atol=0
                                )
                            if device == "cuda":
                                self.assertIsNotNone(moe._routed_state.pointers)
                                if pointers is not None:
                                    self.assertIs(moe._routed_state.pointers, pointers)
                                pointers = moe._routed_state.pointers
                                for (
                                    weight_ptrs,
                                    scale_ptrs,
                                    weight_host,
                                    scale_host,
                                ) in pointers:
                                    self.assertEqual(weight_ptrs.device, values.device)
                                    self.assertEqual(scale_ptrs.device, values.device)
                                    for host in (weight_host, scale_host):
                                        self.assertEqual(host.device.type, "cpu")
                                        self.assertTrue(host.is_pinned())
                        expert = moe.experts[moe.experts_start_idx]
                        with torch.no_grad():
                            weight = expert.w2.weight.detach().clone()
                            weight.view(torch.uint8)[0, 0] = 2 if rank else 6
                        expert.w2.weight = torch.nn.Parameter(
                            weight, requires_grad=False
                        )
                        self.assertIsNone(moe._routed_state.pointers)
                        grouped = moe._grouped_routed(
                            values.repeat(2, 1),
                            torch.arange(2, device=device),
                            torch.ones(2, device=device),
                            counts,
                            torch.tensor(counts, device=device),
                        )
                        want = torch.zeros((1, 32), dtype=torch.bfloat16, device=device)
                        want[:, 0] = 0.75 if rank else 3
                        torch.testing.assert_close(grouped[1], want, rtol=0, atol=0)
                        if device == "cuda":
                            self.assertIsNotNone(moe._routed_state.pointers)
                            self.assertIsNot(moe._routed_state.pointers, pointers)

    def test_rebinding_one_owner_does_not_invalidate_another_shared_owner(self):
        with patch.object(model, "world_size", 1):
            first, values, routing = prepared_moe_literal("cpu")
            second, _, _ = prepared_moe_literal("cpu")
            shared = first.experts[0].w2
            second.experts[0].w2 = shared
            first.prepare_routed_experts()
            second.prepare_routed_experts()
            self.assertIsNotNone(first._routed_state.weights)
            self.assertIsNotNone(second._routed_state.weights)

            replacement = model.Linear(64, 32, dtype=torch.float4_e2m1fn_x2)
            with torch.no_grad():
                replacement.weight.view(torch.uint8).zero_()
                replacement.weight.view(torch.uint8)[0, 0] = 4
                replacement.scale.fill_(1)
            first.experts[0].w2 = replacement
            self.assertIsNone(first._routed_state.weights)
            first._forward_rows(values, routing=routing)
            self.assertIsNotNone(first._routed_state.weights)
            self.assertIsNotNone(second._routed_state.weights)

            shared._apply(
                lambda value: (
                    kernel.fp4_values(value).to(torch.bfloat16)
                    if value.dtype == torch.float4_e2m1fn_x2
                    else value
                )
            )
            self.assertIsNotNone(first._routed_state.weights)
            self.assertIsNone(second._routed_state.weights)
            first._forward_rows(values, routing=routing)
            second._forward_rows(values, routing=routing)

    def test_shared_projection_invalidates_all_live_owners(self):
        with patch.object(model, "world_size", 1):
            first, _, _ = prepared_moe_literal("cpu")
            second, _, _ = prepared_moe_literal("cpu")
            shared = first.experts[0].w2
            second.experts[0].w2 = shared
            first.prepare_routed_experts()
            second.prepare_routed_experts()
            shared._apply(
                lambda value: (
                    kernel.fp4_values(value).to(torch.bfloat16)
                    if value.dtype == torch.float4_e2m1fn_x2
                    else value
                )
            )
            self.assertIsNone(first._routed_state.weights)
            self.assertIsNone(second._routed_state.weights)

    def test_prepare_weights_accepts_torch_uint8_storage(self):
        from oh_my_vllm.operators.projection import workspace as projection

        class Weights(SimpleNamespace):
            def __iter__(self):
                return iter((self.encoded, self.scales, self.encoded, self.scales, 8))

        calls = []
        device = SimpleNamespace(
            empty=lambda shape, dtype: torch.empty(shape, dtype=torch.uint8),
            synchronize_compute=lambda: None,
        )
        module = SimpleNamespace(pack_weights=lambda *args: calls.append(args))
        weights = Weights(
            encoded=torch.zeros((32, 64), dtype=torch.uint8),
            scales=torch.zeros((1, 2), dtype=torch.uint8),
        )
        with (
            patch.object(projection, "require_eager", lambda _: None),
            patch.object(projection, "load", return_value=(module, None, None)),
        ):
            result = projection.prepare_weights(device, weights, bits=8)
            self.assertEqual((result.width, result.bits, result.outputs), (64, 8, 32))
            self.assertTrue(calls)
            with self.assertRaisesRegex(ValueError, "raw weight type"):
                projection.prepare_weights(
                    device,
                    Weights(
                        encoded=torch.zeros((32, 64), dtype=torch.int8),
                        scales=torch.zeros((1, 2), dtype=torch.uint8),
                    ),
                    bits=8,
                )

    def test_channel_gather_keeps_rank_order_for_strided_and_empty_rows(self):
        values = torch.tensor([[0, -1, 1, -1], [2, -1, 3, -1]])[:, ::2]
        expected = torch.tensor(
            [[0, 1, 10, 11, 20, 21, 30, 31], [2, 3, 12, 13, 22, 23, 32, 33]]
        )

        def gather(output, value):
            output.copy_(torch.cat([value + offset for offset in (0, 10, 20, 30)]))

        with (
            patch.object(model, "world_size", 4),
            patch.object(model.dist, "all_gather_into_tensor", side_effect=gather),
        ):
            for value in (values, values[:0], values):
                actual = model._gather_channels(value)
                torch.testing.assert_close(
                    actual, expected[: value.size(0)], rtol=0, atol=0
                )
                self.assertTrue(actual.is_contiguous())
        with patch.object(model, "world_size", 1):
            torch.testing.assert_close(
                model._gather_channels(values), values, rtol=0, atol=0
            )

    def test_adapter_output_dtype_keeps_unrounded_values_and_empty_shape(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            values = torch.zeros((2, 32), dtype=torch.bfloat16, device=device)
            values[:, :2] = torch.tensor([1, 1 / 256], device=device)
            weight = torch.full((32, 16), 0x22, dtype=torch.uint8, device=device)
            scales = torch.ones((32, 1), device=device).to(torch.float8_e8m0fnu)
            weights = [(weight.view(torch.float4_e2m1fn_x2), scales)] * 3
            counts = torch.tensor([1, 0, 1], dtype=torch.int32, device=device)
            for dtype, expected in ((torch.bfloat16, 1), (torch.float32, 1 + 1 / 256)):
                with self.subTest(device=device, dtype=dtype):
                    actual = kernel.grouped_fp4_gemm(
                        values, counts, weights, output_dtype=dtype
                    )
                    torch.testing.assert_close(
                        actual,
                        torch.full((2, 32), expected, dtype=dtype, device=device),
                        rtol=0,
                        atol=0,
                    )
                    empty = kernel.grouped_fp4_gemm(
                        values[:0], counts * 0, weights, output_dtype=dtype
                    )
                    self.assertEqual((empty.shape, empty.dtype), ((0, 32), dtype))
            default = kernel.grouped_fp4_gemm(values, counts, weights)
            self.assertEqual(default.dtype, torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "output dtype"):
                kernel.grouped_fp4_gemm(
                    values, counts, weights, output_dtype=torch.float16
                )

    def test_moe_tp4_rounds_complete_down_outputs_before_six_route_merge(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device), patch.object(model, "world_size", 4):
                expert = model.Expert(
                    128, 256, dtype=torch.float4_e2m1fn_x2, tensor_parallel=True
                ).to(device)
                self.assertEqual(expert.w2.in_features, 256)
                gate_raw = torch.zeros((64, 64), dtype=torch.uint8, device=device)
                gate_raw[0, 0] = 6
                up_raw = torch.zeros_like(gate_raw)
                up_raw[0, 0] = 2
                down_raw = torch.zeros((32, 128), dtype=torch.uint8, device=device)
                down_raw[:, 0] = down_raw[:, 32] = 2
                down_raw[:, 64] = 10
                down_scales = torch.ones((32, 8), device=device)
                down_scales[:, 0] = 1 / 4
                down_scales[:, 2] = 1 / 1024
                down_scales[:, 4] = 1 / 4096
                with torch.no_grad():
                    expert.w1.weight.copy_(gate_raw.view(torch.float4_e2m1fn_x2))
                    expert.w3.weight.copy_(up_raw.view(torch.float4_e2m1fn_x2))
                    expert.w2.weight.copy_(down_raw.view(torch.float4_e2m1fn_x2))
                    expert.w1.scale.fill_(1)
                    expert.w3.scale.fill_(1)
                    expert.w2.scale.copy_(down_scales.to(torch.float8_e8m0fnu))
                moe = SimpleNamespace(
                    experts=[expert] * 6,
                    experts_start_idx=0,
                    experts_end_idx=6,
                    n_routed_experts=6,
                    tensor_parallel=True,
                    expert_parallel=False,
                    gate=lambda x, mask, device=device: (
                        torch.ones((1, 6), device=device),
                        torch.arange(6, device=device).view(1, 6),
                    ),
                    shared_experts=torch.zeros_like,
                )
                moe.prepare_routed_experts = MethodType(
                    model.MoE.prepare_routed_experts, moe
                )
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.zeros((1, 128), dtype=torch.bfloat16, device=device)
                values[:, 0] = 1

                def gather(output, value):
                    output.copy_(value.repeat(4, 1))

                with patch.object(
                    model.dist, "all_gather_into_tensor", side_effect=gather
                ):
                    actual = model.MoE._forward_rows(moe, values)
                # Each complete expert gives 1 + 1/256 - 1/1024, rounded to 1.
                # Rounding only after the six-route merge would give 6.03125.
                torch.testing.assert_close(
                    actual, torch.full_like(values, 6), rtol=0, atol=0
                )

    def test_tp4_fallback_gathers_product_and_merged_output_channels(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device), patch.object(model, "world_size", 4):
                expert = model.Expert(
                    4, 128, dtype=torch.bfloat16, tensor_parallel=True
                ).to(device)
                self.assertEqual(expert.w2.in_features, 128)
                with torch.no_grad():
                    for parameter in expert.parameters():
                        parameter.zero_()
                    expert.w1.weight[0, 0] = expert.w3.weight[0, 0] = 1
                    expert.w2.weight[:, ::32] = 1
                moe = SimpleNamespace(
                    experts=[expert] * 4,
                    experts_start_idx=0,
                    experts_end_idx=4,
                    n_routed_experts=4,
                    tensor_parallel=True,
                    expert_parallel=False,
                    gate=lambda x, mask, device=device: (
                        torch.ones((2, 1), device=device),
                        torch.zeros((2, 1), dtype=torch.long, device=device),
                    ),
                    shared_experts=torch.zeros_like,
                )
                moe.prepare_routed_experts = MethodType(
                    model.MoE.prepare_routed_experts, moe
                )
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.zeros((2, 4), dtype=torch.bfloat16, device=device)
                values[:, 0] = torch.tensor([1, 2], device=device)

                def gather(output, value):
                    if value.dtype == torch.bfloat16:
                        output.copy_(value.repeat(4, 1))
                    else:
                        output.copy_(
                            torch.cat([value * factor for factor in (1, 2, 3, 4)])
                        )

                with patch.object(
                    model.dist, "all_gather_into_tensor", side_effect=gather
                ):
                    actual = model.MoE._forward_rows(moe, values)
                torch.testing.assert_close(
                    actual,
                    torch.tensor(
                        [
                            [2.921875, 5.84375, 8.75, 11.6875],
                            [14.0625, 28.125, 42.25, 56.25],
                        ],
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    rtol=0,
                    atol=0,
                )

    def test_grouped_tp4_gathers_distinct_channels_with_uneven_routes(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            with self.subTest(device=device), patch.object(model, "world_size", 4):
                up_raw = torch.zeros((64, 64), dtype=torch.uint8, device=device)
                up_raw[0, 0] = 2
                down_raw = torch.zeros((32, 128), dtype=torch.uint8, device=device)
                down_raw[:, ::32] = torch.tensor([2, 4, 5, 6], device=device)
                up = SimpleNamespace(
                    weight=up_raw.view(torch.float4_e2m1fn_x2),
                    scale=torch.ones((64, 4), device=device).to(torch.float8_e8m0fnu),
                )
                down = SimpleNamespace(
                    weight=down_raw.view(torch.float4_e2m1fn_x2),
                    scale=torch.ones((32, 8), device=device).to(torch.float8_e8m0fnu),
                )
                expert = SimpleNamespace(w1=up, w3=up, w2=down, swiglu_limit=0)
                moe = SimpleNamespace(
                    experts=[expert] * 4,
                    experts_start_idx=0,
                    experts_end_idx=4,
                    n_routed_experts=4,
                    tensor_parallel=True,
                    expert_parallel=False,
                    shared_experts=torch.zeros_like,
                )
                moe.prepare_routed_experts = MethodType(
                    model.MoE.prepare_routed_experts, moe
                )
                moe._grouped_routed = MethodType(model.MoE._grouped_routed, moe)
                values = torch.ones((3, 128), dtype=torch.bfloat16, device=device)
                weights = torch.ones((3, 2), device=device)
                indices = torch.tensor([[0, 1], [0, 3], [0, 1]], device=device)

                def gather(output, value):
                    output.copy_(torch.cat([value * factor for factor in (1, 2, 3, 4)]))

                with patch.object(
                    model.dist, "all_gather_into_tensor", side_effect=gather
                ):
                    actual = model.MoE._forward_rows(
                        moe, values, routing=(weights, indices)
                    )
                # Quantized channel values are .75, 1.5, 2.25, 3; down weights
                # are 1, 2, 3, 4. Each expert gives 22.5 before two-route merging.
                expected = torch.cat(
                    [
                        torch.full((3, 32), value, dtype=torch.bfloat16, device=device)
                        for value in (45, 90, 135, 180)
                    ],
                    dim=-1,
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ProjectionPrecisionCuda(unittest.TestCase):
    def test_dense_prefill_graph_keeps_new_quantization_storage_and_k32_order(self):
        values = torch.zeros((129, 128), dtype=torch.bfloat16, device="cuda")
        values[:, (0, 32, 64)] = 1
        weight = torch.zeros((1280, 128), device="cuda")
        weight[:, (0, 32)] = 1
        weight[:, 64] = -1
        weight[640:, 32:] = 0
        weight = weight.to(torch.float8_e4m3fn)
        scales = torch.ones((40, 4), device="cuda")
        scales[:, 1] = 2**-25
        scales = scales.to(torch.float8_e8m0fnu)

        def run():
            encoded, input_scales = kernel.act_quant(
                values, 32, "ue8m0", torch.float8_e8m0fnu
            )
            return kernel.fp8_gemm(
                encoded,
                input_scales,
                weight,
                scales,
                block_size=32,
                output_dtype=torch.float32,
            )

        run()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        try:
            with torch.cuda.graph(graph, stream=stream):
                output = run()
            for value in (1, 2, 0.5):
                values[:, (0, 32, 64)] = value
                graph.replay()
                expected = run()
                self.assertTrue(
                    torch.equal(output.view(torch.uint8), expected.view(torch.uint8))
                )
                self.assertEqual(output[:, :640].count_nonzero().item(), 0)
                self.assertTrue(
                    torch.equal(
                        output[:, 640:], torch.full_like(output[:, 640:], value)
                    )
                )
        finally:
            stream.synchronize()
            graph.reset()

    def test_routed_gate_up_share_quantization_and_replay_exactly(self):
        original = kernel.grouped_fp4_gemm

        def separate_quantization(*args, **kwargs):
            kwargs.pop("quantized", None)
            return original(*args, **kwargs)

        for strategy in ("TP4", "TEP4", "DEP4"):
            with self.subTest(strategy=strategy), patch.object(model, "world_size", 1):
                moe, values, _ = prepared_moe_literal("cuda", strategy)
                with torch.no_grad():
                    for index, expert in enumerate(moe.experts):
                        expert.w2.scale.fill_(1 << index)
                values = values.repeat(3, 1)
                tokens = torch.arange(3, device="cuda")
                coefficients = torch.ones(3, device="cuda")
                counts = torch.tensor((1, 0, 2, 0), device="cuda", dtype=torch.int32)

                host_counts = None if strategy == "TP4" else [1, 0, 2, 0]

                def run(
                    moe=moe,
                    values=values,
                    tokens=tokens,
                    coefficients=coefficients,
                    counts=counts,
                    host_counts=host_counts,
                ):
                    return moe._grouped_routed(
                        values, tokens, coefficients, host_counts, counts
                    )[1]

                run()
                with patch.object(
                    kernel, "grouped_fp4_gemm", side_effect=separate_quantization
                ):
                    expected = run()
                quantize = Mock(wraps=kernel.act_quant)
                with (
                    patch.object(kernel, "act_quant", quantize),
                    patch.object(model, "act_quant", quantize),
                ):
                    actual = run()
                self.assertEqual(quantize.call_count, 2)
                self.assertTrue(
                    torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
                )
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        replayed = run()
                    values[:, 0].fill_(2)
                    counts.copy_(
                        torch.tensor((0, 2, 0, 1), device="cuda", dtype=torch.int32)
                    )
                    if host_counts is not None:
                        host_counts[:] = (0, 2, 0, 1)
                    graph.replay()
                    with patch.object(
                        kernel, "grouped_fp4_gemm", side_effect=separate_quantization
                    ):
                        expected = run()
                    self.assertTrue(
                        torch.equal(
                            replayed.view(torch.uint8), expected.view(torch.uint8)
                        )
                    )
                finally:
                    stream.synchronize()
                    graph.reset()
                prepared = moe._routed_state.weights.gate
                encoded, scales = kernel.act_quant(
                    values, 32, "ue8m0", torch.float8_e8m0fnu
                )
                for quantized in ((encoded[:1], scales), (encoded, scales[:1])):
                    with self.assertRaisesRegex(ValueError, "matching FP8 and K32"):
                        kernel.grouped_fp4_gemm(
                            values, counts, prepared, quantized=quantized
                        )

    def test_fp4_projection_decodes_every_packed_four_value_word(self):
        encoded = torch.zeros((4, 32), dtype=torch.float32, device="cuda")
        encoded[:, :4] = torch.eye(4, device="cuda")
        encoded = encoded.to(torch.float8_e4m3fn)
        input_scales = torch.ones((4, 1), device="cuda").to(torch.float8_e8m0fnu)
        levels = torch.tensor(
            (0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6),
            device="cuda",
        )
        weight_scales = torch.ones((32768, 1), device="cuda").to(torch.float8_e8m0fnu)
        for first in (0, 32768):
            words = torch.arange(first, first + 32768, device="cuda")
            raw = torch.zeros((32768, 16), dtype=torch.uint8, device="cuda")
            raw[:, 0] = (words & 255).to(torch.uint8)
            raw[:, 1] = (words >> 8).to(torch.uint8)
            expected = levels[
                (words[None, :] >> (4 * torch.arange(4, device="cuda")[:, None])) & 15
            ]
            for dtype in (torch.bfloat16, torch.float32):
                with self.subTest(first=first, dtype=dtype):
                    actual = kernel.fp4_gemm(
                        encoded,
                        input_scales,
                        raw.view(torch.float4_e2m1fn_x2),
                        weight_scales,
                        act_block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        actual, expected.to(dtype), rtol=0, atol=0
                    )
                    grouped_encoded = encoded.repeat(17, 1)
                    grouped = kernel.grouped_fp4_gemm(
                        grouped_encoded.to(torch.bfloat16),
                        torch.tensor((0, 68), device="cuda", dtype=torch.int32),
                        [(raw.view(torch.float4_e2m1fn_x2), weight_scales)] * 2,
                        output_dtype=dtype,
                        quantized=(grouped_encoded, input_scales.repeat(17, 1)),
                    )
                    torch.testing.assert_close(
                        grouped, expected.to(dtype).repeat(17, 1), rtol=0, atol=0
                    )

    def test_fp4_scales_each_k32_dot_before_accumulation(self):
        # Released FP32 and BF16 results are 646 and 648. Power-of-two scaling
        # preserves these rounded values without overflow or underflow.
        for rows in (1, 16, 17, 31, 32, 33, 63, 64, 65, 125, 126):
            encoded, scales, weight, weight_scales, factors = fp4_literal(rows)
            for dtype, value in ((torch.float32, 646), (torch.bfloat16, 648)):
                with self.subTest(rows=rows, dtype=dtype):
                    output = kernel.fp4_gemm(
                        encoded,
                        scales,
                        weight,
                        weight_scales,
                        torch.float8_e8m0fnu,
                        act_block_size=32,
                        output_dtype=dtype,
                    )
                    torch.testing.assert_close(
                        output, (factors * value).to(dtype), rtol=0, atol=0
                    )

    def test_complete_down_k_keeps_small_terms_in_source_order(self):
        values = torch.zeros((3, 2304), dtype=torch.bfloat16, device="cuda")
        values[:, (0, 576, 1152)] = 1
        raw = torch.zeros((1280, 1152), dtype=torch.uint8, device="cuda")
        raw[:, 0] = raw[:, 288] = 2
        raw[:, 576] = 10
        weight = raw.view(torch.float4_e2m1fn_x2)
        scales = torch.ones((1280, 72), device="cuda")
        scales[:, 18] = 2**-25
        scales = scales.to(torch.float8_e8m0fnu)
        # FP32 source order is (1 + 2**-25) - 1 = 0, not 2**-25.
        actual = kernel.grouped_fp4_gemm(
            values,
            torch.tensor([1, 0, 2], dtype=torch.int32, device="cuda"),
            [(weight, scales)] * 3,
        )
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)
        encoded, input_scales = kernel.act_quant(
            values, 32, "ue8m0", torch.float8_e8m0fnu
        )
        actual = kernel.fp4_gemm(
            encoded, input_scales, weight, scales, act_block_size=32
        )
        torch.testing.assert_close(actual, torch.zeros_like(actual), rtol=0, atol=0)

    def test_grouped_fp4_rounding_replays_with_changed_counts(self):
        for dtype, value in ((torch.bfloat16, 648), (torch.float32, 646)):
            for counts, changed in (
                ((1, 0, 33, 2), (16, 1, 0, 19)),
                ((1, 0, 3, 2), (6, 0, 0, 0)),
                ((1, 0, 13, 2), (0, 16, 0, 0)),
                ((1, 0, 14, 2), (0, 0, 17, 0)),
                ((1, 0, 257, 2), (128, 1, 0, 131)),
                ((1, 0, 1025, 2), (0, 1028, 0, 0)),
            ):
                with self.subTest(dtype=dtype, counts=counts):
                    self.check_grouped_replay(dtype, value, counts, changed)

    def test_padded_grouped_projection_replays_device_live_rows_and_zeroes_tail(self):
        module = kernel._projection_module()
        for dtype in (torch.bfloat16, torch.float32):
            for capacity in (36, 260):
                case = grouped_literal(
                    (1, 0, capacity - 3, 2),
                    max_rows=capacity - 3,
                    dtype=dtype,
                    width=128 if capacity > 64 else 64,
                )
                status = torch.zeros((1,), dtype=torch.int32, device="cuda")
                encoded, scales = case.encoded.clone(), case.scales.clone()

                def run(case=case, status=status, dtype=dtype):
                    return kernel.grouped_fp4_gemm_padded(
                        case.encoded,
                        case.scales,
                        case.counts,
                        case.weights,
                        pointers=case.pointers,
                        status=status,
                        max_rows=case.max_rows,
                        output_dtype=dtype,
                        output=case.output,
                        scratch=case.scratch,
                    )

                run()
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                try:
                    with torch.cuda.graph(graph, stream=stream):
                        run()
                    valid = [(0, 0, 0, 0), (1, 0, 0, 0), (1, 0, 3, 2), (16, 1, 0, 19)]
                    if capacity > 64:
                        valid += [(32, 32, 0, 0), (32, 33, 0, 0), (128, 1, 0, 131)]
                    valid += [(1, 0, capacity - 3, 2)]
                    for counts in valid:
                        live = sum(counts)
                        for execution in ("eager", "graph"):
                            with self.subTest(
                                dtype=dtype,
                                capacity=capacity,
                                counts=counts,
                                execution=execution,
                            ):
                                with torch.cuda.stream(stream):
                                    case.counts.copy_(
                                        torch.tensor(
                                            counts, dtype=torch.int32, device="cuda"
                                        )
                                    )
                                    case.encoded.copy_(encoded)
                                    case.scales.copy_(scales)
                                    case.encoded[live:].fill_(127)
                                    case.scales[live:].fill_(255)
                                    case.output.fill_(-7)
                                    status.zero_()
                                    graph.replay() if execution == "graph" else run()
                                    actual = case.output.clone()
                                stream.synchronize()
                                kernel.require_padded_status(status)
                                reference = torch.empty(
                                    (live, 128), dtype=dtype, device="cuda"
                                )
                                module.grouped_project(
                                    case.encoded[:live],
                                    case.scales[:live],
                                    *case.pointers,
                                    case.counts,
                                    reference,
                                    case.scratch,
                                    case.max_rows,
                                )
                                expected = torch.zeros_like(actual)
                                expected[:live] = reference
                                self.assertTrue(
                                    torch.equal(
                                        actual.view(torch.uint8),
                                        expected.view(torch.uint8),
                                    )
                                )
                    invalid = (
                        (-1, 2, 2, 0),
                        (capacity // 2, capacity - capacity // 2 + 1, 0, 0),
                        (capacity - 2, 0, 0, 0),
                        (2**31 - 1, 2**31 - 1, 0, 0),
                    )
                    for counts in invalid:
                        for execution in ("eager", "graph"):
                            with self.subTest(
                                dtype=dtype,
                                capacity=capacity,
                                invalid=counts,
                                execution=execution,
                            ):
                                with torch.cuda.stream(stream):
                                    case.counts.copy_(
                                        torch.tensor(
                                            counts, dtype=torch.int32, device="cuda"
                                        )
                                    )
                                    status.zero_()
                                    graph.replay() if execution == "graph" else run()
                                stream.synchronize()
                                with self.assertRaisesRegex(
                                    RuntimeError, "device metadata"
                                ):
                                    kernel.require_padded_status(status)
                                self.assertEqual(case.output.count_nonzero().item(), 0)
                    case.counts.copy_(torch.tensor((1, 0, 0, 0), device="cuda"))
                    status.fill_(8)
                    run()
                    self.assertEqual(status.item(), 8)
                    self.assertEqual(case.output.count_nonzero().item(), 0)
                    with self.assertRaisesRegex(ValueError, "shape and dtype"):
                        kernel.grouped_fp4_gemm_padded(
                            case.encoded,
                            case.scales,
                            case.counts,
                            case.weights,
                            pointers=case.pointers,
                            status=status,
                            output=torch.empty(
                                (capacity, 256), dtype=dtype, device="cuda"
                            ),
                            output_dtype=dtype,
                        )
                    for bad_weight, bad_scale in (
                        (case.weights[-1][0][:32], case.weights[-1][1][:32]),
                        (case.weights[-1][0][:, :16], case.weights[-1][1]),
                        (case.weights[-1][0], case.weights[-1][1][:, :1]),
                    ):
                        with (
                            self.subTest(bad_weight=bad_weight.shape),
                            self.assertRaisesRegex(ValueError, "shapes must match"),
                        ):
                            kernel.grouped_fp4_gemm_padded(
                                case.encoded,
                                case.scales,
                                case.counts,
                                [*case.weights[:-1], (bad_weight, bad_scale)],
                                pointers=case.pointers,
                                status=status,
                                output_dtype=dtype,
                            )
                finally:
                    stream.synchronize()
                    graph.reset()

    def test_padded_grouped_zero_capacity_still_rejects_live_counts(self):
        case = grouped_literal((0, 0, 0, 0), max_rows=1)
        status = torch.zeros((1,), dtype=torch.int32, device="cuda")
        module = kernel._projection_module()
        module.grouped_project_padded(
            case.encoded,
            case.scales,
            *case.pointers,
            case.counts,
            case.output,
            case.scratch,
            case.max_rows,
            status,
        )
        kernel.require_padded_status(status)
        case.counts[0] = 1
        module.grouped_project_padded(
            case.encoded,
            case.scales,
            *case.pointers,
            case.counts,
            case.output,
            case.scratch,
            case.max_rows,
            status,
        )
        with self.assertRaisesRegex(RuntimeError, "device metadata"):
            kernel.require_padded_status(status)

    def check_grouped_replay(self, dtype, value, counts, changed):
        module = kernel._projection_module()
        case = grouped_literal(
            counts,
            max_rows=max(33, sum(counts)),
            dtype=dtype,
            width=128 if sum(counts) >= 256 else 64,
        )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run_grouped(module, case)
            eager = case.output.clone()
        stream.synchronize()
        shifts = torch.repeat_interleave(
            torch.tensor([-1, 0, 1, 2], device="cuda"),
            torch.tensor(counts, device="cuda"),
        )
        expected = (case.factors * torch.exp2(shifts[:, None].float()) * value).to(
            dtype
        )
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, stream=stream):
                run_grouped(module, case)
            for replay_counts in (counts, changed):
                with self.subTest(counts=replay_counts), torch.cuda.stream(stream):
                    case.counts.copy_(
                        torch.tensor(replay_counts, device="cuda", dtype=torch.int32)
                    )
                    case.output.fill_(-7)
                    graph.replay()
                    actual = case.output.clone()
                    finished = torch.cuda.Event()
                    finished.record()
                finished.synchronize()
                group_shifts = torch.repeat_interleave(
                    torch.tensor([-1, 0, 1, 2], device="cuda"),
                    torch.tensor(replay_counts, device="cuda"),
                )
                expected = (
                    case.factors * torch.exp2(group_shifts[:, None].float()) * value
                ).to(dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            stream.synchronize()
            graph.reset()

    def test_grouped_output_cannot_alias_encoded_input(self):
        module = kernel._projection_module()
        for dtype in (torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                case = grouped_literal((1, 0, 2, 0), max_rows=2, dtype=dtype)
                storage = torch.empty(
                    3 * 128 * dtype.itemsize, dtype=torch.uint8, device="cuda"
                )
                case.encoded = storage[: 3 * 64].view(3, 64)
                alias = storage.view(dtype).view(3, 128)
                before = storage.clone()
                with self.assertRaisesRegex(Exception, "operator tensors overlap"):
                    run_grouped(module, case, output=alias)
                torch.testing.assert_close(storage, before, rtol=0, atol=0)

    def test_grouped_rejects_invalid_output_metadata_before_writes(self):
        module = kernel._projection_module()
        case = grouped_literal((1, 0, 2, 0), max_rows=2, dtype=torch.float32)
        for output in (
            torch.full((3, 128), -7, dtype=torch.float16, device="cuda"),
            torch.full((3, 128), -7, dtype=torch.float64, device="cuda"),
            torch.full((3, 128), -7, dtype=torch.int32, device="cuda"),
            torch.full((3, 256), -7, dtype=torch.float32, device="cuda")[:, ::2],
            torch.full((2, 128), -7, dtype=torch.float32, device="cuda"),
            torch.full((3, 128), -7, dtype=torch.float32),
        ):
            with self.subTest(
                dtype=output.dtype, shape=output.shape, device=output.device
            ):
                with self.assertRaisesRegex(Exception, "Check failed"):
                    run_grouped(module, case, output=output)
                torch.testing.assert_close(
                    output, torch.full_like(output, -7), rtol=0, atol=0
                )
        for dtype in (torch.bfloat16, torch.float32):
            empty = grouped_literal((0, 0, 0, 0), max_rows=2, dtype=dtype)
            run_grouped(module, empty)

    def test_grouped_invalid_device_counts_produce_nan(self):
        module = kernel._projection_module()
        for dtype in (torch.bfloat16, torch.float32):
            case = grouped_literal((1, 0, 2, 0), max_rows=2, dtype=dtype)
            for counts in ((0, 0, 0, 0), (-1, 2, 2, 0), (3, 0, 0, 0)):
                with self.subTest(dtype=dtype, counts=counts):
                    case.counts.copy_(
                        torch.tensor(counts, device="cuda", dtype=torch.int32)
                    )
                    case.output.fill_(-7)
                    run_grouped(module, case)
                    self.assertTrue(bool(case.output.isnan().all()))


if __name__ == "__main__":
    unittest.main()
