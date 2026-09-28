"""CPU checks for the formal CUDA dispatch gate."""

import pytest
import torch

from development.kernels.variant import verify_fast_dispatch


@pytest.mark.parametrize(
    "operation", ("norm", "add_norm", "gated_norm", "qk", "recurrent", "convolution")
)
def test_formal_variant_gate_requires_exactly_one_fast_dispatch(operation):
    before = {operation: {"fast": 4, "generic": 2}}
    after = {operation: {"fast": 5, "generic": 2}}
    assert verify_fast_dispatch(operation, before, after) == {
        "operation": operation,
        "fast": 1,
        "generic": 0,
        "passed": True,
    }
    for counts in ({"fast": 4, "generic": 3}, {"fast": 5, "generic": 3}):
        with pytest.raises(AssertionError, match="expected exactly one fast"):
            verify_fast_dispatch(operation, before, {operation: counts})


def test_untracked_formal_operation_does_not_claim_fast_dispatch():
    assert verify_fast_dispatch("prepare_attention", {}, {}) is None


@pytest.mark.gpu
def test_cuda_variant_counters_track_fast_and_generic_launches():
    from oh_my_vllm.kernels.attention import append
    from oh_my_vllm.kernels.convolution import causal_conv
    from oh_my_vllm.kernels.cuda_backend import compiled, variant_launch_counts
    from oh_my_vllm.kernels.gdn import normalize_qk, recurrent
    from oh_my_vllm.kernels.normalization import add_rms_norm, rms_norm

    from development.kernels.fixtures import fixture

    def assert_variant(operation, before, selected):
        after = variant_launch_counts()
        for name, counts in before.items():
            for kind in ("fast", "generic"):
                assert after[name][kind] - counts[kind] == int(
                    name == operation and kind == selected
                )

    for operation, config in (
        ("norm", {"operation": "norm", "tokens": 1, "width": 5120}),
        ("add_norm", {"operation": "add_norm", "tokens": 1, "width": 5120}),
        (
            "gated_norm",
            {"operation": "gated_norm", "tokens": 1, "heads": 48, "width": 128},
        ),
        ("qk", {"operation": "qk", "tokens": 16}),
        (
            "recurrent",
            {"operation": "recurrent", "counts": (1,), "state_dtype": "bfloat16"},
        ),
        ("convolution", {"operation": "convolution", "counts": (1,)}),
    ):
        _, candidate = fixture(config)
        before = variant_launch_counts()
        candidate()
        assert verify_fast_dispatch(operation, before, variant_launch_counts())

    device = "cuda"
    bf16 = torch.bfloat16
    cache = torch.zeros((3, 2, 784, 4, 256), device=device, dtype=bf16)
    key = torch.ones((1, 4, 256), device=device, dtype=bf16)
    value = torch.ones_like(key) * 2
    slots = torch.tensor([784], device=device, dtype=torch.int32)
    before = variant_launch_counts()
    append(cache, key, value, slots)
    assert_variant("append", before, "fast")
    torch.testing.assert_close(cache[1, 0, 0], key[0])
    torch.testing.assert_close(cache[1, 1, 0], value[0])

    storage = torch.empty(key.numel() + 1, device=device, dtype=bf16)
    unaligned_key = storage[1:].view_as(key).copy_(key)
    assert unaligned_key.data_ptr() % 16 != 0
    before = variant_launch_counts()
    append(cache, unaligned_key, value, slots)
    assert_variant("append", before, "generic")

    narrow = torch.zeros((1, 128), device=device, dtype=bf16)
    weight = torch.ones(128, device=device)
    before = variant_launch_counts()
    assert not rms_norm(narrow, weight).count_nonzero().item()
    assert_variant("norm", before, "generic")

    before = variant_launch_counts()
    assert not rms_norm(narrow, weight, gate=narrow).count_nonzero().item()
    assert_variant("gated_norm", before, "generic")

    qk_input = torch.zeros((1, 16, 128), device=device, dtype=bf16)
    before = variant_launch_counts()
    q_out, k_out = normalize_qk(qk_input, qk_input.clone())
    assert_variant("qk", before, "generic")
    assert not q_out.count_nonzero().item() and not k_out.count_nonzero().item()

    before = variant_launch_counts()
    summed, normalized = add_rms_norm(narrow, narrow.clone(), weight)
    assert_variant("add_norm", before, "generic")
    assert not summed.count_nonzero().item() and not normalized.count_nonzero().item()

    q = torch.zeros((1, 1, 128), device=device, dtype=bf16)
    state = torch.zeros((3, 1, 128, 128), device=device, dtype=bf16)
    starts = torch.tensor([0, 1], device=device, dtype=torch.int32)
    reads = torch.tensor([0], device=device, dtype=torch.int32)
    writes = torch.tensor([1], device=device, dtype=torch.int32)
    gates = torch.zeros((1, 1), device=device)
    before = variant_launch_counts()
    result = recurrent(q, q, q, gates, gates, state, starts, reads, writes)
    assert_variant("recurrent", before, "generic")
    assert not result.count_nonzero().item()

    conv_weight = torch.zeros((128, 4), device=device, dtype=bf16)
    conv_state = torch.zeros((3, 128, 3), device=device, dtype=bf16)
    before = variant_launch_counts()
    result = causal_conv(narrow, conv_weight, conv_state, reads, starts, reads, writes)
    assert_variant("convolution", before, "generic")
    assert not result.count_nonzero().item()

    with pytest.raises(Exception, match="unknown CUDA variant operation"):
        compiled().variant_launch_count(7, True)
