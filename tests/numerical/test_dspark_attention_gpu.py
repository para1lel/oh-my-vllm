"""DSpark inference-path kernels against rounded-input FP64 references."""

import math

import pytest
import torch
from oh_my_vllm.ir import compile_forward
from oh_my_vllm.ir.dspark import _reference_attention, _reference_norm, _reference_rms
from oh_my_vllm.kernels.dspark_attention import (
    append,
    attention,
    normalize_rope,
    rms_norm,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("rows", [1, 7, 28, 32144, 32768])
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_hidden_rms_checkpoint_rounding(rows, weight_dtype):
    generator = torch.Generator().manual_seed(5120 + rows)
    x = torch.randn(rows, 5120, generator=generator).bfloat16()
    weight = torch.randn(5120, generator=generator).to(weight_dtype)
    expected = _reference_rms(x, weight, 1e-6)
    actual = rms_norm(x.cuda(), weight.cuda())
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=0.03, rtol=0.03
    )


def test_compiled_hidden_rms_capture_replays_updated_inputs():
    x = torch.randn(7, 5120, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(5120, device="cuda", dtype=torch.bfloat16)
    unit = compile_forward(
        lambda rows, multiplier: rms_norm(rows, multiplier), unit="test_dspark_rms"
    )
    unit(x, weight)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = unit(x, weight)
    x.mul_(0.5)
    weight.add_(0.25)
    graph.replay()
    expected = _reference_rms(x.cpu(), weight.cpu(), 1e-6)
    torch.testing.assert_close(
        output.cpu().double(), expected.double(), atol=0.03, rtol=0.03
    )


@pytest.mark.parametrize("heads", [8, 32])
@pytest.mark.parametrize("position", [0, 8192, 262143])
def test_norm_rope_full_rotation_and_strided_projection(heads, position):
    generator = torch.Generator().manual_seed(784 + heads)
    packed = torch.randn(3, heads * 128 + 256, generator=generator).bfloat16().cuda()
    x = packed[:, : heads * 128].view(3, heads, 128)
    weight = torch.randn(128, generator=generator).bfloat16().float().cuda()
    positions = torch.tensor([position, position + 1, position + 2], device="cuda")
    frequency = (1e7 ** (-torch.arange(64, dtype=torch.float32) / 64) / 32).cuda()
    factor = 1 + 0.1 * math.log(32)
    expected = _reference_norm(
        x.cpu(), weight.cpu(), positions.cpu(), frequency.cpu(), factor, 1e-6
    )
    actual = normalize_rope(x, weight, positions, frequency, factor)
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=0.03, rtol=0.03
    )


def _inputs(lengths):
    generator = torch.Generator().manual_seed(784)
    requests = len(lengths)
    pages = sum((length + 783) // 784 for length in lengths)
    cache = torch.full(
        (max(2, pages + 1), 2, 784, 8, 128), float("nan"), dtype=torch.bfloat16
    )
    tables = torch.zeros(
        requests, max(1, (max(lengths) + 783) // 784), dtype=torch.int32
    )
    next_page = 1
    for row, length in enumerate(lengths):
        count = (length + 783) // 784
        mapping = list(range(next_page, next_page + count))[::-1]
        next_page += count
        tables[row, :count] = torch.tensor(mapping, dtype=torch.int32)
        for logical, physical in enumerate(mapping):
            valid = min(784, length - logical * 784)
            cache[physical, :, :valid] = torch.randn(
                2, valid, 8, 128, generator=generator
            ).bfloat16()
    query = torch.randn(requests, 7, 32, 128, generator=generator).bfloat16()
    key = torch.randn(requests, 7, 8, 128, generator=generator).bfloat16()
    value = torch.randn(requests, 7, 8, 128, generator=generator).bfloat16()
    return query, cache, tables, torch.tensor(lengths, dtype=torch.int32), key, value


@pytest.mark.parametrize(
    "lengths", [[0], [1], [783], [784], [785], [4097], [0, 1, 784, 785]]
)
def test_paged_prefix_and_bidirectional_block_ignore_unwritten_cache(lengths):
    inputs = _inputs(lengths)
    expected = _reference_attention(*inputs)
    device = [value.cuda() for value in inputs]
    before = device[1].clone()
    actual = attention(*device)
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=0.03, rtol=0.03
    )
    torch.testing.assert_close(device[1], before, atol=0, rtol=0, equal_nan=True)


def test_compiled_attention_and_context_append_capture_replay():
    inputs = [value.cuda() for value in _inputs([783])]
    query, cache, _tables, lengths, key, value = inputs
    retained_key, retained_value = key[:, 0].contiguous(), value[:, 0].contiguous()
    slots = torch.tensor([784 + 783], device="cuda")

    def run(q, context, pages, counts, block_k, block_v, context_k, context_v, writes):
        append(context, context_k, context_v, writes)
        return attention(q, context, pages, counts, block_k, block_v)

    unit = compile_forward(run, unit="test_dspark_attention")
    args = (*inputs, retained_key, retained_value, slots)
    expected = attention(*inputs).clone()
    actual = unit(*args)
    torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.03)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = unit(*args)
    lengths.fill_(784)
    query.mul_(0.5)
    graph.replay()
    reference = _reference_attention(*(tensor.cpu() for tensor in inputs))
    torch.testing.assert_close(
        output.cpu().double(), reference.double(), atol=0.03, rtol=0.03
    )
    torch.testing.assert_close(cache[1, 0, 783], retained_key[0], atol=0, rtol=0)
    torch.testing.assert_close(cache[1, 1, 783], retained_value[0], atol=0, rtol=0)


@pytest.mark.parametrize("index", [1, 4, 5])
def test_attention_unaligned_contiguous_kv_preserves_canaries(index):
    inputs = [value.cuda() for value in _inputs([1])]
    original = inputs[index]
    storage = torch.full(
        (original.numel() + 2,), 123, device="cuda", dtype=original.dtype
    )
    inputs[index] = storage[1:-1].view(original.shape)
    inputs[index].copy_(original)
    before = storage.clone()
    assert inputs[index].is_contiguous() and inputs[index].data_ptr() % 16
    expected = _reference_attention(*(tensor.cpu() for tensor in inputs))
    actual = attention(*inputs)
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=0.03, rtol=0.03
    )
    torch.testing.assert_close(storage, before, atol=0, rtol=0, equal_nan=True)


def test_attention_maximum_context_uniform_query_uses_all_live_pages():
    length = 262144
    pages = (length + 783) // 784
    cache = torch.full(
        (pages + 1, 2, 784, 8, 128), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    table = torch.arange(pages, 0, -1, device="cuda", dtype=torch.int32)[None]
    total = 0.0
    for logical in range(pages):
        valid = min(784, length - logical * 784)
        physical = pages - logical
        value = float(torch.tensor((logical % 17) / 16).bfloat16())
        cache[physical, 0, :valid] = 0
        cache[physical, 1, :valid] = value
        total += value * valid
    query = torch.zeros(1, 7, 32, 128, device="cuda", dtype=torch.bfloat16)
    block_key = torch.zeros(1, 7, 8, 128, device="cuda", dtype=torch.bfloat16)
    block_value = torch.ones_like(block_key)
    actual = attention(
        query,
        cache,
        table,
        torch.tensor([length], device="cuda", dtype=torch.int32),
        block_key,
        block_value,
    )
    expected = torch.full_like(actual, (total + 7) / (length + 7))
    torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.003)


@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("unaligned", ["input", "weight"])
def test_hidden_rms_unaligned_view_keeps_storage(weight_dtype, unaligned):
    generator = torch.Generator().manual_seed(107)
    arguments = [
        torch.randn(7, 5120, generator=generator).bfloat16().cuda(),
        torch.randn(5120, generator=generator).to(weight_dtype).cuda(),
    ]
    index = 0 if unaligned == "input" else 1
    original = arguments[index]
    storage = torch.full(
        (original.numel() + 2,), 123, dtype=original.dtype, device="cuda"
    )
    arguments[index] = storage[1:-1].view(original.shape)
    arguments[index].copy_(original)
    before = storage.clone()
    expected = _reference_rms(*(x.cpu() for x in arguments), 1e-6)
    actual = rms_norm(*arguments)
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=0.03, rtol=0.03
    )
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


@pytest.mark.parametrize("rows", [3, 1371])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_norm_rope_generic_heads_unaligned_strided_tail(rows, index_dtype):
    generator = torch.Generator().manual_seed(115)
    width = 3 * 128 + 256
    storage = torch.full((rows * width + 2,), 123, dtype=torch.bfloat16, device="cuda")
    packed = storage[1:-1].view(rows, width)
    packed.copy_(torch.randn(rows, width, generator=generator).bfloat16())
    x = packed[:, : 3 * 128].view(rows, 3, 128)
    before = storage.clone()
    weight = torch.randn(128, generator=generator).bfloat16().float().cuda()
    positions = torch.arange(262144 - rows, 262144, dtype=index_dtype, device="cuda")
    frequency = (1e7 ** (-torch.arange(64, dtype=torch.float32) / 64) / 32).cuda()
    factor = 1 + 0.1 * math.log(32)
    expected = _reference_norm(
        x.cpu(), weight.cpu(), positions.cpu(), frequency.cpu(), factor, 1e-6
    )
    actual = normalize_rope(x, weight, positions, frequency, factor)
    torch.testing.assert_close(
        actual.cpu().double(), expected.double(), atol=0.03, rtol=0.03
    )
    torch.testing.assert_close(storage, before, atol=0, rtol=0)


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "rows,layout",
    [
        (7, "aligned"),
        (4101, "aligned"),
        (7, "unaligned"),
        (4101, "unaligned"),
        (7, "alias"),
        (7, "wide_cache"),
    ],
)
def test_append_layout_tail_negative_slots_and_canaries(rows, layout, index_dtype):
    from oh_my_vllm.ir.dspark import _reference_append

    generator = torch.Generator().manual_seed(163)
    pages = (784 + rows + 783) // 784 + 1
    if layout == "wide_cache":
        pages = 84
    shape = (pages, 2, 784, 8, 128)
    storages = []

    def allocate(shape):
        if layout != "unaligned":
            return torch.empty(shape, dtype=torch.bfloat16, device="cuda")
        storage = torch.full(
            (math.prod(shape) + 2,), 123, dtype=torch.bfloat16, device="cuda"
        )
        storages.append(storage)
        return storage[1:-1].view(shape)

    cache = allocate(shape)
    cache.fill_(-17)
    if layout == "alias":
        key, value = cache[0, 0, :rows], cache[0, 1, :rows]
    else:
        key, value = allocate((rows, 8, 128)), allocate((rows, 8, 128))
    key.copy_(torch.randn(rows, 8, 128, generator=generator).bfloat16())
    value.copy_(torch.randn(rows, 8, 128, generator=generator).bfloat16())
    base = 65536 if layout == "wide_cache" else 784
    slots = torch.arange(base, base + rows, dtype=index_dtype, device="cuda")
    slots[0], slots[-1] = -1, -2
    key_before, value_before = key.clone(), value.clone()
    expected = cache.cpu().clone()
    _reference_append(expected, key.cpu(), value.cpu(), slots.cpu())
    append(cache, key, value, slots)
    torch.testing.assert_close(cache.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(key, key_before, atol=0, rtol=0)
    torch.testing.assert_close(value, value_before, atol=0, rtol=0)
    for storage in storages:
        torch.testing.assert_close(
            storage[[0, -1]],
            torch.full((2,), 123, dtype=torch.bfloat16, device="cuda"),
            atol=0,
            rtol=0,
        )
