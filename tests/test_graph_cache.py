"""Graph admission, eviction, and capture ownership invariants."""

import gc
import os
import weakref
from unittest.mock import Mock, patch

import pytest
import torch
from oh_my_vllm.worker.graph_cache import GraphCache
from oh_my_vllm.worker.mtp import MTP


def add(cache, family, key):
    assert cache.should_use(family, (key,))
    return cache.get_or_create(family, (key,), lambda: object())


def test_lru_hit_probation_eviction_and_recent_recapture():
    synchronize = Mock()
    cache = GraphCache(2, admission_hits=2, admission_factor=0, synchronize=synchronize)
    first = add(cache, "target", "first")
    second = add(cache, "target", "second")
    assert cache.should_use("target", ("first",))
    assert cache.get_or_create("target", ("first",), lambda: None) is first
    assert not cache.should_use("target", ("third",))
    third = add(cache, "target", "third")
    assert third is not first and third is not second
    assert cache.contains("target", ("first",))
    assert not cache.contains("target", ("second",))
    synchronize.assert_called_once_with()

    assert not cache.should_use("target", ("second",))
    add(cache, "target", "second")
    assert cache.snapshot()["target_recent_recapture"] == 1
    assert cache.snapshot()["target_budget_eager"] == 2
    assert cache.snapshot()["target_eviction"] == 2
    assert cache.snapshot()["target_capture"] == 4
    assert cache.snapshot()["target_hit"] == 1


def test_default_frequency_admission_protects_hot_residents():
    cache = GraphCache(2, synchronize=Mock())
    add(cache, "target", "hot")
    add(cache, "target", "cold")
    for _ in range(12):
        assert cache.should_use("target", ("hot",))
    for _ in range(3):
        assert not cache.should_use("target", ("candidate",))
    assert cache.should_use("target", ("candidate",))
    add_graph = cache.get_or_create("target", ("candidate",), object)
    assert add_graph is not None
    assert cache.contains("target", ("hot",))
    assert cache.contains("target", ("candidate",))
    assert not cache.contains("target", ("cold",))
    assert cache.snapshot()["target_budget_eager"] == 3


def test_mtp_floor_protects_proposal_and_draft_families():
    cache = GraphCache(
        32,
        family_floors={"draft": 16, "proposal": 4},
        admission_hits=2,
        admission_factor=0,
        synchronize=Mock(),
    )
    for i in range(4):
        add(cache, "proposal", i)
    for i in range(28):
        add(cache, "draft", i)
    assert not cache.should_use("proposal", (4,))
    add(cache, "proposal", 4)
    assert cache.snapshot()["proposal_resident"] == 5
    assert cache.snapshot()["draft_resident"] == 27
    assert cache.contains("proposal", (0,))
    assert not cache.contains("draft", (0,))

    cache = GraphCache(
        32,
        family_floors={"draft": 16, "proposal": 4},
        admission_hits=2,
        admission_factor=0,
        synchronize=Mock(),
    )
    for i in range(16):
        add(cache, "draft", i)
    for i in range(16):
        add(cache, "proposal", i)
    assert not cache.should_use("draft", (16,))
    add(cache, "draft", 16)
    assert cache.snapshot()["draft_resident"] == 17
    assert cache.snapshot()["proposal_resident"] == 15
    assert cache.contains("draft", (0,))
    assert not cache.contains("proposal", (0,))


def test_probation_history_and_capture_history_are_bounded():
    cache = GraphCache(
        1,
        probation_capacity=2,
        recent_capacity=2,
        admission_hits=2,
        admission_factor=0,
        synchronize=Mock(),
    )
    add(cache, "target", 0)
    for i in range(1, 4):
        assert not cache.should_use("target", (i,))
    assert list(cache.probation) == [("target", (2,)), ("target", (3,))]
    assert cache.snapshot()["probation_eviction"] == 1
    add(cache, "target", 3)
    assert ("target", (3,)) not in cache.probation
    add(cache, "target", 2)
    assert cache.snapshot()["capture_history_overflow"] == 1
    assert len(cache.recent_captures) == 2


def test_failed_capture_does_not_register_graph_or_false_capture_count():
    synchronize = Mock()
    cache = GraphCache(
        1,
        admission_hits=2,
        admission_factor=0,
        failure_cooldown=2,
        synchronize=synchronize,
    )
    old = add(cache, "target", 0)
    assert not cache.should_use("target", (1,))
    assert cache.should_use("target", (1,))
    with pytest.raises(RuntimeError, match="capture failed"):
        cache.get_or_create(
            "target",
            (1,),
            lambda: (_ for _ in ()).throw(RuntimeError("capture failed")),
        )
    synchronize.assert_called_once_with()
    assert not cache.contains("target", (1,))
    assert ("target", (1,)) in cache.probation
    assert cache.snapshot()["target_capture"] == 1
    assert cache.snapshot()["target_eviction"] == 0
    assert cache.snapshot()["resident"] == 1
    assert cache.get_or_create("target", (0,), lambda: None) is old
    assert list(cache.graphs) == [("target", (0,))]
    assert not cache.should_use("target", (1,))
    assert not cache.should_use("target", (1,))
    assert cache.should_use("target", (1,))
    cache.get_or_create("target", (1,), lambda: object())
    assert cache.snapshot()["target_capture"] == 2
    assert cache.snapshot()["target_eviction"] == 1
    assert cache.snapshot()["target_capture_failure_eager"] == 2


def test_low_headroom_runs_eager_without_synchronizing_or_evicting():
    free = [GraphCache.MIN_CAPTURE_FREE_BYTES]
    synchronize = Mock()
    cache = GraphCache(
        1,
        admission_hits=2,
        admission_factor=0,
        synchronize=synchronize,
        free_bytes=lambda: free[0],
    )
    old = add(cache, "draft", 0)
    free[0] = 0
    assert not cache.should_use("draft", (1,))
    assert not cache.should_use("draft", (1,))
    assert cache.snapshot()["draft_headroom_eager"] == 2
    assert cache.get_or_create("draft", (0,), object) is old
    synchronize.assert_not_called()
    free[0] = GraphCache.MIN_CAPTURE_FREE_BYTES
    assert not cache.should_use("draft", (1,))
    assert cache.should_use("draft", (1,))
    cache.get_or_create("draft", (1,), object)
    synchronize.assert_called_once_with()


def test_low_headroom_rejects_first_capture_before_cache_is_full():
    free = [GraphCache.MIN_CAPTURE_FREE_BYTES - 1]
    cache = GraphCache(32, free_bytes=lambda: free[0], synchronize=Mock())
    assert not cache.should_use("target", (1,))
    assert cache.snapshot()["target_headroom_eager"] == 1
    assert cache.snapshot()["resident"] == 0
    free[0] += 1
    assert cache.should_use("target", (1,))
    cache.get_or_create("target", (1,), object)
    assert cache.snapshot()["target_capture"] == 1


@pytest.mark.parametrize("capacity", [1, 32])
def test_headroom_drop_between_admission_and_capture_defers_capture(capacity):
    free = [GraphCache.MIN_CAPTURE_FREE_BYTES]
    synchronize = Mock()
    cache = GraphCache(
        capacity,
        admission_hits=1,
        admission_factor=0,
        synchronize=synchronize,
        free_bytes=lambda: free[0],
    )
    if capacity == 1:
        old = add(cache, "target", 0)
    assert cache.should_use("target", (1,))
    free[0] -= 1
    assert cache.get_or_create("target", (1,), object) is None
    assert not cache.contains("target", (1,))
    assert cache.snapshot()["target_headroom_eager"] == 1
    synchronize.assert_not_called()
    if capacity == 1:
        assert cache.get_or_create("target", (0,), object) is old


def test_64_extents_shrinking_batch_and_frequency_state_are_bounded():
    cache = GraphCache(
        32, family_floors={"draft": 16, "proposal": 4}, synchronize=Mock()
    )
    for batch in range(4, 0, -1):
        add(cache, "proposal", (batch, 4096))
    for extent in range(1, 29):
        add(cache, "draft", (5 * (extent % 4 + 1), 4 - extent % 4, extent * 4096))
    assert cache.snapshot()["resident"] == 32
    for extent in range(29, 93):
        key = (5 * (extent % 4 + 1), 4 - extent % 4, extent * 4096)
        assert not cache.should_use("draft", key)
    assert cache.snapshot()["resident"] == 32
    assert cache.snapshot()["draft_eviction"] == 0
    assert len(cache.probation) == 64
    assert len(cache.frequencies) <= 32 + 64
    hot = (5, 4, 64 * 4096)
    for _ in range(2):
        assert not cache.should_use("draft", hot)
    assert cache.should_use("draft", hot)
    cache.get_or_create("draft", hot, object)
    assert cache.contains("draft", hot)
    assert cache.snapshot()["draft_resident"] == 28
    assert cache.snapshot()["proposal_resident"] == 4
    assert cache.snapshot()["draft_eviction"] == 1
    for extent in range(1000, 2000):
        assert not cache.should_use("draft", (1, 1, extent * 4096))
    assert len(cache.probation) <= 256
    assert len(cache.frequencies) <= 512 + 32


def test_failed_capture_cooldown_entries_are_bounded():
    cache = GraphCache(1, probation_capacity=2, synchronize=Mock())
    for key in range(3):
        with pytest.raises(RuntimeError, match="capture failed"):
            cache.get_or_create(
                "draft",
                (key,),
                lambda: (_ for _ in ()).throw(RuntimeError("capture failed")),
            )
    assert list(cache.retry_after) == [("draft", (1,)), ("draft", (2,))]
    assert cache.snapshot()["draft_capture_failure"] == 3


def test_mtp_graph_late_headroom_drop_uses_eager_path():
    class Model:
        mtp = object()

        def __init__(self):
            self.embedding = torch.zeros(1)

        def draft(self, tokens, hidden, batch, cache):
            return hidden + tokens[:, None]

    with patch("oh_my_vllm.worker.mtp.MTPAttention"):
        mtp = MTP(Model(), 1, 4096)
    free = iter([GraphCache.MIN_CAPTURE_FREE_BYTES, 0])
    mtp.graph_cache = GraphCache(free_bytes=lambda: next(free))
    result = mtp._run([3], torch.tensor([[10.0]]), [0, 1], [[1]], [7])
    torch.testing.assert_close(result, torch.tensor([[13.0]]))
    mtp.attention.plan.assert_called_once()
    assert mtp.graph_cache.snapshot()["draft_headroom_eager"] == 1


def test_only_graph_remains_alive_through_capture_then_is_released():
    class Owner:
        pass

    cache = GraphCache(1, admission_hits=2, admission_factor=0, synchronize=Mock())
    old = cache.get_or_create("draft", (0,), Owner)
    old_ref = weakref.ref(old)
    del old
    assert not cache.should_use("draft", (1,))

    def capture():
        assert old_ref() is not None
        assert list(cache.graphs) == [("draft", (0,))]
        return Owner()

    assert cache.should_use("draft", (1,))
    cache.get_or_create("draft", (1,), capture)
    gc.collect()
    assert old_ref() is None
    assert list(cache.graphs) == [("draft", (1,))]


@pytest.mark.parametrize(
    ("capacity", "floors", "probation", "history"),
    [
        (0, {}, 256, 4096),
        (32, {"draft": -1}, 256, 4096),
        (32, {"draft": 20, "proposal": 20}, 256, 4096),
        (32, {}, 0, 4096),
        (32, {}, 256, 0),
    ],
)
def test_invalid_capacity_rejected(capacity, floors, probation, history):
    with pytest.raises(ValueError, match="invalid graph cache"):
        GraphCache(
            capacity,
            family_floors=floors,
            probation_capacity=probation,
            recent_capacity=history,
        )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_real_draft_graph_eviction_preserves_previous_graph_output():
    assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
    assert torch.cuda.device_count() == 1
    assert "B200" in torch.cuda.get_device_name(0)
    free, total = torch.cuda.mem_get_info()
    assert 0 < free < total

    class Model:
        mtp = object()

        def __init__(self):
            self.embedding = torch.zeros(1, device="cuda")

        def draft(self, tokens, hidden, batch, cache):
            return hidden + tokens[:, None] + batch.positions[:, None]

    with patch("oh_my_vllm.worker.mtp.MTPAttention"):
        mtp = MTP(Model(), 4, 4096)
    mtp.graph_cache = GraphCache(
        1, admission_hits=2, admission_factor=0, failure_cooldown=2
    )
    table = [[1]]
    initial = torch.tensor([[10.0]], device="cuda")
    first = mtp._run([3], initial, [0, 1], table, [7])
    next_hidden = first.expand(2, -1)
    # First sighting executes eagerly; the second evicts and captures the new
    # graph while the old graph's static output is still the input tensor.
    mtp._run([4, 5], next_hidden, [0, 2], table, [8, 9])
    with (
        patch(
            "oh_my_vllm.worker.decode_graph.DraftGraph",
            side_effect=RuntimeError("capture failed"),
        ),
        pytest.raises(RuntimeError, match="capture failed"),
    ):
        mtp._run([4, 5], next_hidden, [0, 2], table, [8, 9])
    assert mtp.graph_cache.contains("draft", (1, 1, 4096))
    assert mtp.draft_graph_pool is not None
    assert mtp.graph_cache.snapshot()["draft_eviction"] == 0
    mtp._run([4, 5], next_hidden, [0, 2], table, [8, 9])
    mtp._run([4, 5], next_hidden, [0, 2], table, [8, 9])
    second = mtp._run([4, 5], next_hidden, [0, 2], table, [8, 9])
    final_input = second[:1]
    mtp._run([3], final_input, [0, 1], table, [7])
    final = mtp._run([3], final_input, [0, 1], table, [7])
    torch.testing.assert_close(final, torch.tensor([[42.0]], device="cuda"))
    stats = mtp.graph_cache.snapshot()
    assert stats["draft_capture"] == 3
    assert stats["draft_eviction"] == 2
    assert stats["draft_budget_eager"] == 2
    assert stats["draft_recent_recapture"] == 1
    assert stats["draft_capture_failure_eager"] == 2


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_first_cuda_capture_failure_replaces_unowned_pool_handle():
    assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
    assert "B200" in torch.cuda.get_device_name(0)

    class Model:
        mtp = object()

        def __init__(self):
            self.embedding = torch.zeros(1, device="cuda")
            self.calls = 0

        def draft(self, tokens, hidden, batch, cache):
            self.calls += 1
            result = hidden + tokens[:, None] + batch.positions[:, None]
            if self.calls == 2:
                raise RuntimeError("draft capture failed")
            return result

    with patch("oh_my_vllm.worker.mtp.MTPAttention"):
        mtp = MTP(Model(), 4, 4096)
    mtp.graph_cache.failure_cooldown = 0
    real_handle = torch.cuda.graph_pool_handle
    handles = []

    def new_handle():
        handle = real_handle()
        handles.append(handle)
        return handle

    with patch("torch.cuda.graph_pool_handle", side_effect=new_handle):
        with pytest.raises(RuntimeError, match="draft capture failed"):
            mtp._run([3], torch.tensor([[10.0]], device="cuda"), [0, 1], [[1]], [7])
        assert mtp.draft_graph_pool is None
        assert mtp.graph_cache.snapshot()["resident"] == 0
        output = mtp._run(
            [3], torch.tensor([[10.0]], device="cuda"), [0, 1], [[1]], [7]
        )
    assert len(handles) == 2
    assert handles[0] != handles[1]
    assert mtp.graph_cache.snapshot()["draft_capture"] == 1
    torch.testing.assert_close(output, torch.tensor([[20.0]], device="cuda"))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_real_32_graph_pool_defers_late_capture_at_low_headroom():
    assert os.environ["CUDA_VISIBLE_DEVICES"].startswith("GPU-")
    assert torch.cuda.device_count() == 1
    assert "B200" in torch.cuda.get_device_name(0)

    class Model:
        mtp = object()

        def __init__(self):
            self.embedding = torch.zeros(1, device="cuda")

        def draft(self, tokens, hidden, batch, cache):
            return hidden + tokens[:, None] + batch.positions[:, None]

    def run_shape(mtp, batch_size, total):
        counts = [1] * batch_size
        remaining = total - batch_size
        for i in range(batch_size):
            take = min(remaining, 4)
            counts[i] += take
            remaining -= take
        assert remaining == 0
        starts = [0]
        positions = []
        for count in counts:
            starts.append(starts[-1] + count)
            positions.extend(range(7, 7 + count))
        return mtp._run(
            [1] * total,
            torch.zeros(total, 1, device="cuda"),
            starts,
            [[1] for _ in counts],
            positions,
        )

    with patch("oh_my_vllm.worker.mtp.MTPAttention"):
        mtp = MTP(Model(), 4, 4096)
    shapes = [
        (batch_size, total)
        for batch_size in range(1, 5)
        for total in range(batch_size, 5 * batch_size + 1)
    ]
    for batch_size, total in shapes[:32]:
        run_shape(mtp, batch_size, total)
    assert mtp.graph_cache.snapshot()["draft_resident"] == 32
    assert mtp.graph_cache.snapshot()["draft_capture"] == 32
    mtp.graph_cache.free_bytes = lambda: 0
    output = run_shape(mtp, *shapes[32])
    assert mtp.graph_cache.snapshot()["draft_headroom_eager"] == 1
    assert mtp.graph_cache.snapshot()["draft_eviction"] == 0
    assert mtp.graph_cache.snapshot()["resident"] == 32
    mtp.attention.plan.assert_called_once()
    torch.testing.assert_close(output[0], torch.full_like(output[0], 8))
    old_output = run_shape(mtp, *shapes[0])
    torch.testing.assert_close(old_output, torch.full_like(old_output, 8))
