"""Captured query bounds must distinguish equal totals with unequal row groups."""

from oh_my_vllm.worker.model_runner import RuntimeConfig


def test_dspark_query_bound_prevents_same_total_graph_collision():
    config = RuntimeConfig(
        "target", speculative_mode="dspark", speculative_tokens=7, draft_model="draft"
    )
    asymmetric = config.decode_graph_key([1, 3], 4096)
    symmetric = config.decode_graph_key([2, 2], 4096)
    assert asymmetric[:3] == symmetric[:3]
    assert asymmetric != symmetric
    assert asymmetric[-1] == 5
    assert symmetric[-1] == 2
    assert config.decode_graph_key([3, 1], 4096) == asymmetric
    assert config.decode_graph_key([8, 1], 4096)[-1] == 8
    assert config.decode_graph_key([1, 4], 4096) == config.decode_graph_key(
        [2, 3], 4096
    )


def test_dspark_buckets_cover_each_legal_verification_without_extra_shapes():
    config = RuntimeConfig(
        "target", speculative_mode="dspark", speculative_tokens=7, draft_model="draft"
    )
    bounds = [config.decode_graph_key([count], 4096)[-1] for count in range(1, 9)]
    assert bounds == [1, 2, 5, 5, 5, 8, 8, 8]
    assert all(bound >= count for count, bound in enumerate(bounds, 1))


def test_ordinary_and_mtp_preserve_fixed_query_bounds():
    ordinary = RuntimeConfig("target")
    mtp = RuntimeConfig("target", speculative_tokens=4)
    assert ordinary.decode_graph_key([1, 1], 4096)[-1] == 1
    assert mtp.decode_graph_key([1, 3], 4096)[-1] == 5
    assert mtp.decode_graph_key([2, 2], 4096) == mtp.decode_graph_key([1, 3], 4096)


def test_comparison_keeps_both_modes_target_shapes_and_memory_guard():
    from itertools import product

    from oh_my_vllm.worker.graph_cache import GraphCache

    config = RuntimeConfig(
        "target", speculative_tokens=4, draft_model="draft", comparison=True
    )
    capacity, floors = config.target_graph_budget()
    free = [8 << 30]
    cache = GraphCache(capacity, family_floors=floors, free_bytes=lambda: free[0])
    mtp_keys = [config.decode_graph_key([5] * count, 36864) for count in range(1, 5)]
    dspark = RuntimeConfig(
        "target", speculative_mode="dspark", speculative_tokens=7, draft_model="draft"
    )
    dspark_keys = sorted(
        {
            dspark.decode_graph_key(list(counts), 36864)
            for count in range(1, 5)
            for counts in product(range(1, 6), repeat=count)
        }
    )[:40]
    assert len(dspark_keys) == 40
    for family, keys in (("target", mtp_keys), ("target_dspark", dspark_keys)):
        for key in keys:
            assert cache.should_use(family, key)
            cache.get_or_create(family, key, object)
    for _ in range(3):
        for family, keys in (("target", mtp_keys), ("target_dspark", dspark_keys)):
            for key in keys:
                assert cache.contains(family, key)
                assert cache.should_use(family, key)
    assert cache.snapshot()["target_capture"] == len(mtp_keys)
    assert cache.snapshot()["target_dspark_capture"] == len(dspark_keys)
    assert cache.snapshot()["target_eviction"] == 0
    assert cache.snapshot()["target_dspark_eviction"] == 0
    free[0] = 0
    assert not cache.should_use("target_dspark", (32, 4, 40960, 8))
    assert cache.should_use("target", mtp_keys[0])
    assert cache.snapshot()["target_dspark_headroom_eager"] == 1


def test_target_graph_budgets_keep_single_modes_bounded():
    from oh_my_vllm.worker.graph_cache import GraphCache

    modes = (("none", 0, 32), ("mtp", 4, 32), ("dspark", 7, 64))
    for mode, count, expected in modes:
        config = RuntimeConfig(
            "target",
            speculative_mode=mode,
            speculative_tokens=count,
            draft_model="draft" if mode == "dspark" else None,
        )
        capacity, floors = config.target_graph_budget()
        assert capacity == expected
        cache = GraphCache(capacity, family_floors=floors, synchronize=lambda: None)
        family = "target_dspark" if mode == "dspark" else "target"
        for index in range(capacity * 2):
            key = (index,)
            if cache.should_use(family, key):
                cache.get_or_create(family, key, object)
        assert len(cache.graphs) == expected
        assert cache.snapshot()[f"{family}_budget_eager"] == expected
