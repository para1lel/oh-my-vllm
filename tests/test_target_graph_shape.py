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
