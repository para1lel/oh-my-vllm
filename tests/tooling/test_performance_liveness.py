"""Dead historical features cannot increase the theoretical lower bound."""

import math

import pytest
from oh_my_vllm.performance.analysis import required_outputs, step_work
from oh_my_vllm.performance.dag import b200
from oh_my_vllm.performance.model import Model


def layer_weights(prefix="target.mtp"):
    result = {}

    def tensor(name, shape, dtype="float32"):
        result[name] = dict(
            shape=shape,
            dtype="torch." + dtype,
            bytes=math.prod(shape)
            * {"float32": 4, "bfloat16": 2, "float8_e4m3fn": 1}[dtype],
        )

    for name, width in (
        ("input_norm", 5120),
        ("post_norm", 5120),
        ("q_norm", 256),
        ("k_norm", 256),
    ):
        tensor(prefix + "." + name, [width])
    for name, n, k in (
        ("qkv", 14336, 5120),
        ("out", 5120, 6144),
        ("gate_up", 34816, 5120),
        ("down", 5120, 17408),
    ):
        tensor(prefix + "." + name + ".weight", [n, k], "float8_e4m3fn")
        tensor(prefix + "." + name + ".scale", [n // 128, k // 128])
    return result


@pytest.mark.parametrize("selected", [[], [128], [0, 128], list(range(129))])
def test_fa_projection_counts_only_live_queries_and_shares_quantization(selected):
    model = Model(b200(148, 1965e6, 250249216), layer_weights())
    start = model.node("input")
    end = model.target_layer(
        "target.mtp",
        start,
        [
            dict(
                query=129,
                context=0,
                position_start=1,
                fa_block_table=[1],
                incoming_context=0,
            )
        ],
        state_bytes=2,
        mtp=True,
        output_rows=[selected],
    )
    summary = model.bounds(end)
    expected = 2 * 129 * 5120 * 2048
    expected += (
        2 * len(selected) * (5120 * 12288 + 6144 * 5120 + 5120 * 34816 + 17408 * 5120)
    )
    assert summary["work"]["tensor_fp8"] == expected
    pairs = sum(row + 1 for row in selected)
    assert summary["work"].get("tensor_bf16", 0) == 4 * 24 * 256 * pairs
    expected_quant = 129 * 5120 + len(selected) * (6144 + 5120 + 17408)
    assert summary["relaxed_native_operations"]["fp32_to_fp8"] == expected_quant
    read_names = [
        node.name for node in model.graph.nodes if node.name.startswith("read:")
    ]
    assert "read:target.mtp.qkv.weight[12288:14336]" in read_names
    assert ("read:target.mtp.qkv.weight[0:12288]" in read_names) is bool(selected)
    assert ("read:target.mtp.gate_up.weight" in read_names) is bool(selected)


def target(rid, context, query, *, samples=0, terminal=False, mtp=True):
    return dict(
        request_id=rid,
        context=context,
        query=query,
        sample_rows=samples,
        kept_rows=query,
        drafts=0,
        terminal=terminal,
        mtp=mtp,
        fa_block_table=[1, 2, 3],
    )


def mtp(requests, queries, contexts, outputs):
    return dict(
        kind="mtp_forward",
        requests=requests,
        queries=queries,
        contexts=contexts,
        output_rows=outputs,
        persistent=True,
    )


def test_mixed_batch_uses_selected_endpoints_and_saved_target_boundary():
    source = [
        dict(kind="target", requests=[target(1, 0, 784), target(2, 784, 2, samples=1)]),
        mtp([1, 2], [783, 2], [1, 784], [[], [1]]),
    ]
    result = required_outputs(source)
    assert result[0]["requests"][0]["final_output_rows"] == list(range(784))
    assert result[0]["requests"][1]["final_output_rows"] == [0, 1]
    assert result[1]["live_output_rows"] == [[], [1]]
    assert "final_output_rows" not in source[0]["requests"][0]


def test_terminal_context_tail_has_no_hidden_or_kv_consumer():
    result = required_outputs(
        [
            dict(kind="target", requests=[target(1, 784, 2, samples=2, terminal=True)]),
            mtp([1], [2], [785], [[]]),
        ]
    )
    assert result[1]["live_queries"] == [0]
    assert result[1]["live_output_rows"] == [[]]
    assert result[0]["requests"][0]["final_output_rows"] == [0, 1]


def test_terminal_full_prefix_retains_only_matching_context_and_boundary_features():
    result = required_outputs(
        [
            dict(kind="target", requests=[target(1, 780, 8, samples=1, terminal=True)]),
            mtp([1], [8], [781], [[]]),
        ]
    )
    assert result[1]["live_queries"] == [3]
    assert result[0]["requests"][0]["final_output_rows"] == [0, 1, 2, 3, 7]


def test_plain_target_final_layer_uses_all_required_verify_samples():
    result = required_outputs(
        [
            dict(
                kind="target",
                requests=[
                    target(1, 32768, 8, samples=8, mtp=False),
                    target(2, 0, 624, mtp=False),
                ],
            )
        ]
    )
    assert result[0]["requests"][0]["final_output_rows"] == list(range(8))
    assert result[0]["requests"][1]["final_output_rows"] == []


def test_zero_live_terminal_dspark_injection_is_no_work_through_step_work(monkeypatch):
    def minimal_target(self, requests, state_dtype):
        self.features = self.node("features")
        return self.node("target_sample", (self.features,))

    monkeypatch.setattr(Model, "target", minimal_target)
    operations = [
        dict(
            kind="target",
            requests=[target(1, 784, 1, samples=1, terminal=True, mtp=False)],
            state_dtype="torch.bfloat16",
        ),
        dict(kind="dspark_inject", rows=1, requests=[1], contexts=[784], queries=[1]),
    ]
    summary = step_work(b200(148, 1965e6, 250249216), {}, operations)
    assert not summary["work"]
    assert not summary["incoming_ranges"]


def mtp_model():
    weights = layer_weights()
    for name, shape in (
        ("target.embedding", [248320, 5120]),
        ("target.mtp_fc.weight", [5120, 10240]),
    ):
        weights[name] = dict(
            shape=shape, dtype="torch.bfloat16", bytes=math.prod(shape) * 2
        )
    for name in ("mtp_embedding_norm", "mtp_hidden_norm", "mtp_norm"):
        weights["target." + name] = dict(
            shape=[5120], dtype="torch.float32", bytes=5120 * 4
        )
    return Model(b200(148, 1965e6, 250249216), weights)


def test_mtp_incoming_traffic_uses_absolute_physical_end():
    model = mtp_model()
    parent = model.node("target")
    model.mtp_forward(
        dict(
            queries=[1, 1],
            contexts=[785, 785],
            tables=[[1, 2], [3, 2]],
            unique_token_ids=[7],
        ),
        parent,
    )
    incoming = next(
        n for n in model.graph.nodes if n.name == "read:target.mtp:incoming_kv"
    )
    assert incoming.external_bytes == 1567 * 2 * 4 * 256 * 2


@pytest.mark.parametrize("internal", [False, True])
def test_restored_mtp_target_hidden_has_named_physical_read(internal):
    model = mtp_model()
    model.phase = "prefill" if internal else "decode"
    model.target_requests = [
        dict(request_id=1, context=784, incoming_context=0 if internal else 784)
    ]
    parent = model.node("target")
    model.mtp_forward(
        dict(
            requests=[1],
            queries=[2],
            contexts=[784],
            tables=[[1, 2]],
            unique_token_ids=[7],
        ),
        parent,
    )
    boundary = next(n for n in model.graph.nodes if n.name == "read:mtp_hidden:page:1")
    assert boundary.external_bytes == (0 if internal else 10240)
    sources = next(n for n in model.graph.nodes if n.name == "mtp:hidden_sources")
    assert len(sources.parents) == 2
