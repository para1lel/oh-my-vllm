"""The complete seven-position Markov DAG keeps genuine token feedback."""

import math

import pytest
from oh_my_vllm.performance.dag import b200
from oh_my_vllm.performance.model import Model


def draft_weights():
    weights = {}

    def tensor(name, shape, dtype="bfloat16"):
        weights[name] = {
            "shape": shape,
            "dtype": "torch." + dtype,
            "bytes": math.prod(shape) * (4 if dtype == "float32" else 2),
        }

    for layer in range(5):
        prefix = f"dspark.layers.{layer}"
        for name, shape in (
            ("q", [4096, 5120]),
            ("k", [1024, 5120]),
            ("v", [1024, 5120]),
            ("o", [5120, 4096]),
            ("gate", [17408, 5120]),
            ("up", [17408, 5120]),
            ("down", [5120, 17408]),
            ("input_norm", [5120]),
            ("post_norm", [5120]),
        ):
            tensor(prefix + "." + name, shape)
        tensor(prefix + ".q_norm", [128], "float32")
        tensor(prefix + ".k_norm", [128], "float32")
    tensor("dspark.norm", [5120])
    tensor("target.head", [248320, 5120])
    tensor("dspark.markov_w2", [248320, 256])
    tensor("dspark.confidence_weight", [1, 5376])
    return weights


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("repeated", [False, True])
def test_complete_backbone_keeps_unique_edges_and_seven_feedback_steps(batch, repeated):
    model = Model(b200(148, 1965e6, 250249216), draft_weights())
    parent = model.node("committed_target")
    assert parent == 0
    endpoint = model.dspark_backbone(
        {
            "contexts": [32768] * batch,
            "incoming_contexts": [32768] * batch,
            "tables": [
                list(range(1 + 42 * row, 43 + 42 * row)) for row in range(batch)
            ],
            "anchors": [3] * batch,
            "candidates": [
                [8] * 7 if repeated else list(range(8, 15)) for _ in range(batch)
            ],
        },
        parent,
    )
    nodes = model.graph.nodes
    named = {node.name: (index, node) for index, node in enumerate(nodes)}
    base, _ = named["target.head:matmul"]
    _, first = named["dspark:markov_add:0"]
    assert len(first.parents) == 2
    assert base in first.parents
    for position in range(1, 7):
        _, addition = named[f"dspark:markov_add:{position}"]
        prior, _ = named[f"dspark:greedy:{position - 1}"]
        assert prior in addition.parents
        assert len(addition.parents) == len(set(addition.parents))
    assert sum(node.name == "dspark.markov_w2:matmul" for node in nodes) == 7
    assert sum(node.name == "dspark.confidence_weight:matmul" for node in nodes) == 7
    assert endpoint == named["dspark:greedy:6"][0]
    bound = model.bounds(endpoint)
    assert bound["work"]["tensor_bf16"] > 0
    assert math.isfinite(bound["critical_path_s"])
