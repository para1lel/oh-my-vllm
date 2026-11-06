import datetime
from unittest.mock import patch

import torch
import torch.distributed as dist
from oh_my_vllm.models.deepseek_v41.dep4 import Dep4Executor
from oh_my_vllm.models.deepseek_v41.dispatch import (
    ExpertExchange,
)
from oh_my_vllm.models.deepseek_v41.rank_plan import (
    handle_rank,
    merge_replies,
)

from tests.deepseek_dep4_fixtures import _Model


class _FaultExchange(ExpertExchange):
    def __init__(self, rank, status, *, sparse=False):
        super().__init__(status=status)
        self.rank, self.calls, self.sparse = rank, 0, sparse

    def __call__(self, moe, rows, routing):
        value = super().__call__(moe, rows, routing)
        if self.rank == 2 and self.calls == 0:
            if self.sparse:
                from oh_my_vllm.models.deepseek_v41.kernel import (
                    _sparse_status_accumulator,
                )

                _sparse_status_accumulator(rows.device).fill_(True)
            else:
                self.status.tensor.fill_(4)
        self.calls += 1
        return value


def _failure_message():
    return dict(
        type="execute",
        rpc_id=8,
        step_id=2,
        worker_epoch=7,
        num_batched_tokens=1,
        scheduled=[
            dict(
                request_id=11,
                lease_id=3,
                purpose="advance",
                action="forward",
                sampling={"kind": "target"},
                token_ids=[1],
                forward_start=0,
                forward_end=1,
                accepted_end=1,
                endpoint=1,
                model_state=dict(
                    kind="deepseek_resident_v1",
                    fa_block_table=[1],
                    mamba_block_table=[1],
                    fa_block_copies=[],
                    prefix_state_slot=None,
                    cache_checkpoint=None,
                ),
            )
        ],
        finished_request_ids=[],
        preempted_request_ids=[],
        rank_plan=[
            dict(
                rank=peer,
                request_ids=[11] if peer == 0 else [],
                finished_request_ids=[],
                preempted_request_ids=[],
            )
            for peer in range(4)
        ],
    )


def _round_failure_peer(rank, store, sparse=False):
    from types import SimpleNamespace

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store}",
        rank=rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=30),
    )
    try:
        model = _Model(rank)
        executor = Dep4Executor(model, check_round_status=True)
        executor.sparse_control = sparse
        executor.exchange = _FaultExchange(rank, executor.status, sparse=sparse)
        for layer in (*model.layers, *model.mtp):
            layer.ffn.dep4_exchange = executor.exchange
        commits = []
        message = _failure_message()

        def handle(local):
            for row in local["scheduled"]:
                model(torch.ones((1, 4), dtype=torch.bfloat16))
                commits.append(row["request_id"])
                model(torch.ones((1, 4), dtype=torch.bfloat16))
                commits.append(row["request_id"])
            return dict(type="execute_result", rpc_id=8, outputs=[])

        worker = SimpleNamespace(dep4_executor=executor, handle=handle)
        readbacks = []
        original_cpu = torch.Tensor.cpu

        def readback(value, *args, **kwargs):
            readbacks.append(value.numel())
            return original_cpu(value, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", readback):
            try:
                handle_rank(worker, message, rank)
            except RuntimeError as error:
                reason = (
                    "native sparse attention failure"
                    if sparse
                    else "invalid padded expert metadata"
                )
                assert reason in str(error)
                assert "no cache commit or result" in str(error)
            else:
                raise AssertionError("a failed round cannot produce rank completion")
        assert not commits
        assert executor.target_rounds == 1
        assert executor.active is False
        assert readbacks == [12, 2], readbacks
        errors = {peer: dict(type="error", rpc_id=8) for peer in range(4)}
        try:
            merge_replies(message, errors)
        except ValueError:
            pass
        else:
            raise AssertionError("failed ranks cannot publish an Execute result")
    finally:
        dist.destroy_process_group()
