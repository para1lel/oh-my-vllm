import datetime
from unittest.mock import patch

import torch
import torch.distributed as dist
from oh_my_vllm.models.deepseek_v41.dispatch import (
    ExpertExchange,
)


def _driver_failure(args, proof):
    import json
    from types import SimpleNamespace

    from development.model.check_deepseek_dep4 import initialize_rank

    torch.set_num_threads(1)
    for peer in range(4):
        record = json.loads((args.output / f"proof-rank{peer}.json").read_text())
        assert record["run_id"] == proof.run_id
        assert record["outcome"] == "initializing"
    if args.failure in ("missing", "corrupt", "device", "initialize"):
        original_init = dist.init_process_group

        def device(rank):
            if args.failure == "device" and rank == 2:
                raise RuntimeError("injected device failure")

        def initialize(_backend, **options):
            if args.failure == "initialize" and proof.rank == 2:
                raise ValueError("injected initialization failure")
            return original_init("gloo", **options)

        with (
            patch.object(torch.cuda, "set_device", device),
            patch.object(dist, "init_process_group", initialize),
        ):
            initialize_rank(args, proof)
        return
    dist.init_process_group(
        "gloo",
        init_method=f"file://{args.output / 'rank-store'}",
        rank=proof.rank,
        world_size=4,
        timeout=datetime.timedelta(seconds=240),
    )
    if args.failure == "endpoint":
        if proof.rank == 2:
            raise AssertionError("injected endpoint assertion")
        dist.all_reduce(torch.ones(1))
        raise AssertionError("a peer endpoint failure must terminate this collective")
    exchange = ExpertExchange()
    exchange.prepare_round(600, fixed=True)
    rows = torch.ones((1, 4), dtype=torch.bfloat16)

    def legacy(_exchange, _moe, values, _routing):
        if proof.rank == 2:
            raise ValueError("injected legacy failure")
        dist.all_reduce(torch.ones(1))
        return values.float()

    proof.compare_shadow(
        exchange, legacy, SimpleNamespace(layer_id=0), rows, None, rows.float()
    )
    raise AssertionError("a peer legacy failure must terminate this collective")
