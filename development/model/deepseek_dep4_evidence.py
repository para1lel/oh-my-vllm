import hashlib
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def request_owners(lengths):
    if not 1 <= len(lengths) <= 4:
        raise ValueError("lengths must select between one and four request owners")
    return list(range(len(lengths)))


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def _rank_worker(rank, args, run_id, operation):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="4")
    proof = ExchangeEvidence(args.output, rank, mode=args.mode, run_id=run_id)
    try:
        with proof.observe():
            operation(args, proof)
    except BaseException as error:
        try:
            proof.fail(error)
        finally:
            os._exit(1)


class ExchangeEvidence:
    def __init__(self, output, rank, *, mode, run_id=None):
        self.output = Path(output)
        self.rank = rank
        self.mode = mode
        self.run_id = run_id or uuid.uuid4().hex
        self.counts = dict(
            fixed=0,
            legacy=0,
            fallback=0,
            shadow_legacy=0,
            fixed_waves=0,
            padded_projections=0,
            completed=0,
            failed=0,
            maximum_source_rows=0,
            maximum_local_rows=0,
        )
        self.outcome = "initializing"
        self.error = None
        self.failure_time = None
        self.shadow_differences = []

    def fail(self, error):
        self.outcome = "failed"
        self.failure_time = time.monotonic()
        self.error = dict(type=type(error).__name__, message=str(error))
        self.write()

    def write(self):
        self.output.mkdir(parents=True, exist_ok=True)
        record = dict(
            run_id=self.run_id,
            rank=self.rank,
            mode=self.mode,
            outcome=self.outcome,
            counts=self.counts,
            error=self.error,
            failure_time_monotonic=self.failure_time,
            shadow_differences=self.shadow_differences,
        )
        _write_json(self.output / f"proof-rank{self.rank}.json", record)

    def require_fixed(self):
        if self.mode not in ("fixed", "shadow"):
            return
        counts = self.counts
        if (
            counts["fixed"] == 0
            or counts["legacy"]
            or counts["fallback"]
            or counts["fixed_waves"] == 0
            or counts["padded_projections"] != 3 * counts["fixed_waves"]
        ):
            raise AssertionError(f"fixed exchange path was not proved: {counts}")

    def compare_shadow(self, exchange, original, moe, rows, routing, actual):
        import torch
        import torch.distributed as dist

        exchange.fixed_round = False
        try:
            self.counts["shadow_legacy"] += 1
            expected = original(exchange, moe, rows, routing)
        finally:
            exchange.fixed_round = True
        difference = tensor_difference(expected, actual, torch)
        if difference:
            self.shadow_differences.append(
                dict(layer=moe.layer_id, source_rows=rows.size(0), **difference)
            )
        self.write()
        failed = torch.tensor([bool(difference)], dtype=torch.int32, device=rows.device)
        dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=exchange.group)
        if failed.item():
            raise AssertionError(
                "fixed and legacy expert sums differ on at least one rank"
            )

    @contextmanager
    def observe(self):
        from oh_my_vllm.models.deepseek_v41 import kernel
        from oh_my_vllm.models.deepseek_v41.dispatch import (
            ExpertExchange,
            FixedDispatchBuffers,
        )

        original_exchange = ExpertExchange.__call__
        original_dispatch = FixedDispatchBuffers.dispatch
        original_project = kernel.grouped_fp4_gemm_padded

        def exchange(instance, moe, rows, routing):
            fixed = instance.fixed_round
            self.counts["fixed" if fixed else "legacy"] += 1
            if not fixed and self.mode in ("fixed", "shadow"):
                self.counts["fallback"] += 1
            for key, count in (
                ("maximum_source_rows", instance.source_rows),
                ("maximum_local_rows", rows.size(0)),
            ):
                self.counts[key] = max(self.counts[key], count)
            self.write()
            try:
                if self.mode == "shadow" and fixed:
                    legacy_rows = rows.clone()
                    legacy_routing = tuple(value.clone() for value in routing)
                value = original_exchange(instance, moe, rows, routing)
                if self.mode == "shadow" and fixed:
                    self.compare_shadow(
                        instance,
                        original_exchange,
                        moe,
                        legacy_rows,
                        legacy_routing,
                        value,
                    )
                self.counts["completed"] += 1
                return value
            except BaseException:
                self.counts["failed"] += 1
                raise
            finally:
                self.write()

        def dispatch(instance, *args, **kwargs):
            self.counts["fixed_waves"] += 1
            return original_dispatch(instance, *args, **kwargs)

        def project(*args, **kwargs):
            self.counts["padded_projections"] += 1
            return original_project(*args, **kwargs)

        self.write()
        with (
            patch.object(ExpertExchange, "__call__", exchange),
            patch.object(FixedDispatchBuffers, "dispatch", dispatch),
            patch.object(kernel, "grouped_fp4_gemm_padded", project),
        ):
            try:
                yield self
                self.require_fixed()
                self.outcome = "completed"
            except BaseException as error:
                self.outcome = "failed"
                self.error = dict(type=type(error).__name__, message=str(error))
                raise
            finally:
                self.write()


def tensor_difference(expected, actual, torch):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return dict(reason="tensor metadata differs")
    if torch.equal(
        expected.contiguous().view(torch.uint8), actual.contiguous().view(torch.uint8)
    ):
        return None
    delta = actual.double() - expected.double()
    return dict(
        mismatches=int((actual != expected).sum()),
        max_abs=float(delta.abs().max()) if delta.numel() else 0.0,
        relative_l2=float(
            torch.linalg.vector_norm(delta)
            / torch.linalg.vector_norm(expected.double()).clamp_min(1e-12)
        ),
    )


def checker_identity():
    return {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "check_deepseek_dep4.py",
            "deepseek_dep4_evidence.py",
            "deepseek_dep4_rank.py",
        )
    }
