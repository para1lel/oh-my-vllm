from .deepseek_batch_capture import (
    TargetCapture,
    cache_state,
    differences,
    request_state,
)
from .deepseek_batch_schedule import (
    continuation_rows,
    execute_scheduled,
    require_multiround_gates,
    require_scheduled_requests,
    update_boundaries,
    write_round_report,
)


def multiround(worker, args, torch, batch, prepare, make_row, rank):
    executions = (
        ("reference", "eager", "graph", "graph_fallback")
        if args.graph
        else ("reference", "eager")
    )
    reports = []
    original_limit = worker.decode_graph_limit
    try:
        for mode, same_signature in (
            ("ordinary", True),
            ("dspark_aligned", True),
            ("dspark", False),
            ("remaining", True),
        ):
            for execution in executions:
                runner = MultiRound(
                    worker, args, torch, batch, prepare, make_row, rank, reports
                )
                runner.reset(mode, execution, same_signature, original_limit)
                runner.run()
    finally:
        worker.decode_graph_limit = original_limit
    return reports


class MultiRound:
    def __init__(self, worker, args, torch, batch, prepare, make_row, rank, reports):
        self.worker = worker
        self.args = args
        self.torch = torch
        self.batch = batch
        self.prepare = prepare
        self.make_row = make_row
        self.rank = rank
        self.reports = reports

    def reset(self, mode, execution, same_signature, original_limit):
        self.mode = mode
        self.execution = execution
        self.same_signature = same_signature
        self.worker.decode_graph_limit = (
            1 if self.execution == "graph_fallback" else original_limit
        )
        budgets = {
            key: 2 + index % 2 if self.mode == "remaining" else 6 * self.args.rounds + 8
            for index, key in enumerate(range(1, self.args.batch + 1))
        }
        self.outputs = self.prepare(
            self.mode != "ordinary",
            budgets=budgets,
            through_execute=True,
            same_signature=self.same_signature,
            shared_pair=self.mode == "dspark",
        )
        self.worker.decode_graph_enabled = self.execution.startswith("graph")
        self.worker.batch_decode_enabled = (
            self.args.batch_decode and self.execution != "reference"
        )
        self.previous_verify = set()
        self.continued = self.graph_calls = self.fallback_calls = 0
        self.graph_verifies = self.graph_reuses = self.max_batch = (
            self.max_scheduled
        ) = 0
        self.observed = {}
        self.eviction_steps = 0
        self.remapped = False
        ratios = {
            layer.attn.compress_ratio
            for layer in self.worker.model.layers
            if layer.attn.compress_ratio > 1
        }
        self.crossings = {
            "window": False,
            "bucket": False,
            **{f"compress/{ratio}": False for ratio in ratios},
        }

    def compare_step(self, captured, step):
        actual = dict(
            outputs=self.outputs,
            targets=captured.targets,
            requests=request_state(self.worker),
            state=cache_state(self.worker, self.torch, digest=True),
        )
        path = (
            self.args.output / f"multiround-{self.mode}-rank{self.rank}-step{step}.pt"
        )
        if self.execution == "reference":
            self.torch.save(actual, path)
            changed = []
        else:
            expected = self.torch.load(path, weights_only=True)
            changed = differences(expected, actual, self.torch)
        return actual, changed

    def report_step(self, step, rows, captured, changed, result, keys, evicted, actual):
        report = dict(
            rank=self.rank,
            suite="multiround",
            mode=self.mode,
            execution=self.execution,
            step=step,
            scheduled_requests=len(rows),
            executed_request_ids=sorted(captured.targets),
            batch_decode_enabled=self.worker.batch_decode_enabled,
            differences=changed,
            targets=captured.calls,
            accepted=[item["num_accepted_draft_tokens"] for item in result],
            computed={key: self.worker.requests[key].state_position for key in keys},
            slot_evictions=list(evicted),
            diagnostic_only=not self.args.require_verify_continuation,
        )
        self.reports.append(report)
        write_round_report(self.args, self.rank, report)
        if changed:
            self.torch.save(
                actual, self.args.output / f"multiround-failure-rank{self.rank}.pt"
            )
            raise AssertionError("multiround checkpoint endpoints differ")

    def step(self, step):
        rows = continuation_rows(self.worker, self.outputs, self.make_row, step)
        if not rows:
            return False
        keys = [item["request_id"] for item in rows]
        self.continued += sum(
            key in self.previous_verify
            and bool(self.outputs[key]["new_draft_token_ids"])
            for key in keys
        )
        self.previous_verify = {
            item["request_id"] for item in rows if len(item["token_ids"]) > 1
        }
        evicted = keys if step in (2, 5) else ()
        self.eviction_steps += bool(evicted)
        before_slots = {key: self.worker.requests[key].cache_slot for key in keys}
        with TargetCapture(self.worker, self.torch, self.batch) as captured:
            result = execute_scheduled(
                self.worker, rows, step, self.execution == "reference", evicted
            )
        require_scheduled_requests(rows, result, captured.targets)
        self.max_scheduled = max(self.max_scheduled, len(rows))
        self.outputs = {item["request_id"]: item for item in result}
        for key in keys:
            request = self.worker.requests[key]
            if request.state_position != len(request.history) - 1:
                raise AssertionError(
                    "accepted frontier differs from the uncomputed output"
                )
        update_boundaries(captured.calls, self.observed, self.crossings)
        self.max_batch = max(
            [self.max_batch, *(call["batch_size"] for call in captured.calls)]
        )
        self.remapped |= any(
            self.worker.requests[key].cache_slot != before_slots[key] for key in keys
        )
        actual, changed = self.compare_step(captured, step)
        for call in captured.calls:
            if call["execution"] == "graph":
                self.graph_calls += 1
                self.graph_verifies += call["length"] > 1
                self.graph_reuses += call["captured"] == 0
            elif self.execution.startswith("graph"):
                if call["graphs"] < self.worker.decode_graph_limit:
                    raise AssertionError(
                        "target used Eager before the graph cache limit"
                    )
                self.fallback_calls += 1
        self.report_step(step, rows, captured, changed, result, keys, evicted, actual)
        return True

    def check_coverage(self):
        if self.execution.startswith("graph") and not self.graph_calls:
            raise AssertionError("multiround run did not replay a target graph")
        if (
            self.mode == "ordinary"
            and self.execution == "graph"
            and not self.graph_reuses
        ):
            raise AssertionError("ordinary decode did not reuse a captured graph")
        require_multiround_gates(
            self.mode,
            self.execution,
            continued=self.continued,
            max_batch=self.max_batch,
            crossings=self.crossings,
            require_verify_continuation=self.args.require_verify_continuation,
            require_boundaries=self.args.require_boundaries,
            require_batch=self.args.batch_decode,
            scheduled_requests=self.max_scheduled,
            required_requests=self.args.batch,
        )
        if self.mode.startswith("dspark"):
            if self.execution == "graph" and not self.graph_verifies:
                raise AssertionError(
                    "multiround run did not replay a verification graph"
                )
            if not self.remapped:
                raise AssertionError(
                    "multiround run did not reuse a slot across requests"
                )
        if self.mode == "remaining" and any(
            not request.finish for request in self.worker.requests.values()
        ):
            raise AssertionError("remaining-one/two cases did not finish")
        if (
            self.execution == "graph_fallback"
            and self.mode.startswith("dspark")
            and not self.fallback_calls
        ):
            raise AssertionError("bounded graph cache did not exercise Eager fallback")

    def write_coverage(self):
        coverage = dict(
            rank=self.rank,
            suite="coverage",
            mode=self.mode,
            execution=self.execution,
            continued_verifications=self.continued,
            graph_calls=self.graph_calls,
            graph_verifications=self.graph_verifies,
            reused_graphs=self.graph_reuses,
            fallback_calls=self.fallback_calls,
            remapped=self.remapped,
            boundaries=self.crossings,
            max_target_batch=self.max_batch,
            max_scheduled_requests=self.max_scheduled,
            batch_decode_enabled=self.worker.batch_decode_enabled,
            batch_required=self.args.batch_decode and self.execution != "reference",
            same_signature=self.same_signature,
            shared_pair=self.mode == "dspark",
            diagnostic_only=not self.args.require_verify_continuation,
            comparison_scope="same-source batch/Graph equivalence",
            frozen_mtp_reference_verified=False,
            rust_preemption_recompute_verified=False,
            slot_eviction_endpoint_restore=self.eviction_steps > 0,
            slot_eviction_steps=self.eviction_steps,
        )
        write_round_report(self.args, self.rank, coverage)

    def run(self):
        for step in range(self.args.rounds):
            if not self.step(step):
                break
        self.check_coverage()
        self.write_coverage()
