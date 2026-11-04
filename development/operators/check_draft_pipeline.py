"""Compare Qwen draft lookahead with sequential grammar sampling and KV writes."""

import os
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from oh_my_vllm.models.qwen38 import speculation
from oh_my_vllm.models.qwen38.request import Request
from oh_my_vllm.models.qwen38.sampling import SamplingInputs
from oh_my_vllm.models.qwen38.tokenizer import Tokenization
from oh_my_vllm.operators.runtime.device import Device

from .cases import load_cases
from .fixtures import load_fixture
from .weights import CaseWeights


class DraftLookahead(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = Device()
        cls.case = next(
            case for case in load_cases() if case.id == "mtp.chain.numerics.batch-2"
        )
        cls.fixture = load_fixture(cls.case)
        model = Path(os.environ["MODEL_ROOT"])
        cls.weights = CaseWeights(cls.device, model, cls.case)
        cls.weights.verify(cls.fixture.manifest["checkpoint"])
        cls.tokenizer = Tokenization(model)
        cls.hidden = cls.device.upload(cls.fixture.inputs["hidden"], "bfloat16")

    @classmethod
    def tearDownClass(cls):
        cls.device.synchronize()
        cls.device.close()

    def run_path(
        self,
        pipeline,
        *,
        ending=False,
        stops=None,
        reject=False,
        mixed=False,
        fault=None,
        thinking=False,
    ):
        device, fixture = self.device, self.fixture
        kv = device.upload(fixture.state["kv"], "bfloat16")
        cache = SimpleNamespace(blocks=kv.shape[0], fa=((kv,),))
        tables = fixture.inputs["stage0.tables"].copy()
        requests, scheduled, endpoints = [], [], {}
        for row in range(2):
            options = dict(
                max_tokens=32,
                effort="low" if thinking else "off",
                thinking_token_budget=0 if thinking else -1,
                ignore_eos=not ending,
                format=dict(type="json_schema", schema={"const": None})
                if ending
                else dict(type="json_object"),
                sampling=dict(
                    temperature=0
                    if ending or (mixed and row == 0)
                    else 0.7 + row * 0.3,
                    top_k=20,
                    top_p=0.95,
                    frequency_penalty=0.25 * row,
                    presence_penalty=0.5,
                    repetition_penalty=1.0 if row == 0 else 1.1,
                    repetition_guard=True,
                    seed=42 + row,
                ),
            )
            if stops and row in stops:
                options["stop"] = [stops[row]]
            request = Request(
                self.tokenizer, [int(fixture.inputs["tokens"][row])], options
            )
            request.state_position = int(fixture.inputs["stage0.positions"][0, row])
            request.proposal_offset = 17 + row
            request.counts[:] = fixture.inputs["history"][row]
            request.proposals = device.empty((4, 248320), "float32")
            request.proposal_rows = tuple(
                device.view(request.proposals, (1, 248320), stage * 248320)
                for stage in range(4)
            )
            requests.append(request)
            scheduled.append(
                dict(request_id=row + 1, fa_block_table=tables[row].tolist())
            )
            endpoints[row] = (device.view(self.hidden, (1, 5120), row * 5120),)
        batch = SimpleNamespace(
            requests=requests, scheduled=scheduled, metadata=(None, None, tables)
        )
        records = []
        trace = SimpleNamespace(
            record=lambda *args, **kwargs: records.append((args, kwargs))
        )
        operation = speculation.Speculation(device, self.weights, cache, None, trace)
        update = SamplingInputs.update

        def fail_at_next_prefix(inputs, requests, prefixes, stream):
            if any(prefixes):
                raise ValueError("forced grammar rejection")
            return update(inputs, requests, prefixes, stream)

        with ExitStack() as resources:
            resources.callback(operation.close)
            if not pipeline:
                resources.enter_context(
                    patch.object(speculation, "can_pipeline", return_value=False)
                )
            if reject:
                resources.enter_context(
                    patch.object(SamplingInputs, "update", fail_at_next_prefix)
                )
                with self.assertRaisesRegex(ValueError, "forced grammar rejection"):
                    operation.propose(batch, endpoints, True)
            else:
                # The second call checks replay after the Eager preparation call.
                for iteration in range(2):
                    device.copy(kv, device.upload(fixture.state["kv"], "bfloat16"))
                    if iteration:
                        tables[:] = tables[::-1].copy()
                    for row, request in enumerate(requests):
                        request.drafts = []
                        request.proposal_offset = 17 + row
                        request.state_position = (
                            int(fixture.inputs["stage0.positions"][0, row]) - iteration
                        )
                        scheduled[row]["fa_block_table"] = tables[row].tolist()
                    records.clear()
                    if fault and iteration:
                        self.inject_failure(resources, operation, fault)
                        with self.assertRaisesRegex((OSError, RuntimeError), "forced"):
                            operation.propose(batch, endpoints, True)
                    else:
                        operation.propose(batch, endpoints, True)
            if pipeline:
                self.assertIn(2, operation.pipelines)
            result = dict(
                tokens=[request.drafts.copy() for request in requests],
                offsets=[request.proposal_offset for request in requests],
                presence=[request.presence.copy() for request in requests],
                counts=[request.counts.copy() for request in requests],
                kv=device.download(kv, np.uint16).numpy().copy(),
                records=records.copy(),
                distributions=[
                    device.download(
                        device.view(request.proposals, (len(request.drafts), 248320)),
                        np.float32,
                    )
                    .numpy()
                    .copy()
                    if request.drafts
                    else np.empty((0, 248320), np.float32)
                    for request in requests
                ],
            )
        return result

    def inject_failure(self, resources, operation, fault):
        if fault == "record":
            resources.enter_context(
                patch.object(
                    operation.trace,
                    "record",
                    side_effect=OSError("forced trace failure"),
                )
            )
        elif fault == "graph":
            original = self.device.graph

            def fail_recording(call):
                if getattr(call.__self__, "previous_hidden", None) is not None:
                    raise RuntimeError("forced graph recording failure")
                return original(call)

            resources.enter_context(patch.object(self.device, "graph", fail_recording))
        elif fault == "snapshot":
            original = self.device.module

            class FailSnapshot:
                def __getattr__(self, name):
                    return getattr(original, name)

                def gather_rows(self, *args):
                    raise RuntimeError("forced snapshot failure")

            resources.enter_context(patch.object(self.device, "module", FailSnapshot()))
        else:
            raise ValueError("unsupported pipeline fault")

    def assert_paths(self, expected, actual):
        for key in ("tokens", "offsets", "records"):
            self.assertEqual(expected[key], actual[key], key)
        for key in ("presence", "counts", "distributions"):
            for before, after in zip(expected[key], actual[key], strict=True):
                np.testing.assert_array_equal(before, after, err_msg=key)
        np.testing.assert_array_equal(expected["kv"], actual["kv"])

    def test_grammar_tokens_rng_penalties_and_cache_match(self):
        expected = self.run_path(False)
        actual = self.run_path(True)
        self.assertTrue(all(len(tokens) == 4 for tokens in actual["tokens"]))
        self.assert_paths(expected, actual)
        self.assert_paths(
            self.run_path(False, mixed=True), self.run_path(True, mixed=True)
        )

    def test_thinking_end_switches_to_grammar_without_changing_results(self):
        expected = self.run_path(False, thinking=True)
        actual = self.run_path(True, thinking=True)
        self.assertTrue(
            all(tokens[0] == self.tokenizer.think_end for tokens in actual["tokens"])
        )
        self.assert_paths(expected, actual)

    def test_grammar_end_and_stop_discard_lookahead_cache(self):
        expected = self.run_path(False, ending=True)
        actual = self.run_path(True, ending=True)
        self.assertTrue(any(len(tokens) < 4 for tokens in actual["tokens"]))
        self.assert_paths(expected, actual)
        complete = self.run_path(False)
        stop = self.tokenizer.decode_bytes(complete["tokens"][0][:1]).decode(
            "utf-8", "replace"
        )
        self.assertTrue(stop)
        self.assert_paths(
            self.run_path(False, stops={0: stop}), self.run_path(True, stops={0: stop})
        )

    def test_exception_after_lookahead_restores_only_current_snapshots(self):
        for fault in ("record", "graph", "snapshot"):
            with self.subTest(fault=fault):
                result = self.run_path(True, fault=fault)
                self.assertEqual(result["tokens"], [[], []])
                self.assertEqual(result["offsets"], [17, 18])
                np.testing.assert_array_equal(result["kv"], self.fixture.state["kv"])

    def test_rejected_grammar_restores_unused_kv(self):
        result = self.run_path(True, reject=True)
        self.assertTrue(all(len(tokens) == 1 for tokens in result["tokens"]))
        self.assertEqual(result["offsets"], [18, 19])
        np.testing.assert_array_equal(result["kv"], self.fixture.state["kv"])


if __name__ == "__main__":
    unittest.main()
