import unittest
from types import SimpleNamespace
from unittest.mock import patch

from oh_my_vllm.models.deepseek_v41.trace import execution_modes, runtime_identity

from . import deepseek_accounting as source
from . import deepseek_accounting_packed as packed
from .deepseek_benchmark import runtime_reference
from .deepseek_runtime_trace import assemble_trace
from .test_deepseek_benchmark import run_fixture, runtime_fixture


class RuntimeModes(unittest.TestCase):
    def test_launcher_defaults_case_and_graph_limits_match_runtime_settings(self):
        with (
            patch.dict("os.environ", {}, clear=True),
            patch(
                "oh_my_vllm.models.deepseek_v41.kernel.native_sparse_graph_enabled",
                return_value=True,
            ),
        ):
            self.assertEqual(
                execution_modes(),
                {
                    "batch_decode": False,
                    "decode_graph": False,
                    "decode_graph_limit": 32,
                    "proposal_graph": False,
                },
            )
            for strategy in ("TP4", "TEP4", "DEP4"):
                for limit in (0, 32):
                    with patch.dict(
                        "os.environ",
                        {
                            "OH_MY_VLLM_DEEPSEEK_STRATEGY": strategy,
                            "OH_MY_VLLM_DEEPSEEK_BATCH_DECODE": "1",
                            "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS": str(limit),
                        },
                    ):
                        self.assertEqual(
                            execution_modes(),
                            {
                                "batch_decode": True,
                                "decode_graph": strategy == "TP4",
                                "decode_graph_limit": limit,
                                "proposal_graph": False,
                            },
                        )
            with (
                patch.dict("os.environ", {"OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS": "-1"}),
                self.assertRaises(ValueError),
            ):
                execution_modes()

    def test_runtime_identity_uses_parsed_worker_flags_after_environment_changes(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "OH_MY_VLLM_SOURCE_COMMIT": "cpu-fixture-commit",
                    "OH_MY_VLLM_DEEPSEEK_STRATEGY": "tep4",
                    "OH_MY_VLLM_DEEPSEEK_BATCH_DECODE": "0",
                    "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPH": "0",
                    "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS": "0",
                    "OH_MY_VLLM_DEEPSEEK_MTP_GRAPH": "0",
                },
                clear=True,
            ),
            patch("oh_my_vllm.models.deepseek_v41.trace.NATIVE_MODULES", ()),
            patch(
                "oh_my_vllm.models.deepseek_v41.trace.source_identity",
                return_value="cpu-fixture-source",
            ),
            patch(
                "torch.cuda.get_device_properties",
                return_value=SimpleNamespace(uuid="GPU-cpu-fixture"),
            ),
            patch(
                "oh_my_vllm.models.deepseek_v41.kernel.native_sparse_graph_enabled",
                side_effect=AssertionError("must use parsed Worker graph flag"),
            ),
        ):
            identities = [
                runtime_identity(
                    2,
                    strategy="tp4",
                    batch_decode=True,
                    decode_graph=True,
                    decode_graph_limit=limit,
                    proposal_graph=proposal,
                )
                for limit, proposal in ((0, True), (32, True), (32, False))
            ]
        self.assertNotEqual(identities[0], identities[1])
        self.assertNotEqual(identities[1], identities[2])
        for identity, limit, proposal in zip(
            identities, (0, 32, 32), (True, True, False), strict=True
        ):
            self.assertEqual(
                identity["execution_modes"],
                {
                    "batch_decode": True,
                    "decode_graph": True,
                    "decode_graph_limit": limit,
                    "proposal_graph": proposal,
                },
            )

    def test_proposal_graph_identity_matches_startup_gates_and_zero_limit_fallback(
        self,
    ):
        from oh_my_vllm.models.deepseek_v41.inference import Worker

        cases = (
            ("tp4", "0", "1", True, 32, True, False),
            ("tp4", "1", "1", True, 32, True, True),
            ("tep4", "1", "1", True, 32, False, False),
            ("dep4", "1", "1", True, 32, False, False),
            ("tp4", "1", "0", True, 32, False, False),
            ("tp4", "1", "1", False, 32, False, False),
            ("tp4", "1", "1", True, 0, True, True),
        )
        observed = []
        for strategy, mtp, target, sparse, limit, graph, proposal in cases:
            with (
                self.subTest(
                    strategy=strategy,
                    mtp=mtp,
                    target=target,
                    sparse=sparse,
                    limit=limit,
                ),
                patch.dict(
                    "os.environ",
                    {
                        "OH_MY_VLLM_DEEPSEEK_MTP_GRAPH": mtp,
                        "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPH": target,
                        "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPHS": str(limit),
                    },
                    clear=True,
                ),
                patch(
                    "oh_my_vllm.models.deepseek_v41.kernel._SPARSE_NATIVE_DISABLED",
                    not sparse,
                ),
                patch(
                    "oh_my_vllm.models.deepseek_v41.kernel._SPARSE_NATIVE_FAILED", False
                ),
                patch("oh_my_vllm.models.deepseek_v41.dep4.Dep4Executor"),
            ):
                worker = Worker.__new__(Worker)
                worker.model = SimpleNamespace(layers=())
                worker._configure_execution({"max_num_seqs": 1}, strategy, strategy)
                modes = execution_modes(strategy)
                observed.append(modes)
                self.assertIs(modes["decode_graph"], graph)
                self.assertIs(modes["proposal_graph"], proposal)
                self.assertIs(worker.decode_graph_enabled, graph)
                self.assertIs(worker.proposal_graph_enabled, proposal)
                self.assertEqual(worker.decode_graph_limit, limit)
                if not proposal or limit == 0:
                    self.assertIsNone(worker._proposal_graph_step(None, None, 0, 0))
        self.assertNotEqual(observed[0], observed[1])

    def test_observed_modes_reach_trace_binding_and_changed_rank_is_rejected(self):
        with (
            patch.dict(
                "os.environ",
                {
                    "OH_MY_VLLM_DEEPSEEK_BATCH_DECODE": "1",
                    "OH_MY_VLLM_DEEPSEEK_DECODE_GRAPH": "1",
                    "OH_MY_VLLM_DEEPSEEK_MTP_GRAPH": "1",
                },
                clear=True,
            ),
            patch(
                "oh_my_vllm.models.deepseek_v41.kernel.native_sparse_graph_enabled",
                return_value=True,
            ),
        ):
            modes = {
                strategy: execution_modes(strategy)
                for strategy in ("tp4", "tep4", "dep4")
            }
        self.assertEqual(
            modes["tp4"],
            {
                "batch_decode": True,
                "decode_graph": True,
                "decode_graph_limit": 32,
                "proposal_graph": True,
            },
        )
        self.assertEqual(
            modes["tep4"],
            {
                "batch_decode": True,
                "decode_graph": False,
                "decode_graph_limit": 32,
                "proposal_graph": False,
            },
        )
        case = dict(
            mode="dspark5", strategy="tp4", batch=1, input_tokens=20, output_tokens=4
        )
        harness, events, responses = runtime_fixture(
            case, reference_version=packed.VERSION
        )
        harness["configuration"] = {"execution_modes": modes}
        for rows in events:
            rows[0]["execution"]["runtime"]["execution_modes"] = modes["tp4"].copy()
        trace = assemble_trace(events, responses, harness=harness, case=case)
        self.assertEqual(trace["binding"]["runtime_modes"], modes["tp4"])
        result, _metadata = runtime_reference(
            trace,
            strategy="tp4",
            expected_binding=trace["binding"],
            version=packed.VERSION,
        )
        self.assertEqual(result.version, packed.VERSION)
        for flag in ("batch_decode", "proposal_graph"):
            events[2][0]["execution"]["runtime"]["execution_modes"][flag] = False
            with self.assertRaisesRegex(source.BindingError, "modes"):
                assemble_trace(events, responses, harness=harness, case=case)
            events[2][0]["execution"]["runtime"]["execution_modes"] = modes[
                "tp4"
            ].copy()
        measured, _csv = run_fixture(
            case, harness, events, responses, version=packed.VERSION
        )
        self.assertEqual(measured["status"], "completed")
        self.assertEqual(measured["measurement_identity"], harness)
