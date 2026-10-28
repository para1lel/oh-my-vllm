import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .deepseek_agentx import service_max_num_seqs
from .deepseek_benchmark import (
    effective_cache_configuration,
    effective_service_configuration,
    request_body,
    verify_artifact_identity,
)


class DeepSeekBenchmark(unittest.TestCase):
    def test_agentx_default_sequence_capacity_follows_concurrency(self):
        self.assertEqual(service_max_num_seqs(16), 16)
        self.assertEqual(service_max_num_seqs(64), 64)
        self.assertEqual(service_max_num_seqs(16, 32), 32)
        with self.assertRaises(ValueError):
            service_max_num_seqs(16, 65)

    def test_identity_uses_environment_cache_capacity_when_no_cli_override_exists(self):
        arguments = SimpleNamespace(blocks=None, mamba_blocks=None)
        with patch.dict("os.environ", {"OH_MY_VLLM_NUM_GPU_BLOCKS": "17"}):
            configuration = effective_cache_configuration(arguments)

        self.assertEqual(configuration, {"blocks": 17, "mamba_blocks": 128})

    def test_service_identity_uses_the_effective_sequence_capacity(self):
        self.assertEqual(
            effective_service_configuration(SimpleNamespace(max_num_seqs=16)),
            {"max_num_batched_tokens": 32768, "max_num_seqs": 16},
        )
        self.assertEqual(
            effective_service_configuration(SimpleNamespace()),
            {"max_num_batched_tokens": 32768, "max_num_seqs": 64},
        )

    def test_request_body_uses_the_service_reasoning_field(self):
        body = request_body([{"role": "user", "content": "hello"}], 4, 7)

        self.assertNotIn("effort", body)
        self.assertEqual(body["reasoning_effort"], "off")
        self.assertTrue(body["ignore_eos"])

    def test_verify_artifact_identity_rejects_changed_measurement_inputs(self):
        with (
            patch(
                "development.serving.deepseek_benchmark.artifact_identity",
                return_value={"checkpoint": "changed"},
            ),
            self.assertRaisesRegex(RuntimeError, "changed"),
        ):
            verify_artifact_identity(
                object(),
                object(),
                (),
                {"checkpoint": "expected"},
                workload={},
            )


if __name__ == "__main__":
    unittest.main()
