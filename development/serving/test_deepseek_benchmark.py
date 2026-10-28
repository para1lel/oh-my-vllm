import unittest
from types import SimpleNamespace
from unittest.mock import patch

from .deepseek_benchmark import (
    effective_cache_configuration,
    request_body,
    verify_artifact_identity,
)


class DeepSeekBenchmark(unittest.TestCase):
    def test_identity_uses_environment_cache_capacity_when_no_cli_override_exists(self):
        arguments = SimpleNamespace(blocks=None, mamba_blocks=None)
        with patch.dict("os.environ", {"OH_MY_VLLM_NUM_GPU_BLOCKS": "17"}):
            configuration = effective_cache_configuration(arguments)

        self.assertEqual(configuration, {"blocks": 17, "mamba_blocks": 128})

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
