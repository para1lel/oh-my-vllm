import copy
import hashlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from oh_my_vllm.models.descriptor import validate_descriptor
from oh_my_vllm.worker import zmq_bridge
from oh_my_vllm.worker.placeholder import Worker


def stub_config():
    return dict(
        model_path="stub",
        block_size=784,
        tensor_parallel_size=1,
        num_gpu_blocks=8,
        mamba_blocks=7,
        host_cache_bytes=0,
        max_model_len=128,
        max_num_seqs=4,
        max_num_batched_tokens=784,
        num_speculative_tokens=4,
        speculative_mode="mtp",
        descriptor_version=1,
    )


class ModelReady(unittest.TestCase):
    def test_qwen_ready_returns_a_private_descriptor_with_actual_capacities(self):
        from oh_my_vllm.models.qwen38.inference import Worker as QwenWorker

        from .test_model_descriptor import sample_descriptor

        self.assertEqual(QwenWorker.__module__, "oh_my_vllm.models.qwen38.inference")
        worker = QwenWorker.__new__(QwenWorker)
        worker.descriptor = sample_descriptor(
            fa_capacity_blocks=11, gdn_capacity_blocks=9
        )
        worker.cache = SimpleNamespace(blocks=11, mamba_blocks=9)
        worker.cache_memory = {"policy": "fixed"}
        worker.offload = SimpleNamespace(limit=123)
        result = worker.ready()
        self.assertEqual(result["descriptor"], worker.descriptor)
        self.assertEqual(
            (
                result["logical_num_blocks"],
                result["mamba_blocks"],
                result["host_cache_bytes"],
            ),
            (11, 9, 123),
        )
        result["descriptor"]["cache_schema"]["groups"][0]["layers"].clear()
        self.assertEqual(
            worker.ready()["descriptor"]["cache_schema"]["groups"][0]["layers"],
            [*range(3, 64, 4), 64],
        )

    def test_init_version_rejection_precedes_worker_construction(self):
        for version in (None, True, 0, 2, "1"):
            config = stub_config()
            if version is None:
                config.pop("descriptor_version")
            else:
                config["descriptor_version"] = version
            with patch.object(zmq_bridge, "create_worker") as create:
                with (
                    self.subTest(version=version),
                    self.assertRaisesRegex(ValueError, "descriptor version"),
                ):
                    zmq_bridge.initialize(config)
                create.assert_not_called()
        worker = zmq_bridge.initialize(stub_config())
        self.assertIsInstance(worker, Worker)

    def test_stub_ready_is_explicit_and_binds_resolved_metadata_capacity(self):
        config = stub_config()
        worker = Worker(config)
        reply = worker.ready()
        descriptor = reply["descriptor"]
        validate_descriptor(descriptor)
        self.assertEqual(descriptor["architecture"], "OhMyVllmStubForTesting")
        self.assertEqual(descriptor["model_type"], "oh_my_vllm_stub")
        self.assertEqual(descriptor["max_context_tokens"], 262144)
        self.assertEqual(descriptor["weight_formats"], ["f32"])
        self.assertEqual(
            descriptor["checkpoint_sha256"],
            hashlib.sha256(b"oh-my-vllm/stub/checkpoint/v1").hexdigest(),
        )
        self.assertEqual(
            descriptor["tokenizer_sha256"],
            hashlib.sha256(b"oh-my-vllm/stub/tokenizer/v1").hexdigest(),
        )
        fa, gdn = descriptor["cache_schema"]["groups"]
        self.assertEqual((fa["capacity_blocks"], gdn["capacity_blocks"]), (8, 7))
        for group in (fa, gdn):
            self.assertEqual(group["kind"], "model_checkpoint")
            self.assertEqual(group["layers"], [0])
            self.assertEqual(group["block_tokens"], 784)
        self.assertEqual(fa["storage"]["id"], "oh_my_vllm_stub_fa_metadata")
        self.assertEqual(gdn["storage"]["id"], "oh_my_vllm_stub_gdn_metadata")
        changed = Worker(dict(config, num_gpu_blocks=None, mamba_blocks=None)).ready()
        self.assertEqual(changed["logical_num_blocks"], 1024)
        self.assertEqual(
            [
                g["capacity_blocks"]
                for g in changed["descriptor"]["cache_schema"]["groups"]
            ],
            [1024, 1024],
        )
        for key in ("config_sha256", "checkpoint_sha256", "tokenizer_sha256"):
            self.assertEqual(changed["descriptor"][key], descriptor[key])
        saved = copy.deepcopy(descriptor)
        descriptor["cache_schema"]["groups"][0]["layers"].clear()
        self.assertEqual(worker.ready()["descriptor"], saved)


if __name__ == "__main__":
    unittest.main()
