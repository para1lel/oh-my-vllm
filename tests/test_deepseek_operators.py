import json
import struct
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock

import torch
from oh_my_vllm.models.deepseek_v41.loader import RankLoader
from oh_my_vllm.operators.activation.plan import make_plan as make_activation_plan
from oh_my_vllm.operators.hyperconnection.plan import (
    make_plan as make_hyperconnection_plan,
)
from oh_my_vllm.operators.projection.plan import make_plan as make_projection_plan
from oh_my_vllm.operators.routing.plan import make_plan as make_routing_plan
from oh_my_vllm.operators.sparse_attention.plan import (
    make_plan as make_sparse_attention_plan,
)
from oh_my_vllm.operators.transform.plan import make_plan as make_transform_plan


class DeepSeekOperatorContracts(unittest.TestCase):
    def test_rank_loader_maps_a_host_engram_slice_from_safetensors(self):
        name = "layers.1.engram.embed.weight"
        values = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32)
        raw = values.to(torch.float8_e4m3fn).view(torch.uint8).numpy().tobytes()
        header = json.dumps(
            {
                name: {
                    "dtype": "F8_E4M3",
                    "shape": [8, 32],
                    "data_offsets": [0, len(raw)],
                }
            },
            separators=(",", ":"),
        ).encode()
        header += b" " * (-len(header) % 8)

        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory) / "model.safetensors"
            shard.write_bytes(struct.pack("<Q", len(header)) + header + raw)
            loader = RankLoader.__new__(RankLoader)
            loader.root = Path(directory)
            loader.index = {name: shard.name}
            loader.sources = {name: name}
            loader.layout = {name: {"shape": [8, 32]}}
            loader.specs = {
                name: {
                    "shape": (4, 32),
                    "dtype": "F8_E4M3",
                    "stored_dtype": "F8_E4M3",
                    "axis": 0,
                    "start": 4,
                }
            }
            loader.resources = ExitStack()
            loader.headers = {}
            loader.mappings = {}
            destination = torch.empty(4, 32, dtype=torch.float8_e4m3fn)

            loader._copy_parameter(name, destination)
            expected = destination.float().clone()
            del destination
            loader.close()

        self.assertTrue(
            torch.equal(expected, values.to(torch.float8_e4m3fn)[4:].float())
        )

    def test_rank_loader_reads_sharded_one_dimensional_parameters_from_the_rank_slice(
        self,
    ):
        loader = RankLoader.__new__(RankLoader)
        loader.specs = {
            "layers.0.attn.attn_sink": {
                "shape": (16,),
                "dtype": "F32",
                "stored_dtype": "F32",
                "axis": 0,
                "start": 16,
            }
        }
        loader._read = Mock(return_value=torch.arange(16, dtype=torch.float32))
        destination = torch.empty(16)

        loader._copy_parameter("layers.0.attn.attn_sink", destination)

        self.assertEqual(
            loader._read.call_args.args[1],
            (slice(16, 32),),
        )
        self.assertTrue(torch.equal(destination, torch.arange(16, dtype=torch.float32)))

    def test_native_operator_plans_cover_deepseek_shapes(self):
        self.assertEqual(
            make_activation_plan(rows=32, width=2304, limit=10, routed=True).width,
            2304,
        )
        self.assertEqual(make_routing_plan(rows=64, experts=384).topk, 6)
        self.assertEqual(
            make_sparse_attention_plan(rows=64, heads=64, slots=640).slots,
            640,
        )
        self.assertEqual(
            make_projection_plan(rows=64, outputs=5120, width=5120, bits=8).bits,
            8,
        )
        self.assertEqual(make_hyperconnection_plan(rows=64).rows, 64)
        self.assertEqual(
            make_transform_plan(kind="rms", rows=64, width=5120, epsilon=1e-20).kind,
            "rms",
        )
        self.assertEqual(
            make_transform_plan(kind="rope", rows=64, width=512, heads=64).heads,
            64,
        )

    def test_native_operator_plans_reject_unrelated_shapes(self):
        with self.assertRaises(ValueError):
            make_activation_plan(rows=32, width=5120, limit=10, routed=True)
        with self.assertRaises(ValueError):
            make_routing_plan(rows=64, experts=256)
        with self.assertRaises(ValueError):
            make_sparse_attention_plan(rows=64, heads=32, slots=641)
        with self.assertRaises(ValueError):
            make_projection_plan(rows=64, outputs=5120, width=5120, bits=3)


if __name__ == "__main__":
    unittest.main()
