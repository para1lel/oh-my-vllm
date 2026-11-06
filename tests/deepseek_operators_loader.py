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


class DeepSeekOperatorContractsLoader(unittest.TestCase):
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
        loader.rank = 0
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

    def test_rank_loader_advances_rows_for_axis_one_tiles(self):
        name = "layers.0.attn.wo_b.weight"
        loader = RankLoader.__new__(RankLoader)
        loader.layout = {name: {"shape": [4096, 64]}}
        loader.specs = {
            name: {
                "shape": (4096, 16),
                "dtype": "F32",
                "stored_dtype": "F32",
                "axis": 1,
                "start": 16,
            }
        }
        loader.tile_rows = 2048

        def read(_, selection):
            first, stop = selection[0].start, selection[0].stop
            rows = torch.arange(first, stop, dtype=torch.float32).unsqueeze(1)
            return rows.expand(-1, 16)

        loader._read = Mock(side_effect=read)
        destination = torch.empty(4096, 16)

        loader._copy_parameter(name, destination)

        self.assertEqual(loader._read.call_count, 2)
        self.assertEqual(loader._read.call_args_list[1].args[1][0], slice(2048, 4096))
        self.assertEqual(destination[0, 0], 0)
        self.assertEqual(destination[2048, 0], 2048)

    def test_rank_loader_loads_multiple_parameter_dtypes_through_the_public_boundary(
        self,
    ):
        loader = RankLoader.__new__(RankLoader)
        loader.rank = 0
        loader.specs = {
            "first": {"shape": (2, 3), "dtype": "BF16"},
            "second": {"shape": (4,), "dtype": "F32"},
        }
        values = {
            "first": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            "second": torch.arange(4, dtype=torch.float32),
        }
        loader._read_parameter = Mock(side_effect=values.__getitem__)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Parameter(torch.empty(2, 3, dtype=torch.bfloat16))
                self.second = torch.nn.Parameter(torch.empty(4, dtype=torch.float32))

        model = TinyModel()
        info = loader.load_into(model)

        self.assertEqual(info["parameter_count"], 2)
        torch.testing.assert_close(model.first, values["first"])
        torch.testing.assert_close(model.second, values["second"])
        self.assertEqual(model.first.data_ptr() % 16, 0)
        self.assertEqual(model.second.data_ptr() % 16, 0)

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
