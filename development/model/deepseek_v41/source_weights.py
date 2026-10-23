"""Bounded original-checkpoint loading for the released MP4 source baseline."""

import math

import numpy as np
from oh_my_vllm.models.deepseek_v41.checkpoint import DTYPE_BYTES
from oh_my_vllm.models.deepseek_v41.placement import _placement_rules

from .identity import local_file, open_file, unchanged
from .layout import tensor_layout
from .weights import RAW_DTYPES, Weights

TORCH_DTYPES = {
    "BF16": "bfloat16",
    "F32": "float32",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E8M0": "float8_e8m0fnu",
    "I8": "float4_e2m1fn_x2",
}
_codes = np.arange(256, dtype=np.uint16)
_exponents = ((_codes >> 3) & 15).astype(np.int16)
_FP8 = np.ldexp(
    np.where(_exponents == 0, _codes & 7, (_codes & 7) + 8).astype(np.float32),
    np.where(_exponents == 0, -9, _exponents - 10),
)
_FP8[_codes >= 128] *= -1


def _parameters(config, layout, rank):
    rules = _placement_rules(config, "TEP4")
    result = {}
    for name, stored in layout.items():
        rule = rules[name]
        if rule["placement"] == "consumed_on_load":
            continue
        owner = rule["owner_rank"]
        if (
            owner is not None
            and owner != rank
            and not name.startswith(("vision.", "aligner.", "image_"))
        ):
            continue
        shape = list(stored["shape"])
        axis, start = rule["shard_axis"], 0
        dtype = rule["promoted_dtype"] or stored["dtype"]
        if ".engram.embed." in name:
            axis, shape[0] = 0, (shape[0] + 3) // 4
            start = rank * shape[0]
        else:
            if ".shared_experts." in name or name.startswith(
                ("vision.", "aligner.", "image_")
            ):
                axis = None
            if axis is not None:
                if shape[axis] % 4:
                    raise ValueError(f"Unsupported source shard shape: {name}")
                shape[axis] //= 4
                start = rank * shape[axis]
        if name.startswith("vision.") and name.endswith(
            (".norm1.weight", ".norm2.weight", ".norm.weight")
        ):
            dtype = "F32"
        result[name] = dict(
            shape=shape,
            dtype=dtype,
            stored_dtype=stored["dtype"],
            axis=axis,
            start=start,
        )
    return result


class SourceWeights:
    """config is the root checkpoint config.json, not the source ModelArgs."""

    def __init__(self, root, manifest, config, rank, read_bytes=64 << 20):
        if type(rank) is not int or not 0 <= rank < 4:
            raise ValueError("Source baseline requires one of four ranks")
        if type(read_bytes) is not int or read_bytes <= 0:
            raise ValueError("read_bytes must be a positive integer")
        self.reader = Weights(root, manifest, tensor_layout(config))
        if self.reader._control("config.json", decode=True) != config:
            raise ValueError("Source configuration differs from checkpoint identity")
        self.rank, self.read_bytes = rank, read_bytes
        self.parameters = _parameters(config, self.reader.layout, rank)
        if any(
            math.prod(spec["shape"][1:]) * DTYPE_BYTES[spec["stored_dtype"]]
            > read_bytes
            for spec in self.parameters.values()
        ):
            raise ValueError("read_bytes cannot hold one source row")

    def _read(self, name, selection):
        self.reader._verify_states()
        filename = self.reader.index[name]
        header, base = self.reader._prepare(filename)
        entry = header[name]
        shape = entry["shape"]
        size = math.prod(
            len(range(*s.indices(n))) for s, n in zip(selection, shape, strict=True)
        )
        if size * DTYPE_BYTES[entry["dtype"]] > self.read_bytes:
            raise ValueError("Source tile exceeds read_bytes")
        path = local_file(self.reader.root, filename)
        with open_file(path) as stream:
            unchanged(path, stream, self.reader.states[filename])
            mapped = np.memmap(
                stream,
                mode="r",
                dtype=RAW_DTYPES[entry["dtype"]],
                offset=base + entry["data_offsets"][0],
                shape=tuple(shape),
            )
            try:
                result = np.array(mapped[selection], copy=True)
            finally:
                mapped._mmap.close()
            unchanged(path, stream, self.reader.states[filename])
        return result

    def _wo_a(self, name, raw, first):
        scale_name = name.removesuffix(".weight") + ".scale"
        last = first + raw.shape[0]
        scales = self._read(
            scale_name, (slice(first // 32, (last + 31) // 32), slice(None))
        )
        if np.any((raw & 127) == 127) or np.any(scales == 255):
            raise ValueError(f"Nonfinite wo_a encoding: {name}")
        scale_values = np.ldexp(
            np.ones(scales.shape, dtype=np.float32), scales.astype(np.int16) - 127
        )
        values = _FP8[raw]
        with np.errstate(over="ignore"):
            values.reshape(raw.shape[0], -1, 32)[:] *= scale_values[
                np.arange(first, last) // 32 - first // 32, :, None
            ]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite wo_a conversion: {name}")
        bits = values.view(np.uint32)
        bits += np.uint32(0x7FFF) + ((bits >> 16) & 1)
        result = (bits >> 16).astype("<u2")
        if np.any((result & 0x7F80) == 0x7F80):
            raise ValueError(f"BF16 wo_a overflow: {name}")
        return result

    def tiles(self, name):
        """Yield (local row offset, NumPy storage); only wo_a is converted."""
        if name not in self.parameters:
            raise ValueError(f"Unknown or nonowned source parameter: {name}")
        spec = self.parameters[name]
        source_shape = self.reader.layout[name]["shape"]
        row_bytes = math.prod(spec["shape"][1:]) * DTYPE_BYTES[spec["stored_dtype"]]
        conversion = name.endswith(".wo_a.weight")
        step = max(1, self.read_bytes // (row_bytes * (16 if conversion else 1)))
        for row in range(0, spec["shape"][0], step):
            self.reader._verify_states()
            count = min(step, spec["shape"][0] - row)
            first = row + (spec["start"] if spec["axis"] == 0 else 0)
            stop = min(first + count, source_shape[0])
            selection = [slice(first, stop)] + [slice(None)] * (len(source_shape) - 1)
            if spec["axis"] == 1:
                selection[1] = slice(spec["start"], spec["start"] + spec["shape"][1])
            if stop - first == count:
                raw = self._read(name, tuple(selection))
            else:
                raw = np.full(
                    (count, *spec["shape"][1:]),
                    127 if name.endswith(".scale") else 0,
                    dtype=RAW_DTYPES[spec["stored_dtype"]],
                )
                if stop > first:
                    raw[: stop - first] = self._read(name, tuple(selection))
            if conversion:
                raw = self._wo_a(name, raw, first)
            self.reader._verify_states()
            yield row, raw

    def load_into(self, model):
        """Copy into existing source parameters; aliases and scale links stay intact."""
        destinations = dict(model.named_parameters())
        if destinations.keys() != self.parameters.keys():
            missing = self.parameters.keys() - destinations.keys()
            extra = destinations.keys() - self.parameters.keys()
            raise ValueError(
                f"Source parameters differ: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        for name, spec in self.parameters.items():
            parameter = destinations[name]
            if (
                tuple(parameter.shape) != tuple(spec["shape"])
                or str(parameter.dtype) != "torch." + TORCH_DTYPES[spec["dtype"]]
            ):
                raise ValueError(f"Source parameter shape/dtype differs: {name}")
        import torch

        with torch.no_grad():
            for name, spec in self.parameters.items():
                dtype = (
                    "BF16" if name.endswith(".wo_a.weight") else spec["stored_dtype"]
                )
                for row, raw in self.tiles(name):
                    tile = torch.from_numpy(raw).view(
                        getattr(torch, TORCH_DTYPES[dtype])
                    )
                    destinations[name][row : row + raw.shape[0]].copy_(tile)
        self.reader._verify_states()
        return dict(
            format="released-mp4-baseline",
            world_size=4,
            rank=self.rank,
            parameter_count=len(self.parameters),
            read_bytes=self.read_bytes,
            parameter_bytes=sum(
                math.prod(s["shape"]) * DTYPE_BYTES[s["dtype"]]
                for s in self.parameters.values()
            ),
            checkpoint_layout_sha256=self.reader.manifest["layout_sha256"],
            runtime_feasibility_verified=False,
        )
