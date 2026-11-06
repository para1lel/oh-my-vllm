import json
import math
import mmap
import struct
import warnings
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open

from .layout import tensor_layout
from .loader_layout import _MAPPING as _MAPPING
from .loader_layout import DEFAULT_TILE_ROWS as DEFAULT_TILE_ROWS
from .loader_layout import DIRECT_COPY_BYTES as DIRECT_COPY_BYTES
from .loader_layout import DTYPE_BYTES as DTYPE_BYTES
from .loader_layout import LOAD_CHUNK_BYTES as LOAD_CHUNK_BYTES
from .loader_layout import PARAM_DTYPES as PARAM_DTYPES
from .loader_layout import RAW_DTYPES as RAW_DTYPES
from .loader_layout import _fp8_to_bf16 as _fp8_to_bf16
from .loader_layout import _parameter_specs as _parameter_specs
from .loader_layout import converted_name as converted_name
from .loader_layout import source_index as source_index


class RankLoader:
    """Load only the tensors owned by one rank and keep source reads bounded."""

    def __init__(
        self,
        root,
        config,
        rank,
        *,
        strategy="TEP4",
        tile_rows=DEFAULT_TILE_ROWS,
    ):
        if type(rank) is not int or rank not in range(4):
            raise ValueError("rank must be between zero and three")
        if type(tile_rows) is not int or tile_rows < 1:
            raise ValueError("tile_rows must be positive")
        self.root = Path(root).resolve()
        self.config = config
        self.rank = rank
        self.strategy = strategy
        self.tile_rows = tile_rows
        self.index, self.index_sha256 = source_index(self.root)
        self.layout = tensor_layout(config)
        self.specs = _parameter_specs(config, self.layout, rank, strategy)
        self.sources = {}
        for source_name in self.index:
            name = converted_name(source_name)
            if name is not None:
                self.sources.setdefault(name, source_name)
        self.resources = ExitStack()
        self.headers = {}
        self.mappings = {}
        self.shard_sizes = {}
        self.shards = {
            shard: self.resources.enter_context(
                safe_open(str(self.root / shard), framework="pt", device="cpu")
            )
            for shard in sorted(set(self.index.values()))
        }

    def _read(self, name, selection):
        source_name = self.sources.get(name)
        if source_name is None:
            raise ValueError(f"checkpoint has no source tensor for {name}")
        shard = self.index[source_name]
        with torch.device("cpu"):
            return self.shards[shard].get_slice(source_name)[selection]

    def _read_tile(self, name, spec, first, count):
        shape = self.layout[name]["shape"]
        source_first = first + spec["start"] if spec["axis"] == 0 else first
        source_stop = min(source_first + count, shape[0])
        slices = [slice(source_first, source_stop)] + [slice(None)] * (len(shape) - 1)
        if spec["axis"] == 1:
            slices[1] = slice(spec["start"], spec["start"] + spec["shape"][1])
        raw = self._read(name, tuple(slices))
        if source_stop - source_first < count:
            padding = torch.zeros(
                (count - (source_stop - source_first), *raw.shape[1:]),
                dtype=raw.dtype,
                device=raw.device,
            )
            raw = torch.cat((raw, padding), dim=0)
        return raw

    def _read_parameter(self, name):
        spec = self.specs[name]
        expected = tuple(spec["shape"])
        rows = expected[0] if len(expected) > 1 else 1
        if name.endswith(".wo_a.weight"):
            raw = self._read_tile(name, spec, 0, rows)
            scale_name = name.removesuffix(".weight") + ".scale"
            scale_spec = {
                **spec,
                "stored_dtype": "F8_E8M0",
                "shape": self.layout[scale_name]["shape"],
                "axis": 0,
                "start": spec["start"] // 32,
            }
            scales = self._read_tile(scale_name, scale_spec, 0, (rows + 31) // 32)
            return _fp8_to_bf16(raw, scales)
        if ".engram.embed." in name:
            return self._mapped_parameter(name)
        if len(expected) == 1:
            raw = self._read(
                name,
                (slice(spec["start"], spec["start"] + expected[0]),),
            )
            value = raw.view(PARAM_DTYPES[spec["stored_dtype"]])
            if spec["dtype"] != spec["stored_dtype"]:
                value = value.to(PARAM_DTYPES[spec["dtype"]])
            return value.reshape(expected)
        if rows <= self.tile_rows:
            raw = self._read_tile(name, spec, 0, rows)
            value = raw.view(PARAM_DTYPES[spec["stored_dtype"]])
            if spec["dtype"] != spec["stored_dtype"]:
                value = value.to(PARAM_DTYPES[spec["dtype"]])
            return value.reshape(expected)
        result = torch.empty(
            expected,
            dtype=PARAM_DTYPES[spec["dtype"]],
            device="cpu",
        )
        for first in range(0, rows, self.tile_rows):
            count = min(self.tile_rows, rows - first)
            raw = self._read_tile(name, spec, first, count)
            value = raw.view(PARAM_DTYPES[spec["stored_dtype"]])
            if spec["dtype"] != spec["stored_dtype"]:
                value = value.to(PARAM_DTYPES[spec["dtype"]])
            result[first : first + count].copy_(value)
        return result

    def _copy_parameter(self, name, destination):
        expected = tuple(self.specs[name]["shape"])
        if tuple(destination.shape) != expected:
            raise ValueError(f"source parameter shape differs: {name}")
        value = self._read_parameter(name)
        if ".engram.embed." in name:
            destination.data = value
        else:
            destination.copy_(value.reshape_as(destination))

    def _install_parameter_pools(self, parameters):
        groups = {}
        self._pool_locations = {}
        for name, destination in parameters.items():
            if ".engram.embed." in name:
                continue
            groups.setdefault((destination.device, destination.dtype), []).append(
                (name, destination)
            )

        pools = []
        for (device, dtype), items in groups.items():
            element_size = items[0][1].element_size()
            total = 0
            for _, destination in items:
                total = (total * element_size + 15) // 16 * 16 // element_size
                total += destination.numel()
            storage = torch.empty(total, dtype=dtype, device=device)
            entries = []
            offset = 0
            for name, destination in items:
                offset = (offset * element_size + 15) // 16 * 16 // element_size
                count = destination.numel()
                destination.data = storage[offset : offset + count].view_as(destination)
                entries.append((name, offset, count))
                self._pool_locations[name] = (storage, offset, count)
                offset += count
            pools.append((storage, entries))
        self._parameter_pools = [storage for storage, _ in pools]
        return pools

    @staticmethod
    def _staging_buffer(dtype, count, pinned):
        try:
            return torch.empty(count, dtype=dtype, device="cpu", pin_memory=pinned)
        except RuntimeError:
            return torch.empty(count, dtype=dtype, device="cpu")

    def _load_parameter_pools(self, pools):
        for storage, entries in pools:
            chunk_elements = max(1, LOAD_CHUNK_BYTES // storage.element_size())
            chunk_elements = min(chunk_elements, storage.numel())
            staging = self._staging_buffer(
                storage.dtype,
                chunk_elements,
                storage.device.type == "cuda" and torch.cuda.is_available(),
            )
            storage_offset = 0
            staging_offset = 0

            def flush(storage=storage, staging=staging):
                nonlocal storage_offset, staging_offset
                if staging_offset:
                    storage[storage_offset : storage_offset + staging_offset].copy_(
                        staging[:staging_offset]
                    )
                    storage_offset += staging_offset
                    staging_offset = 0

            for name, _, count in entries:
                value = self._read_parameter(name).reshape(-1)
                if value.numel() != count:
                    raise ValueError(f"source parameter shape differs: {name}")
                if value.numel() * value.element_size() >= DIRECT_COPY_BYTES:
                    flush()
                    storage[storage_offset : storage_offset + count].copy_(value)
                    storage_offset += count
                    continue
                value_offset = 0
                while value_offset < count:
                    copied = min(count - value_offset, chunk_elements - staging_offset)
                    staging[staging_offset : staging_offset + copied].copy_(
                        value[value_offset : value_offset + copied]
                    )
                    staging_offset += copied
                    value_offset += copied
                    if staging_offset == chunk_elements:
                        flush()
            flush()

    def _header(self, shard):
        cached = self.headers.get(shard)
        if cached is None:
            with (self.root / shard).open("rb") as source:
                prefix = source.read(8)
                if len(prefix) != 8:
                    raise ValueError(f"truncated checkpoint header: {shard}")
                header_size = struct.unpack("<Q", prefix)[0]
                header = json.loads(source.read(header_size))
            cached = (8 + header_size, header)
            self.headers[shard] = cached
        return cached

    def _map_shard(self, shard):
        mapped = self.mappings.get(shard)
        if mapped is None:
            source = self.resources.enter_context((self.root / shard).open("rb"))
            mapped = self.resources.enter_context(
                mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ)
            )
            self.mappings[shard] = mapped
        return mapped

    def _source_info(self, name):
        spec = self.specs[name]
        source_name = self.sources.get(name)
        if source_name is None:
            raise ValueError(f"checkpoint has no source tensor for {name}")
        shard = self.index[source_name]
        payload_start, header = self._header(shard)
        record = header.get(source_name)
        if not isinstance(record, dict):
            raise ValueError(f"checkpoint header has no tensor: {source_name}")
        stored_shape = tuple(self.layout[name]["shape"])
        record_shape = record.get("shape")
        if (
            record.get("dtype") != spec["stored_dtype"]
            or not isinstance(record_shape, list)
            or tuple(record_shape) != stored_shape
        ):
            raise ValueError(f"checkpoint tensor metadata differs: {source_name}")
        offsets = record.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) and value >= 0 for value in offsets)
        ):
            raise ValueError(f"invalid checkpoint offsets: {source_name}")
        expected_bytes = math.prod(stored_shape) * DTYPE_BYTES[spec["stored_dtype"]]
        if offsets[1] - offsets[0] != expected_bytes:
            raise ValueError(f"checkpoint tensor byte length differs: {source_name}")
        shard_sizes = getattr(self, "shard_sizes", None)
        if shard_sizes is None:
            shard_sizes = self.shard_sizes = {}
        file_size = shard_sizes.get(shard)
        if file_size is None:
            file_size = (self.root / shard).stat().st_size
            shard_sizes[shard] = file_size
        if payload_start + offsets[1] > file_size:
            raise ValueError(f"checkpoint tensor exceeds its shard: {source_name}")
        return spec, shard, payload_start, offsets, stored_shape

    def _mapped_parameter(self, name):
        spec, shard, payload_start, offsets, stored_shape = self._source_info(name)
        mapped = self._map_shard(shard)
        count = math.prod(stored_shape)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            raw = torch.frombuffer(
                mapped,
                dtype=RAW_DTYPES[spec["stored_dtype"]],
                count=count,
                offset=payload_start + offsets[0],
            ).reshape(stored_shape)
        value = raw.view(PARAM_DTYPES[spec["stored_dtype"]])
        if spec["axis"] is not None:
            available = min(
                spec["shape"][spec["axis"]],
                stored_shape[spec["axis"]] - spec["start"],
            )
            value = value.narrow(spec["axis"], spec["start"], available)
            if available != spec["shape"][spec["axis"]]:
                padded = torch.zeros(
                    spec["shape"],
                    dtype=value.dtype,
                    device="cpu",
                )
                padded.narrow(spec["axis"], 0, available).copy_(value)
                value = padded
        if spec["dtype"] != spec["stored_dtype"]:
            value = value.to(PARAM_DTYPES[spec["dtype"]])
        if tuple(value.shape) != tuple(spec["shape"]):
            raise ValueError(f"source parameter shape differs: {name}")
        return value

    def _device_value(self, name, staging, info, base_offset=0):
        spec, _shard, _payload_start, offsets, stored_shape = info
        raw = staging.narrow(
            0,
            offsets[0] - base_offset,
            offsets[1] - offsets[0],
        )
        raw = raw.view(RAW_DTYPES[spec["stored_dtype"]]).reshape(stored_shape)
        value = raw.view(PARAM_DTYPES[spec["stored_dtype"]])
        if spec["axis"] is not None:
            value = value.narrow(
                spec["axis"],
                spec["start"],
                spec["shape"][spec["axis"]],
            )
        if spec["dtype"] != spec["stored_dtype"]:
            value = value.to(PARAM_DTYPES[spec["dtype"]])
        if tuple(value.shape) != tuple(spec["shape"]):
            raise ValueError(f"source parameter shape differs: {name}")
        return value

    def _load_parameter_pools_by_shard(self):
        names = [
            name for name in self._pool_locations if not name.endswith(".wo_a.weight")
        ]
        if not names:
            return
        infos = {name: self._source_info(name) for name in names}
        by_shard = {}
        for name, info in infos.items():
            _spec, shard, _payload_start, _offsets, _shape = info
            by_shard.setdefault(shard, []).append(name)
        ranges_by_shard = {}
        for shard, shard_names in by_shard.items():
            ranges = []
            for name in sorted(shard_names, key=lambda item: infos[item][3][0]):
                start, end = infos[name][3]
                if not ranges or start - ranges[-1][1] > 4 << 20:
                    ranges.append([start, end, [name]])
                else:
                    ranges[-1][1] = max(ranges[-1][1], end)
                    ranges[-1][2].append(name)
            ranges_by_shard[shard] = ranges
        device = next(
            storage.device
            for storage, _offset, _count in self._pool_locations.values()
            if storage.device.type == "cuda"
        )
        staging = torch.empty(
            max(
                end - start
                for ranges in ranges_by_shard.values()
                for start, end, _ in ranges
            ),
            dtype=torch.uint8,
            device=device,
        )
        for shard, ranges in ranges_by_shard.items():
            payload_start = infos[ranges[0][2][0]][2]
            mapped = self._map_shard(shard)
            for start, end, shard_names in ranges:
                payload_size = end - start
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    source = torch.frombuffer(
                        mapped,
                        dtype=torch.uint8,
                        count=payload_size,
                        offset=payload_start + start,
                    )
                staging[:payload_size].copy_(source)
                for name in shard_names:
                    value = self._device_value(
                        name,
                        staging,
                        infos[name],
                        base_offset=start,
                    ).reshape(-1)
                    storage, offset, count = self._pool_locations[name]
                    if value.numel() != count:
                        raise ValueError(f"source parameter shape differs: {name}")
                    storage[offset : offset + count].copy_(value)
        torch.cuda.synchronize(device)

    def load_into(self, model):
        parameters = dict(model.named_parameters())
        if parameters.keys() != self.specs.keys():
            missing = sorted(self.specs.keys() - parameters.keys())
            extra = sorted(parameters.keys() - self.specs.keys())
            raise ValueError(
                f"source parameters differ: missing={missing}, extra={extra}"
            )
        with torch.no_grad():
            pools = self._install_parameter_pools(parameters)
            for name, destination in parameters.items():
                if ".engram.embed." in name or name.endswith(".wo_a.weight"):
                    self._copy_parameter(name, destination)
            if any(storage.device.type == "cuda" for storage, _ in pools):
                self._load_parameter_pools_by_shard()
                torch.cuda.empty_cache()
            else:
                self._load_parameter_pools(pools)
        parameter_bytes = {
            "dense_device_bytes": 0,
            "routed_expert_device_bytes": 0,
            "host_parameter_bytes": 0,
        }
        for name, parameter in parameters.items():
            if parameter.device.type != "cuda":
                kind = "host_parameter_bytes"
            elif ".ffn.experts." in name:
                kind = "routed_expert_device_bytes"
            else:
                kind = "dense_device_bytes"
            parameter_bytes[kind] += parameter.numel() * parameter.element_size()
        return {
            "format": "original-safetensors-stream",
            "world_size": 4,
            "rank": self.rank,
            "parameter_count": len(parameters),
            **parameter_bytes,
        }

    def close(self):
        self.resources.close()
