import json
from pathlib import Path

import numpy as np

from ..fixtures import MAX_SHARD_BYTES, file_digest


def write_fixture(root, case, arrays, dtypes, writes, checkpoint, reference):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "version": "operator-fixture-v1",
        "case": case.definition(),
        "checkpoint": checkpoint,
        "reference": reference,
        "files": {},
        "tensors": {},
        "writes": writes,
    }
    for name, array in arrays.items():
        array = np.asarray(array)
        record = {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "tensor_dtype": dtypes[name],
            "parts": [],
        }
        flat = array.reshape(-1)
        # The ZIP and NPY headers also count toward the shard size limit.
        capacity = (MAX_SHARD_BYTES - 4096) // array.dtype.itemsize
        for offset in range(0, flat.size, capacity):
            count = min(capacity, flat.size - offset)
            shard = f"{name}.{len(record['parts']):04d}.npz"
            with (root / shard).open("xb") as destination:
                np.savez_compressed(destination, value=flat[offset : offset + count])
            manifest["files"][shard] = file_digest(root / shard)
            record["parts"].append(
                {"file": shard, "key": "value", "offset": offset, "count": count}
            )
        manifest["tensors"][name] = record
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return file_digest(path)
