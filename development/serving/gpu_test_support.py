import ctypes
import fcntl
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


WRAPPER = ROOT / "scripts/with-gpu.sh"


GPU_ENV = (
    "CUDA_VISIBLE_DEVICES",
    "OH_MY_VLLM_GPU_COUNT",
    "OH_MY_VLLM_GPU_UUID",
    "OH_MY_VLLM_GPU_UUIDS",
    "OH_MY_VLLM_GPU_EXCLUDE_INDICES",
    "OH_MY_VLLM_RUN_ID",
)


class DriverFunction:
    def __init__(self, function):
        self.function = function

    def __call__(self, *arguments):
        return self.function(*arguments)


class FakeDriver:
    def __init__(self, identities, count=None, failure=None):
        self.identities = identities
        self.count = len(identities) if count is None else count
        self.failure = failure
        self.cuInit = DriverFunction(lambda flags: int(failure == "init"))
        self.cuDeviceGetCount = DriverFunction(self.get_count)
        self.cuDeviceGet = DriverFunction(self.get_device)
        self.cuDeviceGetUuid_v2 = DriverFunction(self.get_uuid)

    def get_count(self, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = self.count
        return int(self.failure == "count")

    def get_device(self, output, ordinal):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = ordinal + 10
        return int(self.failure == "device")

    def get_uuid(self, output, handle):
        ordinal = handle - 10
        ctypes.memmove(output, uuid.UUID(self.identities[ordinal][4:]).bytes, 16)
        return int(self.failure == "uuid")


class GpuEnvironment(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        identities = sorted(f"GPU-{uuid.uuid4()}" for _ in range(6))
        self.uuids = [identities[index] for index in (0, 3, 1, 2, 4, 5)]
        self.addCleanup(self.remove_owned_locks)
        self.inventory = [
            [index, identity, "NVIDIA B200", 0, 0]
            for index, identity in enumerate(self.uuids)
        ]
        self.fixture = self.directory / "inventory.json"
        self.fake_smi = self.directory / "nvidia-smi"
        self.fake_smi.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "data = json.loads(Path(os.environ['GPU_TEST_FIXTURE']).read_text())\n"
            "query = sys.argv[1]\n"
            "if query.startswith('--query-gpu='):\n"
            "    keys = query.split('=', 1)[1].split(',')\n"
            "    names = ['index', 'uuid', 'name', 'memory.used', 'utilization.gpu']\n"
            "    if 'raw' in data:\n"
            "        print(data['raw'])\n"
            "    else:\n"
            "        for row in data['inventory']:\n"
            "            print(', '.join(str(row[names.index(k)]) for k in keys))\n"
            "elif query == '--query-compute-apps=gpu_uuid':\n"
            "    print(data.get('processes', ''), end='')\n"
            "else:\n"
            "    sys.exit(90)\n"
        )
        self.fake_smi.chmod(0o700)
        self.write_inventory()
        self.environment = {
            key: value for key, value in os.environ.items() if key not in GPU_ENV
        } | {
            "PATH": f"{self.directory}:{os.environ['PATH']}",
            "GPU_TEST_FIXTURE": str(self.fixture),
            "OH_MY_VLLM_GPU_EXCLUDE_INDICES": "0",
        }

    def remove_owned_locks(self):
        for identity in self.uuids:
            for path in Path("/tmp").glob(f"oh-my-vllm-{identity}*.lock"):
                path.unlink()

    def write_inventory(self, **extra):
        self.fixture.write_text(json.dumps({"inventory": self.inventory} | extra))

    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("development.serving.gpu"),
            "the CPU-only group lease module is missing",
        )
        return importlib.import_module("development.serving.gpu")

    def run_wrapper(self, settings=None, command=None, input=None):
        return subprocess.run(
            [str(WRAPPER), *(command or [sys.executable, "-c", "print('EXECUTED')"])],
            env=self.environment | (settings or {}),
            input=input,
            capture_output=True,
            text=True,
            timeout=4,
        )

    def assert_unlocked(self, identities, directory):
        for identity in identities:
            with (directory / f"oh-my-vllm-{identity}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
