import ctypes
import fcntl
import importlib
import json
import os
import select
import signal
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from . import device

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


class GpuGroups(unittest.TestCase):
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

    def test_one_and_four_selection_keep_rank_order_and_exclude_physical_zero(self):
        command = [
            sys.executable,
            "-c",
            "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])",
        ]
        cases = [
            ({}, self.uuids[1]),
            ({"OH_MY_VLLM_GPU_EXCLUDE_INDICES": ""}, self.uuids[0]),
            ({"OH_MY_VLLM_GPU_COUNT": "4"}, ",".join(self.uuids[1:5])),
            ({"OH_MY_VLLM_GPU_UUID": self.uuids[4]}, self.uuids[4]),
            ({"OH_MY_VLLM_GPU_UUIDS": self.uuids[2]}, self.uuids[2]),
            (
                {
                    "OH_MY_VLLM_GPU_COUNT": "4",
                    "OH_MY_VLLM_GPU_UUIDS": ",".join(self.uuids[1:5][::-1]),
                    "CUDA_VISIBLE_DEVICES": "should-be-replaced",
                },
                ",".join(self.uuids[1:5][::-1]),
            ),
        ]
        for settings, expected in cases:
            with self.subTest(settings=settings):
                result = self.run_wrapper(settings, command)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)
                self.assertIn(expected, result.stderr)
                self.assertIn("run=", result.stderr)
        with Path(f"/tmp/oh-my-vllm-{self.uuids[1]}.lock").open("a") as other:
            fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for count, expected in (
                ("1", self.uuids[2]),
                ("4", ",".join(self.uuids[2:6])),
            ):
                with self.subTest(alternate_count=count):
                    try:
                        result = self.run_wrapper(
                            {"OH_MY_VLLM_GPU_COUNT": count}, command
                        )
                    except subprocess.TimeoutExpired:
                        self.fail(
                            f"count {count} waited despite a complete unlocked group"
                        )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout.strip(), expected)
            self.assert_unlocked(self.uuids[2:6], Path("/tmp"))
        self.inventory.reverse()
        self.inventory[4][3] = 65
        self.inventory[3][4] = 1
        self.inventory[2][3] = 64
        self.write_inventory(processes=self.uuids[4] + "\n")
        result = self.run_wrapper(command=command)
        self.assertEqual(result.stdout.strip(), self.uuids[3], result.stderr)

    def test_invalid_settings_and_inventory_fail_before_exec(self):
        cases = (
            [{"OH_MY_VLLM_GPU_COUNT": value} for value in ("", "0", "2", "04", "-1")]
            + [
                {"OH_MY_VLLM_GPU_UUID": self.uuids[0]},
                {"OH_MY_VLLM_GPU_UUID": self.uuids[1], "OH_MY_VLLM_GPU_COUNT": "4"},
                {"OH_MY_VLLM_GPU_UUIDS": ",".join(self.uuids[1:3])},
                {"OH_MY_VLLM_GPU_UUIDS": self.uuids[0]},
                {"OH_MY_VLLM_GPU_UUIDS": self.uuids[1] + ","},
                {"OH_MY_VLLM_GPU_UUIDS": "GPU-../../escape"},
                {"OH_MY_VLLM_GPU_UUID": "GPU-not-a-uuid"},
                {"OH_MY_VLLM_GPU_UUIDS": f"GPU-{uuid.uuid4()}"},
                {
                    "OH_MY_VLLM_GPU_COUNT": "4",
                    "OH_MY_VLLM_GPU_UUIDS": ",".join([self.uuids[1]] * 4),
                },
                {
                    "OH_MY_VLLM_GPU_UUID": self.uuids[1],
                    "OH_MY_VLLM_GPU_UUIDS": self.uuids[1],
                },
            ]
            + [
                {"OH_MY_VLLM_GPU_EXCLUDE_INDICES": value}
                for value in ("-1", "1,", "a", "1.5", " 1")
            ]
        )
        for settings in cases:
            with self.subTest(settings=settings):
                result = self.run_wrapper(settings)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("EXECUTED", result.stdout)
        for raw, processes in (
            ("garbage", ""),
            ("", ""),
            (f"0,{self.uuids[0]},NVIDIA B200,N/A,0", ""),
            (f"-1,{self.uuids[0]},NVIDIA B200,0,0", ""),
            (f"0,{self.uuids[0]},NVIDIA B200,0,101", ""),
            (
                f"0,{self.uuids[0]},NVIDIA B200,0,0\n0,{self.uuids[1]},NVIDIA B200,0,0",
                "",
            ),
            (
                f"0,{self.uuids[0]},NVIDIA B200,0,0\n1,{self.uuids[0]},NVIDIA B200,0,0",
                "",
            ),
            (f"0,{self.uuids[0]},NVIDIA B200,0,0", "[N/A]\n"),
            (f"0,{self.uuids[0]},NVIDIA B200,0,0", self.uuids[0] + ", extra\n"),
            (f'1,"{self.uuids[1]},{self.uuids[2]}",NVIDIA B200,0,0', ""),
        ):
            with self.subTest(raw=raw, processes=processes):
                self.write_inventory(raw=raw, processes=processes)
                result = self.run_wrapper()
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertNotIn("EXECUTED", result.stdout)
        self.inventory[1][2] = "NVIDIA A100"
        self.write_inventory()
        result = self.run_wrapper({"OH_MY_VLLM_GPU_UUID": self.uuids[1]})
        self.assertEqual(result.returncode, 2, result.stderr)

    def test_middle_lock_and_recheck_release_every_lock_before_wait(self):
        gpu = self.module()
        settings = gpu.Settings.from_environment(
            {
                "OH_MY_VLLM_GPU_COUNT": "4",
                "OH_MY_VLLM_GPU_UUIDS": ",".join(self.uuids[1:5]),
                "OH_MY_VLLM_GPU_EXCLUDE_INDICES": "0",
            }
        )
        devices = tuple(gpu.Gpu(*row) for row in self.inventory)
        selected = self.uuids[1:5]
        middle = sorted(selected)[1]
        with (
            patch.object(gpu, "LOCK_DIRECTORY", self.directory),
            patch.object(gpu, "read_inventory", return_value=devices),
            patch.object(gpu, "read_processes", return_value=set()),
        ):
            with (self.directory / f"oh-my-vllm-{middle}.lock").open("a") as other:
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertIsNone(gpu.acquire_group(settings))
                self.assert_unlocked(
                    [identity for identity in selected if identity != middle],
                    self.directory,
                )
            busy = list(devices)
            busy[2] = gpu.Gpu(2, self.uuids[2], "NVIDIA B200", 65, 0)
            for changed, processes in (
                (tuple(busy), set()),
                (devices, {self.uuids[3]}),
            ):
                with (
                    patch.object(gpu, "read_inventory", side_effect=[devices, changed]),
                    patch.object(gpu, "read_processes", side_effect=[set(), processes]),
                ):
                    self.assertIsNone(gpu.acquire_group(settings))
                    self.assert_unlocked(selected, self.directory)
            with patch.object(
                gpu, "read_inventory", side_effect=[devices, ValueError("bad")]
            ):
                with self.assertRaises(ValueError):
                    gpu.acquire_group(settings)
                self.assert_unlocked(selected, self.directory)
            with (
                patch.object(gpu.time, "sleep", side_effect=InterruptedError),
                patch.object(gpu, "read_processes", return_value=set(self.uuids[1:3])),
                self.assertRaises(InterruptedError),
            ):
                gpu.wait_for_group(settings)
            self.assert_unlocked(selected, self.directory)
        fifo = self.directory / f"oh-my-vllm-{middle}.lock"
        fifo.unlink()
        os.mkfifo(fifo)
        probe = (
            "import fcntl, os, sys\n"
            "from pathlib import Path\n"
            "from development.serving import gpu\n"
            "gpu.LOCK_DIRECTORY = Path(sys.argv[1])\n"
            "status = gpu.main([sys.executable, '-c', \"print('EXECUTED')\"])\n"
            "for identity in os.environ['OH_MY_VLLM_GPU_UUIDS'].split(','):\n"
            "    path = gpu.LOCK_DIRECTORY / f'oh-my-vllm-{identity}.lock'\n"
            "    if not path.is_fifo():\n"
            "        with path.open('a') as lock:\n"
            "            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
            "print('partial locks released')\n"
            "sys.exit(status)\n"
        )
        for with_reader in (False, True):
            with self.subTest(fifo_reader=with_reader):
                reader = (
                    os.open(fifo, os.O_RDONLY | os.O_NONBLOCK) if with_reader else None
                )
                try:
                    try:
                        result = subprocess.run(
                            [sys.executable, "-c", probe, str(self.directory)],
                            env=self.environment
                            | {
                                "OH_MY_VLLM_GPU_COUNT": "4",
                                "OH_MY_VLLM_GPU_UUIDS": ",".join(selected),
                            },
                            capture_output=True,
                            text=True,
                            timeout=2,
                        )
                    except subprocess.TimeoutExpired:
                        self.fail("FIFO blocked instead of releasing partial locks")
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertNotIn("EXECUTED", result.stdout)
                    self.assertIn("partial locks released", result.stdout)
                finally:
                    if reader is not None:
                        os.close(reader)

    def test_lock_order_is_separate_from_rank_order_and_busy_devices_are_skipped(self):
        gpu = self.module()
        requested = self.uuids[1:5][::-1]
        settings = gpu.Settings.from_environment(
            {"OH_MY_VLLM_GPU_COUNT": "4", "OH_MY_VLLM_GPU_UUIDS": ",".join(requested)}
        )
        devices = tuple(gpu.Gpu(*row) for row in self.inventory)
        order = []
        original_open = os.open

        def open_lock(path, *arguments, **keywords):
            order.append(Path(path).name)
            return original_open(path, *arguments, **keywords)

        with (
            patch.object(gpu, "LOCK_DIRECTORY", self.directory),
            patch.object(gpu, "read_inventory", return_value=devices),
            patch.object(gpu, "read_processes", return_value=set()),
            patch.object(gpu.os, "open", side_effect=open_lock),
        ):
            lease = gpu.acquire_group(settings)
            self.assertIsNotNone(lease)
            try:
                self.assertEqual([item.uuid for item in lease.devices], requested)
                self.assertEqual(
                    order, [f"oh-my-vllm-{item}.lock" for item in sorted(requested)]
                )
            finally:
                lease.close()
            self.assert_unlocked(requested, self.directory)

    def test_exec_keeps_locks_stdio_pid_exit_status_and_signal_cleanup(self):
        requested = ",".join(self.uuids[1:5][::-1])
        settings = {
            "OH_MY_VLLM_GPU_COUNT": "4",
            "OH_MY_VLLM_GPU_UUIDS": requested,
            "OH_MY_VLLM_RUN_ID": "fixture-run",
        }
        script = (
            "import json, os, sys\n"
            "print(json.dumps([os.getpid(), os.environ['CUDA_VISIBLE_DEVICES'], "
            "os.environ['OH_MY_VLLM_RUN_ID']]), flush=True)\n"
            "print(sys.stdin.readline().strip(), flush=True)\n"
            "print('child-stderr', file=sys.stderr, flush=True)\n"
            "sys.exit(17)\n"
        )
        for terminate in (False, True):
            with self.subTest(terminate=terminate):
                process = subprocess.Popen(
                    [str(WRAPPER), sys.executable, "-c", script],
                    env=self.environment | settings,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                try:
                    ready, _, _ = select.select([process.stdout], [], [], 4)
                    self.assertTrue(ready, "wrapper did not exec the command")
                    identity = json.loads(process.stdout.readline())
                    self.assertEqual(identity, [process.pid, requested, "fixture-run"])
                    for item in self.uuids[1:5]:
                        with (
                            Path(f"/tmp/oh-my-vllm-{item}.lock").open("a") as lock,
                            self.assertRaises(BlockingIOError),
                        ):
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if terminate:
                        process.send_signal(signal.SIGTERM)
                        output, errors = process.communicate(timeout=4)
                        self.assertEqual(process.returncode, -signal.SIGTERM)
                    else:
                        output, errors = process.communicate(
                            "stdin-payload\n", timeout=4
                        )
                        self.assertEqual(process.returncode, 17)
                        self.assertEqual(output.strip(), "stdin-payload")
                        self.assertIn("child-stderr", errors)
                    self.assertIn("fixture-run", errors)
                finally:
                    if process.poll() is None:
                        process.kill()
                    process.communicate(timeout=4)
                self.assert_unlocked(self.uuids[1:5], Path("/tmp"))
        result = self.run_wrapper(settings, ["/missing-gpu-test-command"])
        self.assertEqual(result.returncode, 127)
        self.assert_unlocked(self.uuids[1:5], Path("/tmp"))
        result = self.run_wrapper(settings, [str(self.fixture)])
        self.assertEqual(result.returncode, 126)
        self.assert_unlocked(self.uuids[1:5], Path("/tmp"))
        result = self.run_wrapper(settings, ["/bin/sh", "-c", "kill -PIPE $$; exit 25"])
        self.assertEqual(result.returncode, -signal.SIGPIPE)
        self.assert_unlocked(self.uuids[1:5], Path("/tmp"))

    def test_cuda_logical_mapping_and_count_are_exact_and_single_consumer_rejects_four(
        self,
    ):
        requested = self.uuids[1:5][::-1]
        with (
            patch.dict(
                os.environ, {"CUDA_VISIBLE_DEVICES": ",".join(requested)}, clear=True
            ),
            patch.object(device.ctypes, "CDLL", return_value=FakeDriver(requested)),
        ):
            self.assertTrue(
                hasattr(device, "selected_uuids"), "group CUDA validation is missing"
            )
            self.assertEqual(device.selected_uuids(), tuple(requested))
            with self.assertRaises(ValueError):
                device.selected_uuid()
        for identities, count, failure in (
            (requested[::-1], 4, None),
            (requested, 3, None),
            (requested, 5, None),
            (requested, 4, "init"),
            (requested, 4, "count"),
            (requested, 4, "device"),
            (requested, 4, "uuid"),
        ):
            with (
                self.subTest(count=count, failure=failure, identities=identities),
                patch.dict(
                    os.environ,
                    {"CUDA_VISIBLE_DEVICES": ",".join(requested)},
                    clear=True,
                ),
                patch.object(
                    device.ctypes,
                    "CDLL",
                    return_value=FakeDriver(identities, count, failure),
                ),
                self.assertRaises((ValueError, RuntimeError)),
            ):
                device.selected_uuids()
        with (
            patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": requested[0]}, clear=True),
            patch.object(device.ctypes, "CDLL", return_value=FakeDriver(requested[:1])),
        ):
            self.assertEqual(device.selected_uuid(), requested[0])
        for value in (
            "",
            "0",
            "GPU-invalid",
            requested[0] + ",",
            ",".join([requested[0]] * 4),
        ):
            with (
                self.subTest(value=value),
                patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": value}, clear=True),
                patch.object(
                    device.ctypes,
                    "CDLL",
                    side_effect=AssertionError("CUDA called for bad input"),
                ),
                self.assertRaises(ValueError),
            ):
                device.selected_uuids()
        for settings in (
            {"CUDA_VISIBLE_DEVICES": requested[0], "OH_MY_VLLM_GPU_COUNT": "4"},
            {"CUDA_VISIBLE_DEVICES": ",".join(requested), "OH_MY_VLLM_GPU_COUNT": "1"},
            {
                "CUDA_VISIBLE_DEVICES": ",".join(requested),
                "OH_MY_VLLM_GPU_UUIDS": ",".join(requested[::-1]),
            },
            {
                "CUDA_VISIBLE_DEVICES": requested[0],
                "OH_MY_VLLM_GPU_UUID": requested[1],
            },
        ):
            with (
                self.subTest(settings=settings),
                patch.dict(os.environ, settings, clear=True),
                patch.object(
                    device.ctypes,
                    "CDLL",
                    side_effect=AssertionError("CUDA called for a masked selection"),
                ),
                self.assertRaises(ValueError),
            ):
                device.selected_uuids()


if __name__ == "__main__":
    unittest.main()
