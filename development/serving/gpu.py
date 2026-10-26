import csv
import fcntl
import os
import re
import signal
import stat
import subprocess
import sys
import time
import uuid as uuid_module
from dataclasses import dataclass
from pathlib import Path

LOCK_DIRECTORY = Path("/tmp")
UUID_PATTERN = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}")
UNSIGNED_PATTERN = re.compile(r"[0-9]+")
B200_PATTERN = re.compile(r"\bB200\b")


def parse_uuids(value):
    identities = tuple(value.split(","))
    if not all(UUID_PATTERN.fullmatch(identity) for identity in identities):
        raise ValueError("GPU UUIDs must be full, canonical GPU UUIDs")
    if len(set(identities)) != len(identities):
        raise ValueError("GPU UUIDs must be distinct")
    return identities


@dataclass(frozen=True)
class Settings:
    count: int
    requested: tuple[str, ...]
    excluded: frozenset[int]

    @classmethod
    def from_environment(cls, environment):
        count = environment.get("OH_MY_VLLM_GPU_COUNT", "1")
        if count not in ("1", "4"):
            raise ValueError("OH_MY_VLLM_GPU_COUNT must be 1 or 4")
        single = environment.get("OH_MY_VLLM_GPU_UUID", "")
        multiple = environment.get("OH_MY_VLLM_GPU_UUIDS")
        if single and multiple is not None:
            raise ValueError("OH_MY_VLLM_GPU_UUID and OH_MY_VLLM_GPU_UUIDS conflict")
        if single and count != "1":
            raise ValueError("OH_MY_VLLM_GPU_UUID selects exactly one GPU")
        requested = (
            parse_uuids(multiple if multiple is not None else single)
            if (multiple is not None or single)
            else ()
        )
        if requested and len(requested) != int(count):
            raise ValueError("GPU UUID list length must equal OH_MY_VLLM_GPU_COUNT")
        excluded = environment.get("OH_MY_VLLM_GPU_EXCLUDE_INDICES", "")
        indices = excluded.split(",") if excluded else []
        if not all(UNSIGNED_PATTERN.fullmatch(index) for index in indices):
            raise ValueError(
                "excluded GPU indices must be comma-separated nonnegative integers"
            )
        return cls(int(count), requested, frozenset(int(index) for index in indices))


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    name: str
    memory: int
    utilization: int

    def idle(self, processes):
        return (
            self.memory <= 64 and self.utilization == 0 and self.uuid not in processes
        )


def query_nvidia(query):
    return subprocess.check_output(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"], text=True
    )


def read_inventory():
    output = query_nvidia("--query-gpu=index,uuid,name,memory.used,utilization.gpu")
    devices = []
    indices = set()
    identities = set()
    for row in csv.reader(output.splitlines(), strict=True):
        if len(row) != 5:
            raise ValueError("invalid nvidia-smi inventory row")
        index, identity, name, memory, utilization = (value.strip() for value in row)
        if len(parse_uuids(identity)) != 1:
            raise ValueError("inventory rows must identify exactly one GPU")
        if not name or not all(
            UNSIGNED_PATTERN.fullmatch(value) for value in (index, memory, utilization)
        ):
            raise ValueError("invalid nvidia-smi inventory fields")
        if int(utilization) > 100 or int(index) in indices or identity in identities:
            raise ValueError(
                "invalid or duplicate nvidia-smi inventory identity or utilization"
            )
        indices.add(int(index))
        identities.add(identity)
        devices.append(Gpu(int(index), identity, name, int(memory), int(utilization)))
    if not devices:
        raise ValueError("nvidia-smi returned an empty inventory")
    return tuple(sorted(devices, key=lambda device: device.index))


def read_processes():
    output = query_nvidia("--query-compute-apps=gpu_uuid")
    processes = set()
    for row in output.splitlines():
        identities = parse_uuids(row.strip())
        if len(identities) != 1:
            raise ValueError("invalid nvidia-smi compute process row")
        processes.add(identities[0])
    return processes


def candidate_devices(settings, inventory):
    by_uuid = {device.uuid: device for device in inventory}
    if settings.requested:
        devices = []
        for identity in settings.requested:
            device = by_uuid.get(identity)
            if device is None:
                raise ValueError("requested GPU UUID is not in the device inventory")
            if device.index in settings.excluded:
                raise ValueError("requested GPU physical index is excluded")
            if not B200_PATTERN.search(device.name):
                raise ValueError("requested GPU is not a B200")
            devices.append(device)
        return tuple(devices)
    return tuple(
        device
        for device in sorted(inventory, key=lambda device: device.index)
        if device.index not in settings.excluded and B200_PATTERN.search(device.name)
    )


@dataclass
class Lease:
    devices: tuple[Gpu, ...]
    descriptors: list[int]

    def close(self):
        while self.descriptors:
            os.close(self.descriptors.pop())


def lock_devices(lease):
    for device in sorted(lease.devices, key=lambda device: device.uuid):
        path = LOCK_DIRECTORY / f"oh-my-vllm-{device.uuid}.lock"
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666
        )
        lease.descriptors.append(descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("GPU lock path must be a regular file")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return device.uuid
    return None


def acquire_group(settings):
    candidates = candidate_devices(settings, read_inventory())
    processes = read_processes()
    if settings.requested and any(not device.idle(processes) for device in candidates):
        return None
    candidates = tuple(device for device in candidates if device.idle(processes))
    while len(candidates) >= settings.count:
        lease = Lease(candidates[: settings.count], [])
        transferred = False
        try:
            blocked = lock_devices(lease)
            if blocked is not None:
                if settings.requested:
                    return None
                candidates = tuple(
                    device for device in candidates if device.uuid != blocked
                )
                continue
            inventory = candidate_devices(settings, read_inventory())
            processes = read_processes()
            by_uuid = {device.uuid: device for device in inventory}
            for device in lease.devices:
                current = by_uuid.get(device.uuid)
                if (
                    current is None
                    or current.index != device.index
                    or not current.idle(processes)
                ):
                    return None
            transferred = True
            return lease
        finally:
            if not transferred:
                lease.close()
    return None


def wait_for_group(settings):
    while True:
        lease = acquire_group(settings)
        if lease is not None:
            return lease
        print(
            f"waiting for {settings.count} idle B200 GPU(s)",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(10)


def main(arguments=None):
    arguments = sys.argv[1:] if arguments is None else arguments
    if not arguments:
        print("usage: scripts/with-gpu.sh COMMAND [ARGS...]", file=sys.stderr)
        return 2
    lease = None
    try:
        settings = Settings.from_environment(os.environ)
        lease = wait_for_group(settings)
        identities = ",".join(device.uuid for device in lease.devices)
        os.environ["CUDA_VISIBLE_DEVICES"] = identities
        run_id = os.environ.get("OH_MY_VLLM_RUN_ID") or str(uuid_module.uuid4())
        os.environ["OH_MY_VLLM_RUN_ID"] = run_id
        print(f"run={run_id} gpu={identities}", file=sys.stderr, flush=True)
        for descriptor in lease.descriptors:
            os.set_inheritable(descriptor, True)
        # Python ignores these signals; restore the command's usual exec behavior.
        for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
            if number := getattr(signal, name, None):
                signal.signal(number, signal.SIG_DFL)
        try:
            os.execvp(arguments[0], arguments)
        except FileNotFoundError as error:
            print(str(error), file=sys.stderr)
            return 127
        except OSError as error:
            print(str(error), file=sys.stderr)
            return 126
    except (
        ValueError,
        OSError,
        subprocess.SubprocessError,
        csv.Error,
        UnicodeError,
    ) as error:
        print(f"GPU lease: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    raise SystemExit(main())
