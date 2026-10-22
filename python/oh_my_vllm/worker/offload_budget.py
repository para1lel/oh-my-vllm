from pathlib import Path


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("host memory availability is unknown")


def cgroup_remaining(root, leaf, *, version=2):
    root, leaf = Path(root), Path(leaf)
    leaf.relative_to(root)
    limit_name, used_name = (
        ("memory.max", "memory.current")
        if version == 2
        else ("memory.limit_in_bytes", "memory.usage_in_bytes")
    )
    remaining = []
    for path in (leaf, *leaf.parents):
        limit_file = path / limit_name
        if limit_file.exists():
            limit = limit_file.read_text().strip()
            if limit != "max":
                used = int((path / used_name).read_text().strip())
                remaining.append(max(0, int(limit) - used))
        if path == root:
            break
    return min(remaining) if remaining else None


def mount_path(value):
    for encoded, decoded in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return Path(value)


def process_memory_remaining():
    groups = []
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        _, controllers, path = line.split(":", 2)
        if not controllers or "memory" in controllers.split(","):
            groups.append((2 if not controllers else 1, Path(path)))
    remaining = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields, filesystem = line.split(" - ", 1)
        fields, filesystem = fields.split(), filesystem.split()
        version = 2 if filesystem[0] == "cgroup2" else 1
        if filesystem[0] not in ("cgroup", "cgroup2"):
            continue
        if version == 1 and "memory" not in filesystem[2].split(","):
            continue
        mount_root, mount = mount_path(fields[3]), mount_path(fields[4])
        for group_version, group in groups:
            if group_version != version:
                continue
            if group.is_relative_to(mount_root):
                relative = group.relative_to(mount_root)
            elif group == Path("/"):
                relative = Path(".")
            else:
                continue
            value = cgroup_remaining(mount, mount / relative, version=version)
            if value is not None:
                remaining.append(value)
    return min(remaining) if remaining else None


def host_budget(requested, gpu_bytes):
    if requested is not None and (
        type(requested) is not int or not 0 <= requested <= 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("host_cache_bytes must be a nonnegative integer or null")
    if requested == 0:
        return 0
    available = available_memory()
    remaining = process_memory_remaining()
    if remaining is not None:
        available = min(available, remaining)
    limit = min(gpu_bytes, available // 4)
    return limit if requested is None else min(limit, requested)
