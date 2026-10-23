# GPU group leases

## Scope

The [Phase 2 plan](plan.md#phase-2-add-typed-cache-groups-and-four-rank-control) specifies a lease for one group of four B200 GPUs.
The [parallel strategies](design.md#parallel-strategies) use this same group.
The wrapper also supports the existing single-B200 commands.

This change supplies device selection and cooperative file locks.
It does not supply NCCL communication, distributed worker control, state transactions, or capacity acceptance.
Rust controls scheduling and logical state.
The [execution rules](../execution.md) apply to subsequent GPU computation.

## Command and settings

Use the existing command interface:

```bash
scripts/with-env.sh scripts/with-gpu.sh COMMAND [ARGS...]
```

Set the environment variables before this command.
Put host settings in ignored `LOCAL.md`.

| Variable | Meaning |
|---|---|
| `OH_MY_VLLM_GPU_COUNT` | `1` or `4`. The default is `1`. |
| `OH_MY_VLLM_GPU_UUID` | One full GPU UUID. This setting is valid only with count `1`. |
| `OH_MY_VLLM_GPU_UUIDS` | An ordered, comma-separated list of full GPU UUIDs. Its length must equal the count. |
| `OH_MY_VLLM_GPU_EXCLUDE_INDICES` | Comma-separated nonnegative physical indices. The default list does not include any indices. |
| `OH_MY_VLLM_RUN_ID` | A run ID supplied by the caller. An empty or missing value gets a new UUID. |

Do not combine `OH_MY_VLLM_GPU_UUID` with `OH_MY_VLLM_GPU_UUIDS`.
Use canonical GPU UUIDs with lowercase hexadecimal digits and the `GPU-` prefix.
Do not put spaces or empty elements in lists.

Exclusion has priority over explicit selection.
An explicit list fails if a device has an excluded index, is missing, or is not a B200.
Malformed settings and duplicate UUIDs also fail before command execution.
An explicit device that is busy causes a wait.

The explicit UUID list sets rank order in `CUDA_VISIBLE_DEVICES`.
Automatic selection uses increasing physical index order.
The wrapper sets `CUDA_VISIBLE_DEVICES` from its selection and replaces any previous value.
Lock order does not change rank order.

## Selection and ownership

The [lease module](../../development/serving/gpu.py) reads the full inventory and compute-process list through `nvidia-smi`.
Only B200 devices with at most 64 MiB of used memory, zero utilization, and no compute process are idle.
Malformed inventory or process output causes an error.
Unavailable measurements do not count as idle.

Each attempt selects the full group before it gets locks.
The module gets nonblocking exclusive locks in increasing UUID order.
The lock path is `/tmp/oh-my-vllm-${uuid}.lock`.
This path is compatible with the previous single-GPU wrapper.
Lock files must be regular files.
A FIFO lock path causes an error without a wait.

If one lock is unavailable, the module closes all descriptors from that attempt.
Automatic selection then tries a group without that device before the next wait.
Explicit selection waits for its specified group.
It holds no partial group while it waits.

After the module gets all locks, it reads the full inventory and process list again.
A busy, removed, or changed selected device causes group release.
An invalid explicit selection causes an error after release.

Other failed attempts wait 10 seconds before another attempt.
There is no wait deadline or fairness guarantee.

The wrapper prints the selected UUIDs and run ID to standard error.
It then uses `exec` to start the command with inherited lock descriptors.
Standard input, standard output, standard error, process identity, exit status, and command signals use the command process.
Python's ignored pipe and file-size signals are reset before `exec`.

A missing command exits with status `127`.
Other command execution errors use status `126`.
Selection and inventory errors use status `2`.

The operating system releases each lock when its last inherited descriptor closes.
A parent can keep all lease descriptors open until all GPU children stop.
Those children can use `close_fds=True`.

A child must inherit the descriptors if it can run after its parent exits.
It must also inherit them if its parent closes its descriptors before the child stops.
If ownership moves to a child, that child must keep the lock descriptors until its GPU workers stop.
For these cases, use `pass_fds` when a child starts GPU workers with `close_fds=True`.

Do not remove a lock file while a process can hold its lock.
The wrapper does not change device clocks or stop other users' processes.

Cooperative locks protect only commands that use the same lock files.
Another process can start GPU work after the final inventory check.
The lease cannot prevent that race.

## CUDA identity checks

The [device module](../../development/serving/device.py) supplies `selected_uuids()`.
It accepts one or four distinct full UUIDs in `CUDA_VISIBLE_DEVICES`.
If count or UUID settings are supplied, the visible list must agree with those settings.
It checks that `cuDeviceGetCount` equals the selected count.
It maps each logical ordinal through `cuDeviceGet` and checks its UUID against the selected order.
It does not silently accept a smaller visible set.

`selected_uuid()` accepts one selected GPU only.
Existing single-GPU consumers therefore reject a four-GPU selection.
The CPU checks use a controlled driver fixture.
They do not initialize CUDA on a GPU.
