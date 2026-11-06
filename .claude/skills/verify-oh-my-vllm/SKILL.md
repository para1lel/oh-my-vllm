---
name: verify-oh-my-vllm
description: Verify Chat, Responses, tools, service lifecycle, or offline CLI behavior through the Rust app with recorded evidence.
---

# Verify the Rust app

Run commands from the repository root.
Use the [development environment](../../../docs/development.md) first.
This skill drives the Rust CLI and HTTP app.

**The default helper checks service and transport with a CPU worker fixture.
It does not prove GPU inference, numerical correctness, model capacity, or performance.**

## Launch

### CPU service path

Set `ARTIFACT_ROOT` from ignored `LOCAL.md`.
Keep this directory external to the repository.
Build the service when its source is stable:

```bash
scripts/with-env.sh cargo build --bin oh-my-vllm-zmq-worker
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature chat --binary target/debug/oh-my-vllm-zmq-worker
```

One invocation performs Launch, Doctor, Drive, Evidence, and Cleanup.
The service runs only during that invocation.
The helper selects an available loopback port and different IPC paths in a new scratch directory.
It starts each Rust process in its own process group.
It sets `CUDA_VISIBLE_DEVICES=''` for the helper and its children.

The helper calls [the fixture installer](../../../tests/worker_fixture.py).
It does not import `tests.test_service` or depend on `OH_MY_VLLM_TEST_BINARY`.
The fixture supplies:

- A checkpoint configuration and a worker launcher in scratch storage.
- A synthetic Qwen descriptor with checkpoint and tokenizer identities.
- Prompt token preparation and fixed text, tool, and JSON replies.
- Worker readiness, execution results, cache messages, and worker statistics.

The Rust HTTP server, request parser, scheduler, response storage, and ZMQ/msgpack boundary run through their service paths.
The fixture does not load checkpoint weights, a tokenizer, native modules, or GPU state.
It checks Qwen reply parsing in the shared service.
It does not check DeepSeek model execution.
See [the existing service cases](../../../tests/test_service.py) for the behavior source.

During Rust changes, `--binary` can select an available executable.
The evidence records one executable digest and one source snapshot.
Their build relationship is unverified.
Build and repeat the drive after the final source freeze.

### Checkpoint service path

Use [the service guide](../../../docs/serving.md) for checkpoint inference.
Set `MODEL_ROOT`, `CUDA_HOME`, `OH_MY_VLLM_CUTLASS`, and `OH_MY_VLLM_KERNEL_CACHE` from local configuration.
Use the selected model's guide for GPU count, adapter options, checkpoint files, and acceptance conditions.
Obtain GPU ownership through [the execution rules](../../../docs/execution.md).
Use CUDA 13.4 and the specified native build dependencies.

```bash
scripts/with-env.sh cargo build --release --bin oh-my-vllm-zmq-worker
scripts/with-env.sh scripts/with-gpu.sh python -m development.native_build
VERIFY_SCRATCH="$(mktemp -d /tmp/omv-native-XXXXXX)"
env -u OH_MY_VLLM_WORKER_PYTHON scripts/with-env.sh scripts/with-gpu.sh target/release/oh-my-vllm-zmq-worker --model "$MODEL_ROOT" --socket "$VERIFY_SCRATCH/worker.ipc" serve --listen 127.0.0.1:8000
```

The last command runs in the foreground.
If the example port is not available, use a different listen address from local configuration.
The `env -u` command removes a fixture worker override before the environment wrapper selects the inference interpreter.
This path loads the checkpoint and native modules through the Python inference worker.
The CPU helper does not run this path.
Use [model acceptance](../../../docs/testing.md#model-acceptance) for GPU numerical and performance evidence.

## Doctor

The helper captures the executable's `--help` output and dependency versions.
It polls `GET /diagnostics` until the worker PID is available.
It checks that the reported server PID is its new Rust child.
It checks that the worker PID belongs to that service.

It then reads `GET /v1/models` for the public model ID.
The step is completed when `doctor.json` contains these records.

Use `--launch-timeout` to change the default 15-second launch deadline.
Read `failure.txt`, the process logs, and recorded HTTP errors after a failure.
A port selection race causes a failed launch. Repeat the command with a new scratch directory.

For the checkpoint path, read `/v1/models` and `/diagnostics` at the selected listen address.
Check the selected model's readiness conditions before inference.
A fixture descriptor is not checkpoint readiness evidence.

## Drive

Read the [feature map](features/README.md) for the selected API or CLI behavior.
Select the smallest applicable topic with `--feature`.
Use `--feature all` when shared service behavior changes:

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature all --binary target/debug/oh-my-vllm-zmq-worker
```

The helper sends HTTP requests and captures full responses.
The operations topic also runs the executable's `run` and `bench` commands.
A drive failure returns a nonzero exit code after cleanup.
The helper is an app drive with evidence. It does not add business tests.
Use the repository's [test scope](../../../docs/testing.md) for regression checks.

## Evidence

The helper prints a new evidence directory below `ARTIFACT_ROOT`.
Read `result.json` before reporting completion.
It must show `status=completed`, `cleanup_completed=true`, and `scratch_removed=true`.
Check the requested topic's records and raw responses.

| Record | Contents |
|---|---|
| `manifest.json` | Git state, source digests, executable digest, Python and package versions, scope limits |
| `fixture.json` | Installer identity, substituted setup, configuration, fixed replies |
| `launch.json` | Selected port, IPC path, scratch path |
| `doctor.json` | Owned server and worker readiness, public model ID |
| `http/` | Request method, URL, headers, JSON body, response headers, status, full body, or error |
| `features/` | Observations for each requested topic |
| `processes/` | Commands, process identities, executable digests when observable, stdout, stderr, exit status |
| `cleanup.json` | PID start identities, final observations, adopted-child exit status when available, list of processes to stop |

The source snapshot is not atomic during concurrent edits.
The executable can predate that snapshot.
The manifest states this limit even when the drive completes.
Short commands can exit before executable identity capture.
Worker exit codes can be unavailable when Rust reaps its children.
The cleanup record reports observations without invented exit codes.

Keep evidence external to the repository.
Give its path and the scope limits in the delivery reply.
Keep host settings in `LOCAL.md`.

## Cleanup

The helper first requests graceful shutdown of its Rust children.
It monitors descendants through its own process tree.
It checks PID start identities before it signals a recorded descendant.
It uses a Linux child subreaper to collect orphaned children.

It then removes scratch storage after owned processes stop.
`cleanup.json` must have an empty list of processes to stop.
The external evidence directory stays available after cleanup.

SIGINT and SIGTERM enter cleanup.
SIGKILL cannot run Python cleanup.
After a forced stop, compare saved process identities before any manual process action.
A cleanup failure keeps scratch storage for examination.

For the foreground checkpoint service, use Ctrl-C or SIGTERM on your service process.
Wait for the service and its workers to exit.
Remove your empty socket directory:

```bash
rmdir "$VERIFY_SCRATCH"
```

## Helpers

[verify.py](verify.py) is the executable entry.
Use `--help` for its current options.
[drive.py](drive.py) owns HTTP evidence and topic drives.
[processes.py](processes.py) owns process observation and cleanup.
[evidence.py](evidence.py) records source and executable identities.

These helpers use the standard library and the existing fixture API.
They add no production dependency.

After API or CLI changes, update the feature map and its Chinese translation together.
The user can request `/maintain-verification-skill` for map maintenance.
