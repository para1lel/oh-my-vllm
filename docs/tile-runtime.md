# Task engine

The task engine obeys the resource contract in the [tile design](tile-execution.md).
The task family supplies FP32 copy, scale, and add operations on adjacent buffer elements.
Model execution still uses the native operators.
The [migration plan](tile-migration.md) includes language, sampling, vision, and preparation tasks.

## Plan and submission

The interface is in [mega.py](../python/oh_my_vllm/kernels/mega.py).
`prepare("elementwise", capacity, bindings)` supplies a plan with stable topology and buffer addresses.
Bindings supply the device, FP32 buffers, tasks, and output buffer indices.
Different buffer bindings must not use the same memory. A task can read and write the same binding.

Dependencies are necessary when tasks use the same memory and one task writes to it.
These dependencies also control buffer reuse.

`execute(plan, metadata)` submits work on the caller's stream and supplies a completion object.
`completion.wait()` waits for a CUDA event before it reads device status.
The caller can wait on a different stream.
The completion keeps references to its output buffers.

Before the next submission, the caller must call `completion.wait()` and increase the epoch.
Access to `completion.outputs` also calls `completion.wait()`.
One caller controls each plan. Host submissions to the same plan must not occur at the same time.

`StepMetadata` checks query lengths, page indices, private state slots, and active row limits.
The device task family receives only the epoch and active row count.
Language task bindings must add device page tables and state slots before model integration.
Tasks do not change rows that are not active.

For CUDA Graph execution, capture `plan.launch` with `Device.graph`.
Give `execute` the replay callable through its `replay` argument.
Metadata transfer occurs before graph replay on the caller's stream.
The graph lease prevents plan closure while the graph can use its buffers.
`close(plan)` waits for work in progress and releases plan resources.

## Device progress and errors

The [native engine](../python/oh_my_vllm/kernels/cuda_backend/mega.cu) uses a cooperative launch.
The grid size must not be more than the occupancy limit of the full kernel.
Grid barriers occur between initialization, task execution, and completion publication.
Task execution uses dependencies. It does not use a grid barrier between tasks.

The [queue protocol](../python/oh_my_vllm/kernels/cuda_backend/tile_runtime.cuh) uses a fixed capacity.
Device atomic operations make parent writes available before a child starts.
A full queue records an error and causes termination. A producer does not wait for available queue space.

Incorrect descriptors, epochs that do not increase, and incorrect row counts cause termination.
Graphs with tasks that do not complete also cause termination.
Failure status stays set in subsequent device submissions.

An execution failure prevents plan reuse.
The worker uses the `device` error kind for `TileExecutionError`.
Service controls remove the worker and clear its scheduler epoch.
The step deadline controls termination when a device cannot complete its work.
Tensor state cannot become accepted state after an execution failure.

## Checks

Build the native modules with the [development commands](operators/development.md).
Run the CPU protocol check:

```bash
scripts/with-env.sh python -m unittest development.tile.test_engine
```

Run the GPU protocol fixture with the 600-second step deadline:

```bash
scripts/with-env.sh scripts/with-gpu.sh timeout --kill-after=5s 600s python -m unittest development.tile.gpu_fixture
```

The fixture checks dependencies, changed input data, rows that are not active, epoch reuse, and graph leases.
It also checks queue overflow, errors that stay set, cross-stream completion, and large address offsets.
The large-offset check uses one 8 GiB buffer and accesses its last element.
The fixture supplies no model numerical or performance evidence.

Run Compute Sanitizer with each of `memcheck`, `racecheck`, and `synccheck`:

```bash
scripts/with-env.sh scripts/with-gpu.sh timeout --kill-after=5s 600s "$CUDA_HOME/bin/compute-sanitizer" --tool memcheck --error-exitcode 2 python -m unittest development.tile.gpu_fixture
```

Keep the reports external to the repository.
