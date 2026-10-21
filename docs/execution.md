# GPU execution

## Operator scheduling

Rust controls requests, scheduling, logical KV identities, and state commit.
Python executes the model through CUDA operators and the existing ZMQ DEALER and msgpack boundary.
The [operator boundaries](operators/fusion.md) specify computation and private state ownership.

The worker selects execution once for each Execute message.
If any scheduled row contains prefill, all fragments of that Execute use Eager execution.
A decode-only Execute uses CUDA Graph recording and replay.
This rule also controls sampling, MTP context, proposals, and acceptance.
Vision and weight preparation use Eager execution.

Eager and CUDA Graph execution use the same native operators.
Python calls these operators in model dependency order.
CUDA kernels can fuse operations and use matrix tiles internally.

## Dependent kernel launch

Small FFN and GDN operations use [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/archive/13.1.0/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html) between normalization and input projection.
They also use PDL between output quantization and output projection.
Stream capture records these selected dependencies in the CUDA Graph.
The consumer can prepare its storage and read constant weights before its producer completes execution.
It waits for its producer to complete execution before it reads the inputs or scales that the producer supplies.

The launch signal lets the consumer start. The consumer must wait before it reads the producer's outputs.

At output merging and layer boundaries, a kernel must complete execution before the next kernel starts.
These rules prevent temporary storage reuse before the previous kernels complete execution.
PDL does not change state selection or the Eager and Graph execution rules.

## Graph resources

A workspace owns weights, cache references, metadata, and temporary tensors for its recorded operations.
Weight preparation and workspace construction allocate storage before recording.
Operations with deferred native setup use warmup before recording.
Sampling, MTP acceptance, and prepared draft chains record without warmup.
Recording does not execute their RNG updates.

Buffer addresses stay fixed while a graph exists.
Metadata changes on the caller's stream before replay.
Replay uses that same stream.

Metadata and sampling inputs use reusable page-locked host buffers for uploads.
Each upload takes a copy of its CPU input before it returns.
An event on the caller's stream protects that copy until the transfer completes.
Wait for that event before storage reuse or release.
Do not update these inputs during Graph recording.

Recording uses a different nonblocking stream and restores the caller's stream on exit.
A failed recording releases the resources created before failure.
A completed recording transfers its resource leases to the workspace.
Closure waits for GPU work, destroys the graph executable, and then releases its owners and leases.
An execution failure keeps the existing worker removal and state cleanup rules.

## Numerical and state contracts

Obey the [model contract](operators/model.md) and [full-model numerical evaluation](operators/evaluation.md).
Keep the service, graph resource, memory safety, and state ownership checks in the [test scope](testing.md).
