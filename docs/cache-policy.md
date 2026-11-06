# Model cache policy

The [Rust scheduler](../crates/scheduler/src/lib.rs) uses an explicit model-selected [cache policy](../crates/scheduler/src/policy.rs).
Shared scheduling controls queues, token budgets, preemption, cancellation, and output validation.
It does not interpret model group names, block tables, or physical slot layouts.
KV cache state stays on the GPU. There is no host cache protocol or host storage budget.

## Model operations

The policy supplies admission demand, prefix lookup, valid prefill lengths, reservations, assignments, and resource counts.
Valid prefill lengths obey model state constraints.
Scheduling preferences must not exclude usable residual tokens.
The shared scheduler applies the priorities in the [runtime contract](requirements.md).

`PrefixHit` contains a recovery length and model-selected metadata.
The policy can supply a fallback prefix after allocation fails.
A model can use saved hidden values for a complete prompt hit if that operation is permitted by its cache contract.

`ScheduledRequest<State>` contains shared request fields and a model-selected assignment.
An assignment names state locations. It does not own storage.
The policy pins referenced storage until execution and necessary copies let it release that storage.
Allocation failure must not publish a prefix or keep an unusable request allocation.

## Commit and completion

The scheduler validates the full worker output before calling `commit_cache` for any request.
The policy selects accepted state from validated progress.
Copied assignments and block values cannot authorize storage ownership.
An arena-based policy selects storage through live owners in the [state arena](state-arena.md).

The caller proves device completion before supplying results to the scheduler.
Distributed callers also prove [all-rank completion](distributed.md#transactions-and-publication).
The [wire contract](adapters.md#execution-protocol) binds results to the complete execution identity.

`completion_plan` selects forward computation, endpoint export, or a completed endpoint.
`validate_cache_step` checks model-specific state evidence before commit.
`finish_request` distinguishes success, cancellation, and failure.
Only a validated success can publish the successful request tail.
The [session registry](../crates/scheduler/src/session.rs) supplies admission order for that publication.

## Retirement

`free` releases request ownership only when its caller permits release.
`Scheduler::abort` alone does not prove device completion.
A cancelled submitted request enters `Retiring` while its execution is in flight.
The [lifecycle implementation](../crates/scheduler/src/lifecycle.rs) releases pins only after validated completion clears the execution identity.

Kernel work, auxiliary streams, rollback, exports, and copies must finish before physical release.
Finished-request and preempted-request notifications still release worker state.
Worker replacement first proves that the old worker has exited, then resets state in a new epoch.
Necessary token and metadata transfers obey the [execution rules](execution.md).

## Implementations

The [Qwen policy](../crates/scheduler/src/policy/qwen.rs) uses independent checkpoint and KV page pools with one token radix tree.
Its [cache guide](models/qwen38/cache.md) specifies independent prefix positions, exact session tails, and physical storage.

The [DeepSeek policy](../crates/zmq-worker/src/models/deepseek_v41/cache.rs) uses its resident paired-block assignments and accepted-state rules.
It does not use Qwen's capacity ratio, checkpoint interval, or exact-tail computation rule.
Its [cache guide](models/deepseek_v41/cache.md) specifies the model storage.
A generic policy interface does not prove model mathematics, capacity, or device safety.
