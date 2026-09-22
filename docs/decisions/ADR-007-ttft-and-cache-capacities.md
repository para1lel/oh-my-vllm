# ADR-007: EngineCore TTFT and independent hybrid cache capacities

Status: implementing; user-confirmed 2026-09-22. Final GPU acceptance pending.

## Decision

Measure batch-submission-to-first-retained-token with a monotonic host clock, per
request. Include registration, engine queueing, scheduling, transport and sampling.
Use the median of five per-run maxima, after at least two full warmups. All twelve
workloads must reach TTFT <=110% and throughput >=95% of a newly collected official
vLLM main baseline. The authorized isolated reference collector is the sole
exception to the project's no-vLLM-import rule. Existing baseline JSON is historical.
Freeze upstream e9f169d16b9408bb9ae44f75072b91a5521d733c and its adapted environment.

Support 262144 total tokens without altering block784, recurrent dtypes or tolerances.
The prior shared logical pool allocated both FA and GDN tensors at the same capacity;
large FA capacity therefore duplicated substantial unused recurrent storage.
`--mamba-blocks` configures a separate Rust-owned GDN block pool and Python state
capacity. FA and GDN IDs occupy separate namespaces. Each pool retains the existing
prefix hash, refcount and eviction rules; admission checks both capacities before
allocating. Prefix hits touch both pools before allocation. Without the option the
historical shared pool remains available for controlled capacity/preemption tests.
This is one runner implementation, with configurable cache allocation policy.

The initial long-context diagnostic uses 1400 FA slots and 128 GDN slots (including
immutable null slots). No numerical precision is changed. Capture save/restore and
MTP state ownership remain per tensor type. Fresh-allocation IDs remain unused wire
hints; they must not be interpreted as globally unique IDs across separate pools.

## Sources and verification

Design reference: [vLLM hybrid cache manager](https://docs.vllm.ai/en/stable/design/hybrid_kv_cache_manager/).
Our limited single-model scope permits explicit capacities instead of a general
heterogeneous-memory allocator. Future automatic sizing must preserve admission and
prefix ownership invariants, and cannot move logical allocation into Python.

Rust tests cover separate-pool exhaustion, prefix restoration, MTP crossing,
rejection reuse and migration across prefill chunks. GPU/model tests and complete
long-context/12-row evidence are required before final acceptance.

For the comparison, both engines must keep the full workload resident, with zero
preemptions and the prescribed prefix-hit counts. Raw cache configuration, GPU
identity and CPU affinity are recorded; GPU model/memory/driver and CPU affinity
must match. Byte layouts differ, so matching a raw block-count integer across
engines is not a valid capacity comparison. Check each engine's effective token
capacity against the workload before accepting a baseline configuration.
