# ADR-007: TTFT and different cache capacities

Date: 2026-09-22. Status: accepted and implemented.

## Decision

Use batch-submission-to-first-kept-token TTFT with per-request monotonic clocks.
Include registration, queue time, scheduling, transport, and sampling.
Use the median of five repetition maxima after two full warmups.
The twelve rows must meet 95% throughput,110% TTFT, and 10% spread limits.
Pin baseline source `e9f169d16b9408bb9ae44f75072b91a5521d733c` in its isolated environment.

Keep total context 262144, block 784, state dtypes, and numerical tolerances.
Add `--mamba-blocks` for different Rust GDN allocation and Python tensor capacity.
FA and GDN IDs have different namespaces.
Without the option, keep the shared pool for controlled capacity/preemption tests.
Admission checks the two capacities and rejects impossible requests.

Prefix hits touch the two pools before new allocation.

## Reason and consequences

A large shared pool allocated excess recurrent state alongside FA capacity.
Separate 1400 FA and 128 GDN slots keep long contexts resident with less unused state.
The historical `num_gpu_blocks` unit stays. Python shows `floor(num_gpu_blocks/3)` FA slots.
Block IDs directly address their own group tensors.

Future automatic sizing must keep admission, prefix, reference, and eviction invariants with Rust allocation ownership.

The two engines must keep the full workload resident without preemption.
Match effective token capacity, CPU affinity, and GPU model/memory/driver.
A matching raw block-count integer does not show equivalent capacity.
Record full configuration and measured source identity.
See [requirements](../requirements.md), [testing](../testing.md), and [acceptance](../acceptance.md).

The future roofline policy does not change this active protocol.
