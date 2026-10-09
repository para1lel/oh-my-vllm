# ADR-005: Historical V2 runner

Date: 2026-09-21. Status: superseded by [ADR-006](ADR-006-independent-runtime.md).

## Original decision

The transitional adapter selected the V2 model runner to keep persistent request state and explicit graph lifetime.
Rust kept accepted history, cache ownership, and scheduling.
Admission/resumption supplied full accepted history. Ordinary decode stayed incremental with drafts in different storage.
The transition kept offline/service features, MTP, masks, prefix reuse, and cancellation.

## Replacement

The project runner keeps these state and lifetime contracts with project-owned code.
No V2 adapter, vLLM scheduler, or legacy runner stays.
Use [architecture](../architecture.md), [requirements](../requirements.md), and [acceptance](../acceptance.md) for active behavior.
