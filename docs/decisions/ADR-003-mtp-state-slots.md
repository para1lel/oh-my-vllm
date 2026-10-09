# ADR-003: MTP state slots and precision

Date: 2026-09-19. Status: accepted, still active in the independent runtime.

## Context

The draft layer has no GDN, but target verification must keep additional recurrent states for scheduled drafts.
The initial backend's FP32 state sizing with four drafts increased block size from 784 to 1568.
The project contract fixes block size at 784.

## Decision

Use BF16 GDN state for MTP and FP32 for ordinary execution.
Rust reserves speculative slots and protects the previously accepted state until the next execution consumes it.
The worker returns kept target tokens, accepted draft counts, and next drafts.
Rust rolls back only scheduled rejected drafts.

## Consequences

Precision is part of the measured configuration, not a general MTP accuracy claim.
Keep block 784 and the inference-path FP64 reference tolerances.
Tests include target verification states, GQA/GDN, chunk boundaries, prefix reuse, and coherent text.
Full-model token identity is not a gate.
See [accuracy requirements](../requirements.md#req-acc-001-numerical-accuracy) and [acceptance](../acceptance.md).
