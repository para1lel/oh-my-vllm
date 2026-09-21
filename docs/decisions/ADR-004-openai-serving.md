# ADR-004 — Rust-first OpenAI-compatible serving

**Date:** 2026-09-21
**Status:** Accepted; implemented, GPU acceptance pending

## Context

The user wants local oh-my-pi to call this engine over HTTP and complete an
agentic repository-introduction task. Previously HTTP was outside project scope.

## Decision

Support both OpenAI Chat Completions and Responses, including streaming tool
calls and multi-turn tool results. Prefer Rust for the outer entry points and
lifecycle. Preserve Rust scheduler/KV ownership, Python GPUWorker, single B200,
block size 784 and the existing ZMQ/msgpack boundary.

Support configurable thinking strength and both stateless history replay and
stored Responses continuation. Stored responses live in bounded, expiring memory;
restart persistence is unnecessary. Test both APIs with actual oh-my-pi runs.

## Consequences

HTTP is implemented scope. The service adds online request admission,
streaming, EOS/stop handling and cancellation propagation. Model-specific tool
and thinking output must be converted to each API's wire representation.
API compatibility details and thinking mappings are specified in serving.md;
this decision does not promise every OpenAI platform feature.

The user subsequently authorized implementation: default medium, high→xhigh,
MTP-compatible token constraints, and performance inspection without a new HTTP
throughput gate. Axum and an engine actor own request lifecycle; Python owns
local template preparation and XGrammar mask state only. Stored Responses default
to one-hour TTL, 1,000 records and 256 MiB serialized payload, with no persistence.
Unsupported protocol/schema features fail explicitly. See [serving.md](../serving.md)
for the supported subset and [handoff.md](../handoff.md) for acceptance blockers.
