# ADR-004 — Rust-first OpenAI-compatible serving

**Date:** 2026-09-21
**Status:** Accepted direction; not implemented

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

HTTP is accepted future scope. The offline loop needs online request admission,
streaming, EOS/stop handling and cancellation propagation. Model-specific tool
and thinking output must be converted to each API's wire representation.
API compatibility details and thinking mappings require further specification;
this decision does not promise every OpenAI platform feature.

The user ended the discussion with a documentation-only instruction. No runtime
changes or GPU acceptance runs are authorized by that final instruction.
See [serving.md](../serving.md) for details and unresolved decisions.
