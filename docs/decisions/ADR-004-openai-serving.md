# ADR-004: Rust HTTP service

Date: 2026-09-21. Status: accepted and implemented.

## Decision

Add Chat Completions and Responses to the former offline-only scope.
Keep HTTP, online admission, response storage, cancellation, scheduling, and logical KV in Rust.
Keep templates, tokenizer, grammar, and GPU computation in Python.
Function tools execute in the client.
Keep ZMQ/msgpack, one B200, and block 784.

## Consequences

Supply streams, tool-result history, thinking controls, and stored Responses continuation.
Use bounded expiring memory, without restart persistence.
Stored defaults are one hour,1000 records, and 256 MiB serialized bytes.
Default thinking is medium. High maps to xhigh.

MTP token constraints apply before sampling.
Unsupported API/schema features fail explicitly.

OMP acceptance with the target checkpoint must exercise each API with MTP, file reads, tool results, and an answer that uses those results.
There is no extra HTTP throughput gate.
[Service contracts](../serving.md) define the supported subset and resource limits.
[Acceptance](../acceptance.md) identifies measured evidence.
