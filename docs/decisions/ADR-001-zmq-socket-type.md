# ADR-001: ZMQ DEALER sockets

Date: 2026-09-19. Status: accepted.

## Context

The single Rust driver must exchange bidirectional messages with one Python child.
The pinned `zeromq` crate has no `PairSocket` or `pair-socket` feature.
REQ/REP forces strict alternating replies and restricts pipelining.
A transport change also changes the Python interface.

## Decision

Use DEALER on the two sides with `tokio-runtime` and `ipc-transport`.
Rust binds the IPC endpoint. Python connects.
Use msgpack bodies with explicit type and RPC correlation.

## Consequences

Keep exactly one worker per driver and a unique socket path per live process.
DEALER permits multiple peers.
Do not connect another worker to the same endpoint.
Use owned-process cleanup for stale IPC files.
Remove a stale path only after you make sure that no live owner uses it.

Current framing and asynchronous prepare/execute handling are in the [architecture](../architecture.md#transport-contract).
