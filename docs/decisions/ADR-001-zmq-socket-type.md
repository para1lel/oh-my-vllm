# ADR-001 — ZMQ socket type: DEALER instead of PAIR

**Date:** 2026-09-19
**Status:** Accepted

---

## Context

The Rust binary needs a bidirectional, 1:1 IPC channel to a single Python subprocess. The natural ZMQ socket for a 1:1 channel is `PAIR`. The initial design used `PairSocket` and the `pair-socket` Cargo feature flag.

When implementing the `zmq-worker` crate against `zeromq = "0.6"`, neither the type `PairSocket` nor the feature flag `pair-socket` exists in that version. Attempting to use them produces a compile error.

```
error[E0432]: unresolved import `zeromq::PairSocket`
```

Available types in zeromq 0.6.0 include `DealerSocket`, `RouterSocket`, `PubSocket`, `SubSocket`, `PushSocket`, `PullSocket`, `ReqSocket`, `RepSocket`. There is no `PairSocket`.

Available Cargo features include `tokio-runtime` and `ipc-transport`. There is no `pair-socket` feature.

---

## Decision

Use `DealerSocket` on both sides (Rust and Python). Rust binds; Python connects.

`Cargo.toml`:
```toml
zeromq = { version = "0.6", features = ["tokio-runtime", "ipc-transport"] }
```

`client.rs`:
```rust
let mut sock = zeromq::DealerSocket::new();
sock.bind("ipc:///tmp/oh-my-vllm.ipc").await?;
```

`zmq_bridge.py`:
```python
sock = zmq.Context().socket(zmq.DEALER)
sock.connect("ipc:///tmp/oh-my-vllm.ipc")
```

---

## Consequences

**Positive:**
- Compiles cleanly against zeromq 0.6.0 with the features that actually exist.
- DEALER-DEALER is a valid 1:1 topology for bidirectional messaging when there is exactly one peer on each side.

**Negative / traps:**
- DEALER sockets do not enforce 1:1; a second Python process connecting to the same IPC path would receive messages intended for the first. The current codebase never starts more than one Python worker per binary invocation, so this is not a real risk today, but it is not enforced by the socket type.
- DEALER sockets include a zero-length identity frame before the message body in some configurations. The Python `zmq.DEALER` socket in `pyzmq` does not prepend this frame by default for `recv`/`send` (unlike `zmq.ROUTER`), so the msgpack framing works without changes.
- The IPC file (`/tmp/oh-my-vllm.ipc`) is not cleaned up on crash. The binary must not start if the file exists; the startup path should `rm -f` the socket before binding, or the caller must do so. Current behavior: the binary returns an EADDRINUSE error if the file exists. Callers must run `rm -f /tmp/oh-my-vllm.ipc` between runs.

---

## Alternatives considered

**PAIR socket** — not available in zeromq 0.6.0. Would be the natural choice if available; revisit if the crate is upgraded to a version that adds it.

**REQ/REP** — strict alternating send/receive would simplify framing but prevents pipelining and does not survive a dropped message without reset.

**nanomsg / nng** — different crate, different API surface; would require switching the Python side from `pyzmq` to `pynng`. Not worth the disruption.
