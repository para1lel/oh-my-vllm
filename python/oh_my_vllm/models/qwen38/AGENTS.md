# Qwen adapter

Keep Qwen weights, cache layouts, media positions, request state, and MTP4 behavior in this directory.
Use shared worker mechanisms for transport, constraints, repetition checks, metadata uploads, and execution rules.
The adapter supplies Qwen tool grammar to shared constraints.
Do not put Qwen dimensions or token syntax in shared mechanisms.
The explicit placeholder is a test backend, not a shared mechanism.

Keep the frozen numerical fixtures and their acceptance limits unchanged.
Use the [execution rules](../../../../docs/execution.md) and [test scope](../../../../docs/testing.md).
Keep checkpoint and tokenizer identities bound to the loaded bytes.
Before you add a test, tell the maintainer why it is necessary.
