# Model adapters

Keep each family's weights, computation, cache layouts, codecs, media positions, request state, and speculation in its directory.
The Qwen adapter is the current supported model adapter. Keep its rules in this directory.

Use shared mechanisms for transport, constraints, repetition checks, metadata uploads, and execution rules.
Each adapter supplies its tool grammar. Do not put model dimensions or token syntax in shared mechanisms.
The explicit placeholder is a test backend, not a model fallback.

Keep frozen numerical fixtures and acceptance limits unchanged.

Use the [adapter contract](../../../docs/adapters.md), [execution rules](../../../docs/execution.md), and [test scope](../../../docs/testing.md).
Bind checkpoint and tokenizer identities to loaded bytes.
Remove unused compatibility paths and duplicate tests as the implementation changes.
Before adding a test, tell the maintainer why existing checks are insufficient.
