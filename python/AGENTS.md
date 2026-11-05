# Python worker

Keep the worker process at the same protocol boundary as the inference implementation.
Use the selected model adapter for computation.

Put transport in the bridge.
Use the runtime dependency file for worker packages.

Rust controls request admission, scheduling, and logical block identities.
The worker must return the related RPC ID and one result for each scheduled request.

A prefill chunk before the final chunk supplies no output tokens.
Use different state fields for accepted tokens, new drafts, and request completion.

Free the request state after Abort or a finished-request notification.
Use clear names and small functions.

Before you add a test, tell the user why it is necessary.
