# Collection deadline tails

The one-hour window ends on the collector's monotonic clock.
Native goal pause and RPC abort follow that deadline. The same service then drains.
This collector stop does not change the server's recorded request status.

A server-completed request can have no SSE terminal when the client closes at that boundary.
Only the [deadline evidence checker](../../benchmarks/agentic_deadline_evidence.py) can verify this result.
It must include original control-command bytes, successful acknowledgements, a unique native aborted result, and the same request's original SDK export.
Goal, prompt, session, response, API, provider, model, and clocks must agree.

The wire must record a downstream disconnect and a valid SSE prefix.
Recorded server output must not exceed the original request's output budget.

The [RPC recorder](../../benchmarks/agentic_rpc_evidence.py) timestamps control actions without changing native protocol bytes.
Monotonic clocks determine ordering.
The two-millisecond limit checks timestamp precision and clock offsets.
SDK completion must follow the deadline and abort dispatch.

The proxy records wall and monotonic clocks when the downstream write fails.
That disconnect must follow abort dispatch. Handler cleanup time cannot replace it.
A clock difference of more than two milliseconds fails verification.

Report these results independently from successful streams and server cancellations.
Exclude these results from the request scatter plot. Keep their original metrics and bytes.
Successful streams must have all terminal events and usage that agrees with token counts.
Verification fails if proof is missing or does not bind one request.

The [critical regression cases](../../tests/tooling/test_agentic_deadline_evidence.py) cover the two APIs and reject changed identities, clocks, bytes, acknowledgements, and SDK results.
Historical collections without the captured control timestamps keep their original results.
