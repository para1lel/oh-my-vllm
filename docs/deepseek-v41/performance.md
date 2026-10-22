# DeepSeek V4.1 performance

## Measurement status

DeepSeek implementation and measurement work has not finished.
This document is the result destination for the [validation plan](validation.md).
There are no measured throughput or latency claims here.

## HTTP results

Add measured rows only after correctness acceptance and approval of the strategy matrix.
Each row identifies mode, strategy, batch, input length, output length, and measured request count.
Report full HTTP time, input/output throughput, first-token latency, inter-token latency, and draft acceptance.
Define throughput denominators and acceptance counters with the table.
Report failures and per-rank memory separately from successful throughput.

## AgentX results

Keep one section for each approved concurrency/strategy pair.
Report the request output-rate distribution, prefix cache hit ratio, queueing, and request status counts.
Record the profiling duration, drain limit, dataset identity, and observed HTTP concurrency.
Do not present incomplete requests as successful samples.

## Provenance

Bind results to source, executable, native modules, model, tokenizer, device topology, sampling profile, and commands.
Keep raw measurement artifacts external to the repository.
Use artifact digests or portable identifiers here, not host paths.
Split long tables into linked bilingual documents before the document word limit is exceeded.
