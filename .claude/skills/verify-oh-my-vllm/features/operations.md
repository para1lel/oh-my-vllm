# Diagnostics and offline commands

## Sub-features

- Model discovery through `GET /v1/models`.
- Service, worker, cache, and request state through `GET /diagnostics`.
- A 400 validation response for an unsupported completion count.
- Offline token generation with `run` and prefix reuse with `bench`.
- Owned process shutdown and scratch removal.

See [the CLI entry](../../../../crates/zmq-worker/src/main.rs), [diagnostics](../../../../crates/zmq-worker/src/serving/mod.rs), and [worker lifecycle](../../../../crates/zmq-worker/src/client/lifecycle.rs).

## How to get to it (user POV)

After [Launch](../SKILL.md#launch), request the model list or diagnostics at the service listen address.
For offline work, select `run` or `bench` instead of `serve`.
Put model, socket, and cache options before the subcommand.
Use Ctrl-C or SIGTERM for graceful service shutdown.

## Driving it with verify.py

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature operations --binary target/debug/oh-my-vllm-zmq-worker
```

The [operations drive](../drive.py) records a rejected Chat request with `n=2` and idle diagnostics.
It starts each offline command with its own IPC path and process group.
The run command uses token IDs 1 and 2 with 8 output tokens.
The bench command uses batch 2, input length 786, output length 8, prefix reuse, no warmup, and one repetition.
These are the [existing fixture cases](../../../../tests/test_service.py).
Read `processes/run/`, `processes/bench/`, `features/operations.json`, and `cleanup.json` in the evidence directory.

## Gotchas

Offline output lengths are fixed.
The fixture checks worker messages and logical prefix reuse without GPU computation.
Its bench timing is not inference performance.

The helper verifies ownership with server PID, worker parent, and PID start identities.
Cleanup is completed only when owned processes stop and scratch is removed.
Evidence stays external to scratch.
See [Cleanup](../SKILL.md#cleanup) for failure and forced-stop limits.
