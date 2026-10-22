# AgentX measurement

Use AgentX to measure MTP4 with recorded agent request patterns on one NVIDIA B200.
The harness reconstructs anonymous text from the traces.
This measurement checks service performance, not agent task quality.

## Configuration

Use `agentx-harness==1.0.7` and scenario `inferencex-agentx-mvp`.
Use dataset `semianalysis_cc_traces_weka_062126_256k`, seed `42`, and a profiling duration of `900` seconds.
The trajectory start ratios are `0.0` and `1.0`.

The maximum context length is `262144` tokens.
The harness filters traces by this limit. It does not truncate requests.
The service checks the prepared prompt length and the requested output length.

Concurrency `6` means six active session trees.
Child agents can cause more than six HTTP requests to be active.
The service schedules at most `4` requests with a token budget of `32768`.
The cache uses the [runtime tensor budget](serving.md#cache-memory).

The harness uses streaming responses and `ignore_eos=true`.
The request seed is `42`, with reasoning off and service sampling defaults.
EOS tokens count toward the output limit. They do not stop generation.
Stop strings and explicit minimum output lengths keep their usual behavior.

The scenario keeps trace delays and uses `first_turn_prefix` cache isolation.
It limits system idle gaps to `10` seconds.
Snapshot requests warm the cache before profiling.
Profiling starts after warmup. The final drain lets active requests complete.

Warmup and drain time are additional to the `900` seconds.
The drain limit is `600` seconds.

Use the [upstream procedure](https://github.com/SemiAnalysisAI/agentx-harness/blob/agentx-v1.0.7/docs/tutorials/agentx-mvp.md) for the scenario rules.

## Run

Build the service and native modules with the [service guide](serving.md#start).
Use a separate environment for the harness.
Set `AGENTX_ENV` and `ARTIFACT_ROOT` to new directories external to the repository.
Set `MODEL_ROOT` to the local checkpoint directory.
Use the [GPU wrapper](performance.md#measurement-command) for the measurement.

```bash
scripts/with-env.sh uv venv "$AGENTX_ENV" --python "$CONDA_PREFIX/bin/python"
scripts/with-env.sh uv pip install --python "$AGENTX_ENV/bin/python" -r requirements/agentx.lock
scripts/with-env.sh scripts/with-gpu.sh python -m development.serving.agentx \
  --model "$MODEL_ROOT" \
  --agentx-python "$AGENTX_ENV/bin/python" \
  --artifacts "$ARTIFACT_ROOT" \
  --csv docs/assets/agentx-mtp4.csv \
  --figure docs/assets/agentx-mtp4.png
```

Use new CSV and figure paths when the named files exist.
The command starts and stops the MTP4 service.
It records the source, executable, native module, checkpoint, tokenizer, device, and command identities.
The `agentx` identity entry and `command.json` give the workload configuration.
Shared HTTP accounting fields do not apply to these results.
The execution trace identity gives resolved cache capacities and the memory calculation.

The FA capacity is automatic unless `--blocks` supplies an explicit count.
An explicit count must fit the runtime tensor budget.

The raw records, service log, execution trace, and count report stay external to the repository.
The approved AgentX CSV and figure are repository artifacts for this measurement.

## Request data

One CSV row and one point represent one successful profiling request.
The collector uses the full response ID to join stream usage with service metrics.
It rejects missing cache counts, duplicate IDs, inconsistent lengths, and invalid timers.
The completed request count must agree with the harness summary.
Failed, cancelled, incomplete, and context overflow requests have separate counts in the external report.
The collector rejects a run without a valid scenario result.

| Field | Definition |
|---|---|
| `processing_start_s` | Server processing start minus profiling start, in seconds |
| `output_tokens_per_second` | Output tokens divided by server processing seconds |
| `cache_hit_ratio` | Cached prompt tokens divided by prompt tokens |
| `client_seconds` | Client monotonic time from request start to the last stream event |
| `queue_seconds` | Admission queue time before server processing starts |

Server processing time contains request preparation, later scheduling waits, and generation.
It excludes the preceding admission queue and complete network transport.
The output rate is not a decode-only rate.
Color uses measured prefix reuse with a fixed range from 0% to 100%.
Missing cache data is an error, not a zero value.

Rebuild the image from the CSV:

```bash
scripts/with-env.sh "$AGENTX_ENV/bin/python" -m development.serving.agentx_plot \
  --csv docs/assets/agentx-mtp4.csv \
  --figure docs/assets/agentx-mtp4.png
```
