# End-to-end requirements

Use one NVIDIA B200 and the [model contract](operators/model.md).
Use the complete Rust service and Python inference worker.
Placeholder outputs do not supply inference evidence.

## Configurations

Each batch starts with concurrent HTTP requests.
The input length applies to each request. Each request must generate 8192 tokens.

| Mode | Batch sizes | Input tokens per request |
|---|---|---|
| Ordinary decode | 1, 2, 4 | 32768 |
| MTP4 | 1, 2, 4 | 32768, 65536, 131072 |

These are 12 configurations. Each configuration must pass separately.
Use a token budget of 32768 and a maximum of 4 running requests for this comparison.
Use requests without a cached prefix.
Load the model and complete initialization before timing.

Measure from HTTP submission until the client receives all output from the batch.
Include request preparation, queue time, prefill, decode, constraints, sampling, process communication, and output transport.
Do not use time to first token as the complete latency.

## Performance limit

The measured batch latency must be at most `5 / 3` times its theoretical roofline lower bound.
Use the [B200 peaks and precision rules](operators/evaluation.md#performance-gate).
Calculate necessary work for the complete model, with MTP context, proposals, verification, and acceptance.
Use the full sequence lengths for attention work.
Keep the measured MTP acceptance counts. Do not assume a fixed acceptance rate.

Do not count padding, repeated computation, or unnecessary copies as additional necessary work.
Do not add measured launch time or an empirical latency factor to the lower bound.
Keep source, binary, checkpoint, device, configuration, and timing identities in external artifacts.
Keep the lower-bound calculation available for review.

## Execution

Select decode requests first. Divide the available token budget between prefill requests.
Use CUDA Graph execution for batches that contain only decode requests.
Use ordinary execution for batches that contain prefill requests.

Apply the constraint mask before sampling in target and proposal computation.
Keep constraint state consistent with the accepted token history after speculative rejection.
Use MTP4 acceptance and private cache state selection from the model contract.

Use fusion for producer and consumer work when it decreases the complete execution time.
Multiple CUDA streams are permitted for independent work with explicit dependencies.
Do not use PDL in the inference worker.

Operator optimization is stopped while the complete worker is built.
The [operator workload](operators/evaluation.md) stays available for later operator analysis.
