# ADR-015: Shared service sampling defaults

Status: active. Date: 2026-10-13.

## Problem

The persistent MTP4 test on `249bbb0` failed after `2379.1744` seconds.
OMP detected repeated ordinary text. The reasoning channel did not contain that repeated cycle.
The ordinary text had `23` consecutive copies of a closing-tag sequence.
The server recorded `5035` generated tokens on cancellation, before the explicit service reasoning budget of `8192`.

This was not total-context exhaustion. The input and reserved output were `150941 + 32768` tokens, less than `262144`.

The original request did not supply sampling fields. The service already used checkpoint temperature `1`, top_p `.95`, and top_k `20`.
Presence penalty was `0`, and repetition penalty was `1`.

These supported values agree with Qwen's published thinking defaults.
A preceding grep result had a reasoning closing marker before generation changed to ordinary text.
An input effect is possible. This observation does not prove the cause of repeated output.

Source `98a6eb1` used the shared presence penalty `0.5` and the production reasoning budget `-1`.
Its MTP4 persistent test failed after `1276.9908` seconds.
OMP detected four consecutive copies of a `257`-character cycle in ordinary text.
The reasoning channel did not contain that cycle.

The input and reserved output were `92124 + 32768` tokens. The server recorded `1725` generated tokens on cancellation.
The failed stream had no terminal event or final SDK usage.

The loop markers and repeated text were generated. The recorded request text did not contain those literal markers.
The cause of this repetition is not known.
All eight collection roots were reaped before the next parameter change.
Original failed and canceled records keep their source and status.

## Decision

Use one shared presence penalty default of `1.5` for ordinary, MTP4, DSpark, and test HTTP services.
Keep checkpoint temperature `1`, top_p `.95`, and top_k `20`, with repetition penalty `1` and frequency penalty `0`.
Keep the separate reasoning budget default `-1`. Reasoning effort, total output limits, and context limits keep their existing contracts.

Use `serve --override-generation-config` with a JSON object for supported sampling defaults.
Rust validates these values with the same rules as request sampling.
Reject unknown keys, null defaults, invalid values, and more than `4096` configuration bytes.
CLI validation occurs before worker launch.

An explicit request value takes precedence over service values. Service values take precedence over checkpoint values.
A null request sampling field uses the default. An explicit presence penalty `0` disables the penalty.
An unrelated service change keeps the shared presence penalty default.

OMP Chat, Responses, and automatic compaction use these values when they do not supply sampling fields.
Persistent acceptance uses production defaults, without different sampling or reasoning-budget startup options.
Maximum effort and required output/context limits stay explicit workload settings.
Budget boundary tests and offline greedy measurements keep their specified explicit values.

Service diagnostics report merged service defaults. Request metrics report service and explicit request sampling fields.
These metric fields exclude checkpoint values added by Python.
The CPU fixture separately records Python sampling parameters after checkpoint merge.

## Official guidance and limits

The [vLLM engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/#override-generation-config) specify service overrides merged with model generation configuration.
The [Qwen model card](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices) gives the thinking defaults and an optional presence penalty range from `0` through `2`.
Qwen states that more penalty can decrease quality and cause language mixing.
The selected `1.5` is a project configuration in that range. It is not Qwen's fixed recommended default.

It changes stochastic generation probabilities. New inference and persistent-session acceptance are necessary.

The `presence_penalty` and `frequency_penalty` apply to tokens generated in the current request, with preceding speculative candidates.
The `repetition_penalty` also uses prompt tokens.

A reasoning budget does not detect repetition after reasoning ends.
Repetition can still occur with presence penalty.
Keep OMP's original detector, semantic progress timeout, requests, usage records, and failed evidence.
Project code, dependencies, and build caches stay independent of vLLM.

## Verification

Rust tests check shared defaults, precedence, null omission, explicit zero, bounds, and unsupported fields.
Rust HTTP/ZMQ tests check the two APIs and Python checkpoint merge.
The original integration test failed before this option existed, then passed after implementation.
Full CPU, GPU, budget, phase, operator, persistent-session, and independent resource acceptance are necessary.
