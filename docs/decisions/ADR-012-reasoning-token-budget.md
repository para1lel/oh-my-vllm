# ADR-012: Reasoning token budget

Status: active. Date: 2026-10-12.

## Problem

Reasoning effort controls the Qwen template. It does not set a numeric token limit.
Reasoning can consume the total output budget before the model supplies an answer.
OMP's installed OpenAI adapters do not send a reasoning token budget.

## Decision

Use top-level `thinking_token_budget` for the two HTTP APIs, with the same name as vLLM.
Use `--thinking-token-budget` for the service default. An explicit request value takes precedence.

The default is `-1`, which keeps only the total output limit. Nonnegative values limit generated reasoning content.
Keep effort separate from the numeric budget. Do not map `xhigh` to an invented numeric budget.

At the budget, let target sampling select only the token `</think>`. Continue answer or tool generation after this token.

Keep the total output and context limits effective. The closing token uses part of the total output budget.
Existing reasoning usage includes this token and can be `budget + 1`.
Off does not produce a reasoning block. Zero forces the closing token when thinking is enabled.
This contract is specific to the tested Qwen tokenizer with a distinct, single-token closing marker.

Python previews the committed count for each target verification row, with its preceding draft prefix.
Mask preview does not change the count, history, grammar state, or KV state after rollback.
Only kept output tokens change the committed count.
Before the boundary, thinking output also uses the canonical control-token grammar in [ADR-016](ADR-016-canonical-serving-control-tokens.md).
At the boundary, greedy and stochastic target sampling use the mask, with bonus rows.

DSpark proposals can use a different distribution. Its target verification uses the constrained distribution.
A wrong proposed token has zero target probability and is rejected. The rejection correction supplies the forced closing token.
A singleton target uses a deterministic fallback when floating-point subtraction gives zero residual mass.
Proposal probability validation stays effective.

## Source comparison

The [vLLM reasoning guide](https://docs.vllm.ai/en/stable/features/reasoning_outputs/#thinking-budget-control) documents a per-request budget and forced reasoning termination.
Its default has no separate reasoning limit. It does not recommend one fixed numeric budget for all workloads.
The [SGLang reasoning grammar](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/constrained/reasoner_grammar_backend.py) applies reasoning limits through token masks.
We used these sources to specify the contract. Project code and dependencies do not use those engines.

## OMP integration and limits

The installed OMP is `v18.4.3`.
Observed Chat and Responses requests omit `thinking_token_budget`, with their existing `max` effort mapping to `xhigh`.
A CPU request test shows that model `compat.extraBody` adds the field for Chat but omits it for Responses.
Use the service default to cover the two APIs and automatic compaction without different client patches.
Keep runtime budget configuration in original acceptance records.

The installed Responses adapter stops after `300000` milliseconds without an accepted progress event.
SSE comments and `response.in_progress` do not reset this timer.
A reasoning budget cannot decrease compilation or download time.

The cubin package has the same version as FlashInfer. It installs versioned artifacts before requests.
By default, FlashInfer checks each artifact before use.
Cold-start verification still uses the original client and its timeout.

The comparison environment adds this package to the runtime lock.
The frozen TileLang manifest binds that lock. The DSpark manifest binds the new TileLang manifest hash.
Reference kernel hashes, case IDs, numerical tolerances, and TileFoundry commits stay unchanged.

The budget does not detect repetition after reasoning ends. It cannot guarantee answer quality or eliminate worker failures.
Keep those failures and test the affected workloads again after changes.

## Verification

CPU cases check zero, disabled and unlimited reasoning, total length, typed validation, and mixed requests.
They check draft prefixes, rollback, bonus rows, grammar transitions, and DSpark rejection with different proposal probabilities.
Target-model checks must cover ordinary, MTP4, DSpark, the two APIs, streams, JSON, tools, and the service default.
The one-hour persistent-session workload must keep maximum effort and record the shared production service budget.
Use production defaults unless a test explicitly examines a different budget.
