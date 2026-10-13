# ADR-016: Canonical serving control tokens

Status: implementation in acceptance. Date: 2026-10-13.

## Problem

Python ends reasoning when `Generation.consume` receives the tokenizer's single `</think>` ID.
Rust routes decoded text at the first `</think>` string.
The previous grammar also used a string delimiter.
Ordinary IDs can spell the same string without the dedicated ID.
This spelling could close the grammar and parser while Python continued to count reasoning tokens.
A later finite budget could then fail because the grammar no longer allowed the closing token.

The previous automatic tool grammar started its constraint at `<tool_call>\n<function=`.
Rust recognized `<tool_call>` immediately and accepted more whitespace forms.
Malformed openers could therefore bypass the grammar's schema and function-choice constraints.
These are constructed protocol failures. They do not show the cause of the recorded ordinary-text loops.

## Decision

Validate the single reasoning end ID for each thinking request, with unlimited budgets, too. This ID must differ from EOS.
Use that same ID in `TokenFormat` and `Generation`.
Exclude complete `<think>` and `</think>` strings from reasoning content.
Alternative ordinary IDs cannot close only the grammar and parser.

Give plain thinking output the same reasoning shell, followed by text with no answer-format constraint.
Keep its original whitespace forms after the closing token.
Plain output with thinking off and disabled tools keeps its unmasked path.
JSON and enabled tools keep their existing answer separators and constraints.

Enter automatic and required tool constraints at `<tool_call>`.
Use the builtin function-choice and parallel-call policies.
Canonical generated calls must then satisfy the declared constraints from their first opener.
An explicit `strict=false` keeps its existing unconstrained argument grammar.
Rust XML parsing still applies to those arguments.

Target masks apply to ordinary, MTP4, and DSpark verification rows.
Preview rolls back. Only kept tokens change persistent state.
The reasoning shell rejects EOS before a canonical closing token.
Explicit text stops and total length limits can still finish before that token.

## Repetition investigation

The [diagnostic summary](../../bench/evidence/2026-10-13-constrained-decoding-investigation.json) separates recorded failures, constructed regressions, and conditional GPU measurements.
The two recorded loops occurred after the parser changed to ordinary text while the model discussed a reasoning closing delimiter.
`</thinking>` and `<next_thinking>` are ordinary text, not reasoning control tokens.
The existing body grammar allows these strings.

At some reconstructed prefixes, the target gave its maximum score to another canonical closing token that the body grammar rejected.
Ordinary tag fragments then had high conditional probability after the mask.
At measured late-loop starts, a presence penalty of `0.5` did not decrease the repeated next token's probability to less than `1`.
These measurements cover selected next-token distributions, not each token of a complete loop.

The original sampled IDs, logits, and random state are unavailable.
The diagnostic tokenizes saved output text again and inserts one canonical closing token.
It uses fresh eager target execution with BF16 and FP32 GDN states.
It does not reproduce the original MTP, scheduler, graph, or prefix-cache execution.
The results support a closing delimiter in quoted text as a possible cause of repeated output.
They do not prove one unique cause.

## Engine comparison and limits

[XGrammar structural tags](https://github.com/mlc-ai/xgrammar/blob/main/docs/structural_tag/tool_calling_and_reasoning.md) constrain output structure and support token delimiters.
Structurally valid text can still fail the task. Repeated sentences can still occur.
[vLLM repetition detection](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/#vllm.sampling_params.RepetitionDetectionParams) is a separate, optional request termination mechanism.
Its default is disabled. Detection terminates a request, but the agent task can still be incomplete.

Keep the shared presence penalty at `0.5`, as requested.
Keep checkpoint sampling, the reasoning budget default `-1`, total limits, numerical tolerances, and DSpark weights.
Do not prevent ordinary pseudo-tags globally. JSON strings and tool values can contain markup.
The protocol fixes do not claim a semantic repetition correction.
Long-duration model and independent memory acceptance are necessary.

## Verification

Tokenizer and grammar regressions failed before the protocol changes.
Tests cover alternate delimiter IDs, malformed tool openers, schemas, named functions, parallel limits, and speculative rollback.
They also cover disabled tools, literal markup, original plain whitespace forms, explicit non-strict tools, stop strings, and total length.
Formal acceptance must use the new committed source and shared production defaults.
