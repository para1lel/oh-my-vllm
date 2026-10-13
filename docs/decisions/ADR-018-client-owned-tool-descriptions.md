# ADR-018: Client-owned tool descriptions

Status: Accepted.

## Context

OMP `18.10.3` supplies a Bash tool with `maxLength=48` on its optional `name` parameter.
The existing XML schema validator rejected that description before model execution.
The client owns this argument check and returns a failure to the model when it generates a longer name.

Tool descriptions can contain keywords that give instructions to the model without supplying decoding constraints.
The project must accept those descriptions while keeping explicit strict requests and the existing inference path.

## Decision

Keep each original tool description and parameter schema in the model prompt.
For a tool without explicit `strict=true`, make a separate schema copy for request preparation.
The copy keeps the existing supported XML types, properties, required fields, references, finite choices, and numeric/array constraints.
String length, pattern, format, defaults, examples, and other omitted keywords stay in the original description.

The original schema supplies model instructions. OMP validates returned arguments and supplies errors in subsequent model input.
Client checks on descriptive keywords do not decide framework acceptance.

Use the existing validator and XGrammar compiler on the separate copy.
Do not change GPU computation, sampling, speculative masks, or matcher implementations.
Do not change OMP's native tool definitions, execution, or provider requests for compatibility.

Explicit `strict=true` uses its original schema and existing rejection rules.
Explicit `strict=false` keeps its existing absence of argument schema constraints.
XML must still have a direct closed root object and known, unambiguous parameter types.
JSON answer schemas keep their existing behavior.

## Verification

CPU tests must keep the original descriptions in prompts.
They must accept strings with more characters than the original length limits.
They must keep strict unsupported-schema rejection and existing typed tool constraints.
The OMP executable fixture must reject a `49`-character Bash name against its original `48`-character limit.
It must forward the error and execute the next valid command through each API.

Deterministic fixture responses do not prove target inference or one-hour stability.
Use source-scoped formal service and long-duration evidence for acceptance.
