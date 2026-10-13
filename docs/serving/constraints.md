# Output constraint contracts

## Grammar and speculative tokens

JSON object, JSON Schema, and strict tool grammars apply before sampling.
The constrained answer can follow reasoning.
A request cannot combine JSON answer format with enabled tools.
Strict schemas must have closed objects with each property required.
Unsupported strict schemas fail explicitly.

The supported keyword inventory for JSON answers and explicit strict tools is in `python/oh_my_vllm/worker/serving.py`.
These schemas cannot use unknown string formats or combine pattern/format with length limits.

Without explicit `strict=true`, tool descriptions can contain additional schema keywords.
The model prompt keeps the full original description and parameters.
Request preparation makes a separate copy for the existing XML decoder.
String length, pattern, format, defaults, examples, and other omitted keywords stay as instructions for the model.
OMP validates generated arguments against its original tool definition and returns failures to the model.
Known XML types, required fields, function choice, and parallel-call rules keep their existing behavior.

[ADR-018](../decisions/ADR-018-client-owned-tool-descriptions.md) gives the request preparation policy.

XGrammar supplies Qwen XML tool grammars and JSON answer grammars.

Each thinking request uses its validated single closing token ID, with plain text and unlimited reasoning.
Ordinary IDs that spell the closing string cannot end reasoning.
EOS is masked until that token. Explicit text stops and total length can still finish earlier.
Plain answers keep their whitespace and literal markup after reasoning ends.

Enabled tool constraints start at `<tool_call>`, as in the Rust parser.
The generated call content must satisfy its function-choice, parallel-call, and schema constraints.
Explicit `strict=false` disables tool argument schema constraints. XML parsing still applies.
[The control-token decision](../decisions/ADR-016-canonical-serving-control-tokens.md) gives the contract and repetition limitations.

Each MTP or DSpark draft and bonus position uses its speculative-prefix mask.
Simulated grammar advancement rolls back. Kept output alone advances persistent state.
A draft that the mask rejects fails at its first invalid position.

CPU tests include rollback, rejection, and reasoning-end crossings.
Target-model MTP4 evidence includes the two APIs, off/medium, JSON object, JSON Schema, and strict tools.

DSpark proposals apply grammar masks before candidate selection. The target also applies the numeric reasoning budget.

For stochastic output, acceptance uses full conditional target/proposal probabilities with the configured sampling parameters.
Candidate acceptance is `min(1,p(x)/q(x))`. Rejection uses normalized `max(p-q,0)`, and the bonus uses target distribution `p`.
Verification uses target tokens with maximum scores for greedy sampling.
See [architecture](../architecture/speculation.md#dspark) for context and state commit.

## XML tool restrictions

Tool parameters must have a direct object root, without root `$ref` or `anyOf`.
Properties must have known, unambiguous types.
Explicit open `additionalProperties` is unsupported.
Omission in a non-strict tool closes the object to known properties.
Local property `$ref` and unambiguous `anyOf` are permitted.

Mixed string/nonstring unions are ambiguous in XML and fail.

Explicit strict XML strings cannot use pattern, length, or format constraints.
Other tool descriptions can include these keywords for client validation.
String enum/const cannot contain the parameter closing delimiter.
Candidates cannot differ only in surrounding spaces, newlines, or tabs.
The parser restores padding from the unique schema value.
XML `anyOf` cannot have sibling const/enum.

JSON answer schemas have no XML-specific restrictions.

Unconstrained strings remove one leading/trailing newline from template markup.
Quotes, spaces, extra newlines, and function/tool-close literals stay.
Enum/const strings restore the same schema value, with its newlines.
Parsers recognize closing tags only when they are not in parameter values.

With tools disabled, literal tool tags in ordinary text or JSON answers stay text.
Enabled tools reserve their opener for calls. String arguments can still contain literal markup.

Grammar masks constrain structure. Repeated ordinary text and failed tasks can still occur.
`</thinking>` and `<next_thinking>` are ordinary text, not reasoning delimiters.
Keep the client's original repetition detector and progress timeout in persistent acceptance.
