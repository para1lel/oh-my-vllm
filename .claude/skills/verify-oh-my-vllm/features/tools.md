# Tools and structured output

## Sub-features

- Chat tool calls with the nested `function` form.
- Responses tool calls with the flat function form.
- A required `lookup` call with structured arguments.
- Chat JSON Schema output with a strict object schema.

See [request normalization](../../../../crates/zmq-worker/src/serving/request.rs) and [Qwen output parsing](../../../../crates/zmq-worker/src/models/qwen38/output.rs).

## How to get to it (user POV)

After [Launch](../SKILL.md#launch), send `tools` and `tool_choice` to Chat or Responses.
Chat uses `tools[].function`. Responses puts the function fields in `tools[]`.
Use Chat `response_format` or Responses `text.format` to select a structured output format.
The helper's schema drive uses Chat.

## Driving it with verify.py

```bash
scripts/with-env.sh python .claude/skills/verify-oh-my-vllm/verify.py run --feature tools --binary target/debug/oh-my-vllm-zmq-worker
```

The [tools drive](../drive.py) captures Chat and Responses tool replies.
It checks the Chat function name and parses its arguments as JSON.
It checks the Responses output item type.
It then requests a strict Chat schema with `key` limited to `hello`.
It parses the reply and compares the object with the fixed fixture value.
Read `features/tools.json` and the related `http/` records.

## Gotchas

JSON Schema and enabled tools cannot be used together.
The fixture selects fixed replies from the request controls.
It does not apply GPU grammar masks or perform model sampling.
The tool result is not an external function execution.
Use [the checkpoint path](../SKILL.md#checkpoint-service-path) and model-local constraints checks for generation behavior.
