import json
from pathlib import Path

import llguidance
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer


class Tokenization:
    def __init__(self, path):
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path / "tokenizer.json"))
        config = json.loads((path / "generation_config.json").read_text())
        eos = config["eos_token_id"]
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.valid_tokens = self.tokenizer.get_vocab_size()
        self.grammar = llguidance.LLTokenizer(
            self.tokenizer.to_str(), n_vocab=248320, eos_token=sorted(self.eos)
        )
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"]
        )
        environment.globals["raise_exception"] = self.raise_exception
        environment.filters["tojson"] = lambda value, **options: json.dumps(
            value, ensure_ascii=False, **options
        )
        self.template = environment.from_string(
            (path / "chat_template.jinja").read_text()
        )
        self.think_end = self.tokenizer.token_to_id("</think>")
        self.generation = config

    @staticmethod
    def raise_exception(message):
        raise ValueError(message)

    def decode_bytes(self, tokens):
        return b"".join(
            self.tokenizer.id_to_token(int(token)).encode("utf-8")
            if self.grammar.is_special_token(int(token))
            else self.grammar.decode_bytes([int(token)])
            for token in tokens
        )

    def prepare(self, request):
        messages = list(request["messages"])
        tools = request.get("tools", []) if request.get("tool_choice") != "none" else []
        if tools:
            messages = [
                {
                    "role": "system",
                    "content": (
                        'For each tool call, use <tool_call>{"name":"FUNCTION_NAME",'
                        '"arguments":{...}}</tool_call>. Use JSON arguments. '
                        "Do not use function or parameter XML tags."
                    ),
                },
                *messages,
            ]
        prompt = self.template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=request["effort"] != "off",
            reasoning_effort=request["effort"],
            preserve_thinking=request.get("preserve_thinking", True),
        )
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids
