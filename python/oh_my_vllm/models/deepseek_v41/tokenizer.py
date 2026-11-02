"""DeepSeek tokenizer and chat template ownership."""

import hashlib
import json
from pathlib import Path

import llguidance
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

TOKENIZER_ASSETS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "encoding.py",
)
TOKENIZER_PATHS = {
    "tokenizer.json": "tokenizer.json",
    "tokenizer_config.json": "tokenizer_config.json",
    "chat_template.jinja": "chat_template.jinja",
    "encoding.py": "encoding/encoding.py",
}


class Tokenization:
    def __init__(self, path):
        self.path = Path(path).resolve()
        data = {
            name: (self.path / TOKENIZER_PATHS[name]).read_bytes()
            for name in TOKENIZER_ASSETS
        }
        self.identity = {
            name: hashlib.sha256(value).hexdigest() for name, value in data.items()
        }
        self.tokenizer = Tokenizer.from_str(data["tokenizer.json"].decode("utf-8"))
        config = json.loads(data["tokenizer_config.json"].decode("utf-8"))
        eos = self.tokenizer.token_to_id(
            config["eos_token"]["content"]
            if isinstance(config["eos_token"], dict)
            else config["eos_token"]
        )
        if eos is None:
            raise ValueError("DeepSeek tokenizer has no EOS token")
        self.eos = {eos}
        self.valid_tokens = self.tokenizer.get_vocab_size()
        self.grammar = llguidance.LLTokenizer(
            self.tokenizer.to_str(),
            n_vocab=self.valid_tokens,
            eos_token=sorted(self.eos),
        )
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        environment.globals["raise_exception"] = self.raise_exception
        environment.filters["tojson"] = lambda value, **options: json.dumps(
            value, ensure_ascii=False, **options
        )
        self.template = environment.from_string(
            data["chat_template.jinja"].decode("utf-8")
        )
        think_end = self.tokenizer.token_to_id("</think>")
        if think_end is None:
            raise ValueError("DeepSeek tokenizer has no thinking terminator")
        self.think_end = think_end

    def verify_identity(self, expected):
        if self.identity != expected:
            raise ValueError("loaded tokenizer assets differ from the model contract")
        observed = {
            name: hashlib.sha256(
                (self.path / TOKENIZER_PATHS[name]).read_bytes()
            ).hexdigest()
            for name in TOKENIZER_ASSETS
        }
        if observed != expected:
            raise ValueError("tokenizer asset files changed after loading")

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
        if request.get("tools") and request.get("tool_choice") != "none":
            raise ValueError("DeepSeek tools are not enabled")
        messages = list(request["messages"])
        effort = request.get("effort", "off")
        prompt = self.template.render(
            messages=messages,
            tools=[],
            add_generation_prompt=True,
            enable_thinking=effort != "off",
            thinking_mode="thinking" if effort != "off" else "chat",
            reasoning_effort=request.get("reasoning_effort", "high"),
            drop_thinking=False,
        )
        return self.tokenizer.encode(prompt, add_special_tokens=False).ids
