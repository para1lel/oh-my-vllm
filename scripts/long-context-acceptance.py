"""Real MTP constrained generation and prefix recovery beyond 131072 input tokens."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", default="http://127.0.0.1:18015/v1")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
base = args.base_url.rstrip("/") + "/"
tok = AutoTokenizer.from_pretrained("/data0/shared/Qwen3.8-27B-FP8")
unit = tok.encode(
    "参考资料:北京是中国的首都,拥有丰富的历史文化。", add_special_tokens=False
)
prompt = (
    tok.decode((unit * (131072 // len(unit) + 1))[:131072])
    + "\nReturn only JSON with n=123 and label=verified."
)
schema = {
    "type": "object",
    "properties": {
        "n": {"type": "integer", "enum": [123]},
        "label": {"type": "string", "const": "verified"},
    },
    "required": ["n", "label"],
    "additionalProperties": False,
}
rows = []
for api in ["chat/completions", "responses"]:
    for repeat in range(2):
        body = {"model": "qwen3.8-27b-fp8", "temperature": 0}
        if api == "chat/completions":
            body.update(
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="off",
                max_completion_tokens=256,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "result", "strict": True, "schema": schema},
                },
            )
        else:
            body.update(
                input=prompt,
                reasoning={"effort": "off"},
                max_output_tokens=256,
                store=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "result",
                        "strict": True,
                        "schema": schema,
                    }
                },
            )
        start = time.monotonic()
        with urllib.request.urlopen(
            urllib.request.Request(
                base + api,
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=600,
        ) as r:
            out = json.load(r)
        if api == "chat/completions":
            text = out["choices"][0]["message"]["content"]
            usage = out["usage"]
            count = usage["prompt_tokens"]
            cached = usage["prompt_tokens_details"]["cached_tokens"]
        else:
            text = "".join(
                c.get("text", "")
                for item in out["output"]
                for c in item.get("content", [])
                if c.get("type") == "output_text"
            )
            usage = out["usage"]
            count = usage["input_tokens"]
            cached = usage["input_tokens_details"]["cached_tokens"]
        assert json.loads(text) == {"n": 123, "label": "verified"}, out
        assert count >= 131072, (count, out)
        if repeat:
            assert cached >= 130000, (cached, out)
        rows.append(
            {
                "api": api,
                "repeat": repeat,
                "elapsed_s": time.monotonic() - start,
                "response": out,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        )
        print(api, repeat, count, cached, flush=True)
args.output.write_text(json.dumps(rows, indent=2))
