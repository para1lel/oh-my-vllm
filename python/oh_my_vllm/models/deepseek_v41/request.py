import codecs

from ...worker.constraints import Constraint


class Request:
    def __init__(self, tokenizer, tokens, options, *, fixed_length=False):
        self.tokenizer = tokenizer
        self.prompt_len = len(tokens)
        self.history = list(tokens)
        self.limit = options["max_tokens"]
        self.minimum = self.limit if fixed_length else options.get("min_tokens", 0)
        self.ignore_eos = options.get("ignore_eos", False)
        self.options = options
        self.cache_slot = None
        self.state_position = 0
        self.constraint = Constraint.for_request(tokenizer.grammar, options)
        self.thinking = options.get("effort", "off") != "off"
        self.thinking_count = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.buffer = ""
        self.finish = None

    def stop_position(self, text, first_new):
        positions = [
            text.find(stop, max(0, first_new - len(stop) + 1))
            for stop in self.options.get("stop", [])
        ]
        return min((position for position in positions if position >= 0), default=None)

    @property
    def generated(self):
        return len(self.history) - self.prompt_len

    def append(self, tokens):
        kept = []
        for token in tokens:
            if self.generated >= self.limit or self.finish:
                break
            token = int(token)
            if self.thinking:
                self.thinking_count += 1
                if token == self.tokenizer.think_end:
                    self.thinking = False
            else:
                self.constraint.accept([token])
            self.history.append(token)
            kept.append(token)
            if token in self.tokenizer.eos:
                if not self.ignore_eos:
                    self.finish = "stop"
                    break
                continue
            first_new = len(self.buffer)
            self.buffer += self.decoder.decode(self.tokenizer.decode_bytes([token]))
            if self.generated >= self.minimum:
                position = self.stop_position(self.buffer, first_new)
                if position is not None:
                    self.buffer = self.buffer[:position]
                    self.decoder.reset()
                    self.finish = "stop"
                    break
        if self.generated >= self.limit and self.finish is None:
            self.finish = "length"
        if self.finish:
            self.buffer += self.decoder.decode(b"", final=True)
        end = len(self.buffer)
        if not self.finish:
            for stop in self.options.get("stop", []):
                for size in range(1, min(len(stop), len(self.buffer) + 1)):
                    if self.buffer.endswith(stop[:size]):
                        end = min(end, len(self.buffer) - size)
        text = self.buffer[:end]
        self.buffer = self.buffer[end:]
        return kept, text
