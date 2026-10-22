import codecs
import secrets

import numpy as np

from .constraints import Constraint
from .repetition import HISTORY_WINDOW, cycle_continuations


class Request:
    def __init__(self, tokenizer, tokens, options, *, fixed_length=False):
        self.tokenizer = tokenizer
        self.prompt_len = len(tokens)
        self.history = list(tokens)
        self.limit = options["max_tokens"]
        self.minimum = self.limit if fixed_length else options.get("min_tokens", 0)
        self.ignore_eos = options.get("ignore_eos", False)
        self.options = options
        self.seed = options.get("sampling", {}).get("seed", secrets.randbits(63)) % (
            1 << 64
        )
        self.random_offset = 0
        self.proposal_offset = 0
        self.accept_offset = 0
        self.state_slot = None
        self.state_position = 0
        self.constraint = Constraint.for_request(tokenizer.grammar, options)
        self.thinking = options.get("effort", "off") != "off"
        self.thinking_count = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.buffer = ""
        self.finish = None
        self.media = []
        self.rope_positions = None
        self.rope_delta = 0
        self.counts = np.zeros((2, 248320), dtype=np.int32)
        self.counts[0] = np.bincount(tokens, minlength=248320)
        self.presence = np.zeros(248320 // 8, dtype=np.uint8)
        self.sample_masks = {}
        self.drafts = []
        self.drafts_greedy = True
        self.proposals = None
        self.proposal_rows = ()
        self.mtp_position = 0

    def stop_position(self, text, first_new):
        positions = [
            text.find(stop, max(0, first_new - len(stop) + 1))
            for stop in self.options.get("stop", [])
        ]
        return min((position for position in positions if position >= 0), default=None)

    def stops_after(self, prefix):
        if (
            not self.ignore_eos and any(token in self.tokenizer.eos for token in prefix)
        ) or self.generated + len(prefix) >= self.limit:
            return True
        if not self.options.get("stop"):
            return False
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        decoder.setstate(self.decoder.getstate())
        text = self.buffer
        for index, token in enumerate(prefix):
            if token in self.tokenizer.eos:
                continue
            first_new = len(text)
            text += decoder.decode(self.tokenizer.decode_bytes([token]))
            if (
                self.generated + index + 1 >= self.minimum
                and self.stop_position(text, first_new) is not None
            ):
                return True
        return False

    @property
    def generated(self):
        return len(self.history) - self.prompt_len

    def position(self, position):
        if self.rope_positions is not None and position < self.prompt_len:
            return self.rope_positions[:, position]
        return (position + self.rope_delta,) * 3

    def mask(
        self,
        constraint=None,
        thinking=None,
        generated=None,
        thinking_count=None,
        prefix=(),
        mask=None,
    ):
        constraint = self.constraint if constraint is None else constraint
        thinking = self.thinking if thinking is None else thinking
        generated = self.generated if generated is None else generated
        thinking_count = (
            self.thinking_count if thinking_count is None else thinking_count
        )
        if mask is None:
            mask = np.ones(248320, dtype=np.uint8) if thinking else constraint.mask()
        mask[self.tokenizer.valid_tokens :] = 0
        budget = self.options.get("thinking_token_budget")
        if thinking and budget is not None and budget >= 0 and thinking_count >= budget:
            mask[:] = 0
            mask[self.tokenizer.think_end] = 1
        if thinking or generated < self.minimum:
            mask[list(self.tokenizer.eos)] = 0
        if self.options.get("sampling", {}).get("repetition_guard", True):
            history = self.history[
                max(self.prompt_len, len(self.history) - HISTORY_WINDOW) :
            ] + list(prefix)
            blocked = list(cycle_continuations(history))
            if blocked and np.count_nonzero(mask[blocked]) < np.count_nonzero(mask):
                mask[blocked] = 0
        return mask

    def parameters(self):
        values = {**self.tokenizer.generation, **self.options.get("sampling", {})}
        return np.asarray(
            [
                values.get("temperature", 1),
                values.get("top_k", 20),
                values.get("top_p", 0.95),
                values.get("min_p", 0),
                values.get("frequency_penalty", 0),
                values.get("presence_penalty", 0),
                values.get("repetition_penalty", 1),
            ],
            dtype=np.float32,
        )

    def append(self, tokens):
        self.sample_masks.clear()
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
            self.counts[1, token] += 1
            self.presence[token // 8] |= 1 << (token % 8)
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
