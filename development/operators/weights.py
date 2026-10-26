import json
from dataclasses import replace

from oh_my_vllm.models.qwen38.checkpoint import Checkpoint
from oh_my_vllm.models.qwen38.weights import Weights
from oh_my_vllm.operators.mtp.api import InputWeights
from oh_my_vllm.operators.output.api import Weights as OutputWeights
from oh_my_vllm.operators.vision.api import Weights as VisionWeights

from .bindings import VISION_BLOCK, VISION_MERGER, weight_names


class CaseWeights(Weights):
    def __init__(self, device, path, case):
        self.device, self.checkpoint = device, Checkpoint(path)
        self.config = json.loads((self.checkpoint.path / "config.json").read_text())
        self.names = set()
        self.bindings = ()
        self.visual = None
        family = case.boundary.split(".")[0]
        if family in {"fa", "gdn", "ffn"}:
            self.bindings = tuple(
                self.text_weight(family, layer) for layer in case.weights
            )
        elif family == "output":
            self.norm = self.floats("model.language_model.norm.weight")
            if case.boundary != "output.hidden":
                self.sample = OutputWeights(self.norm, self.upload("lm_head.weight"))
        elif family == "mtp" and case.boundary != "mtp.accept":
            self.embedding = self.upload("model.language_model.embed_tokens.weight")
            self.mtp_input = InputWeights(
                self.embedding,
                self.floats("mtp.pre_fc_norm_hidden.weight"),
                self.floats("mtp.pre_fc_norm_embedding.weight"),
                self.upload("mtp.fc.weight"),
            )
            if case.boundary == "mtp.context":
                self.mtp_attention = self.attention("mtp.layers.0.")
            else:
                self.mtp_layer = self.layer(
                    "mtp.layers.0.",
                    "full_attention",
                )
                self.mtp_sample = OutputWeights(
                    self.floats("mtp.norm.weight"), self.upload("lm_head.weight")
                )
        elif family == "vision":
            self.vision_case(case)
        if sorted(self.names) != weight_names(case):
            raise ValueError("loaded tensor names differ from the fixed weight binding")

    def vision_case(self, case):
        blocks = tuple(
            tuple(
                self.upload(f"model.visual.blocks.{layer}.{name}")
                for name in VISION_BLOCK
            )
            if layer in case.weights
            else ()
            for layer in range(27)
        )
        patch, merger = (), ()
        if case.boundary == "vision.input":
            patch = (
                self.device.view(
                    self.upload("model.visual.patch_embed.proj.weight"), (1152, 1536)
                ),
                self.upload("model.visual.patch_embed.proj.bias"),
                self.upload("model.visual.pos_embed.weight"),
            )
        if case.boundary == "vision.merge":
            merger = tuple(
                self.upload("model.visual.merger." + name) for name in VISION_MERGER
            )
        self.visual = VisionWeights(blocks, patch, merger)

    def upload(self, name):
        self.names.add(name)
        return super().upload(name)

    def floats(self, name):
        self.names.add(name)
        return super().floats(name)

    def concatenate(self, names):
        self.names.update(names)
        return super().concatenate(names)

    def text_weight(self, family, layer):
        prefix = f"model.language_model.layers.{layer}."
        if family == "fa":
            return self.attention(prefix)
        if family == "ffn":
            return self.feed_forward(prefix)
        weight = self.recurrent(prefix)
        if layer == 0:
            weight = replace(
                weight,
                embedding=self.upload("model.language_model.embed_tokens.weight"),
            )
        return weight

    @property
    def identity(self):
        return {"files": dict(self.checkpoint.identity), "weights": sorted(self.names)}

    def verify(self, expected):
        if self.identity != expected:
            raise ValueError(
                "operator checkpoint or weight bindings differ from the fixture"
            )
        self.checkpoint.verify_identity(expected["files"])
