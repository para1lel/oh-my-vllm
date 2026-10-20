import json
from dataclasses import replace

from oh_my_vllm.kernels.mtp import InputWeights
from oh_my_vllm.kernels.sample import Weights as OutputWeights
from oh_my_vllm.worker.checkpoint import Checkpoint
from oh_my_vllm.worker.weights import Weights


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
                self.mtp_attention = self.attention("mtp.layers.0.", prepare=False)
            else:
                self.mtp_layer = self.layer(
                    "mtp.layers.0.",
                    "full_attention",
                    prepare_ffn=False,
                    prepare_fa=False,
                )
                self.mtp_sample = OutputWeights(
                    self.floats("mtp.norm.weight"), self.upload("lm_head.weight")
                )
        elif family == "vision":
            self.vision()

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
