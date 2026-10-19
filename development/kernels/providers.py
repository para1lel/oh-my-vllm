from dataclasses import replace

from oh_my_vllm.kernels.fa import FUNCTIONS as FA_FUNCTIONS
from oh_my_vllm.kernels.ffn import FUNCTIONS as FFN_FUNCTIONS
from oh_my_vllm.kernels.gdn import FUNCTIONS as GDN_FUNCTIONS
from oh_my_vllm.kernels.mtp import FUNCTIONS as MTP_FUNCTIONS
from oh_my_vllm.kernels.vision import FUNCTIONS as VISION_FUNCTIONS
from oh_my_vllm.worker.device import FUNCTIONS as DEVICE_FUNCTIONS

MODULES = {
    "device": DEVICE_FUNCTIONS,
    "ffn": FFN_FUNCTIONS,
    "fa": FA_FUNCTIONS,
    "gdn": GDN_FUNCTIONS,
    "mtp": MTP_FUNCTIONS,
    "sample": ("sample_workspace", "sample", "mtp_accept"),
    "vision": VISION_FUNCTIONS,
}

RUNTIME_MODULES = {
    "ffn": ("ffn",),
    "fa_attention": ("fa",),
    "gdn_attention": ("gdn",),
    "sample": ("sample",),
    "mtp_context": ("mtp", "fa"),
    "mtp_draft": ("mtp", "fa", "ffn", "sample"),
    "vision_block": ("vision",),
}


def prepare(operator, device, checkpoint):
    if operator == "ffn":
        from .ffn import FFNCase, FFNWeights

        weights = FFNWeights(device, checkpoint)
        return lambda shape: FFNCase(device, weights, shape)
    if operator == "fa_attention":
        from .fa import FACase, FAWeights

        weights = FAWeights(device, checkpoint)
        return lambda shape: FACase(device, weights, shape)
    if operator == "gdn_attention":
        from .gdn import GDNCase, GDNWeights

        weights = GDNWeights(device, checkpoint)
        weights.with_embedding(device)
        return lambda shape: GDNCase(device, weights, shape)
    if operator == "vision_block":
        from .vision import VisionCase, VisionWeights

        weights = VisionWeights(device, checkpoint)
        return lambda shape: VisionCase(device, weights, shape)
    if operator == "sample":
        from .sample import SampleCase, SampleWeights

        weights = SampleWeights(device, checkpoint)
        return lambda shape: SampleCase(device, weights, shape)
    if operator in {"mtp_context", "mtp_draft"}:
        from .mtp import MTPContextCase, MTPDraftCase, MTPWeights

        weights = MTPWeights(device, checkpoint)
        if operator == "mtp_context":
            weights.attention("kv_only")
            return lambda shape: MTPContextCase(device, weights, shape)
        from .sample import SampleCase, SampleWeights

        weights.attention("full")
        weights.attention("query_only")
        weights.feed_forward()
        sample_weights = SampleWeights(device, checkpoint, norm="mtp.norm.weight")

        def draft(shape):
            sample_shape = replace(
                shape, operator="sample", variant="sampling", context=0
            )
            sample_case = SampleCase(device, sample_weights, sample_shape)
            return MTPDraftCase(device, weights, shape, sample_case)

        return draft
    raise ValueError(f"native provider is missing: {operator}")
