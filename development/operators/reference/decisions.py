from contextvars import ContextVar

import torch

CURRENT = ContextVar("reference_decisions", default=None)
MINIMUM_MARGIN = 0.01


class DecisionMargins:
    def __init__(self, probability_cap=None):
        self.records = []
        self.probability_cap = probability_cap

    def __enter__(self):
        self.token = CURRENT.set(self)
        return self

    def __exit__(self, *args):
        CURRENT.reset(self.token)


def record(kind, distance, **details):
    current = CURRENT.get()
    if current is not None:
        minimum = max(MINIMUM_MARGIN, details.get("budget", 0) + 32 * 2**-23)
        if distance <= minimum:
            raise ValueError(f"ambiguous {kind} decision: margin {distance}")
        current.records.append({"kind": kind, "distance": distance, **details})


def cdf_margin(probability, draw):
    positive = probability[probability > 0]
    cumulative = positive.cumsum(dim=0) / positive.sum()
    boundaries = torch.cat((cumulative.new_zeros(1), cumulative))
    current = CURRENT.get()
    index = int(torch.searchsorted(cumulative, draw))
    left, right = float(boundaries[index]), float(boundaries[index + 1])
    details = {"draw": draw, "left": left, "right": right}
    if current is not None and current.probability_cap is not None:
        details["budget"] = min(current.probability_cap, (right - left) / 3)
    record("cdf", float((boundaries - draw).abs().min()), **details)


def acceptance_margin(probability, proposal, draw):
    if proposal > 0:
        record("acceptance", abs(probability / proposal - draw))
