"""Greedy DSpark acceptance rules."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Verification:
    accepted_draft_tokens: int
    token_ids: tuple[int, ...]
    rejected: bool


def verify_greedy_drafts(drafts, target_predictions) -> Verification:
    """Accept a matching draft prefix and emit one target token at the boundary."""
    drafts = tuple(int(token) for token in drafts)
    target_predictions = tuple(int(token) for token in target_predictions)
    if len(target_predictions) != len(drafts) + 1:
        raise ValueError("target predictions must include one bonus prediction")

    for index, (draft, target) in enumerate(
        zip(drafts, target_predictions[:-1], strict=True)
    ):
        if draft != target:
            return Verification(index, (*drafts[:index], target), True)
    return Verification(len(drafts), (*drafts, target_predictions[-1]), False)
