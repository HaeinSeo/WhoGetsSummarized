from __future__ import annotations

LABELS = ["SUPPORT", "OPPOSE", "IRRELEVANT", "UNCERTAIN"]

CRITERIA = {
    "SUPPORT": "The review explicitly reports evidence supporting the experience.",
    "OPPOSE": "The review explicitly reports an opposite experience under comparable conditions.",
    "IRRELEVANT": "The review contains no relevant evidence for this experience.",
    "UNCERTAIN": "Ambiguous, mixed, or the conditions do not permit a reliable decision.",
}


class Judge:
    name: str
    version: str  # changes whenever model/prompt/mapping changes -> cache key

    def judge(self, items: list[dict]) -> list[dict]:
        """items: {'experience','condition','review'} -> {'label','probs','abstained','latency_s'}"""
        raise NotImplementedError


def aggregate_labels(labels: list[str]) -> str:
    """Combine window-level labels for one review into a review-level relation."""
    s = set(labels)
    if "SUPPORT" in s and "OPPOSE" in s:
        return "UNCERTAIN"
    if "SUPPORT" in s:
        return "SUPPORT"
    if "OPPOSE" in s:
        return "OPPOSE"
    if "UNCERTAIN" in s:
        return "UNCERTAIN"
    return "IRRELEVANT"
