"""Evidence judges: SUPPORT / OPPOSE / IRRELEVANT / UNCERTAIN for (experience, review) pairs."""
from __future__ import annotations

from ..config import Config
from .base import LABELS, Judge, aggregate_labels  # noqa: F401


def make_judge(cfg: Config, name: str) -> Judge:
    spec = cfg["evidence"]["judges"][name]
    if spec["type"] == "nimble":
        from .nimble import NimbleJudge
        return NimbleJudge(cfg, name, spec)
    if spec["type"] == "nli":
        from .nli import NLIJudge
        return NLIJudge(cfg, name, spec)
    if spec["type"] == "llm":
        from .llm_judge import LLMJudge
        return LLMJudge(cfg, name, spec)
    raise ValueError(f"unknown judge type {spec['type']}")
