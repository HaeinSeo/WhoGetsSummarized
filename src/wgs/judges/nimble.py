"""Nimble choice judge via Ollama's decision endpoint (/v1/systemone).

Nimble is used ONLY to choose among allowed labels. No quotes, decomposition, or free text.
Its probabilities are stored but not assumed calibrated on this data.
"""
from __future__ import annotations

import time

from ..llm import Ollama, OllamaError
from ..storage import short_hash
from .base import CRITERIA, LABELS, Judge


class NimbleJudge(Judge):
    parallel_ok = True

    def __init__(self, cfg, name, spec):
        self.name = name
        self.model = spec["model"]
        self.client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
        digest = self.client.digest(self.model)
        if not digest:
            raise RuntimeError(f"{self.model} not installed: run `ollama pull {self.model}`")
        self.version = f"nimble|{self.model}|{digest}|{short_hash(CRITERIA)}"

    def _payload(self, it: dict) -> dict:
        state = {"experience": it["experience"], "review": it["review"]}
        if it.get("condition"):
            state["condition"] = it["condition"]
        return {"model": self.model, "state": state, "questions": {"relation": {
            "type": "choice",
            "instructions": "Judge the review against the experience under the same conditions.",
            "criteria": CRITERIA}}}

    def judge(self, items):
        out = []
        for it in items:
            t0 = time.perf_counter()
            resp = self.client.systemone(self._payload(it))
            try:
                ans = resp["answers"]["relation"]
                label = ans["choice"]
            except (KeyError, TypeError):
                raise OllamaError(f"unexpected systemone response schema: {str(resp)[:400]}")
            if label not in LABELS:
                raise OllamaError(f"Nimble returned label outside schema: {label}")
            out.append({"label": label, "probs": ans.get("probabilities"), "abstained": False,
                        "latency_s": round(time.perf_counter() - t0, 3)})
        return out
