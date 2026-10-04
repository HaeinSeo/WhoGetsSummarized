"""General LLM judge (default llama3.1:8b) with an enum-constrained JSON answer."""
from __future__ import annotations

from .. import prompts
from ..llm import Ollama
from .base import LABELS, Judge

SCHEMA = {"type": "object", "required": ["label"],
          "properties": {"label": {"type": "string", "enum": LABELS}}}


class LLMJudge(Judge):
    parallel_ok = True

    def __init__(self, cfg, name, spec):
        self.name = name
        self.model = spec["model"]
        self.client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
        digest = self.client.digest(self.model)
        if not digest:
            raise RuntimeError(f"{self.model} not installed")
        self.version = f"llm|{self.model}|{digest}|{prompts.prompt_hash('judge_llm.txt')}"

    def judge(self, items):
        out = []
        for it in items:
            prompt = prompts.render("judge_llm.txt", experience=it["experience"],
                                    condition=it.get("condition") or "(none)", review=it["review"])
            obj, meta = self.client.chat_json(self.model, prompt, SCHEMA, temperature=0.0,
                                              num_ctx=4096, num_predict=20)
            out.append({"label": obj["label"], "probs": None, "abstained": False,
                        "latency_s": meta["wall_s"]})
        return out
