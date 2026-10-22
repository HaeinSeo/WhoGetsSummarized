"""DeBERTa NLI baseline: premise=review, hypothesis=single experience proposition.

Mapping entailment->SUPPORT, contradiction->OPPOSE, neutral->IRRELEVANT is an approximation;
NLI contradiction is NOT the same as an opposite experience. Low-confidence -> UNCERTAIN.
The mapping is validated against human pair gold before use.
"""
from __future__ import annotations

import time

from .base import Judge

MAP = {"entailment": "SUPPORT", "contradiction": "OPPOSE", "neutral": "IRRELEVANT"}


class NLIJudge(Judge):
    batch_ok = True

    def __init__(self, cfg, name, spec, batch: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.name = name
        self.model_id = spec["model_id"]
        self.tau = float(spec.get("abstain_below", 0.5))
        self.device = spec.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id).to(self.device).eval()
        self.id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        if set(self.id2label.values()) != set(MAP):
            raise RuntimeError(f"unexpected NLI labels {self.id2label}")
        self.batch = batch
        rev = getattr(self.model.config, "_commit_hash", None) or "unknown"
        self.version = f"nli|{self.model_id}|{rev}|tau={self.tau}"

    def judge(self, items):
        out = []
        for i in range(0, len(items), self.batch):
            chunk = items[i:i + self.batch]
            t0 = time.perf_counter()
            hyps = []
            for c in chunk:
                cond = c.get("condition")
                hyps.append(f"{c['experience']} ({cond})" if cond else c["experience"])
            enc = self.tok([c["review"] for c in chunk], hyps, truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                p = self.torch.softmax(self.model(**enc).logits, -1).cpu().numpy()
            dt_each = (time.perf_counter() - t0) / len(chunk)
            for row in p:
                probs = {MAP[self.id2label[j]]: float(v) for j, v in enumerate(row)}
                top = max(probs, key=probs.get)
                abstain = probs[top] < self.tau
                out.append({"label": "UNCERTAIN" if abstain else top, "probs": probs,
                            "abstained": bool(abstain), "latency_s": round(dt_each, 4)})
        return out
