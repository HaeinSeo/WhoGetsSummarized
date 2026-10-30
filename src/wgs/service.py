"""Local analysis for arbitrary pasted reviews (MVP).

pasted reviews -> B0 summary (local Ollama) -> atomic claims (same extractor and span rule as the paper)
-> all-pair NLI judging of every claim against every review -> per-claim support counts and spans.

Not yet included: evidence agent v2 (needs experience extraction and planner candidates), URL fetching,
multilingual judging. The judge is the English DeBERTa NLI used in the paper, so results on other
languages or domains are NOT validated. Reviews are never written to disk by this module.
"""
from __future__ import annotations

import re
from functools import lru_cache

from . import prompts
from .config import Config
from .extract import CLAIM_SCHEMA, _validate
from .judges.nli import NLIJudge
from .llm import Ollama

MAX_REVIEWS = 60
MIN_CHARS = 20


def parse_reviews(text: str) -> list[dict]:
    """One review per non-empty line or blank-line-separated block; short fragments are dropped."""
    blocks = re.split(r"\n\s*\n|\r?\n", text.strip())
    out = []
    for b in blocks:
        b = " ".join(b.split())
        if len(b) >= MIN_CHARS:
            out.append(b)
    out = out[:MAX_REVIEWS]
    return [{"anon_id": f"R{i + 1:02d}", "title": "", "text": t} for i, t in enumerate(out)]


@lru_cache(maxsize=1)
def _judge(model_id: str, abstain_below: float, device: str) -> NLIJudge:
    spec = {"model_id": model_id, "abstain_below": abstain_below, "device": device}
    return NLIJudge(None, "nli", spec)


def analyze(cfg: Config, reviews: list[dict], category: str = "product", model: str | None = None) -> dict:
    if len(reviews) < 3:
        raise ValueError("need at least 3 reviews of 20+ characters")
    g = cfg["generation"]
    ex = cfg["extraction"]
    client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
    model = model or g["models"][0]

    # no reviewer labels in the prompt: the generator would otherwise copy them into the summary text
    block = "\n\n".join(f"- {r['text']}" for r in reviews)
    gen = client.chat(model, prompts.render("summary_direct.txt", category=category.replace("_", " "),
                                            reviews=block),
                      temperature=g["temperature"], seed=g["seed"], num_ctx=g["num_ctx"],
                      num_predict=g["num_predict"], thinking=False)
    text = gen["text"].strip()

    obj, _ = client.chat_json(ex["model"], prompts.render("extract_claims.txt", summary=text), CLAIM_SCHEMA,
                              temperature=ex["temperature"], num_ctx=ex["num_ctx"], num_predict=ex["num_predict"])
    claims = [c for c in _validate(obj.get("claims", []), text) if c["valid"]]

    judge_cfg = cfg["evidence"]["judges"]["nli"]
    jd = _judge(judge_cfg["model_id"], float(judge_cfg.get("abstain_below", 0.5)),
                judge_cfg.get("device", "cuda"))
    items, index = [], []
    for ci, c in enumerate(claims):
        for ri, r in enumerate(reviews):
            items.append({"experience": c["proposition"], "condition": c.get("condition") or "", "review": r["text"]})
            index.append((ci, ri))
    labels = jd.judge(items) if items else []

    support: dict[int, list[str]] = {ci: [] for ci in range(len(claims))}
    for (ci, ri), lab in zip(index, labels):
        if lab["label"] == "SUPPORT":
            support[ci].append(reviews[ri]["anon_id"])

    out_claims = []
    for ci, c in enumerate(claims):
        out_claims.append({"start": int(c["span_start"]), "end": int(c["span_end"]),
                           "n_support": len(support[ci]), "reviews": sorted(support[ci]),
                           "proposition": c["proposition"], "polarity": c["polarity"]})
    return {"model": model, "judge": "nli (DeBERTa-v3 MNLI/FEVER/ANLI, English; unvalidated for this input)",
            "n_reviews": len(reviews), "text": text, "word_count": len(text.split()), "claims": out_claims}
