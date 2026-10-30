"""Local analysis for arbitrary pasted reviews.

pasted reviews -> detect language -> B0 summary in that language (local Ollama)
-> atomic claims (same span rule as the paper) -> all-pair NLI judging of every claim against every review
-> per-claim support counts and spans.

English input uses the English DeBERTa NLI from the paper. Other languages use a multilingual XNLI model.
Only the English judge was validated (paper). The multilingual judge passed a 10-pair sanity check only,
so non-English results are NOT validated. Not yet included: evidence agent v2, URL fetching for
non-English stores. Reviews are never written to disk by this module.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

from . import prompts
from .config import Config
from .extract import CLAIM_SCHEMA, _validate
from .judges.nli import NLIJudge
from .llm import Ollama

MAX_REVIEWS = 60
MIN_CHARS = 20
MULTI_NLI = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
LANG_NAMES = {"en": "English", "ko": "Korean", "ja": "Japanese", "zh": "Chinese"}

_HANGUL = re.compile(r"[ᄀ-ᇿ㄰-㆏가-힣]")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")
_LATIN = re.compile(r"[A-Za-z]")


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


def detect_language(text: str) -> str:
    """Script-based guess for one review: ko | ja | zh | en. Kana decides Japanese even when Han is present."""
    h, k, han, lat = (len(_HANGUL.findall(text)), len(_KANA.findall(text)),
                      len(_HAN.findall(text)), len(_LATIN.findall(text)))
    total = h + k + han + lat
    if total == 0:
        return "en"
    if h / total >= 0.3:
        return "ko"
    if k / total >= 0.1:
        return "ja"
    if han / total >= 0.3:
        return "zh"
    return "en"


def detect_input_language(reviews: list[dict]) -> tuple[str, dict]:
    """Majority vote over reviews; returns (code, counts)."""
    counts = Counter(detect_language(r["text"]) for r in reviews)
    return counts.most_common(1)[0][0], dict(counts)


@lru_cache(maxsize=4)
def _judge(model_id: str, abstain_below: float, device: str) -> NLIJudge:
    spec = {"model_id": model_id, "abstain_below": abstain_below, "device": device}
    return NLIJudge(None, "nli", spec)


def _judge_for(cfg: Config, lang: str) -> tuple[NLIJudge, str]:
    if lang == "en":
        spec = cfg["evidence"]["judges"]["nli"]
        model_id = spec["model_id"]
    else:
        spec = {"abstain_below": 0.5, "device": "cuda"}
        model_id = MULTI_NLI
    jd = _judge(model_id, float(spec.get("abstain_below", 0.5)), spec.get("device", "cuda"))
    return jd, model_id


def analyze(cfg: Config, reviews: list[dict], category: str = "product", model: str | None = None,
            language: str = "auto") -> dict:
    if len(reviews) < 3:
        raise ValueError("need at least 3 reviews of 20+ characters")
    g = cfg["generation"]
    ex = cfg["extraction"]
    client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
    model = model or g["models"][0]

    if language == "auto":
        lang, counts = detect_input_language(reviews)
    else:
        lang, counts = language, {}
    lang_name = LANG_NAMES.get(lang, "English")

    # no reviewer labels in the prompt: the generator would otherwise copy them into the summary text
    block = "\n\n".join(f"- {r['text']}" for r in reviews)
    gen = client.chat(model, prompts.render("summary_lang.txt", language=lang_name,
                                            category=category.replace("_", " "), reviews=block),
                      temperature=g["temperature"], seed=g["seed"], num_ctx=g["num_ctx"],
                      num_predict=g["num_predict"], thinking=False)
    text = gen["text"].strip()

    obj, _ = client.chat_json(ex["model"], prompts.render("extract_claims_lang.txt", language=lang_name,
                                                          summary=text), CLAIM_SCHEMA,
                              temperature=ex["temperature"], num_ctx=ex["num_ctx"], num_predict=ex["num_predict"])
    claims = [c for c in _validate(obj.get("claims", []), text) if c["valid"]]

    jd, judge_model = _judge_for(cfg, lang)
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
    validated = lang == "en"
    return {"model": model, "language": lang, "language_name": lang_name, "language_counts": counts,
            "judge": judge_model, "judge_validated": validated,
            "n_reviews": len(reviews), "text": text, "word_count": len(text.split()), "claims": out_claims}
