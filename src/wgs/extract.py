"""Atomic claims from summaries (output side) and experiences from reviews (input side).

Both are separate LLM calls with their own prompts. Spans are validated as exact
(normalized) substrings; failures are kept and flagged, never silently repaired.
"""
from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from . import data, prompts
from .config import Config
from .llm import Ollama
from .storage import JsonlAppender, log_failure, log_step, read_jsonl, short_hash, write_parquet
from .text import locate_span

log = logging.getLogger(__name__)
POLARITY = ["positive", "negative", "mixed", "neutral"]

CLAIM_SCHEMA = {"type": "object", "required": ["claims"], "properties": {"claims": {
    "type": "array", "maxItems": 15, "items": {"type": "object",
        "required": ["source_span", "aspect", "proposition", "polarity", "condition",
                     "quantifier", "claim_type"],
        "properties": {
            "source_span": {"type": "string"}, "aspect": {"type": "string"},
            "proposition": {"type": "string"}, "polarity": {"type": "string", "enum": POLARITY},
            "condition": {"type": "string"}, "quantifier": {"type": "string"},
            "claim_type": {"type": "string", "enum": ["reported_experience", "product_attribute",
                                                      "recommendation", "meta"]}}}}}}

EXP_SCHEMA = {"type": "object", "required": ["experiences"], "properties": {"experiences": {
    "type": "array", "maxItems": 10, "items": {"type": "object",
        "required": ["source_span", "aspect", "proposition", "polarity", "condition"],
        "properties": {
            "source_span": {"type": "string"}, "aspect": {"type": "string"},
            "proposition": {"type": "string"}, "polarity": {"type": "string", "enum": POLARITY},
            "condition": {"type": "string"}}}}}}


def review_string(r: dict) -> str:
    """The review exactly as generators saw it (minus the anon id)."""
    return f"{r['title']}. {r['text']}" if r.get("title") else r["text"]


def _validate(items: list[dict], source: str) -> list[dict]:
    out = []
    for it in items:
        loc = locate_span(source, it.get("source_span", ""))
        reason = None
        if loc is None:
            reason = "span_not_in_source"
        elif not (it.get("proposition") or "").strip():
            reason = "empty_proposition"
        out.append({**it, "span_start": loc[0] if loc else None, "span_end": loc[1] if loc else None,
                    "valid": reason is None, "invalid_reason": reason})
    return out


def _parallel(jobs, fn, workers: int, desc: str, sink, cfg: Config, step: str) -> int:
    """Run fn(job) concurrently (Ollama serves parallel requests); write results in the main thread."""
    n = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(fn, j): j for j in jobs}
        for f in tqdm(as_completed(futs), total=len(futs), desc=f"{desc} x{workers}"):
            try:
                sink.write(f.result())
                n += 1
            except Exception as e:
                log_failure(cfg, step, str(futs[f][0]), repr(e))
    return n


def _client(cfg: Config) -> tuple[Ollama, dict, dict]:
    ex = cfg["extraction"]
    client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
    prov = client.provenance(ex["model"])
    if not prov["digest"]:
        raise RuntimeError(f"extractor {ex['model']} not installed")
    return client, ex, prov


def extract_claims(cfg: Config, limit: int | None = None, variant: str | None = "original") -> int:
    started = dt.datetime.now()
    client, ex, prov = _client(cfg)
    ph = prompts.prompt_hash("extract_claims.txt")
    sums = read_jsonl(cfg.out_dir / "summaries.jsonl")
    if variant:  # controlled-bundle summaries are evaluated by target presence, not claims
        sums = [x for x in sums if x.get("variant", "original") == variant]
    sums = sums[: limit or None]
    def run(job):
        key, s = job
        obj, meta = client.chat_json(ex["model"], prompts.render("extract_claims.txt", summary=s["text"]),
                                     CLAIM_SCHEMA, temperature=ex["temperature"],
                                     num_ctx=ex["num_ctx"], num_predict=ex["num_predict"])
        return {"key": key, "summary_id": s["summary_id"], "extractor": prov["model"],
                "extractor_digest": prov["digest"], "prompt_hash": ph,
                "items": _validate(obj.get("claims", []), s["text"]), "latency_s": meta["wall_s"]}

    with JsonlAppender(cfg.out_dir / "claims_raw.jsonl", "key") as out:
        jobs = [(k, s) for s in sums if (k := short_hash(s["summary_id"], prov["digest"], ph)) not in out]
        n = _parallel(jobs, run, int(ex.get("concurrency", 1)), "extract claims", out, cfg, "extract-claims")
    rows = []
    for r in read_jsonl(cfg.out_dir / "claims_raw.jsonl"):
        for i, it in enumerate(r["items"]):
            rows.append({"claim_id": f"{r['summary_id']}_c{i:02d}", "summary_id": r["summary_id"],
                         "extractor_digest": r["extractor_digest"], **it})
    df = pd.DataFrame(rows)
    write_parquet(df, cfg.out_dir / "summary_claims.parquet")
    log_step(cfg, "extract-claims", {"provenance": prov, "prompt_hash": ph},
             {"new": n, "claims": len(df),
              "invalid": int((~df["valid"]).sum()) if len(df) else 0}, started)
    return n


def extract_experiences(cfg: Config, cohorts=None, splits=None, limit: int | None = None,
                        bundle_file: str | None = None) -> int:
    started = dt.datetime.now()
    client, ex, prov = _client(cfg)
    ph = prompts.prompt_hash("extract_experiences.txt")
    blist = data.bundles(cfg, bundle_file, cohorts, splits)[: limit or None]
    todo = {}
    for b in blist:
        for r in data.bundle_reviews(cfg, b):
            todo.setdefault(r["review_id"], r)
    def run(job):
        key, rid, r = job
        src = review_string(r)
        obj, meta = client.chat_json(ex["model"], prompts.render("extract_experiences.txt", review=src),
                                     EXP_SCHEMA, temperature=ex["temperature"],
                                     num_ctx=ex["num_ctx"], num_predict=ex["num_predict"])
        return {"key": key, "review_id": rid, "extractor_digest": prov["digest"], "prompt_hash": ph,
                "items": _validate(obj.get("experiences", []), src), "latency_s": meta["wall_s"]}

    with JsonlAppender(cfg.out_dir / "experiences_raw.jsonl", "key") as out:
        jobs = [(k, rid, r) for rid, r in todo.items() if (k := short_hash(rid, prov["digest"], ph)) not in out]
        n = _parallel(jobs, run, int(ex.get("concurrency", 1)), "extract experiences", out, cfg,
                      "extract-experiences")
    rows = []
    for r in read_jsonl(cfg.out_dir / "experiences_raw.jsonl"):
        for i, it in enumerate(r["items"]):
            rows.append({"exp_id": f"{r['review_id']}_e{i:02d}", "review_id": r["review_id"], **it})
    df = pd.DataFrame(rows)
    write_parquet(df, cfg.out_dir / "review_experiences.parquet")
    log_step(cfg, "extract-experiences", {"provenance": prov, "prompt_hash": ph},
             {"reviews": len(todo), "new": n, "items": len(df)}, started)
    return n
