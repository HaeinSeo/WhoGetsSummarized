"""Evidence agent v2: draft -> verify -> augment -> rewrite.

Starts from the generator's own direct (B0) summary so coverage is not lost, removes claims with no
supporting review (all-pair NLI edges), and adds up to K verified, specific experiences (agent planner
ranking) that the draft does not express. One constrained rewrite. Human annotations are never used;
K is chosen on the dev split only (see choose_k).
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import data, prompts
from .config import Config
from .evidence import edges_path
from .generate import freq_words
from .judges import make_judge
from .llm import Ollama
from .storage import JsonlAppender, log_failure, log_step, read_jsonl, short_hash, text_hash, write_parquet
from .text import word_count

log = logging.getLogger(__name__)
METHOD = "agent2"


def prepare(cfg: Config, model: str, judge: str, k: int, splits, cohorts=None) -> pd.DataFrame:
    """Per draft: unsupported claim spans and missing verified experiences (no LLM calls)."""
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    d = sums[(sums.method == "direct") & (sums.model_id == model) & (sums.variant == "original")
             & sums.split.isin(splits)]
    if cohorts:
        d = d[d.cohort.isin(cohorts)]
    plans = pd.read_parquet(cfg.out_dir / "plans.parquet")
    plans = plans[(plans.planner == "agent")]
    d = d[d.bundle_id.isin(set(plans.bundle_id))]
    cl = pd.read_parquet(cfg.out_dir / "summary_claims.parquet")
    cl = cl[cl.valid & cl.summary_id.isin(set(d.summary_id))]
    e = pd.read_parquet(edges_path(cfg, judge, "claim", "all"), columns=["hyp_id", "judged", "relation"])
    e = e[e.hyp_id.isin(set(cl.claim_id)) & e.judged]
    supported = set(e[e.relation == "SUPPORT"].hyp_id)
    judged = set(e.hyp_id)
    jd = make_judge(cfg, judge)
    rows = []
    for s in tqdm(d.itertuples(), total=len(d), desc="agent2 prepare"):
        c = cl[cl.summary_id == s.summary_id]
        uns = [x.source_span for x in c.itertuples() if x.claim_id in judged and x.claim_id not in supported]
        p = plans[plans.bundle_id == s.bundle_id]
        p = p[p.support_users >= 1].sort_values("score", ascending=False)
        p = p[p.aspect != "overall"].head(max(k * 3, k))
        miss = []
        if len(p) and k > 0:
            res = jd.judge([{"experience": r.proposition, "condition": r.condition or "", "review": s.text}
                            for r in p.itertuples()])
            for r, lab in zip(p.itertuples(), res):
                if lab["label"] != "SUPPORT" and len(miss) < k:
                    miss.append(r)
        rows.append({"summary_id": s.summary_id, "bundle_id": s.bundle_id, "unsupported": uns,
                     "missing": [{"proposition": m.proposition, "condition": m.condition,
                                  "support_users": int(m.support_users), "quote": m.quote} for m in miss]})
    return pd.DataFrame(rows)


def _render_missing(miss: list[dict], n_total: int) -> str:
    out = []
    for m in miss:
        cond = f" [condition: {m['condition']}]" if m.get("condition") else ""
        q = f" Example: \"{m['quote']}\"" if m.get("quote") else ""
        out.append(f"- {m['proposition']}{cond} - reported by {freq_words(m['support_users'], n_total)}.{q}")
    return chr(10).join(out) if out else "(none)"


def run(cfg: Config, model: str, judge: str, k: int, splits, cohorts=None) -> int:
    started = dt.datetime.now()
    g = cfg["generation"]
    client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
    prov = client.provenance(model)
    prep = prepare(cfg, model, judge, k, splits, cohorts)
    tag = model.replace(":", "_")
    write_parquet(prep.assign(n_unsupported=prep.unsupported.map(len), n_missing=prep.missing.map(len))
                  .drop(columns=["unsupported", "missing"]), cfg.out_dir / f"agent2_prep_{tag}_k{k}.parquet")
    drafts = {s["summary_id"]: s for s in read_jsonl(cfg.out_dir / "summaries.jsonl")}
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    ph = prompts.prompt_hash("agent_augment.txt")
    method = f"agent2_k{k}"
    lo, hi = g["target_words"]
    n = 0
    with JsonlAppender(cfg.out_dir / "summaries.jsonl", "summary_id") as out:
        for r in tqdm(prep.itertuples(), total=len(prep), desc=f"agent2 {model} k={k}"):
            d = drafts[r.summary_id]
            sid = "s" + short_hash(r.summary_id, method, ph, judge, n=15)
            if sid in out:
                continue
            b = bundles[r.bundle_id]
            text, rewritten = d["text"], False
            if r.unsupported or r.missing:
                prompt = prompts.render("agent_augment.txt", category=b["category"].replace("_", " "),
                                        n_reviewers=b["n_reviews"], summary=d["text"],
                                        unsupported=chr(10).join(f"- {u}" for u in r.unsupported) or "(none)",
                                        missing=_render_missing(r.missing, b["n_reviews"]))
                try:
                    res = client.chat(model, prompt, temperature=g["temperature"], seed=g["seed"],
                                      num_ctx=g["num_ctx"], num_predict=g["num_predict"], thinking=g["thinking"])
                    text, rewritten = res["text"], True
                except Exception as e:
                    log_failure(cfg, "agent2", r.summary_id, repr(e))
                    continue
            wc = word_count(text)
            out.write({**{x: d[x] for x in ("bundle_id", "product_id", "cohort", "split", "variant")},
                       "summary_id": sid, "method": method, "model_id": model, "model_digest": prov["digest"],
                       "quantization": prov["quantization"], "prompt_hash": ph, "draft_summary_id": r.summary_id,
                       "decoding": d.get("decoding"), "text": text, "word_count": wc, "length_ok": lo <= wc <= hi,
                       "finish_reason": "stop", "prompt_tokens": None, "output_tokens": None,
                       "possible_ctx_truncation": False, "over_common_budget": False, "had_think_block": False,
                       "agent_k": k, "n_unsupported_removed": len(r.unsupported), "n_missing_added": len(r.missing),
                       "rewritten": rewritten, "text_hash": text_hash(text), "latency_s": None,
                       "created": dt.datetime.now().isoformat(timespec="seconds")})
            n += 1
    log_step(cfg, "agent2", {"model": model, "judge": judge, "k": k, "splits": splits}, {"new": n}, started)
    return n


def choose_k(cfg: Config, judge: str, model: str, ks=(0, 2, 4, 6)) -> dict:
    """DEV ONLY. Largest K whose agent summaries keep supported-claim precision >= the B0 draft's and
    mean length <= the upper word target. Recall is NOT used (it would be circular on dev)."""
    import json
    m = pd.read_parquet(cfg.out_dir / f"metrics_by_summary_{judge}_all_rule_label.parquet")
    m = m[(m.split == "dev") & (m.model_id == model)]
    base = float(m[m.method == "direct"].supported_claim_precision_auto.mean())
    hi = cfg["generation"]["target_words"][1]
    rows = []
    for k in ks:
        x = m[m.method == f"agent2_k{k}"]
        if len(x):
            rows.append({"k": k, "n": len(x), "precision": float(x.supported_claim_precision_auto.mean()),
                         "words": float(x.word_count.mean())})
    ok = [r for r in rows if r["precision"] >= base and r["words"] <= hi]
    pick = max(ok, key=lambda r: r["k"])["k"] if ok else None
    rep = {"model": model, "b0_precision": base, "candidates": rows, "chosen_k": pick,
           "rule": "largest K with precision >= B0 and mean words <= target max (dev split only)"}
    (cfg.out_dir / f"agent2_choose_k_{model.replace(':', '_')}.json").write_text(json.dumps(rep, indent=2))
    return rep
