"""Evidence agent: plan (verified, specific, diverse experiences) -> write -> self-verify -> revise.

Draft = the `plan:agent` summary. Each draft is split into atomic claims (same extractor as the
evaluation), every claim is checked against the bundle's reviews with the in-process NLI judge, and
claims with no supporting review trigger one constrained rewrite. The result is stored as method
`agent`. The human inventory is never used.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from tqdm import tqdm

from . import data, prompts
from .config import Config
from .extract import CLAIM_SCHEMA, review_string
from .generate import render_plan
from .judges import make_judge
from .llm import Ollama
from .storage import JsonlAppender, log_failure, log_step, read_jsonl, short_hash, text_hash
from .text import word_count

log = logging.getLogger(__name__)


def run(cfg: Config, model: str, judge_name: str = "nli", cohorts=None, max_rounds: int = 1) -> int:
    started = dt.datetime.now()
    g, ex = cfg["generation"], cfg["extraction"]
    client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
    prov = client.provenance(model)
    judge = make_judge(cfg, judge_name)
    plans = pd.read_parquet(cfg.out_dir / "plans.parquet")
    plans = plans[(plans.planner == "agent") & plans.selected]
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    drafts = [s for s in read_jsonl(cfg.out_dir / "summaries.jsonl")
              if s["method"] == "plan:agent" and s["model_id"] == model
              and (not cohorts or s["cohort"] in cohorts)]
    lo, hi = g["target_words"]
    ph = prompts.prompt_hash("agent_revise.txt")
    n = 0
    with JsonlAppender(cfg.out_dir / "summaries.jsonl", "summary_id") as out:
        for d in tqdm(drafts, desc=f"agent verify+revise {model}"):
            b = bundles[d["bundle_id"]]
            sid = "s" + short_hash(d["summary_id"], "agent", judge.version, ph, max_rounds, n=15)
            if sid in out:
                continue
            try:
                revs = data.bundle_reviews(cfg, b)
                plan_txt = render_plan(plans[plans.bundle_id == b["bundle_id"]], len(revs))
                text, rounds, history = d["text"], 0, []
                for _ in range(max_rounds + 1):
                    obj, _m = client.chat_json(ex["model"], prompts.render("extract_claims.txt", summary=text),
                                               CLAIM_SCHEMA, temperature=0.0, num_ctx=ex["num_ctx"],
                                               num_predict=ex["num_predict"])
                    claims = [c for c in obj.get("claims", []) if c.get("claim_type") != "meta"]
                    items, owner = [], []
                    for ci, c in enumerate(claims):
                        for r in revs:
                            items.append({"experience": c["proposition"], "condition": c.get("condition", ""),
                                          "review": review_string(r)})
                            owner.append(ci)
                    res = judge.judge(items) if items else []
                    ok = {ci for (ci, r) in zip(owner, res) if r["label"] == "SUPPORT"}
                    unsupported = [c["source_span"] for ci, c in enumerate(claims) if ci not in ok]
                    history.append({"round": rounds, "n_claims": len(claims), "n_unsupported": len(unsupported)})
                    if not unsupported or rounds >= max_rounds:
                        break
                    prompt = prompts.render("agent_revise.txt", category=b["category"].replace("_", " "),
                                            n_reviewers=len(revs), plan=plan_txt, summary=text,
                                            unsupported="\n".join(f"- {u}" for u in unsupported))
                    r = client.chat(model, prompt, temperature=g["temperature"], seed=g["seed"],
                                    num_ctx=g["num_ctx"], num_predict=g["num_predict"], thinking=g["thinking"])
                    text, rounds = r["text"], rounds + 1
            except Exception as e:
                log_failure(cfg, "agent", d["summary_id"], repr(e))
                continue
            wc = word_count(text)
            out.write({**{k: d[k] for k in ("bundle_id", "product_id", "cohort", "split", "variant")},
                       "summary_id": sid, "method": "agent", "model_id": model, "model_digest": prov["digest"],
                       "quantization": prov["quantization"], "prompt_hash": ph, "draft_summary_id": d["summary_id"],
                       "decoding": d.get("decoding"), "text": text, "word_count": wc, "length_ok": lo <= wc <= hi,
                       "finish_reason": "stop", "prompt_tokens": None, "output_tokens": None,
                       "possible_ctx_truncation": False, "over_common_budget": False, "had_think_block": False,
                       "agent_rounds": rounds, "agent_history": history, "verifier": judge.version,
                       "text_hash": text_hash(text), "latency_s": None,
                       "created": dt.datetime.now().isoformat(timespec="seconds")})
            n += 1
    log_step(cfg, "agent", {"model": model, "judge": judge_name, "max_rounds": max_rounds}, {"new": n}, started)
    return n
