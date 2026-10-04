"""P1/P5: summary generation with a frozen generator (Ollama), resumable."""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from tqdm import tqdm

from . import data, prompts
from .config import Config
from .llm import Ollama
from .storage import JsonlAppender, log_failure, log_step, short_hash, text_hash
from .text import word_count

log = logging.getLogger(__name__)


def freq_table(cands: pd.DataFrame) -> str:
    """Aspect x polarity distinct-reviewer counts from AUTOMATIC candidates (B2)."""
    if cands.empty:
        return "(no aspects detected)"
    rows = []
    for (aspect, pol), g in cands.groupby(["aspect", "polarity"]):
        users = len({u for us in g["mention_users"] for u in us})
        rows.append((users, aspect, pol))
    rows.sort(reverse=True)
    return "\n".join(f"- {a} ({p}): {u}" for u, a, p in rows[:25])


def render_plan(plan: pd.DataFrame) -> str:
    lines = []
    for i, p in enumerate(plan.sort_values("rank").itertuples(), 1):
        cond = f" [condition: {p.condition}]" if p.condition else ""
        quote = f" Example: \"{p.quote}\"" if getattr(p, "quote", "") else ""
        lines.append(f"{i}. {p.proposition}{cond} - supported by {p.support_users} reviewer(s); "
                     f"opposite reported by {p.oppose_users}.{quote}")
    return "\n".join(lines) if lines else "(no verified experiences)"


def build_prompt(cfg: Config, bundle: dict, method: str, planner: str | None) -> tuple[str, str]:
    revs = data.bundle_reviews(cfg, bundle)
    kw = {"category": bundle["category"].replace("_", " "), "reviews": prompts.render_reviews(revs)}
    if method == "freqframe":
        cands = data.read_parquet_or_empty(cfg.out_dir / "input_candidates.parquet")
        if cands.empty:
            raise RuntimeError("freqframe needs input_candidates; run `wgs build-candidates`")
        kw.update(freq_table=freq_table(cands[cands.bundle_id == bundle["bundle_id"]]),
                  n_reviewers=len(revs))
    if method == "plan":
        plans = data.read_parquet_or_empty(cfg.out_dir / "plans.parquet")
        if plans.empty:
            raise RuntimeError("plan method needs plans.parquet; run `wgs plan`")
        sel = plans[(plans.bundle_id == bundle["bundle_id"]) & (plans.planner == planner)
                    & plans.selected]
        kw = {"category": kw["category"], "n_reviewers": len(revs), "plan": render_plan(sel)}
    name = prompts.SUMMARY_METHODS[method]
    return prompts.render(name, **kw), prompts.prompt_hash(name)


def generate(cfg: Config, model: str, method: str = "direct", planner: str | None = None,
             cohorts=None, splits=None, limit: int | None = None,
             bundle_file: str | None = None) -> int:
    started = dt.datetime.now()
    g = cfg["generation"]
    client = Ollama(cfg["ollama"]["base_url"], cfg["ollama"]["timeout_s"])
    prov = client.provenance(model)
    if not prov["digest"]:
        raise RuntimeError(f"Ollama model {model} is not installed (ollama pull {model})")
    method_name = f"plan:{planner}" if method == "plan" else method
    blist = data.bundles(cfg, bundle_file, cohorts, splits)[: limit or None]
    n_new = 0
    lo, hi = g["target_words"]
    with JsonlAppender(cfg.out_dir / "summaries.jsonl", "summary_id") as out:
        for b in tqdm(blist, desc=f"generate {model} {method_name}"):
            prompt, phash = build_prompt(cfg, b, method, planner)
            sid = "s" + short_hash(b["bundle_id"], model, prov["digest"], method_name, phash,
                                   g["temperature"], g["seed"], g["num_predict"], text_hash(prompt),
                                   g.get("length_retry", 0), prompts.prompt_hash("length_fix.txt"), n=15)
            if sid in out:
                continue
            try:
                r = client.chat(model, prompt, temperature=g["temperature"], seed=g["seed"],
                                num_ctx=g["num_ctx"], num_predict=g["num_predict"],
                                thinking=g["thinking"])
            except Exception as e:
                log_failure(cfg, "generate", f"{b['bundle_id']}|{model}|{method_name}", repr(e))
                continue
            wc = word_count(r["text"])
            first_wc, retried = wc, False
            if int(g.get("length_retry", 0)) and not (lo <= wc <= hi):
                # identical one-shot length correction for every method (recorded per row)
                fix = prompts.render("length_fix.txt", n=wc, source=prompt, summary=r["text"])
                try:
                    r2 = client.chat(model, fix, temperature=g["temperature"], seed=g["seed"],
                                     num_ctx=g["num_ctx"], num_predict=g["num_predict"],
                                     thinking=g["thinking"])
                    r2["wall_s"] += r["wall_s"]
                    r2["prompt_eval_count"] = r["prompt_eval_count"]
                    r, wc, retried = r2, word_count(r2["text"]), True
                except Exception as e:
                    log_failure(cfg, "generate-length-retry", sid, repr(e))
            out.write({
                "summary_id": sid, "bundle_id": b["bundle_id"], "product_id": b["product_id"],
                "cohort": b["cohort"], "split": b["split"], "variant": b.get("variant", "original"),
                "method": method_name, "model_id": model, "model_digest": prov["digest"],
                "quantization": prov["quantization"], "prompt_hash": phash,
                "decoding": {"temperature": g["temperature"], "top_k": 1, "seed": g["seed"],
                             "num_ctx": g["num_ctx"], "num_predict": g["num_predict"],
                             "thinking": g["thinking"]},
                "text": r["text"], "word_count": wc, "length_ok": lo <= wc <= hi,
                "first_pass_word_count": first_wc, "length_retry_used": retried,
                "finish_reason": r["done_reason"], "prompt_tokens": r["prompt_eval_count"],
                "output_tokens": r["eval_count"],
                "possible_ctx_truncation": (r["prompt_eval_count"] or 0) >= g["num_ctx"] - g["num_predict"],
                "over_common_budget": (r["prompt_eval_count"] or 0) > cfg["bundle"]["common_input_token_limit"],
                "had_think_block": r["had_think_block"], "latency_s": r["wall_s"],
                "created": dt.datetime.now().isoformat(timespec="seconds"),
            })
            n_new += 1
    log_step(cfg, "generate", {"model": model, "method": method_name, "provenance": prov,
                               "cohorts": cohorts, "splits": splits, "limit": limit,
                               "bundle_file": bundle_file},
             {"bundles": len(blist), "new_summaries": n_new,
              "ollama_vram_bytes": client.loaded_vram()}, started)
    return n_new
