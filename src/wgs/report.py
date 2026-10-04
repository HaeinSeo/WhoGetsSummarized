"""Pilot report: observed facts vs automatic estimates vs pending human checks vs gates."""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd

from . import data
from .config import Config
from .storage import read_jsonl

GATES = {"SUPPORT_precision": 0.90, "SUPPORT_recall": 0.80, "recall_at_k": 0.95}


def _md(df, n: int = 40) -> str:
    if df is None or len(df) == 0:
        return "_(none)_\n"
    df = df.head(n)
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def _observed(cfg: Config) -> list[str]:
    out, o = ["## 1. Observed facts\n### Sampling\n"], cfg.out_dir
    rep = cfg.data_dir / "sampling_report.csv"
    out.append(_md(pd.read_csv(rep)) if rep.exists() else "_not run_\n")
    bl = read_jsonl(cfg.data_dir / "bundles.jsonl")
    if bl:
        b = pd.DataFrame(bl)
        out.append("\n### Bundles\n" + _md(b.groupby("cohort").agg(
            bundles=("bundle_id", "size"), reviews_mean=("n_reviews", "mean"),
            est_tokens_mean=("est_input_tokens", "mean"), est_tokens_max=("est_input_tokens", "max"),
            tokenizer_exact=("tokenizer_exact", "all")).reset_index()))
    sm = pd.DataFrame(read_jsonl(o / "summaries.jsonl"))
    if len(sm):
        g = sm.groupby(["model_id", "method", "variant"]).agg(
            n=("summary_id", "size"), words_mean=("word_count", "mean"), length_ok=("length_ok", "mean"),
            stop_rate=("finish_reason", lambda s: (s == "stop").mean()),
            prompt_tokens_max=("prompt_tokens", "max"), over_budget=("over_common_budget", "sum"),
            ctx_trunc=("possible_ctx_truncation", "sum"), think_blocks=("had_think_block", "sum"),
            latency_p50=("latency_s", "median"), latency_p95=("latency_s", lambda s: s.quantile(.95)),
            digest=("model_digest", "first")).reset_index()
        out.append("\n### Generation\n" + _md(g))
    gen = [m for m in read_jsonl(cfg.run_dir / "run_manifest.jsonl") if m["step"] == "generate"]
    if gen:
        out.append("\nOllama-reported loaded VRAM after generate steps (bytes): "
                   + json.dumps({m["params"]["model"]: m["counts"].get("ollama_vram_bytes") for m in gen}) + "\n")
    fails = read_jsonl(cfg.run_dir / "failures.jsonl")
    out.append(f"\n### Failures\n{len(fails)} logged failures (`failures.jsonl`).\n")
    if fails:
        out.append(_md(pd.DataFrame(fails).groupby("step").size().rename("n").reset_index()))
    return out


def _automatic(cfg: Config) -> list[str]:
    out, o = ["\n## 2. Automatic estimates (unvalidated)\n"], cfg.out_dir
    cl = data.read_parquet_or_empty(o / "summary_claims.parquet")
    if len(cl):
        out.append(f"- claims extracted: {len(cl)}; span-invalid rate: {(~cl.valid).mean():.3f}\n")
    for f in sorted(o.glob("metrics_by_summary_*.parquet")):
        m = pd.read_parquet(f)
        cols = [c for c in ["supported_claim_precision_auto", "RR_D", "RR_U", "RG_D_U", "Amp_D",
                            "F_D", "F_D_user", "n_unjudged_pairs"] if c in m]
        out.append(f"\n**{f.stem}** (group labels: {m.group_source.iloc[0]})\n")
        out.append(_md(m.groupby(["model_id", "method"])[cols].mean().reset_index()))
    return out


def _human(cfg: Config) -> list[str]:
    a, rows = cfg.annotations_dir, []
    for kind in ["disclosure", "pair", "inventory", "summary", "selection", "intervention",
                 "intervention_target"]:
        gold = a / ("experience_inventory.parquet" if kind == "inventory" else f"{kind}_gold.parquet")
        agr = a / kind / "_agreement.json"
        ag = json.loads(agr.read_text()) if agr.exists() else {}
        kappa = {k: v for k, v in ag.items() if k.endswith("kappa")}
        rows.append({"kind": kind, "exported": (a / kind).exists(), "gold_ready": gold.exists(),
                     "gold_items": ag.get("gold_items"), "kappa": json.dumps(kappa) if kappa else ""})
    return ["\n## 3. Human annotation status\n", _md(pd.DataFrame(rows))]


def _gates(cfg: Config) -> list[str]:
    out, o = ["\n## 4. Gates for the main experiment\n"], cfg.out_dir
    je = o / "judge_eval.csv"
    if je.exists():
        g = pd.read_csv(je)
        j = g[g.slice == "all_weighted"].copy()
        j["gate_pass"] = ((j.SUPPORT_precision >= GATES["SUPPORT_precision"])
                          & (j.SUPPORT_recall >= GATES["SUPPORT_recall"]))
        out.append(_md(j[["judge", "n", "macro_f1", "SUPPORT_precision", "SUPPORT_recall",
                          "OPPOSE_precision", "OPPOSE_recall", "latency_p50_s", "gate_pass"]]))
        grp = g[g.slice.str.startswith("group=")][["judge", "slice", "n", "SUPPORT_precision", "SUPPORT_recall"]]
        out.append("\nPer-group judge error (D vs nonD):\n" + _md(grp))
    else:
        out.append("- Judge gate: **pending** (pair gold + `wgs eval-judges`).\n")
    rr = o / "retrieval_recall.csv"
    if rr.exists():
        r = pd.read_csv(rr)
        r["gate_pass"] = r.recall_at_k >= GATES["recall_at_k"]
        out.append("\nRetrieval recall@K (exhaustive gold subset):\n" + _md(r))
    else:
        out.append("- Retrieval recall@K gate: **pending** (exhaustive pair gold).\n")
    dc = cfg.data_dir / "product_disclosure_summary.csv"
    out.append("\nD/U co-existence (rule candidates, full-category scan):\n"
               + (_md(pd.read_csv(dc)) if dc.exists() else "_scan-disclosure not run_\n"))
    out.append("\n## 5. Stop / pivot rules (design 17.3)\n"
               "- Few D/U products: incentive analysis becomes secondary.\n"
               "- Humans find few concrete omissions: do not claim positivity bias.\n"
               "- AEP not better than frequency planner: report the simpler method / audit framing.\n"
               "- Group-wise judge error unverifiable: shrink amplification claims.\n")
    return out


def build(cfg: Config) -> str:
    head = [f"# Pilot report: {cfg['run_name']}\n",
            f"Generated {dt.datetime.now():%Y-%m-%d %H:%M} | config hash `{cfg.config_hash()}` | "
            f"design v{cfg.get('design_version')}\n",
            "> Only section 1 is directly observed. Section 2 is automatic and NOT validated against "
            "humans. Draw no performance or bias conclusion before the gates in section 4 pass.\n"]
    text = "\n".join(head + _observed(cfg) + _automatic(cfg) + _human(cfg) + _gates(cfg))
    path = cfg.run_dir / "pilot_report.md"
    path.write_text(text, encoding="utf-8")
    return str(path)
