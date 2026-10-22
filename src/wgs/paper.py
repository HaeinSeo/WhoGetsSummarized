"""Paper artifacts generated ONLY from result files: LaTeX tables, number macros, evidence status.

Nothing is typed by hand: a missing result renders as @missing{...} in red and is listed in
paper/generated/status.md, so the draft can be written now and filled as results arrive.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import REPO_ROOT, Config
from .storage import read_jsonl

OUT = REPO_ROOT / "paper" / "generated"
_BS = chr(92)


def T(text: str) -> str:
    """LaTeX is written with @ for a backslash (@@@@ = row end) to keep this source escape-free."""
    return text.replace("@@@@", _BS * 2).replace("@", _BS)
METHOD_TEX = {"direct": "B0 Direct", "coverage": "B1 Coverage", "freqframe": "B2 Freq.@ prompt",
              "plan:frequency": "B3 Freq.@ plan", "plan:frequency_unverified": "B3 w/o verif.", "plan:agent": "Agent w/o check", "agent": "Plan agent (v1)"}
ORDER = list(METHOD_TEX)


def _f(x, d=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def _ci(r, d=2):
    return f"{_f(r['diff'], d)} [{_f(r['ci_low'], d)}, {_f(r['ci_high'], d)}]"


def _int(n):
    return f"{int(n):,}".replace(",", "{,}")


def _read(p: Path):
    if not p.exists():
        return None
    if p.suffix == ".csv":
        return pd.read_csv(p)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    return pd.read_parquet(p)


class Macros:
    def __init__(self):
        self.lines, self.missing = [], []

    def add(self, name, value, src):
        if value is None or value == "--":
            self.lines.append(f"@newcommand{{@{name}}}{{@missing{{{name}}}}}")
            self.missing.append((name, src))
        else:
            self.lines.append(f"@newcommand{{@{name}}}{{{value}}}  % {src}")


def _scale_and_gold(cfg: Config, M: Macros) -> pd.DataFrame:
    o, a = cfg.out_dir, cfg.annotations_dir
    sm = pd.DataFrame(read_jsonl(o / "summaries.jsonl")) if (o / "summaries.jsonl").exists() else pd.DataFrame()
    b = pd.DataFrame(read_jsonl(cfg.data_dir / "bundles.jsonl"))
    M.add("nTestProducts", _int((b.split == "test").sum()), "bundles.jsonl")
    M.add("nDevProducts", str(int((b.split == "dev").sum())), "bundles.jsonl")
    M.add("reviewsPerBundle", _f(b.n_reviews.mean(), 1), "bundles.jsonl")
    M.add("nSummaries", _int(len(sm)) if len(sm) else None, "summaries.jsonl")
    inv = _read(a / "inventory" / "_agreement.json")
    M.add("invOverlap", _f(100 * inv["overlap_share"], 1) if inv else None, "inventory/_agreement.json")
    M.add("invImpKappa", _f(inv["importance_weighted_kappa"]) if inv else None, "inventory/_agreement.json")
    M.add("invPolKappa", _f(inv["polarity_kappa"]) if inv else None, "inventory/_agreement.json")
    ei = _read(a / "experience_inventory.parquet")
    M.add("nGoldExperiences", _int(len(ei)) if ei is not None else None, "experience_inventory")
    M.add("nGoldAgreed", _int(ei.human_agreed.sum()) if ei is not None else None, "experience_inventory")
    M.add("nGoldProducts", str(ei.bundle_id.nunique()) if ei is not None else None, "experience_inventory")
    for k, name, col in (("pair", "Pair", "relation"), ("disclosure", "Disc", "label")):
        ag = _read(a / k / "_agreement.json") or {}
        M.add(f"kappa{name}", _f(ag.get(f"{col}_cohen_kappa")) if ag else None, f"{k}/_agreement.json")
        M.add(f"n{name}Gold", str(ag["gold_items"]) if "gold_items" in ag else None, f"{k}/_agreement.json")
    return sm


def _measurement(cfg: Config, M: Macros) -> None:
    o = cfg.out_dir
    je = _read(o / "judge_eval.csv")
    rows = []
    if je is not None:
        for r in je[je.slice == "all_weighted"].itertuples():
            rows.append(f"{r.judge} & {_f(r.macro_f1)} & {_f(r.SUPPORT_precision)} & {_f(r.SUPPORT_recall)} & "
                        f"{_f(r.OPPOSE_precision)} & {_f(r.OPPOSE_recall)} & {_f(r.pairs_per_s, 1)} @@@@")
            M.add(f"supP{r.judge.capitalize()}", _f(r.SUPPORT_precision), "judge_eval.csv")
            M.add(f"supR{r.judge.capitalize()}", _f(r.SUPPORT_recall), "judge_eval.csv")
    rr = _read(o / "retrieval_recall.csv")
    for k in (10, 20):
        v = rr[rr.retriever.str.contains("bge") & (rr.relation == "SUPPORT") & (rr.K == k)].recall_at_k if rr is not None else []
        M.add("retRecall" + {10: "Ten", 20: "Twenty"}[k], _f(float(v.mean())) if len(v) else None, "retrieval_recall.csv")
    da = _read(o / "disclosure_rule_accuracy.csv")
    d = da[(da.stratum == "all") & (da.label == "D")] if da is not None else []
    M.add("ruleDPrec", _f(d.precision.iloc[0]) if len(d) else None, "disclosure_rule_accuracy.csv")
    M.add("ruleDRec", _f(d.recall.iloc[0]) if len(d) else None, "disclosure_rule_accuracy.csv")
    t = ["@begin{tabular}{lrrrrrr}", "@toprule",
         "Judge & macro-F1 & SUP P & SUP R & OPP P & OPP R & pairs/s @@@@", "@midrule"]
    t += rows or ["@multicolumn{7}{c}{@missing{judge@_eval}} @@@@"]
    (OUT / "tab_measurement.tex").write_text(T(chr(10).join(t + ["@bottomrule", "@end{tabular}"])), encoding="utf-8")


def _main_table(cfg: Config, sm: pd.DataFrame, judge: str, scope: str) -> None:
    o = cfg.out_dir
    gr = _read(o / f"gold_recall_table_{judge}.csv")
    gb = _read(o / f"gold_recall_paired_bootstrap_{judge}.csv")
    cs = _read(o / f"corrected_support_rates_{judge}.csv")
    keys = []
    if len(sm):
        t = sm[(sm.split == "test") & (sm.variant == "original")]
        keys = sorted({(m, me) for m, me in zip(t.model_id, t.method)},
                      key=lambda k: (k[0], ORDER.index(k[1]) if k[1] in ORDER else 99))
    rows = ["@begin{tabular}{llrrrrr}", "@toprule",
            "Gen. & Method & Words & Rec.$_{gold}$ & Rec.$_{agreed}$ & $@Delta$ vs B0 [95@% CI] & Supp.$_{corr}$ @@@@",
            "@midrule"]
    for model, meth in keys:
        words = sm[(sm.model_id == model) & (sm.method == meth) & (sm.split == "test")].word_count.mean()
        g = gr[(gr.model_id == model) & (gr.method == meth)] if gr is not None else pd.DataFrame()
        rec = g.recall_gold.iloc[0] if len(g) else None
        agr = g.recall_gold_human_agreed.iloc[0] if len(g) and "recall_gold_human_agreed" in g else None
        dl = "--"
        if gb is not None and meth != "direct":
            x = gb[(gb.b == meth) & (gb.metric == "recall_gold_human_agreed")]
            dl = _ci(x.iloc[0]) if len(x) else "--"
        c = cs[(cs.model_id == model) & (cs.method == meth) & (cs.group == "all")] if cs is not None else pd.DataFrame()
        sup = _f(c.corrected_support_rate.iloc[0]) if len(c) else "--"
        rows.append(f"{model} & {("Agent v2 (ours)" if meth.startswith("agent2") else METHOD_TEX.get(meth, meth))} & {_f(words, 0)} & {_f(rec)} & {_f(agr)} & {dl} & {sup} @@@@")
    if not keys:
        rows.append("@multicolumn{7}{c}{@missing{summaries}} @@@@")
    (OUT / "tab_main.tex").write_text(T(chr(10).join(rows + ["@bottomrule", "@end{tabular}"])), encoding="utf-8")


def _interventions(cfg: Config, judge: str, M: Macros) -> None:
    o = cfg.out_dir
    iv = _read(o / f"intervention_metrics_auto_{judge}.csv")
    t = ["@begin{tabular}{llrrr}", "@toprule",
         "Gen. & Method & Joint cov.@ $@uparrow$ & Inconsist.@ $@downarrow$ & Removal err.@ $@downarrow$ @@@@", "@midrule"]
    if iv is not None:
        def wavg(d, c):
            return np.average(d[c], weights=d.n_pairs) if len(d) and c in d and d[c].notna().any() else np.nan
        for (model, meth), x in iv.groupby(["model_id", "method"]):
            pres, rem = x[x.type == "target_preserved"], x[x.type == "target_removed"]
            t.append(f"{model} & {("Agent v2 (ours)" if meth.startswith("agent2") else METHOD_TEX.get(meth, meth))} & {_f(wavg(pres, 'joint_coverage'))} & "
                     f"{_f(wavg(pres, 'inclusion_inconsistency'))} & {_f(wavg(rem, 'removal_error'))} @@@@")
    else:
        t.append("@multicolumn{5}{c}{@missing{intervention@_metrics}} @@@@")
    (OUT / "tab_interventions.tex").write_text(T(chr(10).join(t + ["@bottomrule", "@end{tabular}"])), encoding="utf-8")
    pv = _read(o / f"presence_validation_{judge}.json")
    M.add("presenceAcc", _f(pv["accuracy"]) if pv else None, f"presence_validation_{judge}.json")
    M.add("presenceKappa", _f(pv["kappa"]) if pv and pv.get("kappa") is not None else None,
          f"presence_validation_{judge}.json")


def _status(M: Macros, cfg: Config, judge: str) -> None:
    o, a = cfg.out_dir, cfg.annotations_dir
    need = {
        "C1 two-directional audit (human inventory recall + claim grounding)":
            [a / "experience_inventory.parquet", a / "inventory" / "_agreement.json",
             o / f"gold_recall_table_{judge}.csv", o / f"presence_validation_{judge}.json"],
        "C2 measurement validity (judge over-attribution, retrieval misses, human-calibrated rates)":
            [o / "judge_eval.csv", o / "retrieval_recall.csv", o / f"corrected_support_rates_{judge}.csv",
             a / "pair" / "_agreement.json"],
        "C3 controlled real-review interventions + verified frequency plan":
            [o / f"intervention_metrics_auto_{judge}.csv", o / f"gold_recall_paired_bootstrap_{judge}.csv"],
        "Secondary: disclosure (D/U) representation":
            [o / "disclosure_rule_accuracy.csv", o / f"corrected_support_rates_{judge}.csv"],
    }
    lines = ["# Evidence status per contribution (generated by `wgs paper`)", ""]
    for c, files in need.items():
        miss = [f.name for f in files if not f.exists()]
        lines.append(f"- {'READY  ' if not miss else 'MISSING'} {c}" + (f"  <- {', '.join(miss)}" if miss else ""))
    lines += ["", "## Macros still without a value", ""] + [f"- @{n}  <- {s}" for n, s in M.missing]
    (OUT / "status.md").write_text(T(chr(10).join(lines)), encoding="utf-8")


def build(cfg: Config, judge: str = "nli", scope: str = "all") -> list[str]:
    OUT.mkdir(parents=True, exist_ok=True)
    M = Macros()
    sm = _scale_and_gold(cfg, M)
    _measurement(cfg, M)
    _main_table(cfg, sm, judge, scope)
    _interventions(cfg, judge, M)
    (OUT / "numbers.tex").write_text(T("% generated by `wgs paper` - do not edit" + chr(10) + chr(10).join(M.lines)), encoding="utf-8")
    _status(M, cfg, judge)
    return [str(p) for p in sorted(OUT.glob("*"))]
