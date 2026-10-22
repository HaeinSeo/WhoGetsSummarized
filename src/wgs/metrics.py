"""Representation metrics on the evidence graph + human-gold preservation metrics.

All ratios with a zero denominator are NA (never 0). Edges are post-hoc semantic support,
not generation provenance.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd

from . import data
from .config import Config
from .storage import log_step, read_jsonl, write_parquet

NA = np.nan


def _ratio(a, b):
    return a / b if b else NA


def representation(group_of: dict[str, str], user_of: dict[str, str], support: dict[str, set[str]],
                   groups=("D", "U")) -> dict:
    """group_of: review_id -> group for ALL input reviews of the bundle.
    support: claim_id -> set of review_ids with a SUPPORT edge.
    Returns RR_g, Exposure_g, Share_g, Amp_g, F_g (review & user-dedup), RG_D_U."""
    N = len(group_of)
    linked = set().union(*support.values()) if support else set()
    out = {"n_reviews": N, "n_linked": len(linked)}
    for g in groups:
        members = [r for r, gg in group_of.items() if gg == g]
        n_g, z_g = len(members), sum(r in linked for r in members)
        out[f"N_{g}"] = n_g
        out[f"RR_{g}"] = _ratio(z_g, n_g)
        out[f"Exposure_{g}"] = _ratio(n_g, N)
        out[f"Share_{g}"] = _ratio(z_g, len(linked))
        out[f"Amp_{g}"] = _ratio(out[f"Share_{g}"], out[f"Exposure_{g}"]) \
            if not np.isnan(out[f"Share_{g}"]) and out[f"Exposure_{g}"] else NA
    if len(groups) >= 2:
        a, b = groups[0], groups[1]
        out[f"RG_{a}_{b}"] = out[f"RR_{a}"] - out[f"RR_{b}"]
    # claim-level fractional evidence share (mass 1 per supported claim)
    for dedup in (False, True):
        fs = defaultdict(float)
        supported = [c for c, s in support.items() if s]
        for c in supported:
            units = {}
            for r in support[c]:
                units[user_of[r] if dedup else r] = group_of[r]
            for g in groups:
                fs[g] += sum(1 for v in units.values() if v == g) / len(units)
        sfx = "_user" if dedup else ""
        for g in groups:
            out[f"F_{g}{sfx}"] = _ratio(fs[g], len(supported))
    return out


def summary_metrics(cfg: Config, judge: str, scope: str = "topk:10",
                    group_col: str = "rule_label") -> pd.DataFrame:
    """group_col: 'rule_label' (rule candidates) or 'gold_label' (human disclosure gold, U if unlabeled
    is NOT assumed: unlabeled reviews get group 'unlabeled')."""
    from .evidence import edges_path
    started = dt.datetime.now()
    edges = pd.read_parquet(edges_path(cfg, judge, "claim", scope))
    claims = pd.read_parquet(cfg.out_dir / "summary_claims.parquet")
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    rv = data.reviews(cfg)
    edges = edges.merge(claims[["claim_id", "summary_id"]], left_on="hyp_id", right_on="claim_id")
    if group_col == "gold_label":
        g = pd.read_parquet(cfg.annotations_dir / "disclosure_gold.parquet")
        groups = dict(zip(g.review_id, g.label))
    else:
        groups = rv[group_col].to_dict()
    rows = []
    e_by = dict(tuple(edges.groupby("summary_id")))
    c_by = dict(tuple(claims.groupby("summary_id")))
    for s in sums.itertuples():
        b = bundles.get(s.bundle_id)
        e = e_by.get(s.summary_id)
        if b is None or e is None or e.empty:  # not judged in this edge file -> skip, never 0
            continue
        rids = b["review_ids_ordered"]
        group_of = {r: groups.get(r, "unlabeled") for r in rids}
        user_of = {r: rv.at[r, "user_id"] for r in rids}
        c_all = c_by.get(s.summary_id, claims.iloc[0:0])
        valid = c_all[c_all["valid"]]
        sup_e = e[e.relation == "SUPPORT"].groupby("hyp_id")["review_id"].apply(set).to_dict()
        judged_ids = set(e.loc[e.judged, "hyp_id"])
        support = {c: sup_e.get(c, set()) for c in valid.claim_id}
        judged_claims = [c for c in valid.claim_id if c in judged_ids]
        rep = representation(group_of, user_of, support)
        rows.append({
            "summary_id": s.summary_id, "bundle_id": s.bundle_id, "product_id": s.product_id,
            "cohort": s.cohort, "split": s.split, "variant": s.variant, "method": s.method,
            "model_id": s.model_id, "word_count": s.word_count, "length_ok": s.length_ok,
            "n_claims": len(c_all), "n_valid_claims": len(valid),
            "claim_extraction_invalid_rate": _ratio(len(c_all) - len(valid), len(c_all)),
            "supported_claim_precision_auto": _ratio(sum(bool(support[c]) for c in judged_claims),
                                                     len(judged_claims)),
            "judge": judge, "scope": scope, "group_source": group_col,
            "n_unjudged_pairs": int((~e.judged).sum()), **rep,
        })
    df = pd.DataFrame(rows)
    write_parquet(df, cfg.out_dir / f"metrics_by_summary_{judge}_{scope.replace(':', '')}_{group_col}.parquet")
    log_step(cfg, "metrics", {"judge": judge, "scope": scope, "group_col": group_col}, {"summaries": len(df)}, started)
    return df


def importance_recall(inventory: pd.DataFrame, summary_gold: pd.DataFrame,
                      min_importance: int = 2) -> pd.DataFrame:
    """Recall_imp per (summary): share of gold important experiences faithfully expressed.

    inventory: experience_id, product_id, importance, polarity, minority(bool), conditional(bool)
    summary_gold: summary_id, product_id, experience_id, judgment in
      {faithful, condition_lost, overstated, absent, contradicted}. Only 'faithful' counts.
    Products with no important experience -> NA (excluded, not 0).
    """
    imp = inventory[inventory.importance >= min_importance]
    g = summary_gold.merge(imp[["experience_id", "polarity", "minority", "conditional"]],
                           on="experience_id")
    g["hit"] = (g["judgment"] == "faithful").astype(float)
    out = g.groupby("summary_id").agg(recall_imp=("hit", "mean"), n_important=("hit", "size")).reset_index()
    for col, val, name in [("polarity", "negative", "recall_imp_negative"),
                           ("polarity", "positive", "recall_imp_positive"),
                           ("minority", True, "recall_imp_minority"),
                           ("conditional", True, "recall_imp_conditional")]:
        sub = g[g[col] == val].groupby("summary_id")["hit"].mean().rename(name)
        out = out.merge(sub, on="summary_id", how="left")
    return out


def paired_bootstrap(df: pd.DataFrame, metric: str, method_a: str, method_b: str,
                     unit: str = "product_id", reps: int = 2000, seed: int = 0) -> dict:
    """Paired difference (B - A) of a per-summary metric, resampling products."""
    piv = df[df.method.isin([method_a, method_b])].pivot_table(
        index=[unit, "model_id"], columns="method", values=metric, aggfunc="mean").dropna()
    if piv.empty:
        return {"n_units": 0, "diff": NA, "ci_low": NA, "ci_high": NA}
    diffs = (piv[method_b] - piv[method_a]).groupby(level=0).mean()
    rng = np.random.default_rng(seed)
    vals = diffs.to_numpy()
    boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(reps)]
    return {"metric": metric, "a": method_a, "b": method_b, "n_units": len(vals),
            "diff": float(vals.mean()), "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5))}
