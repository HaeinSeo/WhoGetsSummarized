"""Evaluation that requires human gold: judges, retrieval recall, preservation, interventions.

Every function refuses to report when the needed gold file is missing (no fabricated numbers).
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support
from tqdm import tqdm

from . import data
from .config import Config
from .judges import LABELS
from .storage import log_step, read_jsonl, write_parquet


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"missing gold: {path}. Export, annotate, and `wgs annotate merge` first.")
    return pd.read_parquet(path)


def _cls_report(y, p, w=None) -> dict:
    out = {"n": len(y), "macro_f1": float(f1_score(y, p, labels=LABELS, average="macro",
                                                   sample_weight=w, zero_division=0))}
    pr, rc, f, sup = precision_recall_fscore_support(y, p, labels=LABELS, sample_weight=w, zero_division=0)
    for i, lab in enumerate(LABELS):
        out[f"{lab}_precision"], out[f"{lab}_recall"] = float(pr[i]), float(rc[i])
        out[f"{lab}_support"] = float(sup[i]) if sup is not None else None
    return out


def _coverage_error(df: pd.DataFrame) -> list[dict]:
    """Abstain below confidence threshold -> (coverage, error) curve. Needs probabilities."""
    conf = df["probs"].map(lambda s: max(eval(s).values()) if isinstance(s, str) else np.nan)
    if conf.isna().all():
        return []
    curve = []
    for t in np.linspace(0, 0.95, 20):
        keep = conf >= t
        if keep.sum() == 0:
            break
        curve.append({"threshold": round(float(t), 2), "coverage": float(keep.mean()),
                      "error": float((df.loc[keep, "relation"] != df.loc[keep, "gold"]).mean())})
    return curve


def eval_judges(cfg: Config, judges: list[str], kind: str = "claim") -> pd.DataFrame:
    """Compare judges on the same human pair gold (scope 'gold' edges)."""
    from .evidence import edges_path
    started = dt.datetime.now()
    gold = _need(cfg.annotations_dir / "pair_gold.parquet")
    rv = data.reviews(cfg)
    rows, curves = [], {}
    for j in judges:
        e = pd.read_parquet(edges_path(cfg, j, kind, "gold"))
        m = gold.merge(e[e.judged], on=["hyp_id", "review_id"], how="left")
        missing = int(m["relation_y"].isna().sum())
        m = m.dropna(subset=["relation_y"]).rename(columns={"relation_x": "gold", "relation_y": "relation"})
        if m.empty:
            raise ValueError(f"judge '{j}' has no judged gold pairs (see runs/<run>/failures.jsonl); "
                             f"re-run `wgs judge --judge {j} --scope gold`")
        m["group"] = np.where(m.review_id.map(rv["rule_label"]) == "D", "D", "nonD")
        m["len_bin"] = pd.cut(m.review_id.map(rv["word_count"]), [0, 30, 100, 10**6],
                              labels=["short", "medium", "long"]).astype(str)
        m["rating_band"] = m.review_id.map(rv["rating_band"])
        w = m["sampling_weight"].astype(float)
        lat = m["latency_s"].astype(float)
        base = {"judge": j, "unjudged_gold_pairs": missing,
                "gold_annotator_type": str(gold["annotator_type"].iloc[0]) if "annotator_type" in gold else "unknown",
                "latency_p50_s": float(lat.median()), "latency_p95_s": float(lat.quantile(.95)),
                "pairs_per_s": float(1 / lat.mean()) if lat.mean() else None}
        rows.append({**base, "slice": "all_unweighted", **_cls_report(m.gold, m.relation)})
        rows.append({**base, "slice": "all_weighted", **_cls_report(m.gold, m.relation, w)})
        for col in ("group", "len_bin", "rating_band"):
            for v, g in m.groupby(col):
                rows.append({**base, "slice": f"{col}={v}", **_cls_report(g.gold, g.relation)})
        curves[j] = _coverage_error(m)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir / "judge_eval.csv", index=False)
    (cfg.out_dir / "judge_coverage_error.json").write_text(json.dumps(curves, indent=2))
    log_step(cfg, "eval-judges", {"judges": judges}, {"rows": len(df)}, started)
    return df


def retrieval_recall(cfg: Config, k_values=None, model_tags=(None,)) -> pd.DataFrame:
    """recall@K of gold SUPPORT/OPPOSE pairs, measured on the exhaustive gold subset."""
    from .retrieval import pairs_path
    gold = _need(cfg.annotations_dir / "pair_gold.parquet")
    gold = gold[(gold.stratum == "exhaustive") & gold.relation.isin(["SUPPORT", "OPPOSE"])]
    if gold.empty:
        raise ValueError("no exhaustive-stratum SUPPORT/OPPOSE gold; export pairs with --exhaustive-products")
    rv = data.reviews(cfg)
    rows = []
    for tag in model_tags:
        pr = pd.read_parquet(pairs_path(cfg, "claim", tag))[["hyp_id", "review_id", "rank"]]
        m = gold.drop(columns=["rank"], errors="ignore").merge(pr, on=["hyp_id", "review_id"])
        m["group"] = np.where(m.review_id.map(rv["rule_label"]) == "D", "D", "nonD")
        for k in k_values or cfg["retrieval"]["k_values"]:
            for (rel, grp), g in m.groupby(["relation", "group"]):
                rows.append({"retriever": tag or cfg["retrieval"]["model_id"], "K": k, "relation": rel,
                             "group": grp, "n": len(g), "recall_at_k": float((g["rank"] <= k).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir / "retrieval_recall.csv", index=False)
    return df


def preservation(cfg: Config, baseline: str = "direct", reps: int | None = None) -> pd.DataFrame:
    """Main metric: human important-experience recall + conditional fidelity, per summary,
    and paired product-bootstrap differences of each method vs the baseline."""
    from .metrics import importance_recall, paired_bootstrap
    inv = _need(cfg.annotations_dir / "experience_inventory.parquet")
    sg = _need(cfg.annotations_dir / "summary_gold.parquet")
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    rec = importance_recall(inv, sg)
    expressed = sg[sg.judgment.isin(["faithful", "condition_lost", "overstated"])]
    fid = expressed.groupby("summary_id")["judgment"].apply(lambda s: (s == "faithful").mean()) \
        .rename("conditional_fidelity").reset_index()
    df = sums[["summary_id", "bundle_id", "product_id", "cohort", "method", "model_id", "word_count",
               "latency_s"]].merge(rec, on="summary_id").merge(fid, on="summary_id", how="left")
    write_parquet(df, cfg.out_dir / "preservation_by_summary.parquet")
    reps = reps or int(cfg["evaluation"]["bootstrap_repetitions"])
    comps = []
    for m in sorted(set(df.method) - {baseline}):
        for metric in ("recall_imp", "conditional_fidelity", "recall_imp_minority", "word_count"):
            comps.append(paired_bootstrap(df, metric, baseline, m, reps=reps, seed=cfg.seed % 2**32))
    out = pd.DataFrame(comps)
    out.to_csv(cfg.out_dir / "preservation_paired_bootstrap.csv", index=False)
    return out


def intervention_metrics(cfg: Config) -> pd.DataFrame:
    """Joint coverage / retention / symmetric inconsistency (target-preserved) and removal error
    (target-removed), using human judgments of whether the target appears in each summary.
    Needs intervention_target_gold.parquet (wgs annotate export/merge --kind intervention_target)."""
    g = _need(cfg.annotations_dir / "intervention_target_gold.parquet")
    pairs = pd.DataFrame(read_jsonl(cfg.data_dir / "intervention_pairs.jsonl"))
    valid = _need(cfg.annotations_dir / "intervention_gold.parquet")
    ok = set(valid.loc[valid.valid_pair == "1", "item_id"])
    g = g.merge(pairs[["pair_id", "type", "target_polarity"]], on="pair_id")
    g = g[g.pair_id.isin(ok)]
    rows = []
    for (t, model, method, pol), x in g.groupby(["type", "model_id", "method", "target_polarity"]):
        a, b = x.target_in_original.astype(int), x.target_in_modified.astype(int)
        r = {"type": t, "model_id": model, "method": method, "target_polarity": pol, "n_pairs": len(x)}
        if t == "target_preserved":
            r.update(joint_coverage=float((a & b).mean()),
                     retention=float(b[a == 1].mean()) if (a == 1).any() else np.nan,
                     inclusion_inconsistency=float((a != b).mean()))
        else:
            r.update(removal_error=float(b.mean()))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir / "intervention_metrics.csv", index=False)
    return df


def group_regression(cfg: Config, judge: str, scope: str = "all") -> pd.DataFrame:
    """Stepwise ASSOCIATION models (not causal): z_ipm ~ product FE + model FE + D [+ ...].
    M0: FE + D; M1: + verified + recency; M2: + log length + rating band."""
    import statsmodels.formula.api as smf
    from .evidence import edges_path
    e = pd.read_parquet(edges_path(cfg, judge, "claim", scope))
    claims = pd.read_parquet(cfg.out_dir / "summary_claims.parquet")[["claim_id", "summary_id"]]
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))[["summary_id", "model_id", "method", "bundle_id"]]
    rv = data.reviews(cfg)
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    rows = []
    sup = e[e.relation == "SUPPORT"].merge(claims, left_on="hyp_id", right_on="claim_id")
    linked = set(zip(sup.summary_id, sup.review_id))
    for s in sums.itertuples():
        b = bundles.get(s.bundle_id)
        if b is None or b.get("variant") != "original":
            continue
        for r in b["review_ids_ordered"]:
            rows.append({"summary_id": s.summary_id, "product_id": b["product_id"], "model_id": s.model_id,
                         "method": s.method, "z": int((s.summary_id, r) in linked),
                         "rule_label": rv.at[r, "rule_label"], "D": int(rv.at[r, "rule_label"] == "D"), "verified": int(bool(rv.at[r, "verified_purchase"])),
                         "recency": float(rv.at[r, "timestamp"]) / 3.15e10,
                         "log_len": float(np.log1p(rv.at[r, "word_count"])), "rating_band": rv.at[r, "rating_band"]})
    df = pd.DataFrame(rows)
    df = df[df.rule_label.isin(["D", "U"])]  # S/A excluded from the D-vs-U contrast
    const = df.groupby("product_id")["z"].transform("nunique") == 1
    df_fit = df[~const]
    forms = {"M0": "z ~ C(product_id) + C(model_id) + D",
             "M1": "z ~ C(product_id) + C(model_id) + D + verified + recency",
             "M2": "z ~ C(product_id) + C(model_id) + D + verified + recency + log_len + C(rating_band)"}
    out = []
    for name, f in forms.items():
        for method, x in df_fit.groupby("method"):
            try:
                x = x.reset_index(drop=True)
                res = smf.logit(f, data=x).fit(disp=0, maxiter=200, cov_type="cluster",
                                               cov_kwds={"groups": pd.factorize(x.product_id)[0]})
                ci = res.conf_int().loc["D"]
                out.append({"model": name, "method": method, "beta_D": res.params["D"],
                            "ci_low": ci[0], "ci_high": ci[1], "n": int(res.nobs),
                            "products_dropped_constant_outcome": int(df[const].product_id.nunique())})
            except Exception as ex:
                out.append({"model": name, "method": method, "error": repr(ex)[:200]})
    res = pd.DataFrame(out)
    res.to_csv(cfg.out_dir / f"group_regression_{judge}.csv", index=False)
    return res


def disclosure_rule_accuracy(cfg: Config) -> pd.DataFrame:
    """How well do rule candidates match human disclosure gold? Reported per sampling stratum
    (rule candidates / hard negatives / random U) and overall per class."""
    gold = _need(cfg.annotations_dir / "disclosure_gold.parquet")
    rows = []
    for stratum, g in [("all", gold)] + list(gold.groupby("stratum")):
        for lab in ["D", "S", "U", "A"]:
            pred, true = g.rule_label == lab, g.label == lab
            rows.append({"stratum": stratum, "label": lab, "n": len(g),
                         "rule_count": int(pred.sum()), "gold_count": int(true.sum()),
                         "precision": float((pred & true).sum() / pred.sum()) if pred.sum() else np.nan,
                         "recall": float((pred & true).sum() / true.sum()) if true.sum() else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir / "disclosure_rule_accuracy.csv", index=False)
    return df


def intervention_metrics_auto(cfg: Config, judge_name: str, only_validated: bool = False) -> pd.DataFrame:
    """Scale version of intervention_metrics: target presence in a summary is decided by the
    evidence judge (target proposition vs summary text; SUPPORT = present). Its agreement with
    the human intervention_target gold is reported when that gold exists."""
    from .judges import make_judge
    from .storage import JsonlAppender, short_hash
    judge = make_judge(cfg, judge_name)
    pairs = pd.DataFrame(read_jsonl(cfg.data_dir / "intervention_pairs.jsonl"))
    pairs = pairs[pairs.type.isin(["target_preserved", "target_removed", "aspect_removed"])]
    if only_validated:
        ok = set(_need(cfg.annotations_dir / "intervention_gold.parquet").query("valid_pair == '1'").item_id)
        pairs = pairs[pairs.pair_id.isin(ok)]
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    a = pairs.merge(sums, left_on="original_bundle", right_on="bundle_id")
    m = a.merge(sums[["bundle_id", "model_id", "method", "summary_id", "text"]],
                left_on=["modified_bundle", "model_id", "method"], right_on=["bundle_id", "model_id", "method"],
                suffixes=("", "_mod"))
    cache = JsonlAppender(cfg.out_dir / f"target_presence_{judge_name}.jsonl", "key")
    def present(summary_id, text, target, cond):
        k = short_hash(judge.version, summary_id, target, cond)
        if k not in cache:
            lab = judge.judge([{"experience": target, "condition": cond, "review": text}])[0]["label"]
            cache.write({"key": k, "summary_id": summary_id, "target": target, "label": lab})
        return k
    keys = []
    for r in m.itertuples():
        keys.append((present(r.summary_id, r.text, r.target_text, r.target_condition or ""),
                     present(r.summary_id_mod, r.text_mod, r.target_text, r.target_condition or "")))
    cache.close()
    lab = {x["key"]: x["label"] for x in read_jsonl(cfg.out_dir / f"target_presence_{judge_name}.jsonl")}
    m["target_in_original"] = [int(lab[a] == "SUPPORT") for a, _ in keys]
    m["target_in_modified"] = [int(lab[b] == "SUPPORT") for _, b in keys]
    rows = []
    for (t, model, method, pol), x in m.groupby(["type", "model_id", "method", "target_polarity"]):
        a_, b_ = x.target_in_original, x.target_in_modified
        r = {"type": t, "model_id": model, "method": method, "target_polarity": pol, "n_pairs": len(x),
             "presence_source": f"auto:{judge_name}"}
        if t == "target_preserved":
            r.update(joint_coverage=float((a_ & b_).mean()),
                     retention=float(b_[a_ == 1].mean()) if (a_ == 1).any() else np.nan,
                     inclusion_inconsistency=float((a_ != b_).mean()))
        else:
            r.update(removal_error=float(b_.mean()))
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir / f"intervention_metrics_auto_{judge_name}.csv", index=False)
    gold_p = cfg.annotations_dir / "intervention_target_gold.parquet"
    if gold_p.exists():
        g = pd.read_parquet(gold_p)
        j = g.merge(m[["summary_id", "summary_id_mod", "target_in_original", "target_in_modified"]],
                    on=["summary_id", "summary_id_mod"], suffixes=("_gold", "_auto"))
        agree = pd.concat([(j.target_in_original_gold.astype(int) == j.target_in_original_auto),
                           (j.target_in_modified_gold.astype(int) == j.target_in_modified_auto)])
        pd.DataFrame([{"judge": judge_name, "n": len(agree), "agreement": float(agree.mean())}]).to_csv(
            cfg.out_dir / f"target_presence_agreement_{judge_name}.csv", index=False)
    return df


def auto_recall(cfg: Config, judge_name: str, scope: str = "topk:10", per_bundle: int = 10,
                baseline: str = "direct", reps: int | None = None) -> pd.DataFrame:
    """Scale proxy for important-experience recall (all products with candidates).

    Targets per bundle: verified candidates (support_users >= 1) with specificity >= 0.5, up to
    `per_bundle` chosen by hash (not by frequency, so minority experiences stay in).
    Presence = judge(candidate proposition, summary text) == SUPPORT.
    If human recall_imp exists, the per-summary Spearman correlation with it is reported."""
    from concurrent.futures import ThreadPoolExecutor
    from .judges import make_judge
    from .metrics import paired_bootstrap
    from .planner.features import candidate_table
    from .storage import JsonlAppender, short_hash
    judge = make_judge(cfg, judge_name)
    t = candidate_table(cfg, judge_name, scope)
    t = t[(t.support_users >= 1) & (t.specificity >= 0.5)].copy()
    t["h"] = [short_hash(cfg.seed, "autorec", x) for x in t.hyp_id]
    t = t.sort_values(["bundle_id", "h"]).groupby("bundle_id").head(per_bundle)
    t["minority"] = t.support_share_bundle <= float(cfg["evaluation"]["minority_support_share"])
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    sums = sums[(sums.variant == "original") & sums.bundle_id.isin(set(t.bundle_id))]
    m = sums[["summary_id", "bundle_id", "product_id", "method", "model_id", "text"]].merge(
        t[["bundle_id", "hyp_id", "proposition", "condition", "polarity", "minority"]], on="bundle_id")
    path = cfg.out_dir / f"auto_presence_{judge_name}.jsonl"
    cache = JsonlAppender(path, "key")
    m["key"] = [short_hash(judge.version, a, b) for a, b in zip(m.summary_id, m.hyp_id)]
    todo = m[~m.key.isin(cache.done)]

    def run(r):
        lab = judge.judge([{"experience": r.proposition, "condition": r.condition, "review": r.text}])[0]["label"]
        return {"key": r.key, "summary_id": r.summary_id, "hyp_id": r.hyp_id, "label": lab}
    workers = int(cfg["evidence"].get("concurrency", 1)) if getattr(judge, "parallel_ok", False) else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in tqdm(ex.map(run, todo.itertuples()), total=len(todo), desc=f"auto recall x{workers}"):
            cache.write(res)
    cache.close()
    lab = {x["key"]: x["label"] for x in read_jsonl(path)}
    m["hit"] = [float(lab.get(k) == "SUPPORT") for k in m.key]
    per = m.groupby(["summary_id", "product_id", "method", "model_id"]).agg(
        recall_auto=("hit", "mean"), n_targets=("hit", "size")).reset_index()
    for col, val, name in [("minority", True, "recall_auto_minority"), ("polarity", "negative", "recall_auto_negative")]:
        per = per.merge(m[m[col] == val].groupby("summary_id")["hit"].mean().rename(name), on="summary_id", how="left")
    write_parquet(per, cfg.out_dir / f"auto_recall_{judge_name}.parquet")
    reps = reps or int(cfg["evaluation"]["bootstrap_repetitions"])
    comps = [paired_bootstrap(per, metric, baseline, meth, reps=reps, seed=cfg.seed % 2**32)
             for meth in sorted(set(per.method) - {baseline})
             for metric in ("recall_auto", "recall_auto_minority", "recall_auto_negative")]
    out = pd.DataFrame(comps)
    out.to_csv(cfg.out_dir / f"auto_recall_paired_bootstrap_{judge_name}.csv", index=False)
    hp = cfg.out_dir / "preservation_by_summary.parquet"
    if hp.exists():
        h = pd.read_parquet(hp).merge(per, on="summary_id")
        rho = h[["recall_imp", "recall_auto"]].corr(method="spearman").iloc[0, 1] if len(h) > 2 else np.nan
        pd.DataFrame([{"judge": judge_name, "n_summaries": len(h), "spearman_auto_vs_human": rho}]).to_csv(
            cfg.out_dir / f"auto_recall_validation_{judge_name}.csv", index=False)
    return out


def gold_recall(cfg: Config, judge_name: str, min_importance: int = 2, baseline: str = "direct",
                reps: int | None = None) -> pd.DataFrame:
    """Important-experience recall on the HUMAN inventory (40 products), presence decided by the
    evidence judge (gold experience vs summary text; SUPPORT = expressed). Reported for all gold
    experiences and for the human_agreed subset (found by both annotators; AI-adjudication-free).
    The judge's presence decisions are validated against the human summary-gold sample."""
    from concurrent.futures import ThreadPoolExecutor
    from .judges import make_judge
    from .metrics import paired_bootstrap
    from .storage import JsonlAppender, short_hash
    judge = make_judge(cfg, judge_name)
    inv = _need(cfg.annotations_dir / "experience_inventory.parquet")
    inv = inv[inv.importance >= min_importance]
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    sums = sums[(sums.variant == "original") & sums.bundle_id.isin(set(inv.bundle_id))]
    m = sums[["summary_id", "bundle_id", "product_id", "method", "model_id", "text", "word_count"]].merge(
        inv[["bundle_id", "experience_id", "proposition", "condition", "polarity", "minority", "conditional",
             "human_agreed", "cross_check_status"]], on="bundle_id")
    path = cfg.out_dir / f"gold_presence_{judge_name}.jsonl"
    cache = JsonlAppender(path, "key")
    m["key"] = [short_hash(judge.version, a, b) for a, b in zip(m.summary_id, m.experience_id)]
    todo = m[~m.key.isin(cache.done)].drop_duplicates("key")

    def run(r):
        lab = judge.judge([{"experience": r.proposition, "condition": r.condition, "review": r.text}])[0]["label"]
        return {"key": r.key, "summary_id": r.summary_id, "experience_id": r.experience_id, "label": lab}
    workers = int(cfg["evidence"].get("concurrency", 1)) if getattr(judge, "parallel_ok", False) else 1
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in tqdm(ex.map(run, todo.itertuples()), total=len(todo), desc=f"gold presence x{workers}"):
            cache.write(res)
    cache.close()
    lab = {x["key"]: x["label"] for x in read_jsonl(path)}
    m["hit"] = [float(lab.get(k) == "SUPPORT") for k in m.key]
    write_parquet(m.drop(columns=["text"]), cfg.out_dir / f"gold_presence_{judge_name}.parquet")
    g = ["summary_id", "product_id", "method", "model_id", "word_count"]
    per = m.groupby(g).agg(recall_gold=("hit", "mean"), n_gold=("hit", "size")).reset_index()
    subsets = [("human_agreed", True, "recall_gold_human_agreed"), ("minority", True, "recall_gold_minority"),
               ("polarity", "negative", "recall_gold_negative"), ("conditional", True, "recall_gold_conditional")]
    for col, val, name in subsets:
        per = per.merge(m[m[col] == val].groupby("summary_id")["hit"].mean().rename(name), on="summary_id", how="left")
    write_parquet(per, cfg.out_dir / f"gold_recall_{judge_name}.parquet")
    reps = reps or int(cfg["evaluation"]["bootstrap_repetitions"])
    metrics_ = ["recall_gold", "recall_gold_human_agreed", "recall_gold_minority", "recall_gold_negative", "word_count"]
    out = pd.DataFrame([paired_bootstrap(per, met, baseline, meth, reps=reps, seed=cfg.seed % 2**32)
                        for meth in sorted(set(per.method) - {baseline}) for met in metrics_])
    out.to_csv(cfg.out_dir / f"gold_recall_paired_bootstrap_{judge_name}.csv", index=False)
    per.groupby(["model_id", "method"])[[c for c in per.columns if c.startswith("recall") or c == "word_count"]] \
        .mean().reset_index().to_csv(cfg.out_dir / f"gold_recall_table_{judge_name}.csv", index=False)
    return out


def presence_validation(cfg: Config, judge_name: str) -> dict:
    """Judge presence decisions vs human summary gold on the same (summary, experience) items.
    Human 'faithful' = present; condition_lost / overstated / absent / contradicted = not faithful."""
    sg = _need(cfg.annotations_dir / "summary_gold.parquet")
    auto = _need(cfg.out_dir / f"gold_presence_{judge_name}.parquet")
    j = sg.merge(auto[["summary_id", "experience_id", "hit"]], on=["summary_id", "experience_id"])
    if j.empty:
        raise ValueError("no overlap between summary gold and automatic presence decisions")
    human = (j.judgment == "faithful").astype(int)
    machine = j.hit.astype(int)
    tp = int(((human == 1) & (machine == 1)).sum())
    rep = {"judge": judge_name, "n": len(j), "accuracy": float((human == machine).mean()),
           "precision_faithful": tp / max(1, int(machine.sum())), "recall_faithful": tp / max(1, int(human.sum())),
           "kappa": float(cohen_kappa_score(human, machine)) if human.nunique() > 1 else None,
           "human_faithful_rate": float(human.mean()), "auto_present_rate": float(machine.mean()),
           "by_judgment": j.groupby("judgment")["hit"].mean().round(3).to_dict()}
    (cfg.out_dir / f"presence_validation_{judge_name}.json").write_text(json.dumps(rep, indent=2))
    return rep


def _sens_spec(gold: pd.DataFrame, judged: pd.Series, weights: pd.Series) -> tuple[float, float]:
    y = (gold == "SUPPORT").to_numpy()
    j = (judged == "SUPPORT").to_numpy()
    w = weights.to_numpy(float)
    sens = (w * (y & j)).sum() / max(1e-9, (w * y).sum())
    spec = (w * (~y & ~j)).sum() / max(1e-9, (w * ~y).sum())
    return float(sens), float(spec)


def corrected_support_rates(cfg: Config, judge: str, scope: str = "all", reps: int = 2000) -> pd.DataFrame:
    """Human-calibrated pair-level support rates (Rogan-Gladen correction).

    observed p = share of judged (claim, review) pairs the judge labels SUPPORT, per generator x
    method x review group (D / U / all). Judge sensitivity and specificity for SUPPORT are estimated
    on the human pair gold (dev split, sampling-weighted); corrected = (p + spec - 1)/(sens + spec - 1),
    clipped to [0, 1]. 95% CIs bootstrap the human gold. Assumes judge error rates transfer from
    dev to test (same product population and generators); reported together with the raw rates."""
    from .evidence import edges_path
    gold = _need(cfg.annotations_dir / "pair_gold.parquet")
    ge = pd.read_parquet(edges_path(cfg, judge, "claim", "gold"))
    g = gold.merge(ge[ge.judged][["hyp_id", "review_id", "relation"]], on=["hyp_id", "review_id"],
                   suffixes=("", "_judge"))
    w = g["sampling_weight"].astype(float)
    sens, spec = _sens_spec(g["relation"], g["relation_judge"], w)
    rng = np.random.default_rng(cfg.seed % 2**32)
    boots = []
    for _ in range(reps):
        i = rng.integers(0, len(g), len(g))
        boots.append(_sens_spec(g["relation"].iloc[i], g["relation_judge"].iloc[i], w.iloc[i]))
    e = pd.read_parquet(edges_path(cfg, judge, "claim", scope), columns=["hyp_id", "review_id", "judged", "relation"])
    e = e[e.judged]
    cl = pd.read_parquet(cfg.out_dir / "summary_claims.parquet", columns=["claim_id", "summary_id"])
    sm = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))[["summary_id", "model_id", "method", "split", "variant", "cohort"]]
    e = e.merge(cl, left_on="hyp_id", right_on="claim_id").merge(sm, on="summary_id")
    e = e[(e.split == "test") & (e.variant == "original")]
    e["group"] = e.review_id.map(data.reviews(cfg)["rule_label"])
    rows = []
    def corr(p, s, c):
        d = s + c - 1
        return float(np.clip((p + c - 1) / d, 0, 1)) if d > 0.05 else np.nan
    for (model, meth), x in e.groupby(["model_id", "method"]):
        for grp in ("all", "D", "U"):
            y = x if grp == "all" else x[x.group == grp]
            if y.empty:
                continue
            p = float((y.relation == "SUPPORT").mean())
            cb = [corr(p, s, c) for s, c in boots]
            rows.append({"model_id": model, "method": meth, "group": grp, "n_pairs": len(y),
                         "observed_support_rate": p, "corrected_support_rate": corr(p, sens, spec),
                         "ci_low": float(np.nanpercentile(cb, 2.5)), "ci_high": float(np.nanpercentile(cb, 97.5)),
                         "judge_sensitivity": sens, "judge_specificity": spec, "judge": judge, "scope": scope})
    df = pd.DataFrame(rows)
    df.to_csv(cfg.out_dir / f"corrected_support_rates_{judge}.csv", index=False)
    return df
