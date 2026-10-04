"""Human annotation sheets (export) and gold construction (merge).

Sheets are blinded: no method/model names, no rule D/U tags, no retrieval scores. Keys that
link items back to IDs are written to a separate _key file. Empty labels are NEVER treated as
labels: unlabeled items are excluded from gold and counted.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from . import data
from .config import Config
from .storage import log_step, read_jsonl, short_hash, write_parquet

GUIDELINE_VERSION = "v1.0"
LABEL_COLS = {
    "disclosure": ["label", "vine"],
    "pair": ["relation", "condition_match"],
    "summary": ["judgment"],
    "selection": ["v_evidence_valid", "y_include"],
    "intervention": ["valid_pair"],
    "intervention_target": ["target_in_original", "target_in_modified"],
}
ALLOWED = {
    "label": {"D", "S", "U", "A"}, "vine": {"0", "1"},
    "relation": {"SUPPORT", "OPPOSE", "IRRELEVANT", "UNCERTAIN"},
    "condition_match": {"Y", "N", "NA"},
    "judgment": {"faithful", "condition_lost", "overstated", "absent", "contradicted"},
    "v_evidence_valid": {"0", "1"}, "y_include": {"0", "1"}, "valid_pair": {"0", "1"},
    "target_in_original": {"0", "1"}, "target_in_modified": {"0", "1"},
}


def _dir(cfg: Config, kind: str):
    d = cfg.annotations_dir / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_sheets(cfg: Config, kind: str, sheet: pd.DataFrame, key: pd.DataFrame,
                  extra_cols: list[str] | None = None) -> list[str]:
    d = _dir(cfg, kind)
    for c in LABEL_COLS.get(kind, []) + (extra_cols or []) + ["note"]:
        sheet[c] = ""
    key.to_csv(d / "_key.csv", index=False, encoding="utf-8-sig")
    paths = []
    for a in cfg["annotation"]["annotators"]:
        p = d / f"{a}.csv"
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to overwrite annotations")
        sheet.to_csv(p, index=False, encoding="utf-8-sig")
        paths.append(str(p))
    (d / "_meta.json").write_text(json.dumps({
        "kind": kind, "guideline_version": GUIDELINE_VERSION, "n_items": len(sheet),
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "guide": f"docs/annotation/{kind}.md"}, indent=2), encoding="utf-8")
    return paths


def _review_text(r) -> str:
    return f"{r['title_clean']}. {r['text_clean']}" if r["title_clean"] else r["text_clean"]


# ------------------------------------------------------------------ disclosure (D/S/U/A)
def export_disclosure(cfg: Config, n: int | None = None) -> list[str]:
    n = n or int(cfg["annotation"]["disclosure_items"])
    rv = data.reviews(cfg)
    rv = rv[rv.valid_text & (rv.lang_guess == "en_likely")]
    h = rv["review_id"].map(lambda r: short_hash(cfg.seed, "disc", r))
    rv = rv.assign(h=h).sort_values("h")
    parts = [("rule_candidate", rv[rv.rule_label.isin(["D", "S", "A"])].head(n // 2)),
             ("hard_negative", rv[rv.hard_negative].head(n // 4))]
    used = set(pd.concat([p for _, p in parts])["review_id"])
    parts.append(("random_U", rv[(rv.rule_label == "U") & ~rv.review_id.isin(used)]
                  .head(n - sum(len(p) for _, p in parts))))
    rows = []
    for stratum, p in parts:
        for _, r in p.iterrows():
            rows.append({"item_id": "d" + short_hash(r["review_id"], n=10), "review_id": r["review_id"],
                         "stratum": stratum, "rule_label": r["rule_label"], "text": _review_text(r)})
    df = pd.DataFrame(rows).sample(frac=1, random_state=cfg.seed % 2**32)
    return _write_sheets(cfg, "disclosure", df[["item_id", "text"]].copy(),
                         df[["item_id", "review_id", "stratum", "rule_label"]].copy(),
                         extra_cols=["evidence_span"])


# ------------------------------------------------------------------ pair gold (claim x review)
def _rank_bin(rank: int) -> str:
    return "r01-03" if rank <= 3 else "r04-10" if rank <= 10 else "r11-20" if rank <= 20 else "r21+"


def export_pairs(cfg: Config, n: int | None = None, exhaustive_products: int = 0,
                 kind: str = "claim", splits=None) -> list[str]:
    """Stratified pair sample (rank bins x D-candidate flag) + optional exhaustive products.

    sampling_weight = stratum population / stratum sample size, so natural-distribution
    precision can be re-weighted. Exhaustive products give every pair (for recall@K)."""
    n = n or int(cfg["annotation"]["pair_items"])
    from .retrieval import hypotheses
    pairs = pd.read_parquet(cfg.out_dir / f"retrieval_pairs_{kind}.parquet")
    if splits:  # e.g. judge selection must use dev only, never test
        split_of = {b["bundle_id"]: b["split"] for b in data.all_bundles(cfg)}
        pairs = pairs[pairs.bundle_id.map(split_of).isin(splits)]
        if pairs.empty:
            raise ValueError(f"no retrieval pairs for splits {splits}")
    rv = data.reviews(cfg)
    pairs["rule_label"] = pairs.review_id.map(rv["rule_label"])
    pairs["stratum"] = pairs["rank"].map(_rank_bin) + "|" + np.where(pairs.rule_label == "D", "D", "nonD")
    pairs["h"] = [short_hash(cfg.seed, "pair", a, b) for a, b in zip(pairs.hyp_id, pairs.review_id)]
    chosen = []
    if exhaustive_products:
        bids = sorted(pairs.bundle_id.unique(), key=lambda b: short_hash(cfg.seed, "exh", b))
        ex = pairs[pairs.bundle_id.isin(bids[:exhaustive_products])].assign(stratum="exhaustive",
                                                                           sampling_weight=1.0)
        chosen.append(ex)
        pairs = pairs[~pairs.bundle_id.isin(bids[:exhaustive_products])]
    strata = pairs.groupby("stratum")
    per = max(1, n // max(1, strata.ngroups))
    for s, g in strata:
        take = g.sort_values("h").head(per)
        chosen.append(take.assign(sampling_weight=len(g) / max(1, len(take))))
    sel = pd.concat(chosen, ignore_index=True)
    hyps = hypotheses(cfg, kind).set_index("hyp_id")
    sel["experience"] = sel.hyp_id.map(hyps["text"])
    sel["condition"] = sel.hyp_id.map(hyps["condition"])
    sel["review_text"] = sel.review_id.map(lambda r: _review_text(rv.loc[r]))
    sel["item_id"] = ["p" + short_hash(a, b, n=10) for a, b in zip(sel.hyp_id, sel.review_id)]
    sel = sel.sample(frac=1, random_state=cfg.seed % 2**32)
    return _write_sheets(cfg, "pair", sel[["item_id", "experience", "condition", "review_text"]].copy(),
                         sel[["item_id", "hyp_id", "review_id", "bundle_id", "rank", "score",
                              "stratum", "sampling_weight"]].copy())


# ------------------------------------------------------------------ input experience inventory
def export_inventory(cfg: Config, n_products: int | None = None, cohorts=None) -> list[str]:
    """Per product: a reviews sheet (anon ids) + an empty inventory template.
    Inventory is written independently of any summary; adjudicator produces inventory_gold.csv."""
    n_products = n_products or int(cfg["annotation"]["inventory_products"])
    d = _dir(cfg, "inventory")
    blist = sorted(data.bundles(cfg, None, cohorts), key=lambda b: short_hash(cfg.seed, "inv", b["bundle_id"]))
    keys, paths = [], []
    tmpl_cols = ["experience_id", "aspect", "proposition", "condition", "polarity",
                 "support_anon_ids", "oppose_anon_ids", "importance", "risk_flag", "note"]
    for b in blist[:n_products]:
        revs = data.bundle_reviews(cfg, b)
        pd.DataFrame([{"anon_id": r["anon_id"], "text": (r["title"] + ". " if r["title"] else "") + r["text"]}
                      for r in revs]).to_csv(d / f"{b['bundle_id']}_reviews.csv", index=False, encoding="utf-8-sig")
        for a in cfg["annotation"]["annotators"]:
            p = d / f"{b['bundle_id']}_inventory_{a}.csv"
            if not p.exists():
                pd.DataFrame(columns=tmpl_cols).to_csv(p, index=False, encoding="utf-8-sig")
            paths.append(str(p))
        keys += [{"bundle_id": b["bundle_id"], "anon_id": r["anon_id"], "review_id": r["review_id"],
                  "user_id": r["user_id"]} for r in revs]
    new = pd.DataFrame(keys)
    if (d / "_key.csv").exists():  # repeated exports (e.g. one per cohort) accumulate keys
        new = pd.concat([pd.read_csv(d / "_key.csv", dtype=str, keep_default_na=False), new.astype(str)])
        new = new.drop_duplicates(["bundle_id", "anon_id"])
    new.to_csv(d / "_key.csv", index=False, encoding="utf-8-sig")
    return paths


def build_inventory_gold(cfg: Config) -> pd.DataFrame:
    """Read adjudicated <bundle>_inventory_gold.csv files -> experience_inventory.parquet."""
    d = _dir(cfg, "inventory")
    key = pd.read_csv(d / "_key.csv")
    minority = float(cfg["evaluation"]["minority_support_share"])
    rows = []
    for f in sorted(d.glob("*_inventory_gold.csv")):
        bid = f.name.split("_inventory_gold")[0]
        k = key[key.bundle_id == bid].set_index("anon_id")
        n_users = k["user_id"].nunique()
        inv = pd.read_csv(f, dtype=str, keep_default_na=False).fillna("")
        inv = inv[inv.proposition.str.strip() != ""]
        aspect_users = {}
        for _, e in inv.iterrows():
            ids = [x.strip() for x in e.support_anon_ids.replace(";", ",").split(",") if x.strip()]
            aspect_users.setdefault(e.aspect, set()).update(k.loc[[i for i in ids if i in k.index], "user_id"])
        for _, e in inv.iterrows():
            sup = [x.strip() for x in e.support_anon_ids.replace(";", ",").split(",") if x.strip()]
            opp = [x.strip() for x in e.oppose_anon_ids.replace(";", ",").split(",") if x.strip()]
            bad = [x for x in sup + opp if x not in k.index]
            if bad:
                raise ValueError(f"{f.name}: unknown anon ids {bad}")
            su = set(k.loc[sup, "user_id"])
            share = len(su) / n_users if n_users else np.nan
            rows.append({
                "experience_id": f"{bid}_{e.experience_id}", "bundle_id": bid,
                "aspect": e.aspect, "proposition": e.proposition, "condition": e.condition,
                "polarity": e.polarity, "importance": int(e.importance or 0),
                "risk_flag": e.risk_flag in ("1", "Y", "y", "True"),
                "support_review_ids": list(k.loc[sup, "review_id"]),
                "oppose_review_ids": list(k.loc[opp, "review_id"]),
                "support_users": len(su), "n_input_users": n_users, "support_share": share,
                "aspect_share": len(su) / len(aspect_users[e.aspect]) if aspect_users.get(e.aspect) else np.nan,
                "minority": share <= minority, "conditional": bool(e.condition.strip()),
                "guideline_version": GUIDELINE_VERSION,
            })
    df = pd.DataFrame(rows)
    write_parquet(df, cfg.annotations_dir / "experience_inventory.parquet")
    return df


# ------------------------------------------------------------------ summary gold
def export_summary(cfg: Config, min_importance: int = 1) -> list[str]:
    inv = pd.read_parquet(cfg.annotations_dir / "experience_inventory.parquet")
    inv = inv[inv.importance >= min_importance]
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    sums = sums[sums.bundle_id.isin(inv.bundle_id)]
    m = sums.merge(inv, on="bundle_id")
    m["item_id"] = ["g" + short_hash(a, b, n=10) for a, b in zip(m.summary_id, m.experience_id)]
    m = m.sample(frac=1, random_state=cfg.seed % 2**32)
    sheet = m[["item_id", "text", "proposition", "condition"]].rename(
        columns={"text": "summary", "proposition": "experience"}).copy()
    return _write_sheets(cfg, "summary", sheet, m[["item_id", "summary_id", "experience_id", "bundle_id"]].copy())


# ------------------------------------------------------------------ AEP selection labels
def export_selection(cfg: Config, splits=("train", "dev", "pilot")) -> list[str]:
    cands = pd.read_parquet(cfg.out_dir / "input_candidates.parquet")
    cands = cands[cands.split.isin(splits)]
    budget = int(cfg["planner"]["max_selected_experiences"])
    cands = cands.assign(budget=budget, n_mention_users=cands.mention_users.map(len))
    cands["item_id"] = cands.cand_id
    sheet = cands[["item_id", "bundle_id", "budget", "aspect", "proposition", "condition",
                   "polarity", "n_mention_users", "quote"]].sort_values(["bundle_id", "aspect"]).copy()
    return _write_sheets(cfg, "selection", sheet, cands[["item_id", "cand_id", "bundle_id"]].copy())


def export_interventions(cfg: Config, n: int | None = None) -> list[str]:
    """Validity check of target pairs (D/U resampling pairs have no target and are not shown).
    With n, a random subset is validated (reported as the pair-validity rate)."""
    iv = pd.DataFrame(read_jsonl(cfg.data_dir / "intervention_pairs.jsonl"))
    iv = iv[iv.type.isin(["target_preserved", "target_removed", "aspect_removed"])]
    iv = iv.sample(frac=1, random_state=cfg.seed % 2**32)
    if n:
        iv = iv.head(n)
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    rv = data.reviews(cfg)
    def txt(ids):
        return "\n".join(f"- {_review_text(rv.loc[r])}" for r in ids)
    rows = []
    for p in iv.itertuples():
        o, m = bundles[p.original_bundle], bundles[p.modified_bundle]
        removed = [r for r in o["review_ids_ordered"] if r not in set(m["review_ids_ordered"])]
        added = [r for r in m["review_ids_ordered"] if r not in set(o["review_ids_ordered"])]
        rows.append({"item_id": p.pair_id, "type": p.type, "target_experience": p.target_text,
                     "target_condition": p.target_condition, "removed_reviews": txt(removed),
                     "added_reviews": txt(added)})
    sheet = pd.DataFrame(rows)
    return _write_sheets(cfg, "intervention", sheet, iv[["pair_id"]].rename(columns={"pair_id": "item_id"}))


# ------------------------------------------------------------------ merge -> agreement + gold
def merge(cfg: Config, kind: str) -> dict:
    """Agreement on independent labels, disagreement sheet, and gold from agreed+adjudicated.

    gold parquet: item_id + key columns + <label>_A1, <label>_A2 (independent),
    <label> (final), source in {agreed, adjudicated}. Items with missing labels are excluded.
    """
    started = dt.datetime.now()
    d = _dir(cfg, kind)
    key = pd.read_csv(d / "_key.csv", dtype=str, keep_default_na=False)
    cols = LABEL_COLS[kind]
    anns = cfg["annotation"]["annotators"]
    sheets = {}
    for a in anns:
        s = pd.read_csv(d / f"{a}.csv", dtype=str, keep_default_na=False).fillna("")
        for c in cols:
            s[c] = s[c].str.strip()
            bad = s[(s[c] != "") & ~s[c].isin(ALLOWED[c])]
            if len(bad):
                raise ValueError(f"{a}.csv: invalid {c} values {sorted(set(bad[c]))[:5]}")
        sheets[a] = s.set_index("item_id")
    a1, a2 = anns[0], anns[1]
    both = sheets[a1].join(sheets[a2], lsuffix=f"_{a1}", rsuffix=f"_{a2}", how="inner")
    report = {"kind": kind, "guideline_version": GUIDELINE_VERSION, "items": len(both)}
    complete = pd.Series(True, index=both.index)
    for c in cols:
        complete &= (both[f"{c}_{a1}"] != "") & (both[f"{c}_{a2}"] != "")
    report["labeled_by_both"] = int(complete.sum())
    lab = both[complete]
    disagree = pd.Series(False, index=lab.index)
    for c in cols:
        x, y = lab[f"{c}_{a1}"], lab[f"{c}_{a2}"]
        report[f"{c}_percent_agreement"] = float((x == y).mean()) if len(lab) else None
        report[f"{c}_cohen_kappa"] = float(cohen_kappa_score(x, y)) if len(lab) and len(set(x) | set(y)) > 1 else None
        disagree |= x != y
    dis = lab[disagree].copy()
    adj_path = d / "_adjudicated.csv"
    if len(dis) and not adj_path.exists():
        out = dis[[f"{c}_{a}" for c in cols for a in (a1, a2)]].copy()
        for c in cols:
            out[c] = ""
        out.reset_index().to_csv(d / "_disagreements.csv", index=False, encoding="utf-8-sig")
        shutil.copy(d / "_disagreements.csv", adj_path)
        report["note"] = "fill _adjudicated.csv final columns, then re-run merge"
    gold = lab[~disagree].copy()
    for c in cols:
        gold[c] = gold[f"{c}_{a1}"]
    gold["source"] = "agreed"
    if adj_path.exists():
        adj = pd.read_csv(adj_path, dtype=str, keep_default_na=False).fillna("").set_index("item_id")
        adj = adj[(adj[cols] != "").all(axis=1)]
        extra = lab.loc[lab.index.intersection(adj.index)].copy()
        for c in cols:
            extra[c] = adj.loc[extra.index, c]
        extra["source"] = "adjudicated"
        gold = pd.concat([gold, extra])
        report["unadjudicated"] = int(disagree.sum() - len(extra))
    gold = gold.reset_index().merge(key, on="item_id", how="left")
    gold["guideline_version"] = GUIDELINE_VERSION
    write_parquet(gold, cfg.annotations_dir / f"{kind}_gold.parquet")
    report["gold_items"] = len(gold)
    (d / "_agreement.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log_step(cfg, f"annotate-merge-{kind}", {}, report, started)
    return report


def export_intervention_targets(cfg: Config, n: int | None = None) -> list[str]:
    """For validated target pairs: is the target experience expressed (faithfully) in the summary
    of the original bundle and of the modified bundle? Same model+method on both sides."""
    iv = pd.DataFrame(read_jsonl(cfg.data_dir / "intervention_pairs.jsonl"))
    iv = iv[iv.type.isin(["target_preserved", "target_removed", "aspect_removed"])]
    gold = cfg.annotations_dir / "intervention_gold.parquet"
    if gold.exists():
        ok = set(pd.read_parquet(gold).query("valid_pair == '1'").item_id)
        iv = iv[iv.pair_id.isin(ok)]
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    a = iv.merge(sums, left_on="original_bundle", right_on="bundle_id")
    m = a.merge(sums[["bundle_id", "model_id", "method", "summary_id", "text"]],
                left_on=["modified_bundle", "model_id", "method"], right_on=["bundle_id", "model_id", "method"],
                suffixes=("", "_mod"))
    m["item_id"] = ["t" + short_hash(x, y, n=10) for x, y in zip(m.summary_id, m.summary_id_mod)]
    m = m.sample(frac=1, random_state=cfg.seed % 2**32)
    if n:  # random subset: validates the automatic target-presence judge
        m = m.head(n)
    sheet = m[["item_id", "target_text", "target_condition", "text", "text_mod"]].rename(
        columns={"text": "summary_original", "text_mod": "summary_modified"}).copy()
    key = m[["item_id", "pair_id", "summary_id", "summary_id_mod", "model_id", "method"]].copy()
    return _write_sheets(cfg, "intervention_target", sheet, key)
