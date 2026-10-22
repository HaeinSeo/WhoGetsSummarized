"""Human annotation sheets (export) and gold construction (merge).

Sheets are blinded: no method/model names, no rule D/U tags, no retrieval scores. Keys that
link items back to IDs are written to a separate _key file. Empty labels are NEVER treated as
labels: unlabeled items are excluded from gold and counted.
"""
from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

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
                 kind: str = "claim", splits=None, exhaustive_model: str | None = None) -> list[str]:
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
        ex = pairs[pairs.bundle_id.isin(bids[:exhaustive_products])]
        if exhaustive_model:  # all pairs of ONE generator's summaries keeps the exhaustive set labelable
            cl = pd.read_parquet(cfg.out_dir / "summary_claims.parquet")[["claim_id", "summary_id"]]
            sm = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))[["summary_id", "model_id"]]
            keep = set(cl.merge(sm, on="summary_id").query("model_id == @exhaustive_model").claim_id)
            ex = ex[ex.hyp_id.isin(keep)]
        ex = ex.assign(stratum="exhaustive", sampling_weight=1.0)
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


def _ids(cell: str) -> list[str]:
    return [x.strip() for x in str(cell).replace(";", ",").split(",") if x.strip()]


def build_inventory_gold(cfg: Config, source_dir: str | None = None, adjudicator: str = "human",
                         human_signoff: bool = True) -> pd.DataFrame:
    """Adjudicated <bundle>_inventory_gold.csv files -> experience_inventory.parquet.

    source_dir: folder with the gold CSVs (default annotations/<run>/inventory). If it sits in a
    cross-check folder with adjudication/<bundle>_cross_check.csv and cross_check_summary.csv, each
    experience gets its provenance: decision_type (both_sources / a1_only / a2_only /
    added_from_reviews) and the bundle's cross_check_status. `human_agreed` marks experiences found
    by BOTH human annotators in a two-annotator bundle; it is the primary, AI-independent gold subset.
    """
    from pathlib import Path as _P
    d = _dir(cfg, "inventory")
    src = _P(source_dir) if source_dir else d
    if not src.is_absolute():
        src = cfg.resolve(src)
    key = pd.read_csv(d / "_key.csv", dtype=str, keep_default_na=False)
    minority = float(cfg["evaluation"]["minority_support_share"])
    root = src.parent
    decision = {}
    for f in sorted((root / "adjudication").glob("*_cross_check.csv")):
        a = pd.read_csv(f, dtype=str, keep_default_na=False)
        decision.update({(r.bundle_id, r.final_experience_id): r.decision_type for r in a.itertuples()})
    status = {}
    if (root / "cross_check_summary.csv").exists():
        cs = pd.read_csv(root / "cross_check_summary.csv", dtype=str, keep_default_na=False)
        status = dict(zip(cs.bundle_id, cs.cross_check_status))
    files = sorted(src.glob("*_inventory_gold.csv"))
    if not files:
        raise FileNotFoundError(f"no *_inventory_gold.csv in {src}")
    rows = []
    for f in files:
        bid = f.name.split("_inventory_gold")[0]
        k = key[key.bundle_id == bid].set_index("anon_id")
        if k.empty:
            raise ValueError(f"{f.name}: bundle not in inventory _key.csv")
        n_users = k["user_id"].nunique()
        inv = pd.read_csv(f, dtype=str, keep_default_na=False)
        inv = inv[inv.proposition.str.strip() != ""]
        aspect_users = {}
        for _, e in inv.iterrows():
            aspect_users.setdefault(e.aspect, set()).update(k.loc[[i for i in _ids(e.support_anon_ids) if i in k.index], "user_id"])
        st = status.get(bid, "unknown")
        for _, e in inv.iterrows():
            sup, opp = _ids(e.support_anon_ids), _ids(e.oppose_anon_ids)
            bad = [x for x in sup + opp if x not in k.index]
            if bad:
                raise ValueError(f"{f.name}: unknown anon ids {bad}")
            su = set(k.loc[sup, "user_id"])
            share = len(su) / n_users if n_users else np.nan
            dtype_ = decision.get((bid, e.experience_id), "unknown")
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
                "decision_type": dtype_, "cross_check_status": st,
                "human_agreed": dtype_ == "both_sources" and st == "A1_A2_reviewed",
                "adjudicator_type": adjudicator, "human_signoff": bool(human_signoff),
                "source_dir": str(src), "guideline_version": GUIDELINE_VERSION,
            })
    df = pd.DataFrame(rows)
    write_parquet(df, cfg.annotations_dir / "experience_inventory.parquet")
    return df


# ------------------------------------------------------------------ summary gold
def export_summary(cfg: Config, min_importance: int = 2, n: int | None = None) -> list[str]:
    """(summary, gold experience) presence sheet. With n, a random sample validates the automatic
    presence judge (all pairs of 40 products x ~8 summaries are too many to label by hand)."""
    inv = pd.read_parquet(cfg.annotations_dir / "experience_inventory.parquet")
    inv = inv[inv.importance >= min_importance]
    sums = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
    sums = sums[sums.bundle_id.isin(inv.bundle_id) & (sums.variant == "original")]
    m = sums.merge(inv, on="bundle_id")
    m["item_id"] = ["g" + short_hash(a, b, n=10) for a, b in zip(m.summary_id, m.experience_id)]
    m = m.sample(frac=1, random_state=cfg.seed % 2**32)
    if n:
        m = m.head(n)
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
def merge(cfg: Config, kind: str, annotator_type: str = "human") -> dict:
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
    report = {"kind": kind, "guideline_version": GUIDELINE_VERSION, "items": len(both),
              "annotator_type": annotator_type}
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
    gold["annotator_type"] = annotator_type   # provenance travels with the labels
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


# ------------------------------------------------------------------ hand-off to annotators
ANNOTATOR_README = """[Annotation package for {who}]

1. Open ONLY the files in this zip. Do not look at the other annotator's work or at model summaries.
2. Fill only the empty label columns. Do not edit item_id / text columns or rename files.
3. Labels must be spelled exactly as in the guides (e.g. SUPPORT, not support).
4. Save in Excel as "CSV UTF-8 (comma delimited)".
5. Guides: guides/*.md   (inventory: read <id>_reviews.csv, fill <id>_inventory_{who}.csv)
6. When done, zip the same folder structure and send it back.

[주석 패키지: {who}]
1. 이 압축 파일 안의 파일만 여세요. 다른 주석자의 결과나 AI 요약은 보지 마세요.
2. 비어 있는 라벨 열만 채우고, item_id·원문 열 수정이나 파일 이름 변경은 하지 마세요.
3. 라벨은 가이드의 철자 그대로 쓰세요 (예: SUPPORT, support는 안 됨).
4. 엑셀 저장 형식: "CSV UTF-8(쉼표로 분리)".
5. 기준: guides 폴더 (inventory는 <번호>_reviews.csv를 읽고 <번호>_inventory_{who}.csv를 채움)
6. 끝나면 같은 폴더 구조 그대로 압축해서 돌려보내 주세요.
"""


def pack(cfg: Config, who: str, kinds=None) -> str:
    """Zip only what annotator `who` needs: own sheets, inventory review files, guides.
    Never includes _key / _meta / other annotators' files."""
    import zipfile
    from .config import REPO_ROOT
    if who not in cfg["annotation"]["annotators"]:
        raise ValueError(f"unknown annotator {who}; config has {cfg['annotation']['annotators']}")
    root = cfg.annotations_dir
    out_dir = root / "_handoff"
    out_dir.mkdir(exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    tag = "_".join(kinds) if kinds else "all"
    zpath = out_dir / f"{cfg['run_name']}_{who}_{tag}_{stamp}.zip"
    n = 0
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")]
        if kinds:
            dirs = [p for p in dirs if p.name in set(kinds)]
        for kind_dir in sorted(dirs):
            for f in sorted(kind_dir.iterdir()):
                name = f.name
                own = name == f"{who}.csv" or name.endswith(f"_inventory_{who}.csv") or name.endswith("_reviews.csv")
                if f.is_file() and own and not name.startswith("_"):
                    z.write(f, f"{kind_dir.name}/{name}")
                    n += 1
        for g in sorted((REPO_ROOT / "docs" / "annotation").glob("*.md")):
            z.write(g, f"guides/{g.name}")
        z.writestr("README.txt", ANNOTATOR_README.format(who=who))
    if n == 0:
        zpath.unlink()
        raise FileNotFoundError("no sheets to pack yet (run `annotate export` first)")
    return str(zpath)


def unpack(cfg: Config, who: str, zip_path: str) -> list[str]:
    """Put an annotator's returned sheets back. Only that annotator's own files are accepted;
    item_ids and columns must match the exported template; previous files are backed up."""
    import shutil as _sh
    import zipfile
    root = cfg.annotations_dir
    backup = root / "_backup" / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    done, problems = [], []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            parts = Path(info.filename).parts
            if info.is_dir() or len(parts) < 2:
                continue
            kind, name = parts[-2], parts[-1]
            if not (name == f"{who}.csv" or name.endswith(f"_inventory_{who}.csv")):
                continue  # ignore guides, review files, anything else
            target = root / kind / name
            if not target.exists():
                problems.append(f"{kind}/{name}: no such exported sheet")
                continue
            with z.open(info) as fh:
                new = pd.read_csv(fh, dtype=str, keep_default_na=False)
            old = pd.read_csv(target, dtype=str, keep_default_na=False)
            if list(new.columns) != list(old.columns):
                problems.append(f"{kind}/{name}: columns changed {list(new.columns)}")
                continue
            if "item_id" in old.columns and set(new.item_id) != set(old.item_id):
                if set(old.item_id) <= set(new.item_id):  # sheet was shrunk after hand-off: keep current items
                    new = new[new.item_id.isin(set(old.item_id))]
                else:
                    problems.append(f"{kind}/{name}: item_id set changed")
                    continue
            (backup / kind).mkdir(parents=True, exist_ok=True)
            _sh.copy(target, backup / kind / name)
            new.to_csv(target, index=False, encoding="utf-8-sig")
            done.append(f"{kind}/{name}")
    if problems:
        raise ValueError("not imported:\n" + "\n".join(problems) + f"\n(imported {len(done)} files)")
    return done


def inventory_agreement(cfg: Config, a1_dir: str, a2_dir: str, cross_check_dir: str) -> dict:
    """Agreement between the two independent inventories, read through the adjudication mapping.

    Coverage: of the experiences kept in the final inventory (excluding ones added from reviews),
    how many each annotator found and how many both found. Label agreement (importance, polarity,
    risk) is computed on experiences found by both (a1/a2 ids may list several rows; the max
    importance and the first polarity are used). Bundles missing one annotator are excluded."""
    from pathlib import Path as _P
    a1d, a2d, cc = (cfg.resolve(x) if not _P(x).is_absolute() else _P(x) for x in (a1_dir, a2_dir, cross_check_dir))
    cs = pd.read_csv(cc / "cross_check_summary.csv", dtype=str, keep_default_na=False)
    two = set(cs.loc[cs.cross_check_status == "A1_A2_reviewed", "bundle_id"])
    rows, per_bundle = [], []
    for f in sorted((cc / "adjudication").glob("*_cross_check.csv")):
        a = pd.read_csv(f, dtype=str, keep_default_na=False)
        bid = a.bundle_id.iloc[0]
        if bid not in two:
            continue
        A1 = pd.read_csv(a1d / f"{bid}_inventory_A1.csv", dtype=str, keep_default_na=False).set_index("experience_id")
        A2 = pd.read_csv(a2d / f"{bid}_inventory_A2.csv", dtype=str, keep_default_na=False).set_index("experience_id")
        kept = a[a.decision_type.isin(["both_sources", "a1_only", "a2_only"])]
        per_bundle.append({"bundle_id": bid, "kept": len(kept), "both": int((kept.decision_type == "both_sources").sum()),
                           "a1_only": int((kept.decision_type == "a1_only").sum()),
                           "a2_only": int((kept.decision_type == "a2_only").sum())})
        for r in kept[kept.decision_type == "both_sources"].itertuples():
            i1 = [x for x in _ids(r.a1_experience_ids) if x in A1.index]
            i2 = [x for x in _ids(r.a2_experience_ids) if x in A2.index]
            if not i1 or not i2:
                continue
            imp = lambda t, ids: max(int(t.at[x, "importance"] or 0) for x in ids)
            rows.append({"bundle_id": bid, "imp_a1": imp(A1, i1), "imp_a2": imp(A2, i2),
                         "pol_a1": A1.at[i1[0], "polarity"], "pol_a2": A2.at[i2[0], "polarity"],
                         "risk_a1": max(A1.at[x, "risk_flag"] == "1" for x in i1),
                         "risk_a2": max(A2.at[x, "risk_flag"] == "1" for x in i2)})
    pb, lab = pd.DataFrame(per_bundle), pd.DataFrame(rows)
    tot = pb[["kept", "both", "a1_only", "a2_only"]].sum()
    rep = {
        "bundles_two_annotators": len(pb), "bundles_excluded_single_annotator": sorted(set(cs.bundle_id) - two),
        "experiences_kept": int(tot.kept),
        "found_by_both": int(tot.both), "found_by_A1_only": int(tot.a1_only), "found_by_A2_only": int(tot.a2_only),
        "overlap_share": float(tot.both / tot.kept),
        "A1_coverage": float((tot.both + tot.a1_only) / tot.kept),
        "A2_coverage": float((tot.both + tot.a2_only) / tot.kept),
        "importance_exact_agreement": float((lab.imp_a1 == lab.imp_a2).mean()),
        "importance_weighted_kappa": float(cohen_kappa_score(lab.imp_a1, lab.imp_a2, weights="linear")),
        "polarity_agreement": float((lab.pol_a1 == lab.pol_a2).mean()),
        "polarity_kappa": float(cohen_kappa_score(lab.pol_a1, lab.pol_a2)),
        "risk_agreement": float((lab.risk_a1 == lab.risk_a2).mean()),
        "n_label_pairs": len(lab),
        "note": "final inventory adjudicated by AI (no human sign-off) unless stated otherwise",
    }
    out = cfg.annotations_dir / "inventory" / "_agreement.json"
    out.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    pb.to_csv(cfg.annotations_dir / "inventory" / "_agreement_by_bundle.csv", index=False)
    return rep


def shrink_pairs(cfg: Config, keep_products: int = 2, model: str | None = None) -> dict:
    """Shrink an exported pair sheet IN PLACE without changing item ids: keep every stratified row and
    the exhaustive rows of the first `keep_products` exhaustive products (same order as export),
    optionally only claims from one generator. Labels already entered are preserved; originals are
    backed up. Returned sheets based on the old, larger version are still accepted by unpack."""
    import shutil as _sh
    d = _dir(cfg, "pair")
    key = pd.read_csv(d / "_key.csv", dtype=str, keep_default_na=False)
    ex = key[key.stratum == "exhaustive"]
    order = sorted(ex.bundle_id.unique(), key=lambda b: short_hash(cfg.seed, "exh", b))
    keep_b = set(order[:keep_products])
    ok = ex.bundle_id.isin(keep_b)
    if model:
        cl = pd.read_parquet(cfg.out_dir / "summary_claims.parquet")[["claim_id", "summary_id"]]
        sm = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))[["summary_id", "model_id"]]
        model_claims = set(cl.merge(sm, on="summary_id").query("model_id == @model").claim_id)
        ok &= ex.hyp_id.isin(model_claims)
    keep_items = set(key[key.stratum != "exhaustive"].item_id) | set(ex[ok].item_id)
    backup = cfg.annotations_dir / "_backup" / ("pair_before_shrink_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    backup.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.csv"):
        _sh.copy(f, backup / f.name)
        t = pd.read_csv(f, dtype=str, keep_default_na=False)
        t[t.item_id.isin(keep_items)].to_csv(f, index=False, encoding="utf-8-sig")
    meta = json.loads((d / "_meta.json").read_text(encoding="utf-8"))
    meta.update(n_items=len(keep_items), shrunk={"keep_products": keep_products, "model": model,
                                                 "from_items": len(key), "backup": str(backup)})
    (d / "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"before": len(key), "after": len(keep_items), "exhaustive_products": sorted(keep_b), "backup": str(backup)}


def reset_labels(cfg: Config, kind: str) -> dict:
    """Back up the current A1/A2 sheets of `kind` and blank every label column (and note),
    e.g. when sheets were pre-filled by a model and must be labeled by humans. Item ids and texts
    are unchanged, so the exported key stays valid."""
    import shutil as _sh
    d = _dir(cfg, kind)
    backup = cfg.annotations_dir / "_backup" / f"{kind}_before_reset_{dt.datetime.now():%Y%m%d_%H%M%S}"
    backup.mkdir(parents=True, exist_ok=True)
    cols = LABEL_COLS[kind] + (["evidence_span"] if kind == "disclosure" else []) + ["note"]
    done = {}
    for a in cfg["annotation"]["annotators"]:
        f = d / f"{a}.csv"
        _sh.copy(f, backup / f.name)
        t = pd.read_csv(f, dtype=str, keep_default_na=False)
        filled = int((t[LABEL_COLS[kind][0]] != "").sum())
        for c in cols:
            if c in t:
                t[c] = ""
        t.to_csv(f, index=False, encoding="utf-8-sig")
        done[a] = filled
    for extra in d.glob("AI_ANNOTATION_NOTICE*.txt"):
        _sh.move(str(extra), backup / extra.name)
    return {"kind": kind, "cleared_labels": done, "backup": str(backup)}
