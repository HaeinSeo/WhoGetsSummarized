"""Candidate features for planners. Group signals (D/U, verified, reviewer id) are excluded."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import data
from ..config import Config
from ..evidence import edges_path
from ..extract import review_string
from ..text import specificity, word_count

NUMERIC = ["support_reviews", "support_users", "oppose_users", "uncertain_reviews",
           "support_share_bundle", "support_share_aspect", "uncertain_ratio", "specificity",
           "condition_present", "est_words", "mention_users"]


def candidate_table(cfg: Config, judge: str, scope: str, kind: str = "candidate") -> pd.DataFrame:
    """One row per hypothesis (candidate or probe) with verified evidence counts."""
    from ..retrieval import hypotheses
    hyps = hypotheses(cfg, kind)
    edges = pd.read_parquet(edges_path(cfg, judge, kind, scope))
    rv = data.reviews(cfg)
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    cands = pd.read_parquet(cfg.out_dir / "input_candidates.parquet") if kind == "candidate" else None
    meta = cands.set_index("cand_id") if cands is not None else None
    rows = []
    for h in hyps.itertuples():
        e = edges[(edges.hyp_id == h.hyp_id) & edges.judged]
        sup = e[e.relation == "SUPPORT"].review_id.tolist()
        opp = e[e.relation == "OPPOSE"].review_id.tolist()
        unc = e[e.relation == "UNCERTAIN"].review_id.tolist()
        b = bundles[h.bundle_id]
        n_users = len({rv.at[r, "user_id"] for r in b["review_ids_ordered"]})
        su = {rv.at[r, "user_id"] for r in sup}
        row = {"hyp_id": h.hyp_id, "bundle_id": h.bundle_id, "proposition": h.text,
               "condition": h.condition, "support_ids": sup, "oppose_ids": opp,
               "support_reviews": len(sup), "support_users": len(su),
               "oppose_users": len({rv.at[r, "user_id"] for r in opp}), "uncertain_reviews": len(unc),
               "support_share_bundle": len(su) / n_users if n_users else 0.0,
               "uncertain_ratio": len(unc) / len(e) if len(e) else 0.0,
               "specificity": specificity(h.text, h.condition),
               "condition_present": float(bool(h.condition)),
               "est_words": word_count(h.text) + word_count(h.condition) + 5}
        if meta is not None:
            m = meta.loc[h.hyp_id]
            row.update(aspect=m["aspect"], polarity=m["polarity"], quote=m["quote"],
                       mention_users=len(m["mention_users"]), mention_reviews=list(m["mention_reviews"]))
        else:
            row.update(aspect="", polarity="", quote="", mention_users=0, mention_reviews=[])
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df):
        asp = df.groupby(["bundle_id", "aspect"])["support_users"].transform("sum")
        df["support_share_aspect"] = np.where(asp > 0, df["support_users"] / asp.clip(lower=1), 0.0)
        # quote only if it comes from a verified supporting review
        df["quote"] = [q if any(r in set(s) for r in mr) else ""
                       for q, s, mr in zip(df["quote"], df["support_ids"], df["mention_reviews"])]
    return df


def embed_features(cfg: Config, table: pd.DataFrame, emb) -> np.ndarray:
    """[candidate emb | support-pool mean | oppose-pool mean | numeric] (numeric log-scaled)."""
    rv = data.reviews(cfg)
    def pool(ids):
        if not ids:
            return np.zeros(emb.model.config.hidden_size, np.float32)
        texts = [review_string({"title": rv.at[r, "title_clean"], "text": rv.at[r, "text_clean"]})
                 for r in ids]
        return emb.encode(texts).mean(0)
    c = emb.encode([f"{a.replace('_', ' ')}: {p} {c}".strip() for a, p, c in
                    zip(table["aspect"], table["proposition"], table["condition"])])
    sp = np.stack([pool(list(x)) for x in table["support_ids"]])
    op = np.stack([pool(list(x)) for x in table["oppose_ids"]])
    num = np.log1p(table[NUMERIC].astype(float).clip(lower=0).to_numpy())
    return np.concatenate([c, sp, op, num], axis=1).astype(np.float32)
