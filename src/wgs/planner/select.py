"""Budgeted selection + plan writing shared by all planners (same candidates, same format)."""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from ..config import Config
from ..storage import log_step, short_hash, write_parquet

WORD_BUDGET = 110      # plan items must be expressible in a 100-120 word summary
DEDUP_COS = 0.90


def select(scores: np.ndarray, words: np.ndarray, emb: np.ndarray, max_k: int,
           eligible: np.ndarray, tiebreak: list[str]) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], tiebreak[i]))
    chosen, used = [], 0
    for i in order:
        if not eligible[i] or len(chosen) >= max_k:
            continue
        if used + words[i] > WORD_BUDGET:
            continue
        if any(float(emb[i] @ emb[j]) >= DEDUP_COS for j in chosen):
            continue
        chosen.append(i)
        used += words[i]
    return chosen


def write_plans(cfg: Config, planner: str, table: pd.DataFrame, scores: np.ndarray,
                cand_emb: np.ndarray, eligible: np.ndarray, r=None, q=None) -> pd.DataFrame:
    started = dt.datetime.now()
    max_k = int(cfg["planner"]["max_selected_experiences"])
    rows = []
    for bid, idx in table.groupby("bundle_id").indices.items():
        idx = np.asarray(idx)
        tb = [short_hash(cfg.seed, table.hyp_id.iloc[i]) for i in idx]
        chosen = select(scores[idx], table.est_words.to_numpy()[idx], cand_emb[idx], max_k,
                        eligible[idx], tb)
        rank = {c: k + 1 for k, c in enumerate(chosen)}
        for local, i in enumerate(idx):
            t = table.iloc[i]
            rows.append({"planner": planner, "bundle_id": bid, "cand_id": t.hyp_id,
                         "proposition": t.proposition, "condition": t.condition,
                         "aspect": t.aspect, "polarity": t.polarity,
                         "support_users": int(t.support_users), "oppose_users": int(t.oppose_users),
                         "evidence_ids": list(t.support_ids), "quote": t.quote,
                         "r": None if r is None else float(r[i]), "q": None if q is None else float(q[i]),
                         "score": float(scores[i]), "selected": local in rank,
                         "rank": rank.get(local, 0), "estimated_words": int(t.est_words)})
    new = pd.DataFrame(rows)
    path = cfg.out_dir / "plans.parquet"
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    if len(old):
        old = old[old.planner != planner]
    out = pd.concat([old, new], ignore_index=True)
    write_parquet(out, path)
    log_step(cfg, f"plan-{planner}", {"max_k": max_k, "word_budget": WORD_BUDGET},
             {"bundles": new.bundle_id.nunique(), "selected": int(new.selected.sum())}, started)
    return new


def frequency_scores(table: pd.DataFrame, verified: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """B3: rank by verified distinct supporting users (or raw extraction mentions if not verified)."""
    if verified:
        return table["support_users"].to_numpy(float), (table["support_users"] > 0).to_numpy()
    return table["mention_users"].to_numpy(float), (table["mention_users"] > 0).to_numpy()
