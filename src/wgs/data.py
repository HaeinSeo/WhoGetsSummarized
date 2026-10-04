"""Loaders for run artifacts."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from .config import Config
from .storage import read_jsonl


def bundles(cfg: Config, bundle_file: str | None = None, cohorts=None, splits=None) -> list[dict]:
    path = Path(bundle_file) if bundle_file else cfg.data_dir / "bundles.jsonl"
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"no bundles in {path}; run `wgs sample` first")
    if cohorts:
        rows = [b for b in rows if b["cohort"] in cohorts]
    if splits:
        rows = [b for b in rows if b["split"] in splits]
    return rows


def all_bundles(cfg: Config) -> list[dict]:
    out = read_jsonl(cfg.data_dir / "bundles.jsonl")
    out += read_jsonl(cfg.data_dir / "bundles_interventions.jsonl")
    return out


@lru_cache(maxsize=4)
def _reviews(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).set_index("review_id", drop=False)


def reviews(cfg: Config) -> pd.DataFrame:
    return _reviews(str(cfg.data_dir / "reviews.parquet"))


def bundle_reviews(cfg: Config, bundle: dict) -> list[dict]:
    """Reviews in presentation order with anon ids and the exact included span."""
    rv = reviews(cfg)
    out = []
    for rid, aid, (s, e) in zip(bundle["review_ids_ordered"], bundle["anon_ids"],
                                bundle["included_spans"]):
        r = rv.loc[rid]
        out.append({
            "review_id": rid, "anon_id": aid, "title": r["title_clean"],
            "text": r["text_clean"][s:e], "rating": r["rating"], "user_id": r["user_id"],
            "rule_label": r["rule_label"], "verified_purchase": r["verified_purchase"],
            "rating_band": r["rating_band"], "timestamp": r["timestamp"],
            "word_count": r["word_count"],
        })
    return out


def read_jsonl_cached(path: Path) -> list[dict]:
    return read_jsonl(Path(path))


def read_parquet_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if Path(path).exists() else pd.DataFrame()
