"""P0/P3: cohort sampling, review extraction, preprocessing, and fixed input bundles.

Natural, diagnostic, and disclosure cohorts are drawn separately and are product-disjoint.
Nothing here edits the source data; everything is written under runs/<run>/data_manifests/.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from collections import Counter, defaultdict

import duckdb
import numpy as np
import pandas as pd

from . import prompts
from .config import Config
from .disclosure import (DISCOUNT_RE, FREE_PRODUCT_RE, NEGATED_FREE_RE, NEGATED_VINE_RE,
                         SELF_PAID_RE, VINE_RE, disclosure_flags)
from .storage import log_step, short_hash, write_jsonl, write_parquet
from .text import clean_review_text, language_guess
from .tokens import TokenCounter

log = logging.getLogger(__name__)
OVERSAMPLE = 3          # candidates drawn per requested product (some fail bundle checks)
TEMPLATE_RESERVE = 220  # tokens reserved for frequency tables etc. in longer prompt variants


def _sql_list(items) -> str:
    return ",".join("'" + str(x).replace("'", "''") + "'" for x in items)


def _parquet(cfg: Config, category: str) -> str:
    p = cfg.source_review_parquet / f"{category}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing source parquet: {p}")
    return p.as_posix()


# --------------------------------------------------------------------------- disclosure scan
def scan_disclosure(cfg: Config) -> pd.DataFrame:
    """Full-category scan for rule-based disclosure candidates per product."""
    started = dt.datetime.now()
    con = duckdb.connect()
    frames = []
    d_expr = (f"((regexp_matches(t, '{VINE_RE}') AND NOT regexp_matches(t, '{NEGATED_VINE_RE}')) OR "
              f"((regexp_matches(t, '{FREE_PRODUCT_RE}') OR regexp_matches(t, '{DISCOUNT_RE}')) "
              f"AND NOT regexp_matches(t, '{NEGATED_FREE_RE}')))")
    s_expr = f"regexp_matches(t, '{SELF_PAID_RE}')"
    for cat in cfg["categories"]:
        log.info("scanning %s", cat)
        df = con.sql(f"""
            WITH base AS (
              SELECT parent_asin AS product_id, lower(text) AS t
              FROM read_parquet('{_parquet(cfg, cat)}')
              WHERE text IS NOT NULL AND length(trim(text)) > 0 AND parent_asin IS NOT NULL)
            SELECT '{cat}' AS category, product_id, count(*) AS n_valid,
                   count(*) FILTER (WHERE {d_expr}) AS n_d_candidates,
                   count(*) FILTER (WHERE {s_expr}) AS n_s_candidates
            FROM base GROUP BY product_id
            HAVING n_valid >= 20 AND n_d_candidates > 0
        """).df()
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["n_u_upper"] = out["n_valid"] - out["n_d_candidates"] - out["n_s_candidates"]
    path = write_parquet(out, cfg.data_dir / "product_disclosure_counts.parquet")
    summary = out.groupby("category").agg(
        products_with_d=("product_id", "size"),
        products_d_ge3=("n_d_candidates", lambda s: int((s >= 3).sum())),
        d_candidates=("n_d_candidates", "sum")).reset_index()
    summary.to_csv(cfg.data_dir / "product_disclosure_summary.csv", index=False)
    log_step(cfg, "scan-disclosure", {"categories": cfg["categories"]},
             {"products": len(out), "path": str(path)}, started)
    return summary


# --------------------------------------------------------------------------- product selection
def _candidate_products(cfg: Config, con, cohort: dict, taken: set[str]) -> pd.DataFrame:
    stats = (cfg.eda_dir / "product_review_statistics.parquet").as_posix()
    n = int(cohort["n_per_category"]) * OVERSAMPLE
    where = [f"s.category IN ({_sql_list(cfg['categories'])})", "s.product_id IS NOT NULL",
             f"s.valid_text_review_count >= {int(cohort['min_valid_reviews'])}"]
    join = ""
    if cohort["kind"] == "diagnostic":
        where += [f"s.low_rating_valid_text_count >= {int(cohort['min_low'])}",
                  f"s.high_rating_valid_text_count >= {int(cohort['min_high'])}"]
    elif cohort["kind"] == "disclosure":
        dpath = cfg.data_dir / "product_disclosure_counts.parquet"
        if not dpath.exists():
            log.warning("cohort %s skipped: run `wgs scan-disclosure` first", cohort["name"])
            return pd.DataFrame()
        join = f"JOIN read_parquet('{dpath.as_posix()}') d ON d.category = s.category AND d.product_id = s.product_id"
        where += [f"d.n_d_candidates >= {int(cohort['min_d_candidates'])}",
                  f"d.n_u_upper >= {int(cohort['min_u_reviews'])}"]
    salt = f"{cfg.seed}|{cohort['name']}|"
    df = con.sql(f"""
        SELECT s.category, s.product_id, s.valid_text_review_count, s.low_rating_valid_text_count,
               s.high_rating_valid_text_count, s.avg_rating, s.total_text_chars,
               s.has_low_and_high_ratings, s.product_title,
               md5('{salt}' || s.category || '|' || s.product_id) AS selection_hash
        FROM read_parquet('{stats}') s {join}
        WHERE {' AND '.join(where)}
        QUALIFY row_number() OVER (PARTITION BY s.category ORDER BY selection_hash) <= {n}
    """).df()
    df = df[~df["product_id"].isin(taken)]
    if df.empty:
        return df
    # product ids that occur in more than one category file are ambiguous -> excluded
    multi = con.sql(f"""
        SELECT product_id FROM read_parquet('{stats}')
        WHERE product_id IN ({_sql_list(df['product_id'].unique())})
        GROUP BY product_id HAVING count(DISTINCT category) > 1
    """).df()["product_id"]
    df["cross_category_duplicate"] = df["product_id"].isin(set(multi))
    df["cohort"], df["split"], df["kind"] = cohort["name"], cohort["split"], cohort["kind"]
    return df.sort_values(["category", "selection_hash"]).reset_index(drop=True)


# --------------------------------------------------------------------------- review extraction
def _extract_reviews(cfg: Config, con, products: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for cat, grp in products.groupby("category"):
        frames.append(con.sql(f"""
            SELECT category, parent_asin AS product_id, asin, user_id, rating, review_title,
                   text, timestamp, helpful_vote, verified_purchase, file_row_number AS parquet_row
            FROM read_parquet('{_parquet(cfg, cat)}', file_row_number = true)
            WHERE parent_asin IN ({_sql_list(grp['product_id'].unique())})
        """).df())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def preprocess_reviews(raw: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    df = raw.copy()
    df["review_id"] = [
        "r" + short_hash(c, p, u, t, x, n=19)
        for c, p, u, t, x in zip(df["category"], df["product_id"], df["user_id"],
                                 df["timestamp"], df["text"].fillna(""))
    ]
    # strict duplicates (same category/product/user/time/text) -> one record + multiplicity
    mult = df.groupby("review_id").size().rename("multiplicity")
    df = df.sort_values("parquet_row").drop_duplicates("review_id").merge(mult, on="review_id")
    df["text_clean"] = df["text"].map(clean_review_text)
    df["title_clean"] = df["review_title"].map(clean_review_text)
    df["valid_text"] = df["text_clean"].str.len() > 0
    df["rating_zero"] = df["rating"].fillna(0) <= 0
    df["lang_guess"] = df["text_clean"].map(lambda t: language_guess(
        t, pcfg["min_ascii_ratio"], pcfg["min_en_stopword_ratio"], pcfg["stopword_min_words"]))
    flags = pd.DataFrame([disclosure_flags(f"{a} {b}") for a, b in
                          zip(df["title_clean"], df["text_clean"])], index=df.index)
    df = pd.concat([df, flags], axis=1)
    df["rating_band"] = pd.cut(df["rating"].fillna(0), [-1, 0.5, 2.5, 3.5, 5.1],
                               labels=["zero", "low", "mid", "high"]).astype(str)
    df = df.sort_values(["product_id", "timestamp", "review_id"])
    df["user_review_index"] = df.groupby(["product_id", "user_id"]).cumcount()
    df["word_count"] = df["text_clean"].str.split().str.len().fillna(0).astype(int)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- bundles
def _hash_order(ids, seed, salt=""):
    return sorted(ids, key=lambda r: short_hash(seed, salt, r))


def _ordered_candidates(kind: str, g: pd.DataFrame, cohort: dict, bcfg: dict, seed: int) -> list:
    if kind == "natural":
        g = g.sort_values("timestamp")
        strata = np.array_split(g["review_id"].to_numpy(), max(1, int(bcfg["time_strata"])))
        queues = [_hash_order(list(s), seed, "nat") for s in strata]
        out = []
        while any(queues):
            for q in queues:
                if q:
                    out.append(q.pop(0))
        return out
    rest = _hash_order(g["review_id"].tolist(), seed, kind)
    first = []
    if kind == "diagnostic":
        lows = set(g.loc[g.rating_band == "low", "review_id"])
        highs = set(g.loc[g.rating_band == "high", "review_id"])
        first = ([r for r in rest if r in lows][: int(cohort["min_low"])]
                 + [r for r in rest if r in highs][: int(cohort["min_high"])])
    elif kind == "disclosure":
        ds = set(g.loc[g.rule_label == "D", "review_id"])
        cap = max(int(cohort["min_d_candidates"]), int(bcfg["target_reviews"]) // 3)
        first = [r for r in rest if r in ds][:cap]
    fs = set(first)
    return first + [r for r in rest if r not in fs]


def eligible_reviews(reviews: pd.DataFrame) -> pd.Series:
    return (reviews.valid_text & ~reviews.rating_zero
            & (reviews.lang_guess == "en_likely") & (reviews.user_review_index == 0))


def build_bundle(product: dict, reviews: pd.DataFrame, cohort: dict, bcfg: dict,
                 counter: TokenCounter, seed: int, base_tokens: int) -> tuple[dict | None, dict]:
    elig = reviews[eligible_reviews(reviews)].copy()
    excl = Counter()
    excl["invalid_or_blank"] = int((~reviews.valid_text).sum())
    excl["rating_zero"] = int((reviews.valid_text & reviews.rating_zero).sum())
    excl["non_english_heuristic"] = int((reviews.valid_text & (reviews.lang_guess != "en_likely")).sum())
    excl["repeat_user_review"] = int((reviews.user_review_index > 0).sum())
    elig["tokens"] = [counter.count(prompts.render_reviews([{"anon_id": "R00", "title": t, "text": x}])) + 1
                      for t, x in zip(elig["title_clean"], elig["text_clean"])]
    too_long = elig["tokens"] > int(bcfg["max_review_tokens"])
    excl["too_long"] = int(too_long.sum())
    elig = elig[~too_long]
    order = _ordered_candidates(cohort["kind"], elig, cohort, bcfg, seed)
    tok = dict(zip(elig["review_id"], elig["tokens"]))
    limit = int(bcfg["common_input_token_limit"]) / float(bcfg["cross_tokenizer_margin"])
    used, chosen = base_tokens, []
    for rid in order:
        if len(chosen) >= int(bcfg["target_reviews"]):
            break
        if used + tok[rid] > limit:
            excl["budget_skip"] += 1
            continue
        chosen.append(rid)
        used += tok[rid]
    info = {"eligible": len(elig), "chosen": len(chosen), **excl}
    if len(chosen) < int(bcfg["min_reviews"]):
        return None, {**info, "drop_reason": f"fewer_than_min_reviews({len(chosen)})"}
    presented = _hash_order(chosen, seed, "present")
    bundle_id = "b" + short_hash(product["category"], product["product_id"], cohort["name"],
                                 ",".join(presented), n=15)
    lens = dict(zip(elig["review_id"], elig["text_clean"].str.len()))
    return {
        "bundle_id": bundle_id,
        "product_id": product["product_id"],
        "category": product["category"],
        "cohort": cohort["name"],
        "kind": cohort["kind"],
        "split": cohort["split"],
        "review_ids_ordered": presented,
        "anon_ids": [f"R{i + 1:02d}" for i in range(len(presented))],
        "included_spans": [[0, int(lens[r])] for r in presented],
        "n_reviews": len(presented),
        "est_input_tokens": int(used),
        "tokenizer": counter.name,
        "tokenizer_exact": counter.exact,
        "order_seed": seed,
        "variant": "original",
        "parent_bundle_id": None,
        "exclusions": dict(excl),
    }, info


def base_prompt_tokens(counter: TokenCounter) -> int:
    worst = 0
    for name in prompts.SUMMARY_METHODS.values():
        filled = prompts.load(name)
        for ph in ("{reviews}", "{plan}", "{freq_table}"):
            filled = filled.replace(ph, "")
        worst = max(worst, counter.count(filled))
    return worst + TEMPLATE_RESERVE


def sample(cfg: Config) -> pd.DataFrame:
    started = dt.datetime.now()
    con = duckdb.connect()
    bcfg, seed = cfg["bundle"], cfg.seed
    counter = TokenCounter(bcfg.get("tokenizer"))
    base_tokens = base_prompt_tokens(counter)
    taken: set[str] = set()
    # dev/test separation: never reuse products selected in other runs (e.g. pilot, smoke)
    for run in cfg.get("exclude_products_from_runs") or []:
        prev = cfg.resolve(cfg["paths"]["runs_dir"]) / run / "data_manifests" / "products.parquet"
        if not prev.exists():
            raise FileNotFoundError(f"exclude_products_from_runs: {prev} missing; sample run '{run}' first")
        pdf = pd.read_parquet(prev)
        taken |= set(pdf.loc[pdf.status == "selected", "product_id"])
    n_excluded = len(taken)
    product_rows, review_frames, bundles, report = [], [], [], []
    for cohort in cfg["cohorts"]:
        cand = _candidate_products(cfg, con, cohort, taken)
        if cand.empty:
            report.append({"cohort": cohort["name"], "category": "*", "attempted": 0,
                           "accepted": 0, "drop_reasons": "no candidates / skipped"})
            continue
        n_dup = int(cand.cross_category_duplicate.sum())
        cand = cand[~cand.cross_category_duplicate]
        log.info("cohort %s: extracting reviews for %d candidate products (%d cross-category dups dropped)",
                 cohort["name"], len(cand), n_dup)
        rev = preprocess_reviews(_extract_reviews(cfg, con, cand), cfg["preprocess"])
        by_prod = dict(tuple(rev.groupby("product_id")))
        for cat, grp in cand.groupby("category"):
            accepted = attempted = 0
            reasons = Counter()
            for _, prod in grp.iterrows():
                if accepted >= int(cohort["n_per_category"]):
                    break
                attempted += 1
                prows = by_prod.get(prod.product_id)
                if prows is None or prows.empty:
                    bundle, info = None, {"drop_reason": "no_reviews_found"}
                else:
                    bundle, info = build_bundle(prod.to_dict(), prows, cohort, bcfg, counter,
                                                seed, base_tokens)
                row = {**prod.to_dict(), **info, "attempt_rank": attempted}
                if bundle is None:
                    row["status"] = "dropped"
                    reasons[info.get("drop_reason", "?")] += 1
                else:
                    row["status"], row["bundle_id"] = "selected", bundle["bundle_id"]
                    bundles.append(bundle)
                    review_frames.append(prows.assign(cohort=cohort["name"], split=cohort["split"]))
                    taken.add(prod.product_id)
                    accepted += 1
                product_rows.append(row)
            report.append({"cohort": cohort["name"], "category": cat, "attempted": attempted,
                           "accepted": accepted, "drop_reasons": json.dumps(dict(reasons))})
    products = pd.DataFrame(product_rows)
    reviews = pd.concat(review_frames, ignore_index=True) if review_frames else pd.DataFrame()
    in_bundle = {r for b in bundles for r in b["review_ids_ordered"]}
    if not reviews.empty:
        reviews["in_original_bundle"] = reviews["review_id"].isin(in_bundle)
    write_parquet(products, cfg.data_dir / "products.parquet")
    # raw `text` is dropped; text_clean is what models see (raw stays in source parquet + src_row)
    write_parquet(reviews.drop(columns=["text"], errors="ignore"), cfg.data_dir / "reviews.parquet")
    write_jsonl(cfg.data_dir / "bundles.jsonl", bundles)
    splits = defaultdict(list)
    for b in bundles:
        splits[b["split"]].append(b["product_id"])
    (cfg.data_dir / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    rep = pd.DataFrame(report)
    rep.to_csv(cfg.data_dir / "sampling_report.csv", index=False)
    log_step(cfg, "sample", {"tokenizer": counter.name, "base_prompt_tokens": base_tokens,
                             "excluded_products_from_other_runs": n_excluded},
             {"bundles": len(bundles), "products_attempted": len(products),
              "reviews_extracted": len(reviews)}, started)
    return rep
