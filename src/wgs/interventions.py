"""P6: controlled input changes using REAL reviews of the same product.

target_preserved : keep every review with target-aspect evidence; swap unrelated reviews.
target_removed   : drop all reviews supporting the target (+ paraphrase guard); keep opposing ones.
aspect_removed   : drop support, opposition, and any mention of the target aspect.
du_*             : disclosure-cohort resampling (no_disclosure_subset, size_matched_replacement,
                   ratio sweep only where enough unique real D/U reviews exist).
All pairs start with human_validated=False; only validated pairs enter training/evaluation.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import data, prompts
from .config import Config
from .extract import review_string
from .planner.features import candidate_table
from .sampling import base_prompt_tokens, eligible_reviews
from .storage import log_step, short_hash, write_jsonl
from .tokens import TokenCounter

UNRELATED_SIM = 0.50    # replacement / swapped reviews must be below this similarity to the target
PARAPHRASE_SIM = 0.62   # reviews above this are treated as possible target evidence and removed


class _Ctx:
    def __init__(self, cfg: Config):
        from .retrieval import Embedder
        self.cfg, self.emb = cfg, Embedder(cfg)
        self.rv = data.reviews(cfg)
        self.counter = TokenCounter(cfg["bundle"].get("tokenizer"))
        self.base = base_prompt_tokens(self.counter)
        b = cfg["bundle"]
        self.limit = int(b["common_input_token_limit"]) / float(b["cross_tokenizer_margin"])
        exps = data.read_parquet_or_empty(cfg.out_dir / "review_experiences.parquet")
        self.aspects = exps[exps.valid].groupby("review_id")["aspect"].apply(set).to_dict() if len(exps) else {}

    def text(self, rid):
        r = self.rv.loc[rid]
        return review_string({"title": r["title_clean"], "text": r["text_clean"]})

    def sims(self, target: str, rids: list[str]) -> dict:
        if not rids:
            return {}
        q = self.emb.encode([target], query=True)[0]
        return dict(zip(rids, self.emb.encode([self.text(r) for r in rids]) @ q))

    def tokens(self, rids) -> int:
        return self.base + sum(self.counter.count(prompts.render_reviews(
            [{"anon_id": "R00", "title": self.rv.at[r, "title_clean"], "text": self.rv.at[r, "text_clean"]}])) + 1
            for r in rids)

    def pool(self, b: dict) -> pd.DataFrame:
        g = self.rv[(self.rv.product_id == b["product_id"])]
        g = g[eligible_reviews(g) & ~g.review_id.isin(b["review_ids_ordered"])]
        return g


def _len_bin(n):
    return 0 if n < 30 else 1 if n < 100 else 2


def _replace(ctx: _Ctx, b: dict, out_ids: list[str], candidates: pd.DataFrame, seed_salt: str):
    """Replace out_ids position-wise with matched pool reviews (rating band, then length bin)."""
    rv = ctx.rv
    if candidates.empty:
        return None, {}
    cand = candidates.assign(h=[short_hash(seed_salt, r) for r in candidates.review_id]).sort_values("h")
    used, mapping, matched = set(), {}, 0
    for rid in out_ids:
        band, lb = rv.at[rid, "rating_band"], _len_bin(rv.at[rid, "word_count"])
        pick = None
        for cond in ((cand.rating_band == band) & (cand.word_count.map(_len_bin) == lb),
                     cand.rating_band == band, pd.Series(True, index=cand.index)):
            c = cand[cond & ~cand.review_id.isin(used)]
            if len(c):
                pick = c.review_id.iloc[0]
                matched += int(rv.at[pick, "rating_band"] == band)
                break
        if pick is None:
            return None, {}
        used.add(pick)
        mapping[rid] = pick
    new = [mapping.get(r, r) for r in b["review_ids_ordered"]]
    if ctx.tokens(new) > ctx.limit:
        return None, {}
    return new, {"rating_band_matched": matched, "replaced": len(out_ids),
                 "token_delta": ctx.tokens(new) - ctx.tokens(b["review_ids_ordered"])}


def _bundle_from(b: dict, new_ids: list[str], variant: str, rv) -> dict:
    nb = dict(b)
    nb.update(review_ids_ordered=new_ids, anon_ids=[f"R{i + 1:02d}" for i in range(len(new_ids))],
              included_spans=[[0, len(rv.at[r, "text_clean"])] for r in new_ids], n_reviews=len(new_ids),
              variant=variant, parent_bundle_id=b["bundle_id"],
              bundle_id="b" + short_hash(b["bundle_id"], variant, ",".join(new_ids), n=15))
    return nb


def _target_pairs(ctx: _Ctx, b: dict, t, max_swap: int) -> list[tuple[dict, dict]]:
    rv, ids = ctx.rv, b["review_ids_ordered"]
    target = f"{t.proposition} {t.condition}".strip()
    sim_in = ctx.sims(target, ids)
    aspect_of = lambda r: ctx.aspects.get(r, set())
    sup, opp = set(t.support_ids), set(t.oppose_ids)
    pool = ctx.pool(b)
    sim_pool = ctx.sims(target, pool.review_id.tolist())
    keep = pd.Series([sim_pool[r] < UNRELATED_SIM and t.aspect not in aspect_of(r) for r in pool.review_id],
                     index=pool.index, dtype=bool)
    clean_pool = pool[keep]
    out = []
    base = {"target_cand_id": t.hyp_id, "target_text": t.proposition, "target_condition": t.condition,
            "target_aspect": t.aspect, "target_polarity": t.polarity, "original_bundle": b["bundle_id"],
            "human_validated": False}
    # 1) target_preserved: swap reviews unrelated to the target aspect
    unrelated = [r for r in ids if r not in sup | opp and t.aspect not in aspect_of(r)
                 and sim_in[r] < UNRELATED_SIM]
    k = min(len(unrelated), max_swap)
    if k >= 2:
        swap = sorted(unrelated, key=lambda r: short_hash("swap", t.hyp_id, r))[:k]
        new, mv = _replace(ctx, b, swap, clean_pool, "pres" + t.hyp_id)
        if new:
            nb = _bundle_from(b, new, f"target_preserved:{t.hyp_id}", rv)
            out.append((nb, {**base, "type": "target_preserved", "modified_bundle": nb["bundle_id"],
                             "removed_ids": swap, "added_ids": [r for r in new if r not in ids],
                             "matched_variables": mv}))
    # 2) target_removed (keep opposing) and 3) aspect_removed
    para = {r for r in ids if sim_in[r] >= PARAPHRASE_SIM and r not in opp}
    for typ, drop in (("target_removed", (sup | para) - opp),
                      ("aspect_removed", sup | opp | para | {r for r in ids if t.aspect in aspect_of(r)})):
        drop = [r for r in ids if r in drop]
        if not drop or len(drop) > len(ids) // 2:
            continue
        new, mv = _replace(ctx, b, drop, clean_pool, typ + t.hyp_id)
        if new:
            nb = _bundle_from(b, new, f"{typ}:{t.hyp_id}", rv)
            out.append((nb, {**base, "type": typ, "modified_bundle": nb["bundle_id"], "removed_ids": drop,
                             "added_ids": [r for r in new if r not in ids], "matched_variables": mv}))
    return out


def _du_pairs(ctx: _Ctx, b: dict, ratios=(0.0, 0.1, 0.2, 0.4)) -> list[tuple[dict, dict]]:
    rv, ids = ctx.rv, b["review_ids_ordered"]
    lab = {r: rv.at[r, "rule_label"] for r in ids}
    d_in = [r for r in ids if lab[r] == "D"]
    out, base = [], {"original_bundle": b["bundle_id"], "human_validated": False, "target_cand_id": None,
                     "target_text": None, "target_condition": None, "target_aspect": None,
                     "target_polarity": None}
    if not d_in:
        return out
    keep = [r for r in ids if lab[r] != "D"]
    nb = _bundle_from(b, keep, "du_no_disclosure_subset", rv)
    out.append((nb, {**base, "type": "du_no_disclosure_subset", "modified_bundle": nb["bundle_id"],
                     "removed_ids": d_in, "added_ids": [], "matched_variables": {}}))
    pool = ctx.pool(b)
    new, mv = _replace(ctx, b, d_in, pool[pool.rule_label == "U"], "du_size")
    if new:
        nb = _bundle_from(b, new, "du_size_matched_replacement", rv)
        out.append((nb, {**base, "type": "du_size_matched_replacement", "modified_bundle": nb["bundle_id"],
                         "removed_ids": d_in, "added_ids": [r for r in new if r not in ids],
                         "matched_variables": mv}))
    sub = rv[rv.product_id == b["product_id"]]
    allp = sub[eligible_reviews(sub)]
    D = sorted(allp[allp.rule_label == "D"].review_id, key=lambda r: short_hash("D", r))
    U = sorted(allp[allp.rule_label == "U"].review_id, key=lambda r: short_hash("U", r))
    n = len(ids)
    for ratio in ratios:
        nd = int(round(ratio * n))
        if nd > len(D) or n - nd > len(U):
            continue                          # infeasible with real unique reviews -> skipped
        new = sorted(D[:nd] + U[:n - nd], key=lambda r: short_hash("ord", ratio, r))
        if ctx.tokens(new) > ctx.limit:
            continue
        nb = _bundle_from(b, new, f"du_ratio_{ratio:.2f}", rv)
        out.append((nb, {**base, "type": f"du_ratio_{ratio:.2f}", "modified_bundle": nb["bundle_id"],
                         "removed_ids": [r for r in ids if r not in new], "added_ids": [r for r in new if r not in ids],
                         "matched_variables": {"n_D": nd, "n_U": n - nd}}))
    return out


def build(cfg: Config, judge: str, scope: str, cohorts=None, targets_per_bundle: int = 2,
          max_swap: int = 8) -> int:
    started = dt.datetime.now()
    ctx = _Ctx(cfg)
    table = candidate_table(cfg, judge, scope)
    pairs, nbundles = [], []
    for b in data.bundles(cfg, None, cohorts):
        if b["kind"] == "disclosure":
            res = _du_pairs(ctx, b)
        else:
            t = table[(table.bundle_id == b["bundle_id"]) & (table.support_users >= 1) & (table.specificity >= 0.3)]
            t = t.assign(h=[short_hash(cfg.seed, x) for x in t.hyp_id]).sort_values(["polarity", "h"])
            picks = pd.concat([g.head(1) for _, g in t.groupby("polarity")]).head(targets_per_bundle) if len(t) else t
            res = [x for tt in picks.itertuples() for x in _target_pairs(ctx, b, tt, max_swap)]
        for nb, p in res:
            p["pair_id"] = "iv" + short_hash(p["original_bundle"], p["modified_bundle"], n=12)
            pairs.append(p)
            nbundles.append(nb)
    ctx.emb.save()
    write_jsonl(cfg.data_dir / "bundles_interventions.jsonl", nbundles)
    write_jsonl(cfg.data_dir / "intervention_pairs.jsonl", pairs)
    counts = pd.Series([p["type"] for p in pairs]).value_counts().to_dict() if pairs else {}
    log_step(cfg, "interventions", {"judge": judge, "scope": scope, "cohorts": cohorts}, counts, started)
    return len(pairs)
