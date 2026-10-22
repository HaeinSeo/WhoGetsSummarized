"""BGE retrieval over sentence windows within one product bundle; candidate merging."""
from __future__ import annotations

import datetime as dt
import logging
import pickle
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import data
from .config import Config
from .extract import review_string
from .storage import log_step, short_hash, text_hash, write_parquet
from .text import sentence_windows

log = logging.getLogger(__name__)


class Embedder:
    """CLS-pooled, L2-normalized BGE embeddings with a persistent text-hash cache."""

    def __init__(self, cfg: Config, model_id: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        r = cfg["retrieval"]
        self.model_id = model_id or r["model_id"]
        self.is_bge = "bge" in self.model_id.lower()
        # BGE: CLS pooling + query instruction. Others (e.g. MiniLM): mean pooling, no instruction.
        self.qi = r.get("query_instruction", "") if self.is_bge else ""
        self.device = r.get("device", "cuda") if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self.torch = torch
        self.cache_path = cfg.cache_dir / f"emb_{self.model_id.replace('/', '__')}.pkl"
        self.cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self.cache = pickle.load(f)

    def encode(self, texts: list[str], query: bool = False, batch: int = 64) -> np.ndarray:
        prefix = self.qi if query else ""
        keys = [text_hash(prefix + t) for t in texts]
        missing = [(k, prefix + t) for k, t in zip(keys, texts) if k not in self.cache]
        for i in range(0, len(missing), batch):
            chunk = missing[i:i + batch]
            enc = self.tok([t for _, t in chunk], padding=True, truncation=True, max_length=512,
                           return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                hid = self.model(**enc).last_hidden_state
                if self.is_bge:
                    cls = hid[:, 0]
                else:
                    m = enc["attention_mask"].unsqueeze(-1).to(hid.dtype)
                    cls = (hid * m).sum(1) / m.sum(1).clamp(min=1e-6)
                cls = self.torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
            for (k, _), v in zip(chunk, cls):
                self.cache[k] = v.astype(np.float32)
        if not keys:
            return np.zeros((0, self.model.config.hidden_size), np.float32)
        return np.stack([self.cache[k] for k in keys])

    def save(self) -> None:
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)


def review_windows(r: dict, size: int) -> list[tuple[int, int, str]]:
    src = review_string(r)
    return [(s, e, src[s:e]) for s, e in sentence_windows(src, size)] or [(0, len(src), src)]


def hypotheses(cfg: Config, kind: str) -> pd.DataFrame:
    """kind='claim' -> valid summary claims; 'candidate' -> merged input candidates;
    'probe' -> intervention targets on modified bundles."""
    if kind == "claim":
        cl = pd.read_parquet(cfg.out_dir / "summary_claims.parquet")
        cl = cl[cl["valid"]]
        sm = pd.DataFrame(data.read_jsonl_cached(cfg.out_dir / "summaries.jsonl"))[["summary_id", "bundle_id"]]
        cl = cl.merge(sm, on="summary_id")
        return pd.DataFrame({"hyp_id": cl["claim_id"], "bundle_id": cl["bundle_id"],
                             "text": cl["proposition"], "condition": cl["condition"].fillna(""),
                             "kind": "claim"})
    if kind == "probe":
        # frozen target probes: the original target experience judged against a MODIFIED bundle
        iv = pd.DataFrame(data.read_jsonl_cached(cfg.data_dir / "intervention_pairs.jsonl"))
        return pd.DataFrame({"hyp_id": iv["pair_id"], "bundle_id": iv["modified_bundle"],
                             "text": iv["target_text"], "condition": iv["target_condition"].fillna(""),
                             "kind": "probe"})
    cand = pd.read_parquet(cfg.out_dir / "input_candidates.parquet")
    return pd.DataFrame({"hyp_id": cand["cand_id"], "bundle_id": cand["bundle_id"],
                         "text": cand["proposition"], "condition": cand["condition"].fillna(""),
                         "kind": "candidate"})


def pairs_path(cfg: Config, kind: str, model_id: str | None = None):
    if not model_id or model_id == cfg["retrieval"]["model_id"]:
        return cfg.out_dir / f"retrieval_pairs_{kind}.parquet"
    return cfg.out_dir / f"retrieval_pairs_{kind}__{model_id.split('/')[-1]}.parquet"


def retrieve(cfg: Config, kind: str = "claim", model_id: str | None = None) -> pd.DataFrame:
    started = dt.datetime.now()
    emb = Embedder(cfg, model_id)
    size = int(cfg["retrieval"]["window_sentences"])
    hyps = hypotheses(cfg, kind)
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    rows = []
    for bid, hg in tqdm(hyps.groupby("bundle_id"), desc=f"retrieve {kind}"):
        revs = data.bundle_reviews(cfg, bundles[bid])
        win_meta, win_text = [], []
        for r in revs:
            for s, e, t in review_windows(r, size):
                win_meta.append((r["review_id"], s, e))
                win_text.append(t)
        W = emb.encode(win_text)
        Q = emb.encode([f"{t} {c}".strip() for t, c in zip(hg["text"], hg["condition"])], query=True)
        sims = Q @ W.T
        rid_arr = np.array([m[0] for m in win_meta])
        for qi, h in enumerate(hg.itertuples()):
            per_review: dict[str, list[int]] = {}
            for wi, rid in enumerate(rid_arr):
                per_review.setdefault(rid, []).append(wi)
            # review score = max window similarity; keep the top windows for judging long reviews
            scored = {rid: sorted(wis, key=lambda w: -sims[qi, w]) for rid, wis in per_review.items()}
            ranked = sorted(scored.items(), key=lambda kv: -sims[qi, kv[1][0]])
            for rank, (rid, wis) in enumerate(ranked, 1):
                rows.append({"hyp_id": h.hyp_id, "kind": kind, "bundle_id": bid, "review_id": rid,
                             "rank": rank, "score": float(sims[qi, wis[0]]),
                             "top_windows": [[win_meta[w][1], win_meta[w][2]] for w in wis[:3]]})
    emb.save()
    df = pd.DataFrame(rows)
    write_parquet(df, pairs_path(cfg, kind, model_id))
    log_step(cfg, f"retrieve-{kind}", {"model": emb.model_id, "window_sentences": size},
             {"hypotheses": len(hyps), "pairs": len(df)}, started)
    return df


def build_candidates(cfg: Config, cohorts=None, splits=None) -> pd.DataFrame:
    """Merge per-review extracted experiences into product-level candidate experiences.

    Greedy clustering inside one bundle: same polarity and cosine >= threshold to the cluster
    seed (embedding of aspect + proposition + condition). Distinct conditions/polarities stay apart.
    Mentions here are extraction-based; support is verified later by an evidence judge.
    """
    started = dt.datetime.now()
    emb = Embedder(cfg)
    thr = float(cfg["retrieval"]["candidate_merge_threshold"])
    exps = pd.read_parquet(cfg.out_dir / "review_experiences.parquet")
    exps = exps[exps["valid"]]
    rv = data.reviews(cfg)
    rows = []
    blist = [b for b in data.all_bundles(cfg)
             if (not cohorts or b["cohort"] in cohorts) and (not splits or b["split"] in splits)]
    for b in tqdm(blist, desc="build candidates"):
        rset = set(b["review_ids_ordered"])
        e = exps[exps.review_id.isin(rset)].copy()
        if e.empty:
            continue
        e["key_text"] = (e["aspect"].str.replace("_", " ") + ": " + e["proposition"]
                         + (" (" + e["condition"].fillna("") + ")").where(e["condition"].fillna("") != "", ""))
        e = e.assign(h=[short_hash(cfg.seed, x) for x in e["exp_id"]]).sort_values("h")
        V = emb.encode(e["key_text"].tolist())
        seeds, members = [], []
        for i, pol in enumerate(e["polarity"].tolist()):
            for c, s in enumerate(seeds):
                if e["polarity"].iloc[s] == pol and float(V[i] @ V[s]) >= thr:
                    members[c].append(i)
                    break
            else:
                seeds.append(i)
                members.append([i])
        for s, mem in zip(seeds, members):
            g = e.iloc[mem]
            centroid = V[mem].mean(0)
            medoid = mem[int(np.argmax(V[mem] @ centroid))]
            m = e.iloc[medoid]
            rids = sorted(set(g["review_id"]))
            spans = [x for x in g["source_span"] if isinstance(x, str) and x.strip()]
            rows.append({
                "cand_id": "k" + short_hash(b["bundle_id"], ",".join(sorted(g["exp_id"])), n=14),
                "bundle_id": b["bundle_id"], "product_id": b["product_id"], "cohort": b["cohort"],
                "split": b["split"],
                "aspect": Counter(g["aspect"]).most_common(1)[0][0],
                "proposition": m["proposition"], "condition": m["condition"] or "",
                "polarity": m["polarity"], "member_exp_ids": sorted(g["exp_id"]),
                "mention_reviews": rids,
                "mention_users": sorted({rv.at[r, "user_id"] for r in rids}),
                "quote": min(spans, key=len) if spans else "",
            })
    emb.save()
    df = pd.DataFrame(rows)
    cap = cfg["retrieval"].get("max_candidates_per_bundle")
    if cap and len(df):  # pool cap (chosen on dev): most-mentioned first, hash tie-break
        df["_n"] = df.mention_users.map(len)
        df["_h"] = [short_hash(cfg.seed, x) for x in df.cand_id]
        df = (df.sort_values(["bundle_id", "_n", "_h"], ascending=[True, False, True])
                .groupby("bundle_id").head(int(cap)).drop(columns=["_n", "_h"]).reset_index(drop=True))
    if df.empty:
        raise RuntimeError("no candidates built (are review experiences extracted for these bundles?)")
    prev = cfg.out_dir / "input_candidates.parquet"
    if prev.exists():  # building a subset (e.g. dev) must not drop other bundles' candidates
        old = pd.read_parquet(prev)
        df = pd.concat([old[~old.bundle_id.isin(set(df.bundle_id))], df], ignore_index=True)
    write_parquet(df, prev)
    log_step(cfg, "build-candidates", {"threshold": thr, "max_candidates_per_bundle": cap},
             {"candidates": len(df)}, started)
    return df
