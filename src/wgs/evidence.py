"""P2: judge (hypothesis, review) pairs and build the evidence graph edges.

Pairs outside the judged scope are stored with judged=False and relation=None:
they are UNJUDGED, never treated as IRRELEVANT / represented=0.
"""
from __future__ import annotations

import datetime as dt
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from . import data
from .config import Config
from .extract import review_string
from .judges import aggregate_labels, make_judge
from .retrieval import hypotheses
from .storage import JsonlAppender, log_failure, log_step, read_jsonl, short_hash, write_parquet

log = logging.getLogger(__name__)


def edges_path(cfg: Config, judge: str, kind: str, scope: str):
    tag = scope.replace(":", "")
    return cfg.out_dir / f"evidence_edges_{judge}_{kind}_{tag}.parquet"


def gold_pair_keys(cfg: Config) -> set:
    k = cfg.annotations_dir / "pair" / "_key.csv"
    if not k.exists():
        raise FileNotFoundError("scope 'gold' needs annotations/<run>/pair/_key.csv (wgs annotate export --kind pair)")
    df = pd.read_csv(k, dtype=str)
    return set(zip(df.hyp_id, df.review_id))


def scope_mask(pairs: pd.DataFrame, scope: str, cfg: Config | None = None) -> pd.Series:
    if scope == "gold":
        keys = gold_pair_keys(cfg)
        return pd.Series([(a, b) in keys for a, b in zip(pairs.hyp_id, pairs.review_id)], index=pairs.index)
    if scope == "all":
        return pd.Series(True, index=pairs.index)
    if scope.startswith("topk:"):
        return pairs["rank"] <= int(scope.split(":")[1])
    raise ValueError("scope must be 'all', 'gold' or 'topk:<K>'")


def premises_for(src: str, top_windows, ev: dict) -> list[str]:
    if len(src) <= int(ev["long_review_chars"]):
        return [src]
    return [src[s:e] for s, e in list(top_windows)[: int(ev["windows_per_long_review"])]]


def run_judge(cfg: Config, judge_name: str, kind: str = "claim", scope: str = "topk:10",
              cohorts=None, limit_pairs: int | None = None) -> pd.DataFrame:
    started = dt.datetime.now()
    judge = make_judge(cfg, judge_name)
    ev = cfg["evidence"]
    pairs = pd.read_parquet(cfg.out_dir / f"retrieval_pairs_{kind}.parquet")
    bundles = {b["bundle_id"]: b for b in data.all_bundles(cfg)}
    if cohorts:
        pairs = pairs[pairs.bundle_id.map(lambda b: bundles[b]["cohort"] in cohorts)]
    hyps = hypotheses(cfg, kind).set_index("hyp_id")
    rv = data.reviews(cfg)
    todo = pairs[scope_mask(pairs, scope, cfg)]
    if limit_pairs:
        todo = todo.head(limit_pairs)
    jobs = []
    with JsonlAppender(cfg.out_dir / f"judgments_{judge_name}.jsonl", "key") as cache:
        for p in todo.itertuples():
            h = hyps.loc[p.hyp_id]
            r = rv.loc[p.review_id]
            src = review_string({"title": r["title_clean"], "text": r["text_clean"]})
            prem = premises_for(src, p.top_windows, ev)
            key = short_hash(judge.version, p.hyp_id, p.review_id, h["text"], h["condition"], *prem)
            if key not in cache:
                jobs.append((key, p.hyp_id, p.review_id, h["text"], h["condition"], prem))
        # Ollama judges have a fixed per-request latency; concurrent requests raise throughput.
        workers = int(ev.get("concurrency", 1)) if getattr(judge, "parallel_ok", False) else 1

        def run(job):
            key, hyp_id, review_id, text, cond, prem = job
            res = judge.judge([{"experience": text, "condition": cond, "review": x} for x in prem])
            labels = [x["label"] for x in res]
            return {"key": key, "judge_version": judge.version, "hyp_id": hyp_id,
                    "review_id": review_id, "n_windows": len(prem), "window_labels": labels,
                    "relation": aggregate_labels(labels),
                    "probs": res[0]["probs"] if len(res) == 1 else None,
                    "abstained": any(x["abstained"] for x in res),
                    "latency_s": sum(x["latency_s"] for x in res)}

        n_fail, first_err = 0, None
        if getattr(judge, "batch_ok", False) and jobs:
            # in-process judges (NLI): pack many pairs into one GPU batch
            chunk = 256
            for i in tqdm(range(0, len(jobs), chunk), desc=f"judge {judge_name} {kind} batched"):
                part = jobs[i:i + chunk]
                items, owner = [], []
                for k, (_, _, _, text, cond, prem) in enumerate(part):
                    for x in prem:
                        items.append({"experience": text, "condition": cond, "review": x})
                        owner.append(k)
                res = judge.judge(items)
                for k, (key, hyp_id, review_id, _, _, prem) in enumerate(part):
                    rk = [r for r, o in zip(res, owner) if o == k]
                    labels = [x["label"] for x in rk]
                    cache.write({"key": key, "judge_version": judge.version, "hyp_id": hyp_id,
                                 "review_id": review_id, "n_windows": len(prem), "window_labels": labels,
                                 "relation": aggregate_labels(labels),
                                 "probs": rk[0]["probs"] if len(rk) == 1 else None,
                                 "abstained": any(x["abstained"] for x in rk),
                                 "latency_s": sum(x["latency_s"] for x in rk)})
            jobs = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run, j): j for j in jobs}
            for f in tqdm(as_completed(futs), total=len(futs), desc=f"judge {judge_name} {kind} x{workers}"):
                job = futs[f]
                try:
                    cache.write(f.result())
                except Exception as e:
                    log_failure(cfg, f"judge-{judge_name}", f"{job[1]}|{job[2]}", repr(e))
                    n_fail += 1
                    first_err = first_err or repr(e)
                    if "schema" in str(e) or "not installed" in str(e) or "no longer supported" in str(e):
                        raise
        if jobs and n_fail == len(jobs):  # never write an all-unjudged edge file silently
            raise RuntimeError(f"all {n_fail} {judge_name} judgments failed; first error: {first_err}")
    edges = build_edges(cfg, judge_name, kind, scope, pairs, judge.version)
    log_step(cfg, f"judge-{judge_name}-{kind}", {"scope": scope, "version": judge.version,
                                                 "concurrency": workers},
             {"pairs_in_scope": len(todo), "edges": len(edges),
              "judged": int(edges["judged"].sum())}, started)
    return edges


def build_edges(cfg: Config, judge_name: str, kind: str, scope: str, pairs: pd.DataFrame,
                version: str) -> pd.DataFrame:
    got = {}
    for j in read_jsonl(cfg.out_dir / f"judgments_{judge_name}.jsonl"):
        if j["judge_version"] == version:
            got[(j["hyp_id"], j["review_id"])] = j
    rows = []
    for p, sc in zip(pairs.itertuples(), scope_mask(pairs, scope, cfg)):
        j = got.get((p.hyp_id, p.review_id)) if sc else None
        rows.append({"hyp_id": p.hyp_id, "kind": kind, "bundle_id": p.bundle_id,
                     "review_id": p.review_id, "rank": p.rank, "score": p.score,
                     "judged": j is not None, "relation": j["relation"] if j else None,
                     "abstained": j["abstained"] if j else None,
                     "probs": str(j["probs"]) if j and j["probs"] else None,
                     "latency_s": j["latency_s"] if j else None,
                     "judge": judge_name, "judge_version": version, "scope": scope})
    df = pd.DataFrame(rows)
    write_parquet(df, edges_path(cfg, judge_name, kind, scope))
    return df
