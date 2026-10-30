"""Export a small, text-free demo bundle for the public highlight demo (demo/).

For each selected TEST product: the B0 direct summary and the evidence agent v2 summary of one generator,
each with its claims, the character span of every claim, and how many reviews the all-pair NLI judge marks
as SUPPORT for it. Review IDs are the anonymised R01.. IDs of the bundle. No review text, reviewer IDs or
raw product metadata are written, so the output can sit in a public repository (see the data release policy).

Usage: python scripts/build_demo_data.py --run runs/www --model qwen3:8b --per-category 2 --out demo/data/products.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/www")
    ap.add_argument("--model", default="qwen3:8b")
    ap.add_argument("--agent-method", default=None, help="default: agent2_k<chosen K> from the dev-only choice file")
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--judge", default="nli")
    ap.add_argument("--out", default="demo/data/products.json")
    a = ap.parse_args()

    run = Path(a.run)
    outd = run / "outputs"
    bundles = {b["bundle_id"]: b for b in read_jsonl(run / "data_manifests" / "bundles.jsonl")}

    if a.agent_method is None:
        tag = a.model.replace(":", "_")
        k = json.loads((outd / f"agent2_choose_k_{tag}.json").read_text(encoding="utf-8"))["chosen_k"]
        if k is None:
            raise SystemExit(f"{a.model}: no K chosen on dev, agent not deployed for this generator")
        a.agent_method = f"agent2_k{k}"

    summaries = [s for s in read_jsonl(outd / "summaries.jsonl") if s["split"] == "test" and s["model_id"] == a.model]
    by_method = defaultdict(dict)  # bundle_id -> method -> summary
    for s in summaries:
        if s["method"] in ("direct", a.agent_method):
            by_method[s["bundle_id"]][s["method"]] = s

    # test_natural_core bundles that have both summaries
    cands = [b for b in bundles.values() if b["cohort"] == "test_natural_core"
             and {"direct", a.agent_method} <= set(by_method[b["bundle_id"]])]
    cands.sort(key=lambda b: (b["category"], b["product_id"]))
    chosen, per_cat = [], defaultdict(int)
    for b in cands:
        if per_cat[b["category"]] < a.per_category:
            chosen.append(b)
            per_cat[b["category"]] += 1
    if not chosen:
        raise SystemExit("no bundle has both B0 and agent summaries")

    summary_ids = {by_method[b["bundle_id"]][m]["summary_id"] for b in chosen for m in ("direct", a.agent_method)}
    claims = pd.read_parquet(outd / "summary_claims.parquet")
    claims = claims[claims.valid & claims.summary_id.isin(summary_ids)]
    claim_ids = set(claims.claim_id)
    edges = pd.read_parquet(outd / f"evidence_edges_{a.judge}_claim_all.parquet",
                            columns=["hyp_id", "review_id", "relation", "judged"])
    edges = edges[edges.judged & (edges.relation == "SUPPORT") & edges.hyp_id.isin(claim_ids)]
    support = edges.groupby("hyp_id").review_id.apply(set).to_dict()

    def render(s: dict, anon: dict[str, str]) -> dict:
        cl = claims[claims.summary_id == s["summary_id"]].sort_values("span_start")
        out = []
        for c in cl.itertuples():
            reviews = sorted(anon[r] for r in support.get(c.claim_id, set()) if r in anon)
            out.append({"start": int(c.span_start), "end": int(c.span_end), "n_support": len(reviews),
                        "reviews": reviews})
        return {"text": s["text"], "word_count": int(float(s["word_count"])), "claims": out}

    products = []
    for b in chosen:
        review_ids = b["review_ids_ordered"]
        anon_ids = b["anon_ids"]
        anon = dict(zip(review_ids, anon_ids))
        ms = by_method[b["bundle_id"]]
        products.append({
            "product_id": b["product_id"], "category": b["category"], "n_reviews": int(float(b["n_reviews"])),
            "b0": render(ms["direct"], anon),
            "agent": render(ms[a.agent_method], anon),
        })

    payload = {"model": a.model, "agent_method": a.agent_method, "judge": a.judge,
               "generated_by": "scripts/build_demo_data.py", "products": products}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(products)} products ({a.agent_method}, {a.model}) to {out}")


if __name__ == "__main__":
    main()
