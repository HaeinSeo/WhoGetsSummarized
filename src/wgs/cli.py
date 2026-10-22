"""`wgs` command line. Every step reads configs/<x>.yaml and writes under runs/<run_name>/."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import load_config


def _p(sub, name, help_):
    p = sub.add_parser(name, help=help_)
    p.add_argument("--config", default="configs/pilot.yaml")
    return p


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wgs", description="Who Gets Summarized? pipeline")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _p(sub, "check-env", "P0: check EDA inputs, Ollama models, GPU")
    _p(sub, "scan-disclosure", "P0: full-category rule scan for D/S candidates per product")
    _p(sub, "sample", "P0/P3: cohorts, review IDs, preprocessing, fixed bundles, splits")
    p = _p(sub, "generate", "P1/P5: summaries with a frozen generator")
    p.add_argument("--model", required=True)
    p.add_argument("--method", default="direct", choices=["direct", "coverage", "freqframe", "plan"])
    p.add_argument("--planner", help="plans.parquet planner name for --method plan")
    p.add_argument("--cohort", nargs="*"); p.add_argument("--split", nargs="*")
    p.add_argument("--limit", type=int); p.add_argument("--bundle-file")
    p = _p(sub, "extract-claims", "output side: atomic claims from summaries")
    p.add_argument("--limit", type=int)
    p.add_argument("--variant", default="original", help="'original' (default) or 'all'")
    p = _p(sub, "extract-experiences", "input side: experiences from each bundle review")
    p.add_argument("--cohort", nargs="*"); p.add_argument("--split", nargs="*")
    p.add_argument("--limit", type=int); p.add_argument("--bundle-file")
    p = _p(sub, "build-candidates", "merge review experiences into product-level candidates")
    p.add_argument("--cohort", nargs="*"); p.add_argument("--split", nargs="*")
    p = _p(sub, "retrieve", "BGE sentence-window retrieval inside each bundle")
    p.add_argument("--kind", default="claim", choices=["claim", "candidate", "probe"])
    p.add_argument("--model-id", help="override retriever (e.g. MiniLM sensitivity)")
    p = _p(sub, "judge", "P2: evidence judging (nimble | nli | llama)")
    p.add_argument("--judge", required=True); p.add_argument("--kind", default="claim",
                                                             choices=["claim", "candidate", "probe"])
    p.add_argument("--scope", default="topk:10", help="topk:K | all | gold")
    p.add_argument("--cohort", nargs="*"); p.add_argument("--limit", type=int)
    p = _p(sub, "metrics", "RR/RG/Amp/F + automatic supported-claim precision")
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="topk:10")
    p.add_argument("--group-col", default="rule_label", choices=["rule_label", "gold_label"])
    p = _p(sub, "annotate", "export blinded sheets / merge into gold")
    p.add_argument("action", choices=["export", "merge", "inventory-gold", "pack", "unpack", "shrink-pair", "reset"])
    p.add_argument("--keep-products", type=int, default=2)
    p.add_argument("--kinds", nargs="*", help="pack: only these sheet kinds, e.g. pair disclosure")
    p.add_argument("--model", help="shrink-pair: keep exhaustive rows of this generator only")
    p.add_argument("--annotator", help="pack/unpack: A1 or A2"); p.add_argument("--file", help="unpack: returned zip")
    p.add_argument("--exhaustive-model", help="pair: exhaustive subset uses only this generator's claims")
    p.add_argument("--source-dir", help="inventory-gold: folder with <bundle>_inventory_gold.csv")
    p.add_argument("--adjudicator", default="human", choices=["human", "ai"])
    p.add_argument("--annotator-type", default="human", choices=["human", "ai", "mixed"],
                   help="merge: who produced the A1/A2 labels (recorded with the gold)")
    p.add_argument("--human-signoff", action="store_true", help="inventory-gold: a human reviewed the adjudication")
    p.add_argument("--min-importance", type=int, default=2)
    p.add_argument("--kind", choices=["disclosure", "pair", "inventory", "summary", "selection",
                                      "intervention", "intervention_target"])
    p.add_argument("--n", type=int); p.add_argument("--exhaustive-products", type=int, default=0)
    p.add_argument("--cohort", nargs="*"); p.add_argument("--split", nargs="*")
    p = _p(sub, "eval-judges", "judges vs human pair gold"); p.add_argument("--judges", nargs="+", required=True)
    p = _p(sub, "eval-retrieval", "recall@K on exhaustive gold subset")
    p.add_argument("--model-ids", nargs="*", default=[None])
    p = _p(sub, "eval-auto-recall", "automatic recall of verified specific experiences (scale proxy)")
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="topk:10")
    p.add_argument("--per-bundle", type=int, default=10); p.add_argument("--baseline", default="direct")
    p = _p(sub, "inventory-agreement", "A1 vs A2 inventory agreement through the adjudication mapping")
    p.add_argument("--a1-dir", required=True); p.add_argument("--a2-dir", required=True)
    p.add_argument("--cross-check-dir", required=True)
    p = _p(sub, "eval-gold-recall", "important-experience recall on the human inventory (judge presence)")
    p.add_argument("--judge", required=True); p.add_argument("--min-importance", type=int, default=2)
    p.add_argument("--baseline", default="direct")
    p = _p(sub, "eval-presence", "validate judge presence decisions against human summary gold")
    p.add_argument("--judge", required=True)
    p = _p(sub, "eval-preservation", "important-experience recall + fidelity, paired bootstrap")
    p.add_argument("--baseline", default="direct")
    p = _p(sub, "eval-interventions", "joint coverage / retention / removal error")
    p.add_argument("--auto-judge", help="judge target presence automatically (all pairs) with this judge")
    p.add_argument("--only-validated", action="store_true")
    p = _p(sub, "regress", "stepwise D/U association models (M0-M2)")
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="all")
    p = _p(sub, "plan", "evidence plans (B3 frequency, ablation, or AEP checkpoint)")
    p.add_argument("--planner", required=True, choices=["frequency", "frequency_unverified", "agent", "aep"])
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="all")
    p.add_argument("--checkpoint"); p.add_argument("--name")
    p = _p(sub, "train-aep", "train AEP (requires human selection gold)")
    p.add_argument("--variant", default="full", choices=["local", "context", "full"])
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="all")
    p.add_argument("--lambda-inv", type=float); p.add_argument("--lambda-remove", type=float)
    p = _p(sub, "interventions", "P6: target-preserved / target-removed / D-U resampling bundles")
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="all")
    p.add_argument("--cohort", nargs="*"); p.add_argument("--targets-per-bundle", type=int, default=2)
    _p(sub, "eval-disclosure", "rule D/S/U/A candidates vs human disclosure gold")
    _p(sub, "report", "write runs/<run>/pilot_report.md")
    p = _p(sub, "agent2", "evidence agent v2: verify the B0 draft, add missing verified experiences, rewrite")
    p.add_argument("--model", required=True); p.add_argument("--judge", default="nli")
    p.add_argument("--k", type=int, required=True); p.add_argument("--split", nargs="+", required=True)
    p.add_argument("--cohort", nargs="*")
    p = _p(sub, "agent2-choose-k", "DEV ONLY: choose K for agent2 without using recall")
    p.add_argument("--model", required=True); p.add_argument("--judge", default="nli")
    p = _p(sub, "agent", "evidence agent: verify the plan:agent draft's claims with NLI and revise")
    p.add_argument("--model", required=True); p.add_argument("--judge", default="nli")
    p.add_argument("--cohort", nargs="*"); p.add_argument("--max-rounds", type=int, default=1)
    p = _p(sub, "eval-corrected", "human-calibrated (Rogan-Gladen) pair-level support rates")
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="all")
    p = _p(sub, "paper", "generate paper/generated/*.tex tables, number macros, status.md")
    p.add_argument("--judge", default="nli"); p.add_argument("--scope", default="all")
    p = _p(sub, "show-summaries", "export generated summaries to an Excel-friendly CSV for reading")
    p.add_argument("--split", nargs="*"); p.add_argument("--model"); p.add_argument("--method")
    p.add_argument("--include-inventory-products", action="store_true",
                   help="also show products under human inventory annotation (breaks blinding)")
    p = _p(sub, "figures", "paper figures (PDF with embedded fonts + PNG + CSV) into runs/<run>/figures/<font>/")
    p.add_argument("--font", default="clean", choices=["clean", "comic"])
    p.add_argument("--format", nargs="+", default=["pdf", "png"])
    p.add_argument("--judge", default="nimble")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    cfg = load_config(args.config)
    c = args.cmd
    if c == "check-env":
        from . import checkenv
        res = checkenv.run(cfg)
        for ch in res["checks"]:
            print(f"[{'OK ' if ch['ok'] else 'FAIL'}] {ch['check']}: {ch['detail']}")
        return 0 if res["ok"] else 1
    if c == "scan-disclosure":
        from .sampling import scan_disclosure
        print(scan_disclosure(cfg).to_string(index=False))
    elif c == "sample":
        from .sampling import sample
        print(sample(cfg).to_string(index=False))
    elif c == "generate":
        from .generate import generate
        if args.method == "plan" and not args.planner:
            sys.exit("--planner is required with --method plan")
        print("new summaries:", generate(cfg, args.model, args.method, args.planner, args.cohort,
                                         args.split, args.limit, args.bundle_file))
    elif c == "extract-claims":
        from .extract import extract_claims
        print("new:", extract_claims(cfg, args.limit, None if args.variant == "all" else args.variant))
    elif c == "extract-experiences":
        from .extract import extract_experiences
        print("new:", extract_experiences(cfg, args.cohort, args.split, args.limit, args.bundle_file))
    elif c == "build-candidates":
        from .retrieval import build_candidates
        print("candidates:", len(build_candidates(cfg, args.cohort, args.split)))
    elif c == "retrieve":
        from .retrieval import retrieve
        print("pairs:", len(retrieve(cfg, args.kind, args.model_id)))
    elif c == "judge":
        from .evidence import run_judge
        e = run_judge(cfg, args.judge, args.kind, args.scope, args.cohort, args.limit)
        print(e[e.judged].relation.value_counts().to_string())
    elif c == "metrics":
        from .metrics import summary_metrics
        m = summary_metrics(cfg, args.judge, args.scope, args.group_col)
        print(m.groupby(["model_id", "method"])[["supported_claim_precision_auto", "RR_D", "RR_U"]].mean())
    elif c == "annotate":
        from . import annotate as an
        if args.action == "reset":
            print(json.dumps(an.reset_labels(cfg, args.kind), indent=2))
        elif args.action == "shrink-pair":
            print(json.dumps(an.shrink_pairs(cfg, args.keep_products, args.model), indent=2))
        elif args.action == "pack":
            print(an.pack(cfg, args.annotator, args.kinds))
        elif args.action == "unpack":
            print(chr(10).join(an.unpack(cfg, args.annotator, args.file)))
        elif args.action == "inventory-gold":
            g = an.build_inventory_gold(cfg, args.source_dir, args.adjudicator, args.human_signoff)
            print("experiences:", len(g), "| human_agreed:", int(g.human_agreed.sum()),
                  "| importance 2:", int((g.importance == 2).sum()))
        elif args.action == "merge":
            print(json.dumps(an.merge(cfg, args.kind, args.annotator_type), indent=2))
        else:
            fn = {"disclosure": lambda: an.export_disclosure(cfg, args.n),
                  "pair": lambda: an.export_pairs(cfg, args.n, args.exhaustive_products, splits=args.split,
                                               exhaustive_model=args.exhaustive_model),
                  "inventory": lambda: an.export_inventory(cfg, args.n, args.cohort or None),
                  "summary": lambda: an.export_summary(cfg, args.min_importance, args.n),
                  "selection": lambda: an.export_selection(cfg, tuple(args.split or ("train", "dev"))),
                  "intervention": lambda: an.export_interventions(cfg, args.n),
                  "intervention_target": lambda: an.export_intervention_targets(cfg, args.n)}[args.kind]
            print("\n".join(fn()))
    elif c == "eval-judges":
        from .evaluate import eval_judges
        print(eval_judges(cfg, args.judges).query("slice == 'all_weighted'").to_string(index=False))
    elif c == "eval-retrieval":
        from .evaluate import retrieval_recall
        print(retrieval_recall(cfg, model_tags=args.model_ids).to_string(index=False))
    elif c == "eval-auto-recall":
        from .evaluate import auto_recall
        print(auto_recall(cfg, args.judge, args.scope, args.per_bundle, args.baseline).to_string(index=False))
    elif c == "inventory-agreement":
        from .annotate import inventory_agreement
        print(json.dumps(inventory_agreement(cfg, args.a1_dir, args.a2_dir, args.cross_check_dir), indent=2, ensure_ascii=False))
    elif c == "eval-gold-recall":
        from .evaluate import gold_recall
        print(gold_recall(cfg, args.judge, args.min_importance, args.baseline).to_string(index=False))
    elif c == "eval-presence":
        from .evaluate import presence_validation
        print(json.dumps(presence_validation(cfg, args.judge), indent=2))
    elif c == "eval-preservation":
        from .evaluate import preservation
        print(preservation(cfg, args.baseline).to_string(index=False))
    elif c == "eval-disclosure":
        from .evaluate import disclosure_rule_accuracy
        print(disclosure_rule_accuracy(cfg).to_string(index=False))
    elif c == "eval-interventions":
        from .evaluate import intervention_metrics, intervention_metrics_auto
        if args.auto_judge:
            print(intervention_metrics_auto(cfg, args.auto_judge, args.only_validated).to_string(index=False))
        else:
            print(intervention_metrics(cfg).to_string(index=False))
    elif c == "regress":
        from .evaluate import group_regression
        print(group_regression(cfg, args.judge, args.scope).to_string(index=False))
    elif c == "plan":
        from .planner import run as pr
        if args.planner == "aep":
            if not args.checkpoint:
                sys.exit("--checkpoint is required for --planner aep")
            df = pr.plan_aep(cfg, args.checkpoint, args.judge, args.scope, args.name)
        elif args.planner == "agent":
            df = pr.plan_agent(cfg, args.judge, args.scope)
        else:
            df = pr.plan_frequency(cfg, args.judge, args.scope, args.planner == "frequency")
        print("selected:", int(df.selected.sum()), "bundles:", df.bundle_id.nunique())
    elif c == "train-aep":
        from .planner.run import train_aep
        print(train_aep(cfg, args.judge, args.scope, args.variant, args.lambda_inv,
                        args.lambda_remove).tail(10).to_string(index=False))
    elif c == "interventions":
        from .interventions import build
        print("pairs:", build(cfg, args.judge, args.scope, args.cohort, args.targets_per_bundle))
    elif c == "figures":
        from .figures import make_all
        made = make_all(cfg, args.font, tuple(args.format), args.judge)
        print(chr(10).join(made) if made else "no figure inputs yet (run the eval-* steps first)")
    elif c == "show-summaries":
        import pandas as pd
        from .storage import read_jsonl
        sm = pd.DataFrame(read_jsonl(cfg.out_dir / "summaries.jsonl"))
        if args.split:
            sm = sm[sm.split.isin(args.split)]
        if args.model:
            sm = sm[sm.model_id == args.model]
        if args.method:
            sm = sm[sm.method == args.method]
        key = cfg.annotations_dir / "inventory" / "_key.csv"
        hidden = 0
        if key.exists() and not args.include_inventory_products:
            inv = set(pd.read_csv(key, dtype=str).bundle_id)
            hidden = int(sm.bundle_id.isin(inv).sum())
            sm = sm[~sm.bundle_id.isin(inv)]
        cols = ["split", "cohort", "product_id", "model_id", "method", "word_count", "text"]
        out = cfg.out_dir / "summaries_view.csv"
        sm[cols].sort_values(["product_id", "model_id", "method"]).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"{len(sm)} summaries -> {out}  (hidden: {hidden} from inventory-annotation products)")
    elif c == "eval-corrected":
        from .evaluate import corrected_support_rates
        print(corrected_support_rates(cfg, args.judge, args.scope).to_string(index=False))
    elif c == "paper":
        from .paper import build as build_paper
        print(chr(10).join(build_paper(cfg, args.judge, args.scope)))
    elif c == "agent":
        from .agent import run as run_agent
        print("new agent summaries:", run_agent(cfg, args.model, args.judge, args.cohort, args.max_rounds))
    elif c == "agent2":
        from .agent2 import run as run_agent2
        print("new agent2 summaries:", run_agent2(cfg, args.model, args.judge, args.k, args.split, args.cohort))
    elif c == "agent2-choose-k":
        from .agent2 import choose_k
        print(json.dumps(choose_k(cfg, args.judge, args.model), indent=2))
    elif c == "report":
        from .report import build
        print(build(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
