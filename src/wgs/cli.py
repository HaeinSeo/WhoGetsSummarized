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
    p.add_argument("action", choices=["export", "merge", "inventory-gold"])
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
    p = _p(sub, "eval-preservation", "important-experience recall + fidelity, paired bootstrap")
    p.add_argument("--baseline", default="direct")
    p = _p(sub, "eval-interventions", "joint coverage / retention / removal error")
    p.add_argument("--auto-judge", help="judge target presence automatically (all pairs) with this judge")
    p.add_argument("--only-validated", action="store_true")
    p = _p(sub, "regress", "stepwise D/U association models (M0-M2)")
    p.add_argument("--judge", required=True); p.add_argument("--scope", default="all")
    p = _p(sub, "plan", "evidence plans (B3 frequency, ablation, or AEP checkpoint)")
    p.add_argument("--planner", required=True, choices=["frequency", "frequency_unverified", "aep"])
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
        if args.action == "inventory-gold":
            print("experiences:", len(an.build_inventory_gold(cfg)))
        elif args.action == "merge":
            print(json.dumps(an.merge(cfg, args.kind), indent=2))
        else:
            fn = {"disclosure": lambda: an.export_disclosure(cfg, args.n),
                  "pair": lambda: an.export_pairs(cfg, args.n, args.exhaustive_products, splits=args.split),
                  "inventory": lambda: an.export_inventory(cfg, args.n, args.cohort or None),
                  "summary": lambda: an.export_summary(cfg),
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
    elif c == "report":
        from .report import build
        print(build(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
