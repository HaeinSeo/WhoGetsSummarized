import json

import numpy as np
import pandas as pd
import pytest

from wgs.config import Config

pytest.importorskip("matplotlib")


def fake_run(tmp_path):
    raw = {"run_name": "fig", "seed": 1, "paths": {"runs_dir": str(tmp_path), "annotations_dir": str(tmp_path / "a")}}
    cfg = Config(raw=raw, path=tmp_path / "c.yaml")
    o = cfg.out_dir
    meths = ["coverage", "freqframe", "plan:frequency", "plan:frequency_unverified"]
    rows = [{"metric": m, "a": "direct", "b": b, "n_units": 100, "diff": d, "ci_low": d - .05, "ci_high": d + .05}
            for m in ("recall_auto", "recall_auto_minority") for b, d in zip(meths, (.02, .04, .12, .07))]
    pd.DataFrame(rows).to_csv(o / "auto_recall_paired_bootstrap_nimble.csv", index=False)
    iv = [{"type": t, "model_id": mo, "method": me, "target_polarity": p, "n_pairs": 40,
           "joint_coverage": .5 + .1 * i, "removal_error": .3 - .05 * i}
          for t in ("target_preserved", "target_removed") for i, mo in enumerate(("qwen3:8b", "gemma4:12b"))
          for me in ("direct", "coverage") for p in ("positive", "negative")]
    pd.DataFrame(iv).to_csv(o / "intervention_metrics_auto_nimble.csv", index=False)
    rng = np.random.default_rng(0)
    m = pd.DataFrame([{"product_id": f"p{i}", "cohort": "test_disclosure", "variant": "original", "model_id": mo,
                       "method": me, "RR_D": rng.uniform(.4, .9), "RR_U": rng.uniform(.3, .7)}
                      for i in range(30) for mo in ("qwen3:8b", "gemma4:12b") for me in ("direct", "coverage")])
    m.to_parquet(o / "metrics_by_summary_nimble_all_rule_label.parquet")
    je = [{"judge": j, "slice": "all_weighted", "macro_f1": .7, "SUPPORT_precision": .9 - .05 * k,
           "SUPPORT_recall": .8, "OPPOSE_precision": .6, "OPPOSE_recall": .5} for k, j in enumerate(("nimble", "nli", "llama"))]
    pd.DataFrame(je).to_csv(o / "judge_eval.csv", index=False)
    (o / "judge_coverage_error.json").write_text(json.dumps(
        {j: [{"threshold": t, "coverage": 1 - t, "error": .3 - .25 * t} for t in (0, .3, .6, .9)] for j in ("nimble", "nli")}))
    return cfg


@pytest.mark.parametrize("font", ["clean", "comic"])
def test_figures_render_with_embedded_truetype(tmp_path, font):
    from wgs.figures import make_all
    cfg = fake_run(tmp_path)
    made = make_all(cfg, font, ("pdf", "png"))
    pdfs = [p for p in made if p.endswith(".pdf")]
    assert len(pdfs) == 4                       # recall, interventions, D/U, judges (human gold absent -> skipped)
    for p in pdfs:
        blob = open(p, "rb").read()
        assert b"/FontFile2" in blob and b"/Type3" not in blob
