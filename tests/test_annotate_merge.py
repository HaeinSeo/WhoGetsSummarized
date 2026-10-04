import pandas as pd

from wgs import annotate
from wgs.config import Config


def _cfg(tmp_path):
    raw = {"run_name": "t", "seed": 1, "design_version": "2.0",
           "paths": {"runs_dir": str(tmp_path / "runs"), "annotations_dir": str(tmp_path / "ann")},
           "annotation": {"annotators": ["A1", "A2"]}}
    return Config(raw=raw, path=tmp_path / "c.yaml")


def test_merge_agreement_adjudication_and_blanks(tmp_path):
    cfg = _cfg(tmp_path)
    d = cfg.annotations_dir / "pair"
    d.mkdir(parents=True)
    pd.DataFrame({"item_id": ["p1", "p2", "p3"], "hyp_id": ["h1", "h2", "h3"],
                  "review_id": ["r1", "r2", "r3"], "sampling_weight": ["1", "1", "1"]}).to_csv(d / "_key.csv", index=False)
    base = {"item_id": ["p1", "p2", "p3"], "experience": ["e"] * 3, "note": [""] * 3}
    pd.DataFrame({**base, "relation": ["SUPPORT", "OPPOSE", ""], "condition_match": ["Y", "Y", ""]}).to_csv(d / "A1.csv", index=False)
    pd.DataFrame({**base, "relation": ["SUPPORT", "IRRELEVANT", "SUPPORT"], "condition_match": ["Y", "NA", "Y"]}).to_csv(d / "A2.csv", index=False)
    rep = annotate.merge(cfg, "pair")
    assert rep["labeled_by_both"] == 2          # blank label (p3) never becomes gold
    gold = pd.read_parquet(cfg.annotations_dir / "pair_gold.parquet")
    assert list(gold.item_id) == ["p1"]         # disagreement waits for adjudication
    adj = pd.read_csv(d / "_adjudicated.csv", dtype=str).fillna("")
    adj.loc[adj.item_id == "p2", ["relation", "condition_match"]] = ["OPPOSE", "Y"]
    adj.to_csv(d / "_adjudicated.csv", index=False)
    annotate.merge(cfg, "pair")
    gold = pd.read_parquet(cfg.annotations_dir / "pair_gold.parquet").set_index("item_id")
    assert gold.at["p2", "relation"] == "OPPOSE" and gold.at["p2", "source"] == "adjudicated"
    assert gold.at["p2", "relation_A2"] == "IRRELEVANT"   # independent labels kept


def test_invalid_label_rejected(tmp_path):
    cfg = _cfg(tmp_path)
    d = cfg.annotations_dir / "pair"
    d.mkdir(parents=True)
    pd.DataFrame({"item_id": ["p1"], "hyp_id": ["h"], "review_id": ["r"]}).to_csv(d / "_key.csv", index=False)
    for a in ("A1", "A2"):
        pd.DataFrame({"item_id": ["p1"], "relation": ["support"], "condition_match": ["Y"]}).to_csv(d / f"{a}.csv", index=False)
    try:
        annotate.merge(cfg, "pair")
        assert False, "lowercase label must be rejected"
    except ValueError:
        pass
