import math

import pandas as pd

from wgs.disclosure import disclosure_flags
from wgs.judges.base import aggregate_labels
from wgs.metrics import importance_recall, representation
from wgs.planner.select import select
from wgs.text import locate_span, sentence_windows, strip_think


def test_disclosure_rules():
    assert disclosure_flags("I received this product for free in exchange for my honest review.")["rule_label"] == "D"
    assert disclosure_flags("Vine Customer Review of Free Product")["V_flag"]
    assert disclosure_flags("I bought this myself and paid full price.")["rule_label"] == "S"
    assert disclosure_flags("Great charger, fast free shipping")["rule_label"] == "U"
    assert disclosure_flags("Great charger, fast free shipping")["hard_negative"]
    amb = disclosure_flags("in exchange for an honest review... not free, I paid full price")
    assert amb["rule_label"] == "A"


def test_locate_span_exact_and_normalized():
    s = "Some users reported   overheating during long gaming sessions."
    assert locate_span(s, "overheating during long") is not None
    a, b = locate_span(s, "users reported overheating")
    assert s[a:b].split() == ["users", "reported", "overheating"]
    assert locate_span(s, "battery died") is None


def test_windows_and_think():
    w = sentence_windows("A one. B two. C three.", 2)
    assert len(w) == 2
    assert strip_think("<think>x</think> Hello") == ("Hello", True)


def test_representation_design_example():
    # design doc 10.2: 20 inputs, D=4 U=16; linked D=3 U=6
    group = {f"d{i}": "D" for i in range(4)} | {f"u{i}": "U" for i in range(16)}
    users = {r: r for r in group}
    support = {"c1": {"d0", "d1", "d2", "u0", "u1", "u2"}, "c2": {"u3", "u4", "u5"}}
    m = representation(group, users, support)
    assert math.isclose(m["RR_D"], 0.75) and math.isclose(m["RR_U"], 0.375)
    assert math.isclose(m["RG_D_U"], 0.375)
    assert math.isclose(m["Amp_D"], (3 / 9) / 0.2)
    assert math.isclose(m["F_D"] + m["F_U"], 1.0)


def test_representation_na_for_missing_group():
    m = representation({"u0": "U"}, {"u0": "x"}, {"c": {"u0"}})
    assert math.isnan(m["RR_D"]) and math.isnan(m["RG_D_U"])


def test_aggregate_labels():
    assert aggregate_labels(["IRRELEVANT", "SUPPORT"]) == "SUPPORT"
    assert aggregate_labels(["SUPPORT", "OPPOSE"]) == "UNCERTAIN"
    assert aggregate_labels(["IRRELEVANT"]) == "IRRELEVANT"


def test_importance_recall_na_and_condition_loss():
    inv = pd.DataFrame({"experience_id": ["e1", "e2", "e3"], "importance": [2, 2, 0],
                        "polarity": ["negative", "positive", "positive"],
                        "minority": [True, False, False], "conditional": [True, False, False]})
    sg = pd.DataFrame({"summary_id": ["s1", "s1", "s1"], "experience_id": ["e1", "e2", "e3"],
                       "judgment": ["condition_lost", "faithful", "faithful"]})
    r = importance_recall(inv, sg).set_index("summary_id")
    assert r.at["s1", "recall_imp"] == 0.5          # condition loss is not a success
    assert r.at["s1", "recall_imp_minority"] == 0.0


def test_select_budget_and_dedup():
    import numpy as np
    emb = np.eye(4)
    emb[1] = emb[0]                                  # duplicate of 0
    chosen = select(np.array([4, 3, 2, 1.]), np.array([20, 20, 20, 200]), emb, 6,
                    np.array([True] * 4), list("abcd"))
    assert chosen == [0, 2]                          # 1 deduped, 3 over word budget
