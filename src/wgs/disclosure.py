"""Rule-based CANDIDATES for compensation-disclosure labels (D/S/U/A).

These are observed-text candidates, not gold and not hidden incentive status.
Patterns are copied from the EDA (scripts/eda_review_summary_v1.py) so that
DuckDB (RE2) and Python `re` scans agree.
"""
from __future__ import annotations

import re

VINE_RE = r"\b(amazon vine|vine voice|vine customer review|vine program)\b"
NEGATED_VINE_RE = (r"\b(not (an? )?(amazon )?vine|not part of (the )?(amazon )?vine|"
                   r"never (joined|participated in) (the )?(amazon )?vine)\b")
FREE_PRODUCT_RE = (r"\b(received (this |the )?(product|item).{0,60}(for free|free of charge|complimentary)|"
                   r"free (product|item) (in exchange|for review)|complimentary (product|item)|"
                   r"in exchange for (an? )?(honest |unbiased )?review)\b")
DISCOUNT_RE = (r"\b(discounted product|discount in exchange for|rebate for (my )?review|"
               r"refund(ed)? after review|reimbursed for (my )?review)\b")
SELF_PAID_RE = (r"\b(purchased (it|this|these)? with my own money|paid full price|"
                r"bought (it|this|these) myself|not sponsored|no discount|own dime)\b")
NEGATED_FREE_RE = (r"\b(not free|wasn.t free|not a free product|no free product|"
                   r"didn.t receive.{0,70}for free|did not receive.{0,70}for free|"
                   r"never received.{0,70}for free|wasn.t given.{0,70}for free|"
                   r"didn.t receive.{0,40}discount|did not receive.{0,40}discount|"
                   r"free shipping|gift for|as a gift|on sale|sale price)\b")
# Hard negatives for annotation sampling (look similar, usually not disclosure)
HARD_NEGATIVE_RE = r"\b(free shipping|as a gift|gift for|on sale|sale price|coupon|discount|free)\b"

_C = {k: re.compile(v, re.I) for k, v in {
    "vine": VINE_RE, "neg_vine": NEGATED_VINE_RE, "free": FREE_PRODUCT_RE,
    "discount": DISCOUNT_RE, "self": SELF_PAID_RE, "neg_free": NEGATED_FREE_RE,
    "hard_neg": HARD_NEGATIVE_RE,
}.items()}


def disclosure_flags(text: str | None) -> dict:
    t = (text or "").lower()
    vine = bool(_C["vine"].search(t)) and not _C["neg_vine"].search(t)
    free_hit = bool(_C["free"].search(t)) or bool(_C["discount"].search(t))
    neg_free = bool(_C["neg_free"].search(t))
    d = vine or (free_hit and not neg_free)
    s = bool(_C["self"].search(t))
    ambiguous = (d and s) or (free_hit and neg_free and not vine)
    if ambiguous:
        label = "A"
    elif d:
        label = "D"
    elif s:
        label = "S"
    else:
        label = "U"
    m = (_C["vine"].search(t) or _C["free"].search(t) or _C["discount"].search(t)
         or _C["self"].search(t))
    return {
        "D_flag": d, "S_flag": s, "V_flag": vine, "ambiguous_flag": ambiguous,
        "rule_label": label,
        "rule_evidence": (text or "")[max(0, m.start() - 60): m.end() + 60] if m else "",
        "hard_negative": label == "U" and bool(_C["hard_neg"].search(t)),
    }
