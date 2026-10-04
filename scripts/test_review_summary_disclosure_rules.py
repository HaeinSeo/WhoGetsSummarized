import duckdb

from eda_review_summary_v1 import (
    DISCOUNT_RE,
    FREE_PRODUCT_RE,
    NEGATED_FREE_RE,
    NEGATED_VINE_RE,
    SELF_PAID_RE,
    VINE_RE,
)


def matches(con: duckdb.DuckDBPyConnection, text: str, pattern: str) -> bool:
    return con.execute("SELECT regexp_matches(?, ?)", [text.lower(), pattern]).fetchone()[0]


def main() -> None:
    con = duckdb.connect(":memory:")
    cases = [
        ("I received this product for free in exchange for an honest review.", "free"),
        ("I did not receive this product for free. I paid for it.", None),
        ("Free shipping was nice, but I paid for the item.", None),
        ("I bought this as a gift for my friend on sale.", None),
        ("This is an Amazon Vine review.", "vine"),
        ("I am not an Amazon Vine reviewer.", None),
        ("I received a rebate for my review.", "discount"),
        ("I did not receive a discount in exchange for my review.", None),
        ("I paid full price with my own money.", "self_paid"),
    ]
    for text, expected in cases:
        actual = None
        if matches(con, text, VINE_RE) and not matches(con, text, NEGATED_VINE_RE):
            actual = "vine"
        elif matches(con, text, FREE_PRODUCT_RE) and not matches(con, text, NEGATED_FREE_RE):
            actual = "free"
        elif matches(con, text, DISCOUNT_RE) and not matches(con, text, NEGATED_FREE_RE):
            actual = "discount"
        elif matches(con, text, SELF_PAID_RE):
            actual = "self_paid"
        assert actual == expected, (text, actual, expected)
    print(f"disclosure rule verification passed: {len(cases)} cases")
    con.close()


if __name__ == "__main__":
    main()
