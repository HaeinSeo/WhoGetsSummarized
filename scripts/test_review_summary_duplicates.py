from pathlib import Path

import duckdb
import pandas as pd

from eda_review_summary_v1 import (
    set_review_parquet_view,
    write_duplicate_pair_part,
    write_reviewer_activity_part,
)


def main() -> None:
    project = Path(r"E:\WhoGetsSummarized")
    full = project / "outputs" / "review_summary_eda_v1" / "full"
    out = project / "outputs" / "review_summary_eda_v1" / "duplicate_verification"
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    con.execute("SET enable_progress_bar=false")
    set_review_parquet_view(con, full / "source_parquet" / "review" / "All_Beauty.parquet")
    target = out / "duplicate_user_product_report.csv"
    write_duplicate_pair_part(con, target, "All_Beauty")
    result = pd.read_csv(target).iloc[0]
    reference = con.execute(
        """
        WITH pair_details AS (
          SELECT user_id, parent_asin, count(*) AS pair_review_count,
                 count(DISTINCT lower(trim(coalesce(text,'')))) AS distinct_text_count,
                 count(DISTINCT timestamp) AS distinct_timestamp_count,
                 count(DISTINCT rating) AS distinct_rating_count
          FROM reviews WHERE user_id IS NOT NULL AND parent_asin IS NOT NULL
          GROUP BY user_id, parent_asin HAVING count(*) > 1
        )
        SELECT count(*), coalesce(sum(pair_review_count), 0), max(pair_review_count),
               count(*) FILTER (WHERE distinct_text_count = 1 AND distinct_timestamp_count = 1),
               count(*) FILTER (WHERE distinct_text_count = 1 AND distinct_timestamp_count > 1),
               count(*) FILTER (WHERE distinct_text_count > 1),
               count(*) FILTER (WHERE distinct_rating_count > 1)
        FROM pair_details
        """
    ).fetchone()
    columns = [
        "repeated_user_product_pairs", "reviews_in_repeated_user_product_pairs",
        "max_reviews_in_one_user_product_pair", "same_text_same_time_pairs",
        "same_text_different_time_pairs", "changed_text_pairs", "changed_rating_pairs",
    ]
    actual = tuple(int(result[column]) for column in columns)
    assert actual == reference, (actual, reference)
    user_target = out / "reviewer_activity_distribution.csv"
    write_reviewer_activity_part(con, user_target, "All_Beauty")
    users = int(pd.read_csv(user_target).loc[0, "users"])
    expected_users = con.execute("SELECT count(DISTINCT user_id) FROM reviews").fetchone()[0]
    assert users == expected_users, (users, expected_users)
    print(f"duplicate verification passed: repeated pairs={actual[0]}, active users={users}")
    con.close()


if __name__ == "__main__":
    main()
