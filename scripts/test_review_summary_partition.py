from pathlib import Path

import duckdb

from eda_review_summary_v1 import (
    connect,
    set_review_parquet_view,
    write_product_stats_partitioned,
)


def main() -> None:
    project = Path(r"E:\WhoGetsSummarized")
    full = project / "outputs" / "review_summary_eda_v1" / "full"
    out = project / "outputs" / "review_summary_eda_v1" / "partition_verification" / "All_Beauty"
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    con = connect(out, Path(r"C:\Users\haein\AppData\Local\Temp\WhoGetsSummarized_eda_partition_test"))
    con.execute(
        "CREATE TEMP TABLE meta_cat AS SELECT "
        "NULL::VARCHAR AS parent_asin, NULL::VARCHAR AS main_category, "
        "NULL::VARCHAR AS product_title, NULL::VARCHAR AS brand_or_store, "
        "NULL::VARCHAR AS categories_json, NULL::BIGINT AS metadata_rows_for_parent "
        "WHERE false"
    )
    set_review_parquet_view(con, full / "source_parquet" / "review" / "All_Beauty.parquet")
    write_product_stats_partitioned(con, out, checkpoints, "All_Beauty", False, 2)
    result = con.execute(
        "SELECT count(*), sum(review_count), sum(valid_text_review_count), "
        "sum(low_rating_count), sum(high_rating_count) "
        "FROM read_parquet(?)", [str(out / "product_review_statistics.parquet")]
    ).fetchone()
    reference = con.execute(
        "SELECT count(*), sum(review_count), sum(valid_text_review_count), "
        "sum(low_rating_count), sum(high_rating_count) "
        "FROM read_parquet(?)",
        [str(full / "category_parts" / "All_Beauty" / "product_review_statistics.parquet")],
    ).fetchone()
    assert result == reference, (result, reference)
    temporal_count = con.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(out / "product_temporal_rating_change.parquet")],
    ).fetchone()[0]
    reference_temporal_count = con.execute(
        "SELECT count(*) FROM read_parquet(?)",
        [str(full / "category_parts" / "All_Beauty" / "product_temporal_rating_change.parquet")],
    ).fetchone()[0]
    assert temporal_count == reference_temporal_count
    print(f"partition verification passed: product rows={result[0]}, review rows={result[1]}, temporal rows={temporal_count}")
    con.close()


if __name__ == "__main__":
    main()
