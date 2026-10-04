import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd


REQUIRED = [
    "file_inventory.csv",
    "schema_report.json",
    "data_quality_report.csv",
    "metadata_join_report.csv",
    "metadata_join_by_category.csv",
    "overall_summary.csv",
    "category_summary.csv",
    "monthly_summary.csv",
    "product_review_statistics.parquet",
    "candidate_product_summary.csv",
    "candidate_products.csv",
    "random_product_sample.csv",
    "diagnostic_product_sample.csv",
    "review_bundle_sample.parquet",
    "bundle_coverage_report.csv",
    "annotation_sample.csv",
    "annotation_sample_blinded.csv",
    "annotation_guide.md",
    "eda_report.md",
]


def validate(root: Path, full: bool) -> dict:
    failures = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    for name in REQUIRED:
        path = root / name
        check(path.exists() and path.stat().st_size > 0, f"missing or empty: {name}")
    if failures:
        return {"status": "failed", "failures": failures}

    inventory = pd.read_csv(root / "file_inventory.csv")
    quality = pd.read_csv(root / "data_quality_report.csv")
    overall = pd.read_csv(root / "overall_summary.csv").iloc[0]
    category = pd.read_csv(root / "category_summary.csv")
    join = pd.read_csv(root / "metadata_join_report.csv").iloc[0]
    candidates = pd.read_csv(root / "candidate_product_summary.csv")
    random_products = pd.read_csv(root / "random_product_sample.csv")
    diagnostic_products = pd.read_csv(root / "diagnostic_product_sample.csv")
    coverage = pd.read_csv(root / "bundle_coverage_report.csv")
    annotation = pd.read_csv(root / "annotation_sample.csv")
    blinded = pd.read_csv(root / "annotation_sample_blinded.csv")

    check(len(inventory) == 68, "file inventory should contain 34 review and 34 metadata files")
    check(int(category.review_count.sum()) == int(overall.review_count), "category review total differs from overall")
    check(int(quality.loc[quality.kind == "review", "row_count"].sum()) == int(overall.review_count), "file review total differs from overall")
    check(int(join.review_rows) == int(overall.review_count), "join review total differs from overall")
    check(int(join.matched_review_rows) <= int(join.review_rows), "matched rows exceed review rows")
    check(int(join.raw_join_rows_if_not_deduplicated) >= int(join.matched_review_rows), "raw join rows below matched rows")
    check(not random_products.duplicated(["category", "product_id"]).any(), "random products are duplicated")
    check(not diagnostic_products.duplicated(["category", "product_id", "candidate_reason"]).any(), "diagnostic rows are duplicated")
    check(int(coverage.bundle_reviews.max()) <= 100, "review bundle exceeds 100 reviews per product")
    check(set(annotation.sample_id) == set(blinded.sample_id), "blinded review IDs differ")
    check("candidate_type" not in blinded.columns and "evidence_sentence" not in blinded.columns, "blinded file reveals rule fields")
    for field in ["human_aspect", "human_sentiment", "human_issue", "human_importance", "annotator_notes"]:
        check(field in annotation.columns and annotation[field].isna().all(), f"human field is filled or missing: {field}")
    if full:
        check(500 <= len(annotation) <= 1000, "full annotation sample is outside 500-1000 reviews")
        relaxed_path = root / "cross_category_relaxed_duplicate_summary.csv"
        if relaxed_path.exists():
            relaxed = pd.read_csv(relaxed_path).iloc[0]
            bucket_files = sorted((root / "cross_category_fingerprint_reports").glob("*.csv"))
            check(len(bucket_files) == 64, "relaxed duplicate fingerprint bucket count differs from 64")
            if bucket_files:
                bucket_rows = [pd.read_csv(path) for path in bucket_files]
                bucket_groups = sum(len(rows) for rows in bucket_rows)
                bucket_review_rows = sum(int(rows.row_count.sum()) for rows in bucket_rows)
                check(bucket_groups == int(relaxed.fingerprint_keys_across_categories),
                      "relaxed duplicate fingerprint group total differs")
                check(bucket_review_rows == int(relaxed.rows_in_fingerprint_groups),
                      "relaxed duplicate fingerprint row total differs")
            check((root / "cross_category_relaxed_duplicate_method.json").exists(),
                  "relaxed duplicate method description is missing")

    con = duckdb.connect(":memory:")
    stats_path = (root / "product_review_statistics.parquet").as_posix().replace("'", "''")
    bundle_path = (root / "review_bundle_sample.parquet").as_posix().replace("'", "''")
    product_metrics = con.execute(
        f"""SELECT sum(review_count), count(DISTINCT product_id),
                   count(*) FILTER (WHERE valid_text_review_count >= 20),
                   count(*) FILTER (WHERE rating_entropy < -0.000001 OR rating_entropy > ln(5) + 0.000001)
            FROM read_parquet('{stats_path}')"""
    ).fetchone()
    if full:
        check(int(product_metrics[0]) == int(overall.review_count), "product review rows differ from full review rows")
        product_category_rows = con.execute(
            f"SELECT category, sum(review_count) AS review_count "
            f"FROM read_parquet('{stats_path}') GROUP BY category"
        ).df()
        expected_category_rows = dict(zip(category.category, category.review_count.astype(int)))
        actual_category_rows = dict(zip(product_category_rows.category, product_category_rows.review_count.astype(int)))
        check(expected_category_rows == actual_category_rows, "product review totals differ by category")
    else:
        check(int(product_metrics[0]) <= int(overall.review_count), "product review rows exceed overall reviews")
    check(int(product_metrics[1]) == int(overall.unique_products), "unique product count differs")
    check(int(product_metrics[2]) == int(candidates.products_valid_reviews_min20.sum()), "min20 product count differs")
    check(int(product_metrics[3]) == 0, "rating entropy outside valid range")
    bundle_count = con.execute(f"SELECT count(*) FROM read_parquet('{bundle_path}')").fetchone()[0]
    check(int(bundle_count) == int(coverage.bundle_reviews.sum()), "bundle rows differ from coverage report")
    con.close()

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "review_rows": int(overall.review_count),
        "unique_products": int(overall.unique_products),
        "unique_users": int(overall.unique_users),
        "annotation_rows": len(annotation),
        "bundle_rows": int(bundle_count),
        "low_rating_lost_bundles": int(coverage.low_rating_lost.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    result = validate(args.root, args.full)
    destination = args.root / "validation_report.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
