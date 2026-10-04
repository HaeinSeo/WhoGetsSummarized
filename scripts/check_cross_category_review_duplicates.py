import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from eda_review_summary_v1 import (
    checkpoint_done,
    connect,
    create_parquet_views,
    qpath,
    run_step,
    setup_logging,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(r"E:\WhoGetsSummarized\outputs\review_summary_eda_v1\full"))
    parser.add_argument("--temp-dir", type=Path, default=None)
    parser.add_argument("--buckets", type=int, default=64)
    args = parser.parse_args()
    if args.buckets < 2:
        raise ValueError("--buckets must be at least 2")
    root = args.root.resolve()
    setup_logging(root / "logs")
    checkpoints = root / "checkpoints"
    fingerprint_dir = root / "cross_category_fingerprints"
    report_dir = root / "cross_category_fingerprint_reports"
    report_dir.mkdir(exist_ok=True)
    con = connect(root, args.temp_dir)
    create_parquet_views(con, root / "source_parquet" / "review", root / "source_parquet" / "metadata_product")

    split_name = f"cross_category_fingerprints_{args.buckets}"
    if not checkpoint_done(checkpoints, split_name):
        if fingerprint_dir.exists():
            resolved = fingerprint_dir.resolve()
            if not resolved.is_relative_to(root) or fingerprint_dir.is_symlink():
                raise ValueError(f"unsafe fingerprint directory: {fingerprint_dir}")
            shutil.rmtree(fingerprint_dir)
        run_step(
            checkpoints, split_name, False,
            lambda: con.execute(
                f"""
                COPY (
                  WITH keyed AS (
                    SELECT md5(coalesce(user_id,'') || '|' || coalesce(timestamp::VARCHAR,'') || '|' ||
                               coalesce(rating::VARCHAR,'') || '|' || lower(trim(coalesce(text,'')))) AS review_key,
                           category, parent_asin
                    FROM reviews
                  )
                  SELECT review_key, category, parent_asin,
                         (hash(review_key) % {args.buckets})::INTEGER AS fp_bucket
                  FROM keyed
                ) TO '{qpath(fingerprint_dir)}' (FORMAT PARQUET, PARTITION_BY(fp_bucket))
                """
            ),
        )

    for bucket in range(args.buckets):
        target = report_dir / f"{bucket}.csv"
        source = fingerprint_dir / f"fp_bucket={bucket}" / "*.parquet"
        run_step(
            checkpoints, f"cross_category_fingerprint_bucket_{bucket}", False,
            lambda source=source, target=target: con.execute(
                f"""
                COPY (
                  SELECT review_key, count(*) AS row_count,
                         min(category) AS category_min, max(category) AS category_max,
                         min(parent_asin) AS parent_asin_min,
                         max(parent_asin) AS parent_asin_max
                  FROM read_parquet('{qpath(source)}')
                  GROUP BY review_key
                  HAVING min(category) <> max(category)
                ) TO '{qpath(target)}' (HEADER, DELIMITER ',')
                """
            ),
        )

    dtypes = {"review_key": "string", "parent_asin_min": "string", "parent_asin_max": "string"}
    parts = [pd.read_csv(report_dir / f"{bucket}.csv", dtype=dtypes) for bucket in range(args.buckets)]
    hits = pd.concat(parts, ignore_index=True)
    summary = pd.DataFrame([{
        "fingerprint_keys_across_categories": len(hits),
        "rows_in_fingerprint_groups": int(hits["row_count"].sum()) if not hits.empty else 0,
        "different_parent_key_groups": int((hits["parent_asin_min"] != hits["parent_asin_max"]).sum()) if not hits.empty else 0,
    }])
    summary.to_csv(root / "cross_category_relaxed_duplicate_summary.csv", index=False)
    hits.sort_values("row_count", ascending=False).head(1000).to_csv(
        root / "cross_category_relaxed_duplicate_examples.csv", index=False)
    (root / "cross_category_relaxed_duplicate_method.json").write_text(
        json.dumps({
            "fingerprint": "MD5(user_id | timestamp | rating | lower(trim(text)))",
            "parent_asin_in_fingerprint": False,
            "detection": "same fingerprint in at least two categories",
            "buckets": args.buckets,
            "caveats": [
                "A cryptographic hash collision is theoretically possible.",
                "A matching fingerprint is a duplicate candidate, not a verified same experience.",
                "Different wording or timestamps are not detected.",
            ],
        }, indent=2), encoding="utf-8",
    )
    logging.info("relaxed cross-category duplicate fingerprint groups: %s", len(hits))
    print(summary.to_string(index=False))
    con.close()


if __name__ == "__main__":
    main()
