import argparse
import csv
import datetime as dt
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import psutil


REVIEW_COLUMNS = {
    "rating": "DOUBLE",
    "title": "VARCHAR",
    "text": "VARCHAR",
    "images": "JSON",
    "asin": "VARCHAR",
    "parent_asin": "VARCHAR",
    "user_id": "VARCHAR",
    "timestamp": "BIGINT",
    "helpful_vote": "BIGINT",
    "verified_purchase": "BOOLEAN",
}

META_COLUMNS = {
    "main_category": "VARCHAR",
    "title": "VARCHAR",
    "average_rating": "DOUBLE",
    "rating_number": "BIGINT",
    "features": "JSON",
    "description": "JSON",
    "price": "VARCHAR",
    "images": "JSON",
    "videos": "JSON",
    "store": "VARCHAR",
    "categories": "JSON",
    "details": "JSON",
    "parent_asin": "VARCHAR",
    "bought_together": "JSON",
}

DRAWBACK_RE = r"\b(but|however|although|though|except|wish|unfortunately|downside|con\b|problem|issue|complain|complaint|disappoint|disappointed|poor|bad|terrible|broken|breaks|broke|defect|leak|leaked|smell|small|large|hard to|difficult|doesn.t|didn.t|not)\b"
POSITIVE_RE = r"\b(good|great|excellent|love|liked|works|worked|nice|easy|useful|perfect|recommend|comfortable|quality|happy|pleased)\b"
ATTRIBUTE_RE = r"\b(battery|durability|durable|broke|broken|size|small|large|fit|fits|usability|easy to use|hard to use|material|smell|taste|leak|noise|screen|charger|delivery|packaging|return|refund)\b"
COMPLAINT_RE = r"\b(problem|issue|complaint|complain|bad|poor|terrible|disappoint|broken|broke|defect|not work|doesn.t work|stopped working|leak|hard to|difficult|too small|too large)\b"
VINE_RE = r"\b(amazon vine|vine voice|vine customer review|vine program)\b"
NEGATED_VINE_RE = r"\b(not (an? )?(amazon )?vine|not part of (the )?(amazon )?vine|never (joined|participated in) (the )?(amazon )?vine)\b"
FREE_PRODUCT_RE = r"\b(received (this |the )?(product|item).{0,60}(for free|free of charge|complimentary)|free (product|item) (in exchange|for review)|complimentary (product|item)|in exchange for (an? )?(honest |unbiased )?review)\b"
DISCOUNT_RE = r"\b(discounted product|discount in exchange for|rebate for (my )?review|refund(ed)? after review|reimbursed for (my )?review)\b"
SELF_PAID_RE = r"\b(purchased (it|this|these)? with my own money|paid full price|bought (it|this|these) myself|not sponsored|no discount|own dime)\b"
NEGATED_FREE_RE = r"\b(not free|wasn.t free|not a free product|no free product|didn.t receive.{0,70}for free|did not receive.{0,70}for free|never received.{0,70}for free|wasn.t given.{0,70}for free|didn.t receive.{0,40}discount|did not receive.{0,40}discount|free shipping|gift for|as a gift|on sale|sale price)\b"


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def qpath(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def cols_literal(cols: dict[str, str]) -> str:
    return "{" + ", ".join(f"'{k}':'{v}'" for k, v in cols.items()) + "}"


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "eda_run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def checkpoint_done(checkpoint_dir: Path, name: str) -> bool:
    return (checkpoint_dir / f"{name}.done").exists()


def mark_done(checkpoint_dir: Path, name: str) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / f"{name}.done").write_text(now_stamp(), encoding="utf-8")


def run_step(checkpoint_dir: Path, name: str, force: bool, fn):
    if checkpoint_done(checkpoint_dir, name) and not force:
        logging.info("skip checkpoint %s", name)
        return
    logging.info("start %s", name)
    try:
        fn()
    except Exception:
        logging.exception("failed %s", name)
        with (checkpoint_dir.parent / "logs" / "errors.log").open("a", encoding="utf-8") as error_log:
            error_log.write(f"{now_stamp()} failed {name}\n")
        raise
    mark_done(checkpoint_dir, name)
    logging.info("done %s", name)


def category_from_review_file(path: Path) -> str:
    return path.stem


def category_from_meta_file(path: Path) -> str:
    return re.sub(r"^meta_", "", path.stem)


def make_smoke_inputs(project: Path, out_dir: Path, n: int, force: bool) -> tuple[Path, Path]:
    smoke_root = out_dir / "smoke_inputs"
    review_out = smoke_root / "review"
    meta_out = smoke_root / "metadata_product"
    if smoke_root.exists() and force:
        shutil.rmtree(smoke_root)
    review_out.mkdir(parents=True, exist_ok=True)
    meta_out.mkdir(parents=True, exist_ok=True)

    def copy_heads(src_dir: Path, dst_dir: Path):
        for src in sorted(src_dir.glob("*.jsonl")):
            dst = dst_dir / src.name
            if dst.exists() and not force:
                continue
            with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8", newline="\n") as fout:
                for i, line in enumerate(fin):
                    if i >= n:
                        break
                    fout.write(line)

    copy_heads(project / "data" / "review", review_out)
    copy_heads(project / "data" / "metadata_product", meta_out)
    return review_out, meta_out


def inventory_and_schema(project: Path, review_dir: Path, meta_dir: Path, out_dir: Path, mode: str, limit_per_file: int | None) -> None:
    rows = []
    for kind, root in [("review", review_dir), ("metadata", meta_dir)]:
        for path in sorted(root.glob("*.jsonl")):
            category = category_from_review_file(path) if kind == "review" else category_from_meta_file(path)
            rows.append(
                {
                    "mode": mode,
                    "kind": kind,
                    "category": category,
                    "path": str(path),
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "is_compressed": path.suffix.lower() in {".gz", ".zip", ".zst", ".bz2", ".xz"},
                    "bytes": path.stat().st_size,
                    "modified_time": dt.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                    "sample_limit_per_file": limit_per_file,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "file_inventory.csv", index=False, encoding="utf-8-sig")

    schema = {"mode": mode, "generated_at": now_stamp(), "review": {}, "metadata": {}, "notes": []}
    for kind, root in [("review", review_dir), ("metadata", meta_dir)]:
        for path in sorted(root.glob("*.jsonl")):
            counts = {}
            examples = {}
            parse_errors = []
            total = 0
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= min(limit_per_file or 1000, 1000):
                        break
                    total += 1
                    try:
                        obj = json.loads(line)
                    except Exception as exc:
                        parse_errors.append({"line": i + 1, "error": str(exc)[:300]})
                        continue
                    for k, v in obj.items():
                        counts.setdefault(k, {})
                        typ = type(v).__name__
                        counts[k][typ] = counts[k].get(typ, 0) + 1
                        if k not in examples and v is not None:
                            examples[k] = repr(v)[:300]
            schema[kind][path.name] = {"sampled_rows": total, "field_types": counts, "examples": examples, "sample_parse_errors": parse_errors[:20]}
    (out_dir / "schema_report.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def connect(out_dir: Path, temp_dir_override: Path | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(out_dir / "eda_work.duckdb"))
    temp_dir = temp_dir_override.resolve() if temp_dir_override else out_dir / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{qpath(temp_dir)}'")
    if temp_dir_override:
        free_gb = psutil.disk_usage(str(temp_dir.anchor)).free / (1024 ** 3)
        temp_limit_gb = max(1, min(120, int(free_gb * 0.65)))
        con.execute(f"SET max_temp_directory_size='{temp_limit_gb}GB'")
        logging.info("DuckDB temp directory %s; limit %s GB; drive free %.1f GB",
                     temp_dir, temp_limit_gb, free_gb)
    con.execute("PRAGMA threads=2")
    con.execute("PRAGMA disable_progress_bar")
    con.execute("SET preserve_insertion_order=false")
    available_gb = psutil.virtual_memory().available / (1024 ** 3)
    memory_gb = max(2, min(6, int(available_gb * 0.85)))
    con.execute(f"PRAGMA memory_limit='{memory_gb}GB'")
    logging.info("DuckDB memory limit %s GB; available RAM %.1f GB", memory_gb, available_gb)
    return con


def prepare_full_parquet(con: duckdb.DuckDBPyConnection, project: Path, out_dir: Path, force: bool) -> tuple[Path, Path]:
    cache = out_dir / "source_parquet"
    review_cache = cache / "review"
    meta_cache = cache / "metadata_product"
    review_cache.mkdir(parents=True, exist_ok=True)
    meta_cache.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "checkpoints"

    for kind, source_dir, target_dir in [
        ("review", project / "data" / "review", review_cache),
        ("metadata", project / "data" / "metadata_product", meta_cache),
    ]:
        for source in sorted(source_dir.glob("*.jsonl")):
            target = target_dir / f"{source.stem}.parquet"
            temporary = target_dir / f"{source.stem}.partial.parquet"
            checkpoint = f"parquet_{kind}_{source.stem}"
            if checkpoint_done(checkpoint_dir, checkpoint) and target.exists() and not force:
                logging.info("skip checkpoint %s", checkpoint)
                continue
            category = category_from_review_file(source) if kind == "review" else category_from_meta_file(source)
            free_gb = psutil.disk_usage(str(project.drive + "\\")).free / (1024 ** 3)
            if free_gb < 30:
                raise RuntimeError(f"Only {free_gb:.1f} GB free on {project.drive}; stopping before {source.name}")
            logging.info("convert %s %s (%.2f GB)", kind, category, source.stat().st_size / (1024 ** 3))
            if kind == "review":
                query = f"""
                    SELECT '{category}' AS category, filename, rating, title AS review_title,
                           text, asin, parent_asin, user_id, timestamp,
                           CASE WHEN timestamp IS NULL THEN NULL
                                WHEN timestamp > 100000000000 THEN CAST(to_timestamp(timestamp / 1000.0) AS TIMESTAMP)
                                ELSE CAST(to_timestamp(timestamp) AS TIMESTAMP) END AS review_datetime,
                           helpful_vote, verified_purchase
                    FROM read_json('{qpath(source)}', format='newline_delimited', filename=true,
                                   columns={cols_literal(REVIEW_COLUMNS)})
                """
            else:
                query = f"""
                    SELECT '{category}' AS meta_file_category, filename, parent_asin,
                           main_category, title AS product_title, store AS brand_or_store,
                           categories::VARCHAR AS categories_json,
                           average_rating AS metadata_average_rating,
                           rating_number AS metadata_rating_number, details::VARCHAR AS details_json
                    FROM read_json('{qpath(source)}', format='newline_delimited', filename=true,
                                   columns={cols_literal(META_COLUMNS)})
                """
            try:
                con.execute(f"COPY ({query}) TO '{qpath(temporary)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
                temporary.replace(target)
                mark_done(checkpoint_dir, checkpoint)
                logging.info("converted %s %s (%.2f GB)", kind, category, target.stat().st_size / (1024 ** 3))
            except Exception:
                logging.exception("failed converting %s", source)
                with (out_dir / "logs" / "errors.log").open("a", encoding="utf-8") as error_log:
                    error_log.write(f"{now_stamp()} failed conversion {source}\n")
                raise
    return review_cache, meta_cache


def create_parquet_views(con: duckdb.DuckDBPyConnection, review_cache: Path, meta_cache: Path) -> None:
    con.execute("DROP VIEW IF EXISTS reviews")
    con.execute("DROP VIEW IF EXISTS metadata_raw")
    con.execute(f"CREATE VIEW reviews AS SELECT * FROM read_parquet('{qpath(review_cache / '*.parquet')}')")
    con.execute(f"CREATE VIEW metadata_raw AS SELECT * FROM read_parquet('{qpath(meta_cache / '*.parquet')}')")


def set_review_parquet_view(con: duckdb.DuckDBPyConnection, review_path: Path) -> None:
    con.execute("DROP VIEW IF EXISTS reviews")
    con.execute(f"CREATE VIEW reviews AS SELECT * FROM read_parquet('{qpath(review_path)}')")


def merge_part_csv(out_dir: Path, categories: list[str], filename: str) -> None:
    parts = [pd.read_csv(out_dir / "category_parts" / category / filename) for category in categories]
    pd.concat(parts, ignore_index=True).to_csv(out_dir / filename, index=False, encoding="utf-8-sig")


def create_views(con: duckdb.DuckDBPyConnection, review_dir: Path, meta_dir: Path) -> None:
    review_glob = qpath(review_dir / "*.jsonl")
    meta_glob = qpath(meta_dir / "*.jsonl")
    con.execute("DROP VIEW IF EXISTS reviews")
    con.execute("DROP VIEW IF EXISTS metadata_raw")
    con.execute(
        f"""
        CREATE VIEW reviews AS
        SELECT
          regexp_replace(regexp_extract(filename, '([^/\\\\]+)\\.jsonl$', 1), '^meta_', '') AS category,
          filename,
          rating,
          title AS review_title,
          text,
          asin,
          parent_asin,
          user_id,
          timestamp,
          CASE
            WHEN timestamp IS NULL THEN NULL
            WHEN timestamp > 100000000000 THEN CAST(to_timestamp(timestamp / 1000.0) AS TIMESTAMP)
            ELSE CAST(to_timestamp(timestamp) AS TIMESTAMP)
          END AS review_datetime,
          helpful_vote,
          verified_purchase
        FROM read_json('{review_glob}', format='newline_delimited', filename=true, union_by_name=true, columns={cols_literal(REVIEW_COLUMNS)})
        """
    )
    con.execute(
        f"""
        CREATE VIEW metadata_raw AS
        SELECT
          regexp_replace(regexp_extract(filename, '([^/\\\\]+)\\.jsonl$', 1), '^meta_', '') AS meta_file_category,
          filename,
          parent_asin,
          main_category,
          title AS product_title,
          store AS brand_or_store,
          categories::VARCHAR AS categories_json,
          average_rating AS metadata_average_rating,
          rating_number AS metadata_rating_number,
          details::VARCHAR AS details_json
        FROM read_json('{meta_glob}', format='newline_delimited', filename=true, union_by_name=true, columns={cols_literal(META_COLUMNS)})
        """
    )


def write_data_quality(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    fields = [
        "rating", "review_title", "text", "asin", "parent_asin", "user_id",
        "timestamp", "helpful_vote", "verified_purchase", "main_category",
        "product_title", "brand_or_store", "categories_json",
        "metadata_average_rating", "metadata_rating_number", "details_json",
    ]
    review_fields = {
        "rating": "rating IS NULL",
        "review_title": "review_title IS NULL OR review_title = ''",
        "text": "text IS NULL OR text = ''",
        "asin": "asin IS NULL OR asin = ''",
        "parent_asin": "parent_asin IS NULL OR parent_asin = ''",
        "user_id": "user_id IS NULL OR user_id = ''",
        "timestamp": "timestamp IS NULL",
        "helpful_vote": "helpful_vote IS NULL",
        "verified_purchase": "verified_purchase IS NULL",
    }
    meta_fields = {
        "parent_asin": "parent_asin IS NULL OR parent_asin = ''",
        "main_category": "main_category IS NULL OR main_category = ''",
        "product_title": "product_title IS NULL OR product_title = ''",
        "brand_or_store": "brand_or_store IS NULL OR brand_or_store = ''",
        "categories_json": "categories_json IS NULL OR categories_json = '[]'",
        "metadata_average_rating": "metadata_average_rating IS NULL",
        "metadata_rating_number": "metadata_rating_number IS NULL",
        "details_json": "details_json IS NULL OR details_json = '{}'",
    }

    def missing_columns(expressions: dict[str, str]) -> str:
        return ",\n".join(
            f"avg(CASE WHEN {expressions[field]} THEN 1.0 ELSE 0.0 END) AS missing_{field}_rate"
            if field in expressions else f"NULL::DOUBLE AS missing_{field}_rate"
            for field in fields
        )

    con.execute(
        f"""
        COPY (
          SELECT 'review' AS kind, category, count(*) AS row_count,
            0 AS parse_error_count,
            {missing_columns(review_fields)},
            min(timestamp) AS min_timestamp_raw,
            max(timestamp) AS max_timestamp_raw,
            min(review_datetime) AS min_review_datetime,
            max(review_datetime) AS max_review_datetime,
            count(*) FILTER (WHERE timestamp >= 100000000000) AS timestamp_milliseconds_rows,
            count(*) FILTER (WHERE timestamp < 100000000000) AS timestamp_seconds_rows
          FROM reviews
          GROUP BY category
          UNION ALL
          SELECT 'metadata' AS kind, meta_file_category AS category, count(*) AS row_count,
            0 AS parse_error_count,
            {missing_columns(meta_fields)},
            NULL::BIGINT AS min_timestamp_raw,
            NULL::BIGINT AS max_timestamp_raw,
            NULL::TIMESTAMP AS min_review_datetime,
            NULL::TIMESTAMP AS max_review_datetime,
            NULL::BIGINT AS timestamp_milliseconds_rows,
            NULL::BIGINT AS timestamp_seconds_rows
          FROM metadata_raw
          GROUP BY meta_file_category
        ) TO '{qpath(out_dir / "data_quality_report.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_metadata_tables(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    con.execute("DROP TABLE IF EXISTS meta_key_counts")
    con.execute(
        """
        CREATE TABLE meta_key_counts AS
        SELECT parent_asin, count(*) AS metadata_rows_for_parent
        FROM metadata_raw
        WHERE parent_asin IS NOT NULL AND parent_asin <> ''
        GROUP BY parent_asin
        """
    )
    con.execute("DROP TABLE IF EXISTS metadata_dedup")
    has_duplicate_keys = con.execute(
        "SELECT EXISTS(SELECT 1 FROM meta_key_counts WHERE metadata_rows_for_parent > 1)"
    ).fetchone()[0]
    if has_duplicate_keys:
        con.execute(
            """
            CREATE TABLE metadata_dedup AS
            SELECT parent_asin, any_value(main_category) AS main_category,
                   any_value(product_title) AS product_title,
                   any_value(brand_or_store) AS brand_or_store,
                   any_value(categories_json) AS categories_json,
                   any_value(meta_file_category) AS meta_file_category,
                   avg(metadata_average_rating) AS metadata_average_rating_avg,
                   max(metadata_rating_number) AS metadata_rating_number_max,
                   count(*) AS metadata_rows_for_parent
            FROM metadata_raw
            WHERE parent_asin IS NOT NULL AND parent_asin <> ''
            GROUP BY parent_asin
            """
        )
    else:
        logging.info("metadata parent_asin keys are unique; materializing without aggregation")
        con.execute(
            """
            CREATE TABLE metadata_dedup AS
            SELECT parent_asin, main_category, product_title, brand_or_store,
                   categories_json, meta_file_category,
                   metadata_average_rating AS metadata_average_rating_avg,
                   metadata_rating_number AS metadata_rating_number_max,
                   1 AS metadata_rows_for_parent
            FROM metadata_raw
            WHERE parent_asin IS NOT NULL AND parent_asin <> ''
            """
        )
    con.execute(
        f"""
        COPY (
          SELECT * FROM metadata_dedup
        ) TO '{qpath(out_dir / "metadata_dedup.parquet")}' (FORMAT PARQUET)
        """
    )


def write_metadata_join_report(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    stats_path = qpath(out_dir / "product_review_statistics.parquet")
    dq_path = qpath(out_dir / "data_quality_report.csv")
    con.execute(
        f"""
        COPY (
          SELECT
            (SELECT sum(row_count) FROM read_csv_auto('{dq_path}') WHERE kind='metadata') AS metadata_rows,
            (SELECT count(*) FROM meta_key_counts) AS metadata_unique_parent_asin,
            (SELECT count(*) FROM meta_key_counts WHERE metadata_rows_for_parent > 1) AS duplicated_metadata_keys,
            (SELECT max(metadata_rows_for_parent) FROM meta_key_counts) AS max_metadata_rows_per_parent,
            (SELECT sum(row_count) FROM read_csv_auto('{dq_path}') WHERE kind='review') AS review_rows,
            (SELECT count(DISTINCT product_id) FROM read_parquet('{stats_path}')) AS review_unique_parent_asin,
            (SELECT sum(review_count) FROM read_parquet('{stats_path}') WHERE metadata_rows_for_parent IS NOT NULL) AS matched_review_rows,
            (SELECT count(DISTINCT product_id) FROM read_parquet('{stats_path}') WHERE metadata_rows_for_parent IS NOT NULL) AS matched_review_parent_asin,
            (SELECT sum(review_count * coalesce(metadata_rows_for_parent, 0)) FROM read_parquet('{stats_path}')) AS raw_join_rows_if_not_deduplicated
        ) TO '{qpath(out_dir / "metadata_join_report.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_core_eda(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT *, CASE WHEN text IS NULL OR length(trim(text)) = 0 THEN 0
                           ELSE array_length(regexp_split_to_array(trim(text), '\\s+')) END AS text_word_count
            FROM reviews
          )
          SELECT
            category,
            count(*) AS review_count,
            count(DISTINCT parent_asin) AS unique_products,
            count(DISTINCT user_id) AS unique_users,
            min(review_datetime) AS first_review_datetime,
            max(review_datetime) AS last_review_datetime,
            avg(rating) AS avg_rating,
            stddev_samp(rating) AS sd_rating,
            sum(CASE WHEN rating=1 THEN 1 ELSE 0 END) AS rating_1_count,
            sum(CASE WHEN rating=2 THEN 1 ELSE 0 END) AS rating_2_count,
            sum(CASE WHEN rating=3 THEN 1 ELSE 0 END) AS rating_3_count,
            sum(CASE WHEN rating=4 THEN 1 ELSE 0 END) AS rating_4_count,
            sum(CASE WHEN rating=5 THEN 1 ELSE 0 END) AS rating_5_count,
            count(*) FILTER (WHERE verified_purchase = true) AS verified_true_count,
            count(*) FILTER (WHERE verified_purchase = false) AS verified_false_count,
            count(*) FILTER (WHERE verified_purchase IS NULL) AS verified_missing_count,
            avg(CASE WHEN verified_purchase THEN 1 WHEN verified_purchase=false THEN 0 ELSE NULL END) AS verified_purchase_rate,
            avg(length(coalesce(text, ''))) AS avg_text_chars,
            median(length(coalesce(text, ''))) AS median_text_chars,
            quantile_cont(length(coalesce(text, '')), 0.10) AS p10_text_chars,
            quantile_cont(length(coalesce(text, '')), 0.90) AS p90_text_chars,
            quantile_cont(length(coalesce(text, '')), 0.99) AS p99_text_chars,
            avg(text_word_count) AS avg_text_words,
            median(text_word_count) AS median_text_words,
            quantile_cont(text_word_count, 0.10) AS p10_text_words,
            quantile_cont(text_word_count, 0.90) AS p90_text_words,
            quantile_cont(text_word_count, 0.99) AS p99_text_words,
            sum(CASE WHEN text IS NULL OR length(trim(text))=0 THEN 1 ELSE 0 END) AS empty_text_count,
            sum(CASE WHEN text IS NOT NULL AND length(trim(text)) BETWEEN 1 AND 20 THEN 1 ELSE 0 END) AS very_short_text_count,
            approx_count_distinct(md5(lower(trim(coalesce(text, ''))))) AS approx_distinct_texts
          FROM base
          GROUP BY category
        ) TO '{qpath(out_dir / "category_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )
    write_core_eda_secondary(con, out_dir)


def write_core_eda_secondary(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          SELECT
            category,
            date_part('year', review_datetime)::INTEGER AS review_year,
            date_part('month', review_datetime)::INTEGER AS review_month,
            date_trunc('month', review_datetime)::DATE AS month_start,
            count(*) AS review_count,
            count(DISTINCT parent_asin) AS unique_products,
            avg(rating) AS avg_rating,
            sum(CASE WHEN rating IN (1,2) THEN 1 ELSE 0 END) AS low_rating_count,
            sum(CASE WHEN rating=3 THEN 1 ELSE 0 END) AS mid_rating_count,
            sum(CASE WHEN rating IN (4,5) THEN 1 ELSE 0 END) AS high_rating_count
          FROM reviews
          WHERE review_datetime IS NOT NULL
          GROUP BY category, review_year, review_month, month_start
          ORDER BY category, month_start
        ) TO '{qpath(out_dir / "monthly_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT
            category,
            CASE
              WHEN text IS NULL OR length(trim(text))=0 THEN 'empty'
              WHEN regexp_matches(text, '[가-힣]') THEN 'contains_korean'
              WHEN regexp_matches(text, '[ぁ-ヿ㐀-鿿]') THEN 'contains_cjk'
              WHEN regexp_matches(text, '[áéíóúñü¿¡àèìòùçâêîôûãõ]') THEN 'contains_latin_diacritic'
              WHEN regexp_matches(text, '^[\\x00-\\x7F]+$') THEN 'ascii_likely_english_or_code'
              ELSE 'other_non_ascii'
            END AS language_heuristic,
            count(*) AS sampled_review_count
          FROM (
            SELECT * FROM reviews
            WHERE abs(hash(coalesce(user_id,'') || coalesce(parent_asin,'') || coalesce(timestamp::VARCHAR,''))) % 1000 = 0
          )
          GROUP BY category, language_heuristic
          ORDER BY category, sampled_review_count DESC
        ) TO '{qpath(out_dir / "language_sample_estimate.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          WITH selected AS (
            SELECT category, lower(trim(text)) AS exact_key,
                   regexp_replace(lower(trim(text)), '[^a-z0-9]+', '', 'g') AS normalized_key
            FROM reviews
            WHERE text IS NOT NULL AND length(trim(text)) > 0
              AND regexp_matches(text, '^[\\x00-\\x7F]+$')
              AND hash(lower(trim(text))) % 1000 = 0
          )
          SELECT category, count(*) AS sampled_nonempty_reviews,
                 count(DISTINCT exact_key) AS unique_exact_texts,
                 count(DISTINCT normalized_key) AS unique_normalized_texts,
                 1000 AS inverse_sampling_fraction
          FROM selected GROUP BY category ORDER BY category
        ) TO '{qpath(out_dir / "text_similarity_sample_report.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_core_eda_large(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    cursor = con.execute(
        """
        SELECT count(*) AS review_count,
               min(review_datetime) AS first_review_datetime,
               max(review_datetime) AS last_review_datetime,
               avg(rating) AS avg_rating, stddev_samp(rating) AS sd_rating,
               count(*) FILTER (WHERE rating=1) AS rating_1_count,
               count(*) FILTER (WHERE rating=2) AS rating_2_count,
               count(*) FILTER (WHERE rating=3) AS rating_3_count,
               count(*) FILTER (WHERE rating=4) AS rating_4_count,
               count(*) FILTER (WHERE rating=5) AS rating_5_count,
               count(*) FILTER (WHERE verified_purchase=true) AS verified_true_count,
               count(*) FILTER (WHERE verified_purchase=false) AS verified_false_count,
               count(*) FILTER (WHERE verified_purchase IS NULL) AS verified_missing_count,
               avg(CASE WHEN verified_purchase THEN 1 WHEN verified_purchase=false THEN 0 ELSE NULL END) AS verified_purchase_rate,
               avg(length(coalesce(text,''))) AS avg_text_chars,
               count(*) FILTER (WHERE text IS NULL OR length(trim(text))=0) AS empty_text_count,
               count(*) FILTER (WHERE text IS NOT NULL AND length(trim(text)) BETWEEN 1 AND 20) AS very_short_text_count
        FROM reviews
        """
    )
    values = dict(zip([column[0] for column in cursor.description], cursor.fetchone()))
    values["category"] = out_dir.name
    logging.info("large core %s: base metrics complete", out_dir.name)

    values["unique_products"] = con.execute("SELECT count(DISTINCT parent_asin) FROM reviews").fetchone()[0]
    logging.info("large core %s: unique products complete", out_dir.name)
    values["unique_users"] = con.execute("SELECT count(DISTINCT user_id) FROM reviews").fetchone()[0]
    logging.info("large core %s: unique users complete", out_dir.name)

    chars = con.execute(
        """
        SELECT median(length(coalesce(text,''))),
               quantile_cont(length(coalesce(text,'')), 0.10),
               quantile_cont(length(coalesce(text,'')), 0.90),
               quantile_cont(length(coalesce(text,'')), 0.99)
        FROM reviews
        """
    ).fetchone()
    values.update(zip(["median_text_chars", "p10_text_chars", "p90_text_chars", "p99_text_chars"], chars))
    logging.info("large core %s: character quantiles complete", out_dir.name)

    words = con.execute(
        """
        WITH lengths AS (
          SELECT CASE WHEN text IS NULL OR length(trim(text))=0 THEN 0
                      ELSE array_length(regexp_split_to_array(trim(text), '\\s+')) END AS word_count
          FROM reviews
        )
        SELECT avg(word_count), median(word_count),
               quantile_cont(word_count, 0.10), quantile_cont(word_count, 0.90),
               quantile_cont(word_count, 0.99)
        FROM lengths
        """
    ).fetchone()
    values.update(zip(["avg_text_words", "median_text_words", "p10_text_words", "p90_text_words", "p99_text_words"], words))
    logging.info("large core %s: word quantiles complete", out_dir.name)

    values["approx_distinct_texts"] = con.execute(
        "SELECT approx_count_distinct(md5(lower(trim(coalesce(text,''))))) FROM reviews"
    ).fetchone()[0]
    pd.DataFrame([values]).to_csv(out_dir / "category_summary.csv", index=False, encoding="utf-8-sig")
    logging.info("large core %s: category summary complete", out_dir.name)
    write_core_eda_secondary(con, out_dir)


def write_core_eda_batches(con: duckdb.DuckDBPyConnection, out_dir: Path, force: bool) -> None:
    review_cache = out_dir / "source_parquet" / "review"
    categories = [path.stem for path in sorted(review_cache.glob("*.parquet"))]
    for category in categories:
        part_dir = out_dir / "category_parts" / category
        part_dir.mkdir(parents=True, exist_ok=True)
        review_path = review_cache / f"{category}.parquet"
        set_review_parquet_view(con, review_path)
        writer = write_core_eda_large if review_path.stat().st_size >= 6 * 1024 ** 3 else write_core_eda
        run_step(out_dir / "checkpoints", f"core_{category}", force, lambda: writer(con, part_dir))
    for filename in [
        "category_summary.csv", "monthly_summary.csv", "language_sample_estimate.csv",
        "text_similarity_sample_report.csv",
    ]:
        merge_part_csv(out_dir, categories, filename)


def write_product_stats(con: duckdb.DuckDBPyConnection, out_dir: Path, metadata_relation: str = "metadata_dedup") -> None:
    con.execute(
        f"""
        COPY (
          WITH product_base AS (
            SELECT
              category,
              parent_asin AS product_id,
              count(*) AS review_count,
              sum(CASE WHEN text IS NOT NULL AND length(trim(text)) > 0 THEN 1 ELSE 0 END) AS valid_text_review_count,
              count(DISTINCT user_id) AS unique_reviewer_count,
              avg(rating) AS avg_rating,
              stddev_samp(rating) AS sd_rating,
              sum(CASE WHEN rating=1 THEN 1 ELSE 0 END) AS rating_1_count,
              sum(CASE WHEN rating=2 THEN 1 ELSE 0 END) AS rating_2_count,
              sum(CASE WHEN rating=3 THEN 1 ELSE 0 END) AS rating_3_count,
              sum(CASE WHEN rating=4 THEN 1 ELSE 0 END) AS rating_4_count,
              sum(CASE WHEN rating=5 THEN 1 ELSE 0 END) AS rating_5_count,
              sum(CASE WHEN rating IN (1,2) THEN 1 ELSE 0 END) AS low_rating_count,
              sum(CASE WHEN rating=3 THEN 1 ELSE 0 END) AS mid_rating_count,
              sum(CASE WHEN rating IN (4,5) THEN 1 ELSE 0 END) AS high_rating_count,
              sum(CASE WHEN rating IN (1,2) AND text IS NOT NULL AND length(trim(text)) > 0 THEN 1 ELSE 0 END) AS low_rating_valid_text_count,
              sum(CASE WHEN rating=3 AND text IS NOT NULL AND length(trim(text)) > 0 THEN 1 ELSE 0 END) AS mid_rating_valid_text_count,
              sum(CASE WHEN rating IN (4,5) AND text IS NOT NULL AND length(trim(text)) > 0 THEN 1 ELSE 0 END) AS high_rating_valid_text_count,
              sum(length(coalesce(text,''))) AS total_text_chars,
              min(review_datetime) AS first_review_datetime,
              max(review_datetime) AS last_review_datetime,
              date_diff('day', min(review_datetime), max(review_datetime)) AS review_span_days,
              arg_min(rating, review_datetime) AS first_observed_rating,
              arg_max(rating, review_datetime) AS last_observed_rating,
              arg_max(rating, review_datetime) - arg_min(rating, review_datetime) AS observed_last_minus_first_rating
            FROM reviews
            WHERE parent_asin IS NOT NULL AND parent_asin <> ''
            GROUP BY category, parent_asin
          )
          SELECT
            p.*,
            rating_1_count::DOUBLE / nullif(review_count,0) AS rating_1_share,
            rating_2_count::DOUBLE / nullif(review_count,0) AS rating_2_share,
            rating_3_count::DOUBLE / nullif(review_count,0) AS rating_3_share,
            rating_4_count::DOUBLE / nullif(review_count,0) AS rating_4_share,
            rating_5_count::DOUBLE / nullif(review_count,0) AS rating_5_share,
            (low_rating_count > 0 AND high_rating_count > 0) AS has_low_and_high_ratings,
            -1 * (
              CASE WHEN rating_1_count > 0 THEN (rating_1_count::DOUBLE/review_count) * ln(rating_1_count::DOUBLE/review_count) ELSE 0 END +
              CASE WHEN rating_2_count > 0 THEN (rating_2_count::DOUBLE/review_count) * ln(rating_2_count::DOUBLE/review_count) ELSE 0 END +
              CASE WHEN rating_3_count > 0 THEN (rating_3_count::DOUBLE/review_count) * ln(rating_3_count::DOUBLE/review_count) ELSE 0 END +
              CASE WHEN rating_4_count > 0 THEN (rating_4_count::DOUBLE/review_count) * ln(rating_4_count::DOUBLE/review_count) ELSE 0 END +
              CASE WHEN rating_5_count > 0 THEN (rating_5_count::DOUBLE/review_count) * ln(rating_5_count::DOUBLE/review_count) ELSE 0 END
            ) AS rating_entropy,
            m.main_category,
            m.product_title,
            m.brand_or_store,
            m.categories_json,
            m.metadata_rows_for_parent
          FROM product_base p
          LEFT JOIN {metadata_relation} m ON p.product_id = m.parent_asin
        ) TO '{qpath(out_dir / "product_review_statistics.parquet")}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT
            category,
            product_id,
            review_year,
            count(*) AS review_count,
            avg(rating) AS avg_rating,
            count(*) FILTER (WHERE rating IN (1, 2)) AS low_rating_count,
            count(*) FILTER (WHERE rating = 3) AS mid_rating_count,
            count(*) FILTER (WHERE rating IN (4, 5)) AS high_rating_count
          FROM (
            SELECT category, parent_asin AS product_id, rating,
                   date_part('year', review_datetime)::INTEGER AS review_year
            FROM reviews
            WHERE parent_asin IS NOT NULL AND parent_asin <> '' AND review_datetime IS NOT NULL
          )
          GROUP BY category, product_id, review_year
        ) TO '{qpath(out_dir / "product_temporal_rating_change.parquet")}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT category, count(*) AS products,
                 avg(review_count) AS mean_reviews_per_product,
                 median(review_count) AS median_reviews_per_product,
                 quantile_cont(review_count, 0.90) AS p90_reviews_per_product,
                 quantile_cont(review_count, 0.99) AS p99_reviews_per_product,
                 max(review_count) AS max_reviews_per_product
          FROM read_parquet('{qpath(out_dir / "product_review_statistics.parquet")}')
          GROUP BY category ORDER BY category
        ) TO '{qpath(out_dir / "product_review_count_distribution.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_product_stats_partitioned(
    con: duckdb.DuckDBPyConnection, out_dir: Path, checkpoint_dir: Path,
    category: str, force: bool, bucket_count: int = 16,
) -> None:
    bucket_dir = out_dir / "review_buckets"
    split_name = f"review_buckets_{category}"
    if force or not checkpoint_done(checkpoint_dir, split_name):
        if bucket_dir.exists():
            resolved = bucket_dir.resolve()
            if not resolved.is_relative_to(out_dir.resolve()) or bucket_dir.is_symlink():
                raise ValueError(f"unsafe bucket directory: {bucket_dir}")
            shutil.rmtree(bucket_dir)
        run_step(
            checkpoint_dir, split_name, force,
            lambda: con.execute(
                f"COPY (SELECT *, (hash(parent_asin) % {bucket_count})::INTEGER AS review_bucket "
                f"FROM reviews WHERE parent_asin IS NOT NULL AND parent_asin <> '') "
                f"TO '{qpath(bucket_dir)}' (FORMAT PARQUET, PARTITION_BY(review_bucket))"
            ),
        )
    bucket_results = out_dir / "bucket_results"
    for bucket in range(bucket_count):
        bucket_out = bucket_results / str(bucket)
        bucket_out.mkdir(parents=True, exist_ok=True)
        bucket_name = f"product_{category}_bucket_{bucket}"
        if checkpoint_done(checkpoint_dir, bucket_name) and not force:
            continue
        set_review_parquet_view(con, bucket_dir / f"review_bucket={bucket}" / "*.parquet")
        con.execute("DROP VIEW IF EXISTS meta_bucket")
        con.execute(
            f"CREATE TEMP VIEW meta_bucket AS SELECT * FROM meta_cat "
            f"WHERE hash(parent_asin) % {bucket_count} = {bucket}"
        )
        run_step(checkpoint_dir, bucket_name, force,
                 lambda: write_product_stats(con, bucket_out, "meta_bucket"))
        con.execute("DROP VIEW meta_bucket")
    for filename in ["product_review_statistics.parquet", "product_temporal_rating_change.parquet"]:
        source_glob = bucket_results / "*" / filename
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{qpath(source_glob)}')) "
            f"TO '{qpath(out_dir / filename)}' (FORMAT PARQUET)"
        )
    con.execute(
        f"COPY (SELECT category, count(*) AS products, "
        f"avg(review_count) AS mean_reviews_per_product, "
        f"median(review_count) AS median_reviews_per_product, "
        f"quantile_cont(review_count, 0.90) AS p90_reviews_per_product, "
        f"quantile_cont(review_count, 0.99) AS p99_reviews_per_product, "
        f"max(review_count) AS max_reviews_per_product "
        f"FROM read_parquet('{qpath(out_dir / 'product_review_statistics.parquet')}') "
        f"GROUP BY category ORDER BY category) "
        f"TO '{qpath(out_dir / 'product_review_count_distribution.csv')}' (HEADER, DELIMITER ',')"
    )


def write_product_stats_batches(con: duckdb.DuckDBPyConnection, out_dir: Path, force: bool) -> None:
    review_cache = out_dir / "source_parquet" / "review"
    categories = [path.stem for path in sorted(review_cache.glob("*.parquet"))]
    for category in categories:
        part_dir = out_dir / "category_parts" / category
        part_dir.mkdir(parents=True, exist_ok=True)
        if checkpoint_done(out_dir / "checkpoints", f"product_{category}") and not force:
            logging.info("skip checkpoint product_%s", category)
            continue
        set_review_parquet_view(con, review_cache / f"{category}.parquet")
        con.execute("DROP TABLE IF EXISTS meta_cat")
        con.execute("DROP TABLE IF EXISTS category_keys")
        con.execute(
            "CREATE TEMP TABLE category_keys AS SELECT DISTINCT parent_asin FROM reviews "
            "WHERE parent_asin IS NOT NULL AND parent_asin <> ''"
        )
        con.execute(
            "CREATE TEMP TABLE meta_cat AS SELECT m.* FROM metadata_dedup m "
            "JOIN category_keys k ON m.parent_asin = k.parent_asin"
        )
        if review_cache.joinpath(f"{category}.parquet").stat().st_size >= 5 * 1024 ** 3:
            run_step(out_dir / "checkpoints", f"product_{category}", force,
                     lambda: write_product_stats_partitioned(
                         con, part_dir, out_dir / "checkpoints", category, force))
        else:
            run_step(out_dir / "checkpoints", f"product_{category}", force,
                     lambda: write_product_stats(con, part_dir, "meta_cat"))
        con.execute("DROP TABLE meta_cat")
        con.execute("DROP TABLE category_keys")
    for filename in ["product_review_statistics.parquet", "product_temporal_rating_change.parquet"]:
        source_glob = out_dir / "category_parts" / "*" / filename
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{qpath(source_glob)}')) "
            f"TO '{qpath(out_dir / filename)}' (FORMAT PARQUET)"
        )
    merge_part_csv(out_dir, categories, "product_review_count_distribution.csv")
    create_parquet_views(con, review_cache, out_dir / "source_parquet" / "metadata_product")


def write_overall_summary(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          SELECT
            (SELECT sum(review_count) FROM read_csv_auto('{qpath(out_dir / "category_summary.csv")}')) AS review_count,
            (SELECT count(DISTINCT product_id) FROM read_parquet('{qpath(out_dir / "product_review_statistics.parquet")}')) AS unique_products,
            (SELECT count(DISTINCT user_id) FROM reviews WHERE user_id IS NOT NULL AND user_id <> '') AS unique_users
        ) TO '{qpath(out_dir / "overall_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_metadata_join_by_category(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          WITH product_join AS (
            SELECT category, count(*) AS products_with_key,
                   count(*) FILTER (WHERE metadata_rows_for_parent IS NOT NULL) AS matched_products,
                   sum(review_count) FILTER (WHERE metadata_rows_for_parent IS NOT NULL) AS matched_review_rows
            FROM read_parquet('{qpath(out_dir / "product_review_statistics.parquet")}')
            GROUP BY category
          )
          SELECT p.category, c.review_count AS review_rows, p.products_with_key,
                 p.matched_products, coalesce(p.matched_review_rows, 0) AS matched_review_rows,
                 p.matched_products::DOUBLE / nullif(p.products_with_key, 0) AS product_join_rate,
                 coalesce(p.matched_review_rows, 0)::DOUBLE / nullif(c.review_count, 0) AS review_join_rate
          FROM product_join p JOIN read_csv_auto('{qpath(out_dir / "category_summary.csv")}') c USING(category)
          ORDER BY p.category
        ) TO '{qpath(out_dir / "metadata_join_by_category.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_duplicate_reports(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    con.execute(
        f"""
        COPY (
          SELECT category,
            count(*) AS repeated_user_product_pairs,
            sum(pair_review_count) AS reviews_in_repeated_user_product_pairs,
            max(pair_review_count) AS max_reviews_in_one_user_product_pair,
            count(*) FILTER (WHERE distinct_text_count = 1 AND distinct_timestamp_count = 1) AS same_text_same_time_pairs,
            count(*) FILTER (WHERE distinct_text_count = 1 AND distinct_timestamp_count > 1) AS same_text_different_time_pairs,
            count(*) FILTER (WHERE distinct_text_count > 1) AS changed_text_pairs,
            count(*) FILTER (WHERE distinct_rating_count > 1) AS changed_rating_pairs
          FROM (
            SELECT category, user_id, parent_asin, count(*) AS pair_review_count,
                   count(DISTINCT lower(trim(coalesce(text,'')))) AS distinct_text_count,
                   count(DISTINCT timestamp) AS distinct_timestamp_count,
                   count(DISTINCT rating) AS distinct_rating_count
            FROM reviews
            WHERE user_id IS NOT NULL AND parent_asin IS NOT NULL
            GROUP BY category, user_id, parent_asin
            HAVING count(*) > 1
          )
          GROUP BY category
          ORDER BY category
        ) TO '{qpath(out_dir / "duplicate_user_product_report.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          WITH per_user AS (
            SELECT category, user_id, count(*) AS reviews_written
            FROM reviews WHERE user_id IS NOT NULL AND user_id <> ''
            GROUP BY category, user_id
          )
          SELECT category, count(*) AS users,
                 avg(reviews_written) AS mean_reviews_per_user,
                 median(reviews_written) AS median_reviews_per_user,
                 quantile_cont(reviews_written, 0.90) AS p90_reviews_per_user,
                 quantile_cont(reviews_written, 0.99) AS p99_reviews_per_user,
                 max(reviews_written) AS max_reviews_per_user,
                 count(*) FILTER (WHERE reviews_written = 1) AS one_review_users,
                 count(*) FILTER (WHERE reviews_written >= 10) AS ten_plus_review_users
          FROM per_user GROUP BY category ORDER BY category
        ) TO '{qpath(out_dir / "reviewer_activity_distribution.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute("DROP TABLE IF EXISTS cross_category_duplicate_groups")
    con.execute(
        """
        CREATE TEMP TABLE cross_category_duplicate_groups AS
          WITH keys AS (
            SELECT
              md5(coalesce(user_id,'') || '|' || coalesce(parent_asin,'') || '|' || coalesce(timestamp::VARCHAR,'') || '|' || coalesce(rating::VARCHAR,'') || '|' || lower(trim(coalesce(text,'')))) AS review_key,
              category
            FROM reviews
          ),
          grouped AS (
            SELECT review_key, count(*) AS row_count, count(DISTINCT category) AS category_count, string_agg(DISTINCT category, ';') AS categories
            FROM keys
            GROUP BY review_key
            HAVING count(DISTINCT category) > 1
          )
          SELECT * FROM grouped
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT * FROM cross_category_duplicate_groups ORDER BY row_count DESC LIMIT 1000
        ) TO '{qpath(out_dir / "cross_category_duplicate_review_examples.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT count(*) AS duplicate_review_keys_across_categories,
                 coalesce(sum(row_count), 0) AS rows_in_cross_category_duplicate_keys
          FROM cross_category_duplicate_groups
        ) TO '{qpath(out_dir / "cross_category_duplicate_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_duplicate_pair_part(con: duckdb.DuckDBPyConnection, target: Path, category: str) -> None:
    con.execute("DROP TABLE IF EXISTS repeated_pairs")
    con.execute(
        "CREATE TEMP TABLE repeated_pairs AS "
        "SELECT user_id, parent_asin FROM reviews "
        "WHERE user_id IS NOT NULL AND parent_asin IS NOT NULL "
        "GROUP BY user_id, parent_asin HAVING count(*) > 1"
    )
    con.execute(
        f"""
        COPY (
          WITH pair_details AS (
            SELECT r.user_id, r.parent_asin, count(*) AS pair_review_count,
                   count(DISTINCT lower(trim(coalesce(r.text,'')))) AS distinct_text_count,
                   count(DISTINCT r.timestamp) AS distinct_timestamp_count,
                   count(DISTINCT r.rating) AS distinct_rating_count
            FROM reviews r JOIN repeated_pairs p USING(user_id, parent_asin)
            GROUP BY r.user_id, r.parent_asin
          )
          SELECT '{category}' AS category,
                 count(*) AS repeated_user_product_pairs,
                 coalesce(sum(pair_review_count), 0) AS reviews_in_repeated_user_product_pairs,
                 max(pair_review_count) AS max_reviews_in_one_user_product_pair,
                 count(*) FILTER (WHERE distinct_text_count = 1 AND distinct_timestamp_count = 1) AS same_text_same_time_pairs,
                 count(*) FILTER (WHERE distinct_text_count = 1 AND distinct_timestamp_count > 1) AS same_text_different_time_pairs,
                 count(*) FILTER (WHERE distinct_text_count > 1) AS changed_text_pairs,
                 count(*) FILTER (WHERE distinct_rating_count > 1) AS changed_rating_pairs
          FROM pair_details
        ) TO '{qpath(target)}' (HEADER, DELIMITER ',')
        """
    )
    con.execute("DROP TABLE repeated_pairs")


def write_reviewer_activity_part(con: duckdb.DuckDBPyConnection, target: Path, category: str) -> None:
    con.execute(
        f"""
        COPY (
          WITH per_user AS (
            SELECT user_id, count(*) AS reviews_written
            FROM reviews WHERE user_id IS NOT NULL AND user_id <> ''
            GROUP BY user_id
          )
          SELECT '{category}' AS category, count(*) AS users,
                 avg(reviews_written) AS mean_reviews_per_user,
                 median(reviews_written) AS median_reviews_per_user,
                 quantile_cont(reviews_written, 0.90) AS p90_reviews_per_user,
                 quantile_cont(reviews_written, 0.99) AS p99_reviews_per_user,
                 max(reviews_written) AS max_reviews_per_user,
                 count(*) FILTER (WHERE reviews_written = 1) AS one_review_users,
                 count(*) FILTER (WHERE reviews_written >= 10) AS ten_plus_review_users
          FROM per_user
        ) TO '{qpath(target)}' (HEADER, DELIMITER ',')
        """
    )


def write_cross_category_duplicates(con: duckdb.DuckDBPyConnection, out_dir: Path) -> None:
    category = pd.read_csv(out_dir / "category_summary.csv")
    overall = pd.read_csv(out_dir / "overall_summary.csv").iloc[0]
    if int(category.unique_products.sum()) == int(overall.unique_products):
        pd.DataFrame([{
            "duplicate_review_keys_across_categories": 0,
            "rows_in_cross_category_duplicate_keys": 0,
        }]).to_csv(out_dir / "cross_category_duplicate_summary.csv", index=False)
        pd.DataFrame(columns=["review_key", "row_count", "category_count", "categories"]).to_csv(
            out_dir / "cross_category_duplicate_review_examples.csv", index=False)
        return
    con.execute("DROP TABLE IF EXISTS shared_parent_asin")
    con.execute(
        f"CREATE TEMP TABLE shared_parent_asin AS "
        f"SELECT product_id AS parent_asin FROM read_parquet('{qpath(out_dir / 'product_review_statistics.parquet')}') "
        f"GROUP BY product_id HAVING count(DISTINCT category) > 1"
    )
    con.execute("DROP TABLE IF EXISTS cross_category_duplicate_groups")
    con.execute(
        """
        CREATE TEMP TABLE cross_category_duplicate_groups AS
          WITH keys AS (
            SELECT md5(coalesce(r.user_id,'') || '|' || coalesce(r.parent_asin,'') || '|' ||
                       coalesce(r.timestamp::VARCHAR,'') || '|' || coalesce(r.rating::VARCHAR,'') || '|' ||
                       lower(trim(coalesce(r.text,'')))) AS review_key,
                   r.category
            FROM reviews r JOIN shared_parent_asin s USING(parent_asin)
          )
          SELECT review_key, count(*) AS row_count,
                 count(DISTINCT category) AS category_count,
                 string_agg(DISTINCT category, ';') AS categories
          FROM keys GROUP BY review_key HAVING count(DISTINCT category) > 1
        """
    )
    con.execute(
        f"COPY (SELECT * FROM cross_category_duplicate_groups ORDER BY row_count DESC LIMIT 1000) "
        f"TO '{qpath(out_dir / 'cross_category_duplicate_review_examples.csv')}' (HEADER, DELIMITER ',')"
    )
    con.execute(
        f"COPY (SELECT count(*) AS duplicate_review_keys_across_categories, "
        f"coalesce(sum(row_count), 0) AS rows_in_cross_category_duplicate_keys "
        f"FROM cross_category_duplicate_groups) "
        f"TO '{qpath(out_dir / 'cross_category_duplicate_summary.csv')}' (HEADER, DELIMITER ',')"
    )


def write_duplicate_reports_batches(con: duckdb.DuckDBPyConnection, out_dir: Path, force: bool) -> None:
    review_cache = out_dir / "source_parquet" / "review"
    categories = [path.stem for path in sorted(review_cache.glob("*.parquet"))]
    checkpoint_dir = out_dir / "checkpoints"
    count_columns = [
        "repeated_user_product_pairs", "reviews_in_repeated_user_product_pairs",
        "same_text_same_time_pairs", "same_text_different_time_pairs",
        "changed_text_pairs", "changed_rating_pairs",
    ]
    for category in categories:
        part_dir = out_dir / "category_parts" / category
        bucket_dir = part_dir / "review_buckets"
        if bucket_dir.exists() and checkpoint_done(checkpoint_dir, f"review_buckets_{category}"):
            bucket_reports = []
            for bucket in range(16):
                bucket_report = part_dir / "duplicate_buckets" / f"{bucket}.csv"
                bucket_report.parent.mkdir(exist_ok=True)
                if not checkpoint_done(checkpoint_dir, f"duplicate_pairs_{category}_bucket_{bucket}") or force:
                    set_review_parquet_view(con, bucket_dir / f"review_bucket={bucket}" / "*.parquet")
                    run_step(
                        checkpoint_dir, f"duplicate_pairs_{category}_bucket_{bucket}", force,
                        lambda target=bucket_report: write_duplicate_pair_part(con, target, category),
                    )
                bucket_reports.append(pd.read_csv(bucket_report))
            merged = pd.concat(bucket_reports, ignore_index=True)
            row = {"category": category}
            row.update({column: int(merged[column].sum()) for column in count_columns})
            row["max_reviews_in_one_user_product_pair"] = merged[
                "max_reviews_in_one_user_product_pair"
            ].max()
            pd.DataFrame([row]).to_csv(part_dir / "duplicate_user_product_report.csv", index=False)
        else:
            if not checkpoint_done(checkpoint_dir, f"duplicate_pairs_{category}") or force:
                set_review_parquet_view(con, review_cache / f"{category}.parquet")
                run_step(
                    checkpoint_dir, f"duplicate_pairs_{category}", force,
                    lambda: write_duplicate_pair_part(
                        con, part_dir / "duplicate_user_product_report.csv", category),
                )
        if not checkpoint_done(checkpoint_dir, f"reviewer_activity_{category}") or force:
            set_review_parquet_view(con, review_cache / f"{category}.parquet")
            run_step(
                checkpoint_dir, f"reviewer_activity_{category}", force,
                lambda: write_reviewer_activity_part(
                    con, part_dir / "reviewer_activity_distribution.csv", category),
            )
    merge_part_csv(out_dir, categories, "duplicate_user_product_report.csv")
    merge_part_csv(out_dir, categories, "reviewer_activity_distribution.csv")
    create_parquet_views(con, review_cache, out_dir / "source_parquet" / "metadata_product")
    write_cross_category_duplicates(con, out_dir)


def write_candidate_tables(con: duckdb.DuckDBPyConnection, out_dir: Path, seed: int) -> None:
    stats_path = qpath(out_dir / "product_review_statistics.parquet")
    con.execute(
        f"""
        COPY (
          WITH p AS (SELECT * FROM read_parquet('{stats_path}')),
          thresholds AS (
            SELECT quantile_cont(total_text_chars, 0.10) AS low_text_threshold,
                   quantile_cont(total_text_chars, 0.90) AS high_text_threshold
            FROM p WHERE valid_text_review_count >= 20
          ),
          flags AS (
            SELECT *,
              valid_text_review_count >= 20 AS min20,
              valid_text_review_count >= 50 AS min50,
              valid_text_review_count >= 100 AS min100,
              has_low_and_high_ratings AS has_low_high,
              (high_rating_count::DOUBLE / nullif(review_count,0) >= 0.70 AND low_rating_count BETWEEN 1 AND greatest(1, floor(review_count * 0.20))) AS high_dominant_with_some_low,
              (rating_entropy >= 1.25 AND low_rating_count > 0 AND mid_rating_count > 0 AND high_rating_count > 0) AS relatively_balanced_rating_distribution,
              low_text_threshold, high_text_threshold
            FROM p, thresholds
          )
          SELECT
            category,
            count(*) AS products_total,
            sum(CASE WHEN min20 THEN 1 ELSE 0 END) AS products_valid_reviews_min20,
            sum(CASE WHEN min50 THEN 1 ELSE 0 END) AS products_valid_reviews_min50,
            sum(CASE WHEN min100 THEN 1 ELSE 0 END) AS products_valid_reviews_min100,
            sum(CASE WHEN min20 AND has_low_high THEN 1 ELSE 0 END) AS min20_products_with_low_and_high,
            sum(CASE WHEN min50 AND has_low_high THEN 1 ELSE 0 END) AS min50_products_with_low_and_high,
            sum(CASE WHEN min100 AND has_low_high THEN 1 ELSE 0 END) AS min100_products_with_low_and_high,
            sum(CASE WHEN min20 THEN low_rating_valid_text_count ELSE 0 END) AS low_rating_text_reviews_in_min20_products,
            sum(CASE WHEN has_low_high THEN 1 ELSE 0 END) AS products_with_low_and_high,
            sum(CASE WHEN high_dominant_with_some_low THEN 1 ELSE 0 END) AS high_dominant_with_some_low_products,
            sum(CASE WHEN relatively_balanced_rating_distribution THEN 1 ELSE 0 END) AS relatively_balanced_rating_distribution_products,
            sum(CASE WHEN min20 AND total_text_chars <= low_text_threshold THEN 1 ELSE 0 END) AS min20_low_total_text_products,
            sum(CASE WHEN min20 AND total_text_chars >= high_text_threshold THEN 1 ELSE 0 END) AS min20_high_total_text_products,
            max(low_text_threshold) AS global_low_text_threshold,
            max(high_text_threshold) AS global_high_text_threshold,
            quantile_cont(total_text_chars, 0.10) FILTER (WHERE min20) AS p10_total_text_chars,
            quantile_cont(total_text_chars, 0.90) FILTER (WHERE min20) AS p90_total_text_chars
          FROM flags
          GROUP BY category
          ORDER BY category
        ) TO '{qpath(out_dir / "candidate_product_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          WITH p AS (SELECT * FROM read_parquet('{stats_path}')),
          thresholds AS (
            SELECT
              quantile_cont(total_text_chars, 0.10) AS low_text_threshold,
              quantile_cont(total_text_chars, 0.90) AS high_text_threshold
            FROM p WHERE valid_text_review_count >= 20
          ),
          flags AS (
            SELECT p.*,
              CASE
                WHEN valid_text_review_count >= 100 AND has_low_and_high_ratings THEN 'min100_low_high'
                WHEN valid_text_review_count >= 50 AND high_rating_count::DOUBLE / nullif(review_count,0) >= 0.70 AND low_rating_count > 0 THEN 'min50_high_dominant_some_low'
                WHEN valid_text_review_count >= 20 AND rating_entropy >= 1.25 AND low_rating_count > 0 AND mid_rating_count > 0 AND high_rating_count > 0 THEN 'min20_rating_distribution_diagnostic'
                WHEN total_text_chars <= low_text_threshold AND valid_text_review_count >= 20 THEN 'min20_low_total_text'
                WHEN total_text_chars >= high_text_threshold AND valid_text_review_count >= 20 THEN 'min20_high_total_text'
                ELSE NULL
              END AS candidate_reason
            FROM p, thresholds
          )
          SELECT
            abs(hash(product_id || '{seed}'::VARCHAR)) AS selection_hash,
            *
          FROM flags
          WHERE candidate_reason IS NOT NULL
          ORDER BY candidate_reason, selection_hash
        ) TO '{qpath(out_dir / "candidate_products.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT 'random_min20' AS sample_group, category, product_id,
                 valid_text_review_count, low_rating_count, mid_rating_count,
                 high_rating_count, total_text_chars, rating_entropy,
                 abs(hash(category || product_id || '{seed}'::VARCHAR)) AS selection_hash
          FROM read_parquet('{stats_path}')
          WHERE valid_text_review_count >= 20
          ORDER BY selection_hash
          LIMIT 200
        ) TO '{qpath(out_dir / "random_product_sample.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          WITH ranked AS (
            SELECT 'diagnostic' AS sample_group, category, product_id,
                   candidate_reason, valid_text_review_count, low_rating_count,
                   mid_rating_count, high_rating_count, total_text_chars,
                   rating_entropy, selection_hash,
                   row_number() OVER (PARTITION BY category, candidate_reason ORDER BY selection_hash) AS stratum_rank
            FROM read_csv_auto(
              '{qpath(out_dir / "candidate_products.csv")}',
              types={{'selection_hash':'UBIGINT', 'product_id':'VARCHAR',
                      'category':'VARCHAR', 'candidate_reason':'VARCHAR'}}
            )
          )
          SELECT * EXCLUDE(stratum_rank) FROM ranked WHERE stratum_rank <= 5
          ORDER BY category, candidate_reason, selection_hash
        ) TO '{qpath(out_dir / "diagnostic_product_sample.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT category,
                 count(*) FILTER (WHERE valid_text_review_count = 0) AS no_valid_text_products,
                 count(*) FILTER (WHERE valid_text_review_count BETWEEN 1 AND 19) AS valid_text_1_to_19_products,
                 count(*) FILTER (WHERE valid_text_review_count >= 20) AS eligible_min20_products,
                 count(*) FILTER (WHERE product_title IS NULL) AS no_matched_product_title_products
          FROM read_parquet('{stats_path}')
          GROUP BY category ORDER BY category
        ) TO '{qpath(out_dir / "candidate_exclusion_reasons.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_review_bundle_sample(con: duckdb.DuckDBPyConnection, out_dir: Path, seed: int) -> None:
    con.execute(
        f"""
        COPY (
          WITH selected AS (
            SELECT category, product_id, string_agg(sample_group, ';' ORDER BY sample_group) AS sample_group
            FROM (
              SELECT category, product_id, sample_group FROM read_csv_auto(
                '{qpath(out_dir / "random_product_sample.csv")}',
                types={{'selection_hash':'UBIGINT', 'product_id':'VARCHAR'}})
              UNION ALL
              SELECT category, product_id, sample_group FROM read_csv_auto(
                '{qpath(out_dir / "diagnostic_product_sample.csv")}',
                types={{'selection_hash':'UBIGINT', 'product_id':'VARCHAR'}})
            )
            GROUP BY category, product_id
          ), eligible AS (
            SELECT s.sample_group, r.category, r.parent_asin AS product_id, r.rating,
                   r.review_title, r.text, r.timestamp, r.review_datetime,
                   r.user_id, r.verified_purchase,
                   CASE WHEN r.rating <= 2 THEN 'low'
                        WHEN r.rating >= 4 THEN 'high' ELSE 'middle' END AS rating_band,
                   abs(hash(coalesce(r.user_id,'') || coalesce(r.timestamp::VARCHAR,'') ||
                            coalesce(r.text,'') || '{seed}'::VARCHAR)) AS selection_hash
            FROM reviews r
            JOIN selected s ON r.category = s.category AND r.parent_asin = s.product_id
            WHERE r.text IS NOT NULL AND length(trim(r.text)) > 0
          ), band_ranked AS (
            SELECT *, row_number() OVER (
              PARTITION BY category, product_id, rating_band ORDER BY selection_hash
            ) AS band_rank
            FROM eligible
          ), prioritized AS (
            SELECT *,
                   CASE WHEN rating_band = 'low' AND band_rank = 1 THEN 0
                        WHEN rating_band = 'high' AND band_rank = 1 THEN 1
                        WHEN rating_band = 'low' AND band_rank <= 20 THEN 2
                        ELSE 3 END AS priority
            FROM band_ranked
          ), final_ranked AS (
            SELECT *, row_number() OVER (
              PARTITION BY category, product_id ORDER BY priority, selection_hash
            ) AS bundle_rank
            FROM prioritized
          )
          SELECT sample_group, category, product_id, rating, review_title,
                 text, timestamp, review_datetime, user_id, verified_purchase,
                 rating_band, bundle_rank
          FROM final_ranked WHERE bundle_rank <= 100
        ) TO '{qpath(out_dir / "review_bundle_sample.parquet")}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
          WITH full_counts AS (
            SELECT category, product_id, review_count, valid_text_review_count,
                   low_rating_count, high_rating_count
            FROM read_parquet('{qpath(out_dir / "product_review_statistics.parquet")}')
          ), bundle_counts AS (
            SELECT category, product_id, count(*) AS bundle_reviews,
                   count(*) FILTER (WHERE rating <= 2) AS bundle_low_rating_reviews,
                   count(*) FILTER (WHERE rating >= 4) AS bundle_high_rating_reviews
            FROM read_parquet('{qpath(out_dir / "review_bundle_sample.parquet")}')
            GROUP BY category, product_id
          )
          SELECT b.*, f.review_count, f.valid_text_review_count,
                 f.low_rating_count, f.high_rating_count,
                 (f.low_rating_count > 0 AND b.bundle_low_rating_reviews = 0) AS low_rating_lost,
                 (f.high_rating_count > 0 AND b.bundle_high_rating_reviews = 0) AS high_rating_lost
          FROM bundle_counts b JOIN full_counts f USING(category, product_id)
          ORDER BY category, product_id
        ) TO '{qpath(out_dir / "bundle_coverage_report.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          WITH marked AS (
            SELECT category, product_id, rating, text,
                   regexp_matches(lower(text), '{COMPLAINT_RE}') AS complaint_marker,
                   unnest([
                     CASE WHEN regexp_matches(lower(text), '\\b(battery|charger|charge)\\b') THEN 'battery_or_charging' END,
                     CASE WHEN regexp_matches(lower(text), '\\b(durable|durability|broke|broken|defect)\\b') THEN 'durability' END,
                     CASE WHEN regexp_matches(lower(text), '\\b(size|small|large|fit)\\b') THEN 'size_or_fit' END,
                     CASE WHEN regexp_matches(lower(text), '\\b(usability|easy to use|hard to use|difficult)\\b') THEN 'usability' END,
                     CASE WHEN regexp_matches(lower(text), '\\b(leak|smell|taste|noise|screen|packaging|delivery|refund|return)\\b') THEN 'other_specific_attribute' END
                   ]) AS aspect_marker
            FROM read_parquet('{qpath(out_dir / "review_bundle_sample.parquet")}')
          )
          SELECT category, product_id, aspect_marker,
                 count(*) AS candidate_review_mentions,
                 CASE WHEN count(*) >= 2 THEN 'repeated_marker_reviews'
                      ELSE 'single_marker_review' END AS occurrence_class,
                 any_value(substr(text, 1, 300)) AS example_excerpt
          FROM marked
          WHERE complaint_marker AND aspect_marker IS NOT NULL
          GROUP BY category, product_id, aspect_marker
          ORDER BY category, product_id, aspect_marker
        ) TO '{qpath(out_dir / "bundle_issue_marker_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )


def write_rule_samples(con: duckdb.DuckDBPyConnection, out_dir: Path, seed: int) -> None:
    con.execute("DROP TABLE IF EXISTS rule_base_sample")
    con.execute(
        f"""
        CREATE TEMP TABLE rule_base_sample AS
        SELECT * FROM reviews
        WHERE text IS NOT NULL AND length(trim(text)) > 0
          AND hash(coalesce(user_id,'') || coalesce(parent_asin,'') ||
                   coalesce(timestamp::VARCHAR,'') || '{seed}'::VARCHAR) % 1000 = 0
        """
    )
    case_sql = f"""
      WITH base AS (
        SELECT
          category,
          parent_asin AS product_id,
          rating,
          review_title,
          text,
          timestamp,
          review_datetime,
          user_id,
          lower(coalesce(review_title,'') || ' ' || coalesce(text,'')) AS txt
        FROM rule_base_sample
        WHERE text IS NOT NULL AND length(trim(text)) > 0
      ),
      cases AS (
        SELECT *, 'high_rating_with_drawback_marker' AS candidate_type FROM base
        WHERE rating >= 4 AND regexp_matches(txt, '{DRAWBACK_RE}')
        UNION ALL
        SELECT *, 'low_rating_with_positive_marker' AS candidate_type FROM base
        WHERE rating <= 2 AND regexp_matches(txt, '{POSITIVE_RE}')
        UNION ALL
        SELECT *, 'mixed_positive_drawback_markers' AS candidate_type FROM base
        WHERE regexp_matches(txt, '{POSITIVE_RE}') AND regexp_matches(txt, '{DRAWBACK_RE}')
        UNION ALL
        SELECT *, 'attribute_complaint_marker' AS candidate_type FROM base
        WHERE regexp_matches(txt, '{ATTRIBUTE_RE}') AND regexp_matches(txt, '{COMPLAINT_RE}')
      )
      SELECT
        md5(category || '|' || coalesce(product_id,'') || '|' || coalesce(user_id,'') || '|' || coalesce(timestamp::VARCHAR,'') || '|' || candidate_type) AS sample_id,
        product_id,
        category,
        rating,
        review_title,
        text,
        timestamp,
        candidate_type,
        regexp_extract(coalesce(text,''), '([^.!?]*(but|however|wish|unfortunately|problem|issue|complaint|broken|broke|defect|battery|durability|size|small|large|hard to|difficult|leak|refund)[^.!?]*[.!?]?)', 1, 'i') AS evidence_sentence,
        abs(hash(coalesce(product_id,'') || coalesce(user_id,'') || coalesce(timestamp::VARCHAR,'') || candidate_type || '{seed}'::VARCHAR)) AS sample_hash
      FROM cases
    """
    con.execute(
        f"""
        COPY (
          SELECT * EXCLUDE(sample_hash)
          FROM ({case_sql})
          ORDER BY sample_hash
          LIMIT 5000
        ) TO '{qpath(out_dir / "exploratory_rule_review_samples.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT category, parent_asin AS product_id, rating, review_title, text, timestamp, user_id,
              lower(coalesce(review_title,'') || ' ' || coalesce(text,'')) AS txt
            FROM rule_base_sample
            WHERE text IS NOT NULL AND length(trim(text)) > 0
          ),
          hits AS (
            SELECT *,
              CASE
                WHEN regexp_matches(txt, '{VINE_RE}') AND NOT regexp_matches(txt, '{NEGATED_VINE_RE}') THEN 'vine_disclosure_candidate'
                WHEN regexp_matches(txt, '{FREE_PRODUCT_RE}') AND NOT regexp_matches(txt, '{NEGATED_FREE_RE}') THEN 'free_product_or_exchange_candidate'
                WHEN regexp_matches(txt, '{DISCOUNT_RE}') AND NOT regexp_matches(txt, '{NEGATED_FREE_RE}') THEN 'discount_rebate_candidate'
                WHEN regexp_matches(txt, '{SELF_PAID_RE}') THEN 'self_paid_statement_candidate'
                ELSE NULL
              END AS rule_id
            FROM base
          )
          SELECT
            md5(category || '|' || coalesce(product_id,'') || '|' || coalesce(user_id,'') || '|' || coalesce(timestamp::VARCHAR,'') || '|' || rule_id) AS sample_id,
            category,
            product_id,
            rating,
            timestamp,
            rule_id,
            regexp_extract(coalesce(text,''), '([^.!?]*(free product|discount|complimentary|exchange for|honest review|unbiased review|vine|rebate|reimbursed|own money|paid full price|not sponsored|no discount)[^.!?]*[.!?]?)', 1, 'i') AS evidence_sentence,
            review_title,
            text
          FROM hits
          WHERE rule_id IS NOT NULL
          ORDER BY abs(hash(coalesce(product_id,'') || coalesce(user_id,'') || coalesce(timestamp::VARCHAR,'') || rule_id || '{seed}'::VARCHAR))
          LIMIT 5000
        ) TO '{qpath(out_dir / "compensation_disclosure_candidates.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          SELECT category, candidate_type, count(*) AS sampled_candidate_rows,
                 1000 AS inverse_sampling_fraction
          FROM ({case_sql})
          GROUP BY category, candidate_type
          ORDER BY category, candidate_type
        ) TO '{qpath(out_dir / "exploratory_rule_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )
    con.execute(
        f"""
        COPY (
          WITH base AS (
            SELECT category, rating, verified_purchase,
                   lower(coalesce(review_title,'') || ' ' || coalesce(text,'')) AS txt
            FROM rule_base_sample
          ), hits AS (
            SELECT category, rating, verified_purchase,
                   CASE
                     WHEN regexp_matches(txt, '{VINE_RE}') AND NOT regexp_matches(txt, '{NEGATED_VINE_RE}') THEN 'vine_disclosure_candidate'
                     WHEN regexp_matches(txt, '{FREE_PRODUCT_RE}') AND NOT regexp_matches(txt, '{NEGATED_FREE_RE}')
                       THEN 'free_product_or_exchange_candidate'
                     WHEN regexp_matches(txt, '{DISCOUNT_RE}') AND NOT regexp_matches(txt, '{NEGATED_FREE_RE}')
                       THEN 'discount_rebate_candidate'
                     WHEN regexp_matches(txt, '{SELF_PAID_RE}') THEN 'self_paid_statement_candidate'
                     ELSE NULL END AS rule_id
            FROM base
          )
          SELECT category, rating, verified_purchase, rule_id,
                 count(*) AS sampled_candidate_rows,
                 1000 AS inverse_sampling_fraction
          FROM hits WHERE rule_id IS NOT NULL
          GROUP BY category, rating, verified_purchase, rule_id
          ORDER BY category, rule_id, rating, verified_purchase
        ) TO '{qpath(out_dir / "compensation_disclosure_summary.csv")}' (HEADER, DELIMITER ',')
        """
    )


def evidence_from_text(title: str, body: str, pattern: str) -> str:
    joined = " ".join(part for part in [title, body] if part and part != "nan")
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", joined) if part.strip()]
    for sentence in sentences:
        if re.search(pattern, sentence, flags=re.IGNORECASE):
            return sentence
    return sentences[0] if sentences else ""


def finalize_evidence_files(out_dir: Path) -> None:
    cases = [
        (
            "exploratory_rule_review_samples.csv", "candidate_type",
            {
                "high_rating_with_drawback_marker": DRAWBACK_RE,
                "low_rating_with_positive_marker": POSITIVE_RE,
                "mixed_positive_drawback_markers": f"({POSITIVE_RE}|{DRAWBACK_RE})",
                "attribute_complaint_marker": f"({ATTRIBUTE_RE}|{COMPLAINT_RE})",
            },
        ),
        (
            "compensation_disclosure_candidates.csv", "rule_id",
            {
                "vine_disclosure_candidate": VINE_RE,
                "free_product_or_exchange_candidate": FREE_PRODUCT_RE,
                "discount_rebate_candidate": DISCOUNT_RE,
                "self_paid_statement_candidate": SELF_PAID_RE,
            },
        ),
    ]
    for filename, kind_column, patterns in cases:
        path = out_dir / filename
        rows = pd.read_csv(path, dtype={"product_id": "string"})
        missing = rows["evidence_sentence"].isna() | rows["evidence_sentence"].eq("")
        if not missing.any():
            continue
        for idx in rows.index[missing]:
            row = rows.loc[idx]
            rows.at[idx, "evidence_sentence"] = evidence_from_text(
                str(row["review_title"]), str(row["text"]), patterns[str(row[kind_column])]
            )
        rows.to_csv(path, index=False, encoding="utf-8")
        logging.info("filled %s missing evidence sentences in %s", int(missing.sum()), filename)


def build_annotation_files(out_dir: Path, seed: int) -> None:
    src = pd.read_csv(out_dir / "exploratory_rule_review_samples.csv")
    if src.empty:
        ann = src.copy()
    else:
        src = src.drop_duplicates(["category", "product_id", "timestamp", "text"]).copy()
        strata = [
            group.sample(n=1, random_state=seed)
            for _, group in src.groupby(["candidate_type", "category", "rating"], dropna=False)
        ]
        first_pass = pd.concat(strata, ignore_index=True)
        if len(first_pass) > 800:
            first_pass = first_pass.sample(n=800, random_state=seed)
        remaining = src.loc[~src["sample_id"].isin(first_pass["sample_id"])]
        fill_count = min(max(0, 800 - len(first_pass)), len(remaining))
        extra = remaining.sample(n=fill_count, random_state=seed) if fill_count else remaining.head(0)
        ann = pd.concat([first_pass, extra], ignore_index=True).sample(frac=1, random_state=seed)
    for col in ["human_aspect", "human_sentiment", "human_issue", "human_importance", "annotator_notes"]:
        ann[col] = ""
    keep = [
        "sample_id",
        "product_id",
        "category",
        "rating",
        "review_title",
        "text",
        "timestamp",
        "candidate_type",
        "evidence_sentence",
        "human_aspect",
        "human_sentiment",
        "human_issue",
        "human_importance",
        "annotator_notes",
    ]
    ann[keep].to_csv(out_dir / "annotation_sample.csv", index=False, encoding="utf-8-sig")
    hidden = ann[keep].drop(columns=["candidate_type", "evidence_sentence"])
    hidden.to_csv(out_dir / "annotation_sample_blinded.csv", index=False, encoding="utf-8-sig")

    guide = """# 수동 검수 안내: Review Summary EDA v1

AI 요약 실험 전에 리뷰 원문과 규칙 후보의 적합성을 확인하는 검수 파일이다. 검수 결과도 전체 Amazon 리뷰의 정답 분포는 아니다.

## 입력 필드

- `human_aspect`: 언급된 속성. 예: 배터리, 내구성, 크기, 사용성, 냄새, 포장, 배송, 가격, 착용감, 맛, 화면, 고객지원, 기타. 복수 속성은 세미콜론으로 구분한다.
- `human_sentiment`: 문장 내용 기준 `positive`, `negative`, `mixed`, `neutral/unclear` 중 선택한다. 별점을 그대로 옮기지 않는다.
- `human_issue`: 구체적인 장점·단점이나 경험을 짧게 적는다. 근거가 없으면 비워 둔다.
- `human_importance`: 이 단계에는 정답 기준이 없다. 원문에 판단 근거가 충분할 때만 `low`, `medium`, `high`를 시험적으로 적고, 불확실하면 비워 둔다.
- `annotator_notes`: 모호함, 반어법, 맥락 부족, 보상 문구, 규칙의 오탐·누락 등을 기록한다.

## 검수 순서와 주의

1. 가능하면 `annotation_sample_blinded.csv`의 원문을 먼저 보고 사람 입력 필드를 채운다. 이후 일반 파일의 `candidate_type`과 `evidence_sentence`를 열어 규칙 후보와 비교한다.
2. 한 리뷰의 장점과 단점, 높은 별점의 단점, 낮은 별점의 장점을 각각 확인한다. 별점은 문장 감성 라벨이 아니다.
3. 규칙 ID와 키워드는 감성·속성·중요도의 정답이 아니다. 드물게 등장한 불만도 자동으로 중요하다고 판단하지 않는다.
4. `verified_purchase=true`는 무보상 구매의 증거가 아니며, 공개 문구가 없다는 사실도 무보상의 증거가 아니다.
"""
    (out_dir / "annotation_guide.md").write_text(guide, encoding="utf-8")
    selection_rules = f"""# 제품·리뷰 표본 선정 기준

Seed: `{seed}`. 해시 기반 선정에 이 값을 고정했다. smoke test는 원본 파일마다 앞 10,000행이므로 전체 분포 추정에 쓰지 않는다.

## 전체 제품 집계와 조건 비교

- 제품 키는 `parent_asin`이며, 제품별 통계는 텍스트가 빈 리뷰도 포함한다. 유효 텍스트 리뷰 수는 `text`가 null이 아니고 공백을 제거했을 때 비어 있지 않은 리뷰 수다.
- 유효 텍스트 20 / 50 / 100개 이상 조건을 각각 비교하며 하나로 최종 확정하지 않는다.
- 고·저평점 공존: 1~2점 리뷰와 4~5점 리뷰가 각각 1개 이상.
- 고평점 우세·일부 저평점 요약 조건: 전체 리뷰의 4~5점 비율 70% 이상, 1~2점 수는 1개 이상이고 전체의 최대 20% 이하.
- 별점 분포 진단 조건: 엔트로피 1.25 이상이며 저·중·고평점 리뷰가 모두 존재. 감성 양극화의 확정 라벨이 아니다.
- 적은/많은 텍스트 총량: 유효 텍스트 20개 이상 제품의 전체 텍스트 문자 수에서 전역 10·90분위수를 기준으로 분류한다.

## 추출 표본

- 무작위 제품 표본: 유효 텍스트 20개 이상 제품 중 seed를 포함한 해시 순서의 앞 200개.
- 진단 제품 표본: `candidate_products.csv`에서 카테고리·우선순위별 사유당 최대 5개. 사유는 유효 100개+고저 공존, 유효 50개+고평점 70% 이상+저평점 1개 이상, 유효 20개+엔트로피 조건, 유효 20개+낮은 텍스트 총량, 유효 20개+높은 텍스트 총량 순으로 한 제품에 첫 사유 하나를 부여한다. 두 번째 진단 사유는 위의 '최대 20%' 요약 조건보다 넓다.
- 무작위 표본과 진단 표본은 분리해 저장한다. 진단 표본의 분포를 전체 Amazon의 분포처럼 읽지 않는다.
- 리뷰 묶음: 선택 제품당 비어 있지 않은 텍스트 리뷰 최대 100개. 저평점·고평점 각각 1개를 우선하고 저평점 최대 20개를 먼저 배치한 뒤 seed 해시 순서로 채운다. `bundle_coverage_report.csv`에서 원제품의 별점 구간이 묶음에서 사라졌는지 확인한다.
- 검수 표본: 결정적 1/1,000 리뷰 표본의 규칙 후보 중 최대 800개. 사례 유형·카테고리·별점 층에서 먼저 1개씩 뽑고 남은 수를 seed로 채운다. 규칙 후보 수는 전체 모집단의 실제 불만·보상 빈도가 아니다.

## 제외와 비제외

- 제품 키가 없는 리뷰는 제품별 통계에서 제외한다. 유효 텍스트 20개 미만 제품은 제품 표본에서 제외하며, `candidate_exclusion_reasons.csv`에 규모를 기록한다.
- 제품명 메타데이터가 없다는 이유만으로 제외하지 않는다. 동일 사용자·제품의 반복 기록과 보상 문구 후보도 자동 제외하지 않는다.
"""
    (out_dir / "selection_rules.md").write_text(selection_rules, encoding="utf-8")


def write_figures(out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    cat = pd.read_csv(out_dir / "category_summary.csv")
    if not cat.empty:
        top = cat.sort_values("review_count", ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        plt.barh(top["category"], top["review_count"])
        plt.gca().invert_yaxis()
        plt.title("Top Categories by Review Count")
        plt.tight_layout()
        plt.savefig(fig_dir / "top_categories_review_count.png", dpi=150)
        plt.close()

        rating_cols = [f"rating_{i}_count" for i in range(1, 6)]
        rating_totals = cat[rating_cols].sum()
        plt.figure(figsize=(7, 5))
        plt.bar([str(i) for i in range(1, 6)], rating_totals.values)
        plt.xlabel("Rating")
        plt.ylabel("Reviews")
        plt.title("Overall Rating Distribution")
        plt.tight_layout()
        plt.savefig(fig_dir / "overall_rating_distribution.png", dpi=150)
        plt.close()

    monthly = pd.read_csv(out_dir / "monthly_summary.csv")
    if not monthly.empty:
        overall = monthly.groupby("month_start", as_index=False)["review_count"].sum()
        plt.figure(figsize=(11, 5))
        plt.plot(pd.to_datetime(overall["month_start"]), overall["review_count"])
        plt.title("Monthly Review Counts")
        plt.tight_layout()
        plt.savefig(fig_dir / "monthly_review_counts.png", dpi=150)
        plt.close()


def table_value(path: Path, column: str, default=None):
    try:
        df = pd.read_csv(path)
        if df.empty or column not in df.columns:
            return default
        return df[column].sum()
    except Exception:
        return default


def write_report(out_dir: Path, mode: str, seed: int) -> None:
    cat = pd.read_csv(out_dir / "category_summary.csv")
    join = pd.read_csv(out_dir / "metadata_join_report.csv")
    cand = pd.read_csv(out_dir / "candidate_product_summary.csv")
    dq = pd.read_csv(out_dir / "data_quality_report.csv")
    dup = pd.read_csv(out_dir / "duplicate_user_product_report.csv")
    cross_dup = pd.read_csv(out_dir / "cross_category_duplicate_summary.csv")
    overall = pd.read_csv(out_dir / "overall_summary.csv")
    similarity = pd.read_csv(out_dir / "text_similarity_sample_report.csv")
    compensation = pd.read_csv(out_dir / "compensation_disclosure_summary.csv")
    random_sample = pd.read_csv(out_dir / "random_product_sample.csv")
    diagnostic_sample = pd.read_csv(out_dir / "diagnostic_product_sample.csv")
    bundle_coverage = pd.read_csv(out_dir / "bundle_coverage_report.csv")
    annotation = pd.read_csv(out_dir / "annotation_sample.csv")
    total_reviews = int(overall.loc[0, "review_count"])
    total_products = int(overall.loc[0, "unique_products"])
    total_users = int(overall.loc[0, "unique_users"])
    min20 = int(cand["products_valid_reviews_min20"].sum()) if not cand.empty else 0
    min50 = int(cand["products_valid_reviews_min50"].sum()) if not cand.empty else 0
    min100 = int(cand["products_valid_reviews_min100"].sum()) if not cand.empty else 0
    low_high = int(cand["products_with_low_and_high"].sum()) if not cand.empty else 0
    min20_low_high = int(cand["min20_products_with_low_and_high"].sum()) if not cand.empty else 0
    min50_low_high = int(cand["min50_products_with_low_and_high"].sum()) if not cand.empty else 0
    min100_low_high = int(cand["min100_products_with_low_and_high"].sum()) if not cand.empty else 0
    min20_low_text = int(cand["low_rating_text_reviews_in_min20_products"].sum()) if not cand.empty else 0
    low_text_products = int(cand["min20_low_total_text_products"].sum()) if not cand.empty else 0
    high_text_products = int(cand["min20_high_total_text_products"].sum()) if not cand.empty else 0
    high_some_low = int(cand["high_dominant_with_some_low_products"].sum()) if not cand.empty else 0
    balanced = int(cand["relatively_balanced_rating_distribution_products"].sum()) if not cand.empty else 0
    matched_review_rows = int(join.loc[0, "matched_review_rows"]) if not join.empty else 0
    review_rows = int(join.loc[0, "review_rows"]) if not join.empty else 0
    match_rate = matched_review_rows / review_rows if review_rows else 0
    matched_products = int(join.loc[0, "matched_review_parent_asin"])
    review_products = int(join.loc[0, "review_unique_parent_asin"])
    product_match_rate = matched_products / review_products if review_products else 0
    raw_join_rows = int(join.loc[0, "raw_join_rows_if_not_deduplicated"])
    metadata_dupe_keys = int(join.loc[0, "duplicated_metadata_keys"])
    dup_pairs = int(dup["repeated_user_product_pairs"].sum()) if not dup.empty and "repeated_user_product_pairs" in dup else 0
    bundle_low_lost = int(bundle_coverage["low_rating_lost"].sum()) if not bundle_coverage.empty else 0
    duplicate_across_categories = int(cross_dup.loc[0, "duplicate_review_keys_across_categories"])
    relaxed_path = out_dir / "cross_category_relaxed_duplicate_summary.csv"
    if relaxed_path.exists():
        relaxed = pd.read_csv(relaxed_path).iloc[0]
        relaxed_note = (
            f"- 제품 키를 제외하고 사용자·시각·별점·정규화 본문의 MD5 지문이 둘 이상의 카테고리에 나타난 후보는 "
            f"{int(relaxed['fingerprint_keys_across_categories']):,}개 지문, "
            f"{int(relaxed['rows_in_fingerprint_groups']):,}행이다. "
            "해시 충돌 가능성과 서로 다른 제품에 같은 문구를 쓴 경우가 있으므로 실제 동일 경험으로 확정하지 않는다."
        )
    else:
        relaxed_note = ""
    sample_ascii_rows = int(similarity["sampled_nonempty_reviews"].sum()) if not similarity.empty else 0
    sample_exact_duplicate_rows = int((similarity["sampled_nonempty_reviews"] - similarity["unique_exact_texts"]).sum()) if not similarity.empty else 0
    sample_normalized_duplicate_rows = int((similarity["sampled_nonempty_reviews"] - similarity["unique_normalized_texts"]).sum()) if not similarity.empty else 0
    compensation_hits = int(compensation["sampled_candidate_rows"].sum()) if not compensation.empty else 0
    review_dq = dq.loc[dq["kind"] == "review"]
    first_date = review_dq["min_review_datetime"].dropna().astype(str).min()
    last_date = review_dq["max_review_datetime"].dropna().astype(str).max()
    ms_rows = int(review_dq["timestamp_milliseconds_rows"].sum())
    sec_rows = int(review_dq["timestamp_seconds_rows"].sum())
    parse_errors = int(dq["parse_error_count"].sum())

    top_cats = cat.sort_values("review_count", ascending=False).head(10)[["category", "review_count", "unique_products", "avg_rating"]]
    cand_top = cand.sort_values("products_valid_reviews_min50", ascending=False).head(10)

    report = f"""# AI 리뷰 요약 연구를 위한 1차 EDA

생성: {now_stamp()} | 실행: `{mode}` | seed: `{seed}`

## A. 요약 연구에 사용할 수 있는 규모

- 리뷰 {total_reviews:,}건, 고유 `parent_asin` {total_products:,}개, 고유 사용자 {total_users:,}명. 카테고리 간 중복은 전체 고유 수에서 한 번만 셌다.
- 유효 텍스트 리뷰가 20 / 50 / 100개 이상인 카테고리-제품 조합: {min20:,} / {min50:,} / {min100:,}개.
- 리뷰 작성 범위: {first_date} ~ {last_date}. 데이터셋 이름의 2023년으로 날짜를 제한하지 않았다.
- 메타데이터 연결: 리뷰 {matched_review_rows:,}/{review_rows:,}건 ({match_rate:.2%}), 제품 키 {matched_products:,}/{review_products:,}개 ({product_match_rate:.2%}). 카테고리별 연결률은 `metadata_join_by_category.csv`에 있다. 제품명은 `title`, 세부 카테고리는 `categories`, 브랜드 관련 대리 필드는 `store`다. `store`를 검증된 브랜드명으로 보지 않는다.

리뷰 수 상위 카테고리:

{top_cats.to_markdown(index=False)}

## B. 장단점이 함께 존재할 가능성이 있는 제품

- 1~2점과 4~5점 리뷰가 모두 있는 카테고리-제품 조합: {low_high:,}개. 유효 텍스트 20 / 50 / 100개 기준에서는 {min20_low_high:,} / {min50_low_high:,} / {min100_low_high:,}개.
- 고평점이 70% 이상이며 저평점이 소수인 조합: {high_some_low:,}개. 별점 분포 엔트로피가 1.25 이상이고 저·중·고평점이 모두 있는 진단 조건: {balanced:,}개.

이 숫자는 별점 구성만 나타낸다. 문장 감성이나 의견의 양극화를 확정하지 않는다. `candidate_product_summary.csv`의 조건별 수는 서로 겹칠 수 있지만, `candidate_products.csv`의 진단 사유는 우선순위에 따라 제품당 하나만 부여했다. 진단 사유의 고평점 우세 조건은 저평점 비율 상한이 없는 넓은 조건이며, 위의 '저평점이 소수' 집계와 정확히 같은 집합은 아니다.

## C. 소수 저평점 의견을 분석할 표본 규모

- 유효 텍스트 20개 이상 제품에 포함된 1~2점 텍스트 리뷰: {min20_low_text:,}건. 제품별·카테고리별 분포는 `candidate_product_summary.csv`를 참조한다.
- 유효 텍스트 20개 이상 조합의 텍스트 총량 하위·상위 10% 기준에 드는 조합: {low_text_products:,} / {high_text_products:,}개. 기준 문자 수와 카테고리 분포는 후보 요약 파일에 있다.
- 무작위 제품 표본 {len(random_sample):,}개와 조건별 진단 표본 {len(diagnostic_sample):,}행을 별도 저장했다. 진단 표본은 의도적으로 조건을 강화해 뽑았으며 Amazon 전체 분포가 아니다.
- 각 선택 제품에서 최대 100개 리뷰를 뽑았고, 원제품에는 저평점이 있지만 묶음에는 하나도 없는 제품은 {bundle_low_lost:,}개였다. 상세 원인은 `bundle_coverage_report.csv`에서 확인한다. 원래 저평점 리뷰에 텍스트가 없을 수도 있다.

유효 텍스트 50개 이상 제품이 많은 카테고리:

{cand_top[["category", "products_valid_reviews_min50", "min50_products_with_low_and_high"]].to_markdown(index=False)}

## D. 누락, 중복, 표본 편향

- 원본은 리뷰 34개와 메타데이터 34개의 JSONL 파일이다. 압축본·압축 해제본의 중복 처리는 없었다. 리뷰 스키마에는 `timestamp`가 있고 `sort_timestamp`는 확인되지 않았다.
- 엄격한 JSONL 읽기에서 파싱 오류 {parse_errors:,}건. `schema_report.json`의 자료형은 파일당 앞 1,000행 표본 기준이며, 필드별 결측률은 `data_quality_report.csv`에 있다. timestamp는 밀리초 단위 {ms_rows:,}건, 초 단위 {sec_rows:,}건으로 판별했다.
- 메타데이터 `parent_asin` 중복 키 {metadata_dupe_keys:,}개. 중복 제거 전 inner join 결과는 {raw_join_rows:,}행이 되며, 제품 통계는 키 중복 제거 후 연결했다.
- 동일 사용자·제품의 반복 기록 쌍 {dup_pairs:,}개. 같은 텍스트/시각인지, 다른 시각·텍스트·별점인지 `duplicate_user_product_report.csv`에 구분했고 자동 삭제하지 않았다. `parent_asin`을 포함한 엄격한 키의 카테고리 간 동일 리뷰는 {duplicate_across_categories:,}개다. 카테고리 간 공유되는 `parent_asin`이 없다는 전수 키 집계로 확인했으며, 서로 다른 제품 키에 복사된 문구의 전수 중복을 뜻하지는 않는다.
{relaxed_note}
- ASCII 텍스트의 결정적 1/1,000 표본 {sample_ascii_rows:,}건에서 정확히 같은 텍스트로 중복되는 행은 {sample_exact_duplicate_rows:,}건, 대소문자·구두점·공백 정규화 후 중복되는 행은 {sample_normalized_duplicate_rows:,}건이다. 이는 의미적 유사도 판정이 아니다.
- 언어 구성은 `language_sample_estimate.csv`의 재현 가능한 문자 기반 휴리스틱이다. 비ASCII 문자를 포함한 영어도 별도 범주가 될 수 있다. 규칙·보상 문구 탐색 역시 1/1,000 표본 기반이므로 전체 빈도로 읽지 않는다.
- smoke test는 파일 앞 10,000행을 사용했다. 그 분포를 전체 데이터 분포로 해석하지 않는다.

## E. 후속 실험 전 수동 검수

`annotation_sample.csv`에 {len(annotation):,}개 리뷰를 담았고, 규칙 유형과 근거 문구를 숨긴 `annotation_sample_blinded.csv`도 제공한다. 사람 입력 필드는 비어 있다. 규칙 단서 문장을 찾지 못한 경우 `evidence_sentence`에는 원문 첫 문장을 넣었으므로 반드시 전체 텍스트와 대조해야 한다. 높은 별점의 단점, 낮은 별점의 장점, 한 리뷰의 혼합 의견, 속성별 불만 후보와 단발·반복 문구를 원문으로 확인해야 한다. 중요도에는 아직 정답 라벨이 없다.

보상 공개·내돈내산 자기보고 규칙의 표본 후보는 {compensation_hits:,}건이다. `compensation_disclosure_summary.csv`와 근거 문장을 대조해야 한다. `verified_purchase=true`는 무보상 구매의 증거가 아니며, 공개 문구가 없다고 무보상으로 확정하지 않는다. 후보를 자동 제외하지 않았다.

## F. 실험용 제품·리뷰 묶음 권고

1. `parent_asin`을 제품 키로 사용하고 메타데이터 키를 중복 제거한 뒤 연결한다.
2. 유효 리뷰 최소 20 / 50 / 100개 조건을 병렬로 유지한다. 20개 조건은 범위가 넓고 50·100개 조건은 제품 내 비교에 더 많은 리뷰를 제공하지만 제품·카테고리 수가 줄어든다.
3. 고·저평점 공존, 고평점 우세 속 소수 저평점, 별점 분포 진단 조건과 텍스트 총량 조건을 별도 층으로 유지한다. 한 기준을 최종 데이터셋으로 확정하지 않는다.
4. 리뷰 묶음은 카테고리와 별점 층화 후 저평점 보존을 검사한다. 규칙 후보는 수동 검수 후에만 요약 실험의 관찰 변수로 사용한다.

## 산출물 해석

그림은 `figures/`, 진행 기록은 `logs/`, 재개 기록은 `checkpoints/`에 있다. 제품별 연도 통계는 `product_temporal_rating_change.parquet`, 표본 속성 문구 반복은 `bundle_issue_marker_summary.csv`에 있다. 별점은 문장 감성의 대리 지표이고 규칙 후보는 불만·중요도의 정답이 아니다. 이 EDA만으로 AI 요약의 긍정 편향을 입증할 수 없다.
"""
    (out_dir / "eda_report.md").write_text(report, encoding="utf-8")


def write_run_metadata(project: Path, out_dir: Path, mode: str, limit_per_file: int | None, seed: int) -> None:
    usage = psutil.disk_usage(str(project.drive + "\\"))
    vm = psutil.virtual_memory()
    metadata = {
        "generated_at": now_stamp(),
        "project": str(project),
        "mode": mode,
        "limit_per_file": limit_per_file,
        "seed": seed,
        "python": sys.version,
        "duckdb": duckdb.__version__,
        "cpu_count": os.cpu_count(),
        "memory_total_bytes": vm.total,
        "memory_available_bytes_at_start": vm.available,
        "drive_total_bytes": usage.total,
        "drive_free_bytes_at_start": usage.free,
        "notes": [
            "No original data or previous outputs are overwritten.",
            "No external LLM API or large model download is used.",
            "Rules are exploratory candidate finders, not sentiment/aspect/importance labels.",
        ],
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=r"E:\WhoGetsSummarized")
    parser.add_argument("--out", default=r"E:\WhoGetsSummarized\outputs\review_summary_eda_v1")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--limit-per-file", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--temp-dir", type=Path, default=None,
                        help="Optional DuckDB spill directory on a faster drive; final outputs remain under --out")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    base_out = Path(args.out).resolve()
    out_dir = base_out / ("smoke_test" if args.mode == "smoke" else "full")
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / "logs")
    checkpoint_dir = out_dir / "checkpoints"
    write_run_metadata(project, out_dir, args.mode, args.limit_per_file if args.mode == "smoke" else None, args.seed)

    if args.mode == "smoke":
        review_dir, meta_dir = make_smoke_inputs(project, out_dir, args.limit_per_file, args.force)
        limit = args.limit_per_file
    else:
        review_dir = project / "data" / "review"
        meta_dir = project / "data" / "metadata_product"
        limit = None

    run_step(checkpoint_dir, "inventory_schema", args.force, lambda: inventory_and_schema(project, review_dir, meta_dir, out_dir, args.mode, limit))

    con = connect(out_dir, args.temp_dir)
    if args.mode == "full":
        review_cache, meta_cache = prepare_full_parquet(con, project, out_dir, args.force)
        create_parquet_views(con, review_cache, meta_cache)
    else:
        create_views(con, review_dir, meta_dir)
    run_step(checkpoint_dir, "data_quality", args.force, lambda: write_data_quality(con, out_dir))
    run_step(checkpoint_dir, "metadata_join", args.force, lambda: write_metadata_tables(con, out_dir))
    if args.mode == "full":
        run_step(checkpoint_dir, "core_eda", args.force, lambda: write_core_eda_batches(con, out_dir, args.force))
        run_step(checkpoint_dir, "product_stats", args.force, lambda: write_product_stats_batches(con, out_dir, args.force))
    else:
        run_step(checkpoint_dir, "core_eda", args.force, lambda: write_core_eda(con, out_dir))
        run_step(checkpoint_dir, "product_stats", args.force, lambda: write_product_stats(con, out_dir))
    run_step(checkpoint_dir, "metadata_join_report", args.force, lambda: write_metadata_join_report(con, out_dir))
    run_step(checkpoint_dir, "overall_summary", args.force, lambda: write_overall_summary(con, out_dir))
    run_step(checkpoint_dir, "metadata_join_by_category", args.force, lambda: write_metadata_join_by_category(con, out_dir))
    if args.mode == "full":
        run_step(checkpoint_dir, "duplicates", args.force,
                 lambda: write_duplicate_reports_batches(con, out_dir, args.force))
    else:
        run_step(checkpoint_dir, "duplicates", args.force, lambda: write_duplicate_reports(con, out_dir))
    run_step(checkpoint_dir, "candidate_products", args.force, lambda: write_candidate_tables(con, out_dir, args.seed))
    run_step(checkpoint_dir, "review_bundle_sample", args.force, lambda: write_review_bundle_sample(con, out_dir, args.seed))
    run_step(checkpoint_dir, "rule_samples", args.force, lambda: write_rule_samples(con, out_dir, args.seed))
    finalize_evidence_files(out_dir)
    run_step(checkpoint_dir, "annotation_files", args.force, lambda: build_annotation_files(out_dir, args.seed))
    run_step(checkpoint_dir, "figures", args.force, lambda: write_figures(out_dir))
    run_step(checkpoint_dir, "report", args.force, lambda: write_report(out_dir, args.mode, args.seed))
    con.close()
    logging.info("complete %s outputs at %s", args.mode, out_dir)


if __name__ == "__main__":
    main()
