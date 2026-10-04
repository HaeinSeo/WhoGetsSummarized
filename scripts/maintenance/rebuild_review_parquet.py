"""Rebuild one category's review parquet from the raw JSONL (same query as the EDA).

Used when an EDA parquet fails integrity checks. Writes <cat>.rebuilt.parquet, verifies every
row group and the row count, then keeps the old file as <cat>.parquet.corrupt (never deleted).
Usage: python scripts/maintenance/rebuild_review_parquet.py Electronics
"""
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
COLS = ("{'rating':'DOUBLE','title':'VARCHAR','text':'VARCHAR','images':'JSON','asin':'VARCHAR',"
        "'parent_asin':'VARCHAR','user_id':'VARCHAR','timestamp':'BIGINT','helpful_vote':'BIGINT',"
        "'verified_purchase':'BOOLEAN'}")


def check(path: Path) -> int:
    pf = pq.ParquetFile(path)
    for i in range(pf.metadata.num_row_groups):
        pf.read_row_group(i)
    return pf.metadata.num_rows


def main(cat: str) -> None:
    src = ROOT / "data" / "review" / f"{cat}.jsonl"
    dst_dir = ROOT / "outputs" / "review_summary_eda_v1" / "full" / "source_parquet" / "review"
    tmp, dst = dst_dir / f"{cat}.rebuilt.parquet", dst_dir / f"{cat}.parquet"
    con = duckdb.connect()
    con.execute(f"PRAGMA temp_directory='{(dst_dir / '_tmp').as_posix()}'")
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("SET preserve_insertion_order=true")
    con.execute(f"""
        COPY (SELECT '{cat}' AS category, filename, rating, title AS review_title, text, asin,
                     parent_asin, user_id, timestamp,
                     CASE WHEN timestamp IS NULL THEN NULL
                          WHEN timestamp > 100000000000 THEN CAST(to_timestamp(timestamp / 1000.0) AS TIMESTAMP)
                          ELSE CAST(to_timestamp(timestamp) AS TIMESTAMP) END AS review_datetime,
                     helpful_vote, verified_purchase
              FROM read_json('{src.as_posix()}', format='newline_delimited', filename=true, columns={COLS}))
        TO '{tmp.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)""")
    n_new = check(tmp)
    n_old = pq.ParquetFile(dst).metadata.num_rows if dst.exists() else None
    print(f"rebuilt rows={n_new} old_rows={n_old}")
    if n_old is not None and n_old != n_new:
        sys.exit("row count differs; leaving both files in place for inspection")
    if dst.exists():
        dst.replace(dst_dir / f"{cat}.parquet.corrupt")
    tmp.replace(dst)
    print("ok:", dst)


if __name__ == "__main__":
    main(sys.argv[1])
