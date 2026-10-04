from pathlib import Path

import duckdb

import eda_review_summary_v1 as eda


project = Path(r"E:\WhoGetsSummarized")
smoke = project / "outputs" / "review_summary_eda_v1" / "verification" / "smoke_test"
fixture = project / "outputs" / "review_summary_eda_v1" / "batch_verification"
review_dir = fixture / "source_parquet" / "review"
meta_dir = fixture / "source_parquet" / "metadata_product"
review_dir.mkdir(parents=True, exist_ok=True)
meta_dir.mkdir(parents=True, exist_ok=True)
categories = ["All_Beauty", "Amazon_Fashion"]

con = eda.connect(fixture)
eda.create_views(con, smoke / "smoke_inputs" / "review", smoke / "smoke_inputs" / "metadata_product")
for category in categories:
    con.execute(
        f"COPY (SELECT * FROM reviews WHERE category='{category}') "
        f"TO '{eda.qpath(review_dir / (category + '.parquet'))}' (FORMAT PARQUET)"
    )
    con.execute(
        f"COPY (SELECT * FROM metadata_raw WHERE meta_file_category='{category}') "
        f"TO '{eda.qpath(meta_dir / ('meta_' + category + '.parquet'))}' (FORMAT PARQUET)"
    )
con.execute(
    f"CREATE TABLE metadata_dedup AS SELECT * FROM read_parquet('{eda.qpath(smoke / 'metadata_dedup.parquet')}')"
)
eda.write_core_eda_batches(con, fixture, force=True)
eda.write_product_stats_batches(con, fixture, force=True)
review_count = con.execute(
    f"SELECT sum(review_count) FROM read_parquet('{eda.qpath(fixture / 'product_review_statistics.parquet')}')"
).fetchone()[0]
assert review_count == 20000, review_count
assert len(list((fixture / "category_parts").glob("*/category_summary.csv"))) == 2
con.close()
print("batch smoke passed: 2 categories, 20,000 reviews")
