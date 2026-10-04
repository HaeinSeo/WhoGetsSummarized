# Runbook (design v2.0)

Commands are identical in PowerShell and bash. `C=configs/pilot.yaml` unless noted.
Each stage ends with a **gate**. Do not move on (and do not look at test data) until it passes.

## P0 - inputs and samples

| # | Command | Output |
|---|---|---|
| 0.1 | `python -m wgs check-env` | `runs/pilot/env_check.json` |
| 0.2 | `python -m wgs sample --config configs/smoke.yaml` | ~12-product smoke run in `runs/smoke/` |
| 0.3 | `python -m wgs scan-disclosure` | `product_disclosure_counts.parquet`, `product_disclosure_summary.csv` |
| 0.4 | `python -m wgs sample` | `products.parquet`, `reviews.parquet`, `bundles.jsonl`, `splits.json`, `sampling_report.csv` |

Gate: review IDs unique, drop reasons explained, D/U co-existence counted.
(0.3 scans four full categories; run it before 0.4 so the disclosure cohort is built.)

## P1 - direct summaries and extraction (smoke first: add `--config configs/smoke.yaml`)

| # | Command |
|---|---|
| 1.1 | `python -m wgs generate --model qwen3:8b --limit 10` (smoke: inspect text, tokens, latency) |
| 1.2 | `python -m wgs generate --model qwen3:8b` |
| 1.3 | `python -m wgs generate --model gemma4:12b` |
| 1.4 | `python -m wgs extract-claims` |
| 1.5 | `python -m wgs extract-experiences` |
| 1.6 | `python -m wgs build-candidates` |

Gate: `length_ok`, `possible_ctx_truncation`, `over_common_budget` and claim span-invalid
rate are acceptable; real omissions are visible in a manual read of ~20 summaries.

## P2 - evidence graph and measurement validity

| # | Command |
|---|---|
| 2.1 | `python -m wgs retrieve --kind claim` |
| 2.2 | `python -m wgs retrieve --kind claim --model-id sentence-transformers/all-MiniLM-L6-v2` |
| 2.3 | `python -m wgs annotate export --kind pair --n 500 --exhaustive-products 10` |
| 2.4 | `python -m wgs annotate export --kind disclosure --n 1000` |
| 2.5 | (humans) fill `annotations/pilot/{pair,disclosure}/A1.csv` and `A2.csv` independently |
| 2.6 | `python -m wgs annotate merge --kind pair` then fill `_adjudicated.csv`, re-run merge |
| 2.7 | `python -m wgs annotate merge --kind disclosure` (same adjudication loop) |
| 2.8 | `python -m wgs judge --judge nimble --scope gold` |
| 2.9 | `python -m wgs judge --judge nli --scope gold` |
| 2.10 | `python -m wgs judge --judge llama --scope gold` |
| 2.11 | `python -m wgs eval-judges --judges nimble nli llama` |
| 2.12 | `python -m wgs eval-retrieval --model-ids BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2` |
| 2.13 | `python -m wgs report` |

Gate (design 11.3): primary judge SUPPORT precision >= .90 and recall >= .80 on dev,
recall@K >= .95 on the exhaustive subset, D vs nonD judge error reported with CI.
The judge is fixed here and never swapped after seeing test results.

## P3 - freeze

Copy `configs/pilot.yaml` to `configs/main_frozen.yaml`; set `run_name: main`, cohort sizes
(e.g. natural 50/category, diagnostic 25/category, disclosure as feasible), and add cohorts
with `split: train` (600 products) and `split: dev` (100). Fix generator tags (Gemma fallback
decided now), judge, K, and token budget. Commit the file. Then:
`python -m wgs scan-disclosure --config configs/main_frozen.yaml` and
`python -m wgs sample --config configs/main_frozen.yaml`.

## P4 - baselines and AEP (train/dev only)

| # | Command (all with `--config configs/main_frozen.yaml`) |
|---|---|
| 4.1 | `extract-experiences --split train dev`, `build-candidates --split train dev` |
| 4.2 | `retrieve --kind candidate`, `judge --judge nimble --kind candidate --scope all` |
| 4.3 | `annotate export --kind selection`; humans label v/y; `annotate merge --kind selection` |
| 4.4 | `interventions --judge nimble --cohort <train/dev cohorts>`; `annotate export --kind intervention`; label; merge |
| 4.5 | `extract-experiences --bundle-file runs/main/data_manifests/bundles_interventions.jsonl`, `build-candidates`, `retrieve --kind candidate`, `retrieve --kind probe`, `judge --judge nimble --kind candidate --scope all`, `judge --judge nimble --kind probe --scope all` |
| 4.6 | `train-aep --judge nimble --variant local`, `--variant context`, `--variant full`; ablations `--lambda-inv 0`, `--lambda-remove 0` |

## P5 - main runs on test (same bundles for every method and generator)

First build verified candidates for test bundles (same automatic pipeline as train/dev; no gold):
`extract-experiences --split test`, `build-candidates --split test`, `retrieve --kind candidate`,
`judge --judge nimble --kind candidate --scope all`.

For each generator `M` in {qwen3:8b, gemma4:12b} (and later llama3.1:8b):
`generate --model M --split test --method direct|coverage|freqframe`,
`plan --planner frequency --judge nimble` then `generate --model M --split test --method plan --planner frequency`,
`plan --planner frequency_unverified ...` (no-verification ablation),
`plan --planner aep --checkpoint runs/main/checkpoints/<ckpt>.pt --name aep_full ...` then
`generate --model M --split test --method plan --planner aep_full` (same for local/context).

## P6 - evaluation

`extract-claims`, `retrieve --kind claim`, `judge --judge nimble --scope all`,
`metrics --judge nimble --scope all` (and `--group-col gold_label`),
`annotate export --kind inventory` -> humans -> `annotate inventory-gold`,
`annotate export --kind summary` -> humans -> `annotate merge --kind summary`,
`eval-preservation --baseline direct`,
`interventions` on test cohorts -> validate -> generate on `bundles_interventions.jsonl` ->
`annotate export --kind intervention_target` -> merge -> `eval-interventions`,
`regress --judge nimble --scope all`, `report`.

## P7 - robustness (only if a remaining risk needs it)

`generate --model llama3.1:8b ...`, 8192-token budget, input order seeds, judge swap.
