# Who Gets Summarized?

Evidence-preserving evaluation and planning for AI product-review summaries
(Amazon Reviews 2023). Target venue: The Web Conference short paper.

> **Status: pre-results.** This repository contains the pipeline and protocol (design v2.0).
> It contains no experimental findings. Numbers appear only after the steps below are run,
> and automatic estimates are reported separately from human-validated results.

**What the pipeline measures**

1. *Output → input*: are atomic claims in a summary supported by input reviews
   (review–claim evidence graph; SUPPORT / OPPOSE / IRRELEVANT / UNCERTAIN)?
2. *Input → output*: do human-identified important experiences in the input survive in the
   summary (important-experience recall, conditional fidelity)?
3. *Controlled changes*: target-preserved swaps, target-removed swaps, and D/U
   (disclosure / no-disclosure) resampling using real reviews of the same product.
4. *Method*: AEP (Aspect Evidence Planner), a small trainable selector over verified
   evidence, compared with prompt and frequency baselines under the same budget.

## Repository layout

```
configs/            pilot.yaml (P0-P2), smoke.yaml (~12 products), main_frozen.yaml (create at P3)
src/wgs/            the pipeline (`python -m wgs <step>`)
  sampling.py       cohorts, stable review IDs, preprocessing, fixed bundles, splits
  generate.py       frozen-generator summaries via Ollama (B0 direct, B1 coverage, B2 freqframe, plan)
  extract.py        atomic claims (output) and per-review experiences (input), span-validated
  retrieval.py      BGE sentence-window retrieval; candidate experience merging
  judges/           nimble (/v1/systemone choice), DeBERTa NLI, Llama judge - swappable adapters
  evidence.py       judged evidence edges (unjudged pairs stay unjudged, never "irrelevant")
  metrics.py        RR / RG / Exposure / Share / Amp / F, recall_imp, paired bootstrap
  annotate.py       blinded sheets, agreement (kappa), adjudication, gold parquet
  evaluate.py       judge vs gold, retrieval recall@K, preservation, interventions, regression
  planner/          frequency baselines (B3), AEP local/context/full (B4-B6)
  interventions.py  controlled bundle pairs
  report.py         pilot report (observed / automatic / human-pending / gates)
  prompts/*.txt     all prompts (content-hashed into every output row)
docs/               methodology mapping and annotation guidelines
scripts/            EDA scripts that produced outputs/review_summary_eda_v1/
tests/              unit tests (`python -m pytest`)
```

Not committed (see `.gitignore`): `data/` (raw Amazon files), `outputs/` (EDA, ~155 GB),
`runs/` (pipeline outputs), `annotations/` (human labels; publish adjudicated gold
deliberately after checking the dataset terms), model checkpoints.

## Setup

```bash
python -m pip install -e ".[ml,eda,dev]"
```

```bash
ollama pull qwen3:8b
```

```bash
ollama pull gemma4:12b
```

```bash
ollama pull llama3.1:8b
```

```bash
ollama pull nimble:9b
```

On a drive without ownership info (e.g. exFAT), git needs:
`git config --global --add safe.directory E:/WhoGetsSummarized`.

Data: download Amazon Reviews 2023 (McAuley Lab) review and metadata JSONL files into
`data/review/` and `data/metadata_product/`, then run the EDA
(`python scripts/eda_review_summary_v1.py --mode full --project . --out outputs/review_summary_eda_v1`), which writes
`outputs/review_summary_eda_v1/full/` including `source_parquet/` and
`product_review_statistics.parquet` used below.

## Running the pipeline

All commands take `--config` (default `configs/pilot.yaml`) and write to `runs/<run_name>/`.
Every step appends to `run_manifest.jsonl` (config hash, git commit, model digest, prompt
hash, counts) and to `failures.jsonl`; long steps resume from their JSONL caches.
Ordered command lists:
- [docs/RUNBOOK_WWW.md](docs/RUNBOOK_WWW.md) - WWW short-paper scope (`configs/www_short.yaml`)
- [docs/RUNBOOK_FULL.md](docs/RUNBOOK_FULL.md) - full design scale incl. AEP training (`configs/main_frozen.yaml`)
- [docs/RUNBOOK.md](docs/RUNBOOK.md) - pilot-first variant (`configs/pilot.yaml`)

## License and data terms

Code license: to be chosen by the authors before release. Review text is subject to the
Amazon Reviews 2023 terms; release IDs, splits, guidelines, prompts, and metric code rather
than raw review text unless redistribution is permitted.
