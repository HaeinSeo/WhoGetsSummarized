# Design v2.0 -> code map

The full research plan (design v2.0, 2026-10-03) should be stored as `docs/research_plan_v2.md`.
This file records where each design decision is implemented.

| Design section | Decision | Implementation |
|---|---|---|
| 4.1 units | stable review_id (not user+product) | `sampling.preprocess_reviews`: sha1(category, parent_asin, user, timestamp, text) + `src_row` |
| 4.3 preprocessing | strict dups -> 1 + multiplicity; repeat user/product not independent | `multiplicity`, `user_review_index` (only first per user enters a bundle) |
| 4.3 language | heuristic, not ground truth | `text.language_guess` (alphabetic ASCII share, stopwords for >=8 words) |
| 5 disclosure | D/S/V/A flags kept separately; U is not "organic" | `disclosure.disclosure_flags`, human gold via `annotate --kind disclosure` |
| 6.2 cohorts | natural / diagnostic / disclosure, product-disjoint | `configs/*.yaml: cohorts`, `sampling._candidate_products` |
| 6.3 bundles | same ids/order/spans for all generators; 4,096-token budget; no truncation | `sampling.build_bundle`; actual tokens checked per model (`prompt_tokens`, `over_common_budget`) |
| 7 models | Ollama tags + digests recorded | `llm.Ollama.provenance`; `summaries.jsonl: model_digest, quantization` |
| 7.1 decoding | greedy, thinking off, 100-120 words | `generation:` config; `top_k=1`, `think=false`, `length_ok`, `finish_reason` |
| 8.1 claims | span must be a substring | `extract._validate` (`valid`, `invalid_reason`) |
| 8.2 inventory | independent of outputs, human gold | `annotate export --kind inventory` -> `inventory-gold` |
| 9.1 retrieval | within product, review-level max over windows, K=10/20/all | `retrieval.retrieve`, `evidence.scope_mask` |
| 9.2 labels | SUPPORT/OPPOSE/IRRELEVANT/UNCERTAIN; OPPOSE != contradiction | `judges/base.py`; NLI mapping validated in `eval-judges` |
| 9.3 Nimble | choice only, probs stored, digest pinned | `judges/nimble.py` |
| 10 metrics | RR, RG, Exposure, Share, Amp, F (review and user), NA on zero denominators | `metrics.representation` (unit-tested on the design example) |
| 10.4 recall_imp | condition loss is not success; NA if no important experience | `metrics.importance_recall` |
| 11 gold | two annotators, kappa, adjudication, guideline version | `annotate.merge` |
| 12 AEP | r (local) x q (budget/context), L_sup + L_inv + L_remove, frozen probe | `planner/aep.py`, `planner/run.py` |
| 13 baselines | B0 direct, B1 coverage, B2 freqframe, B3 frequency, B4-B6 AEP; no-verification ablation | `prompts/*.txt`, `planner/select.py` |
| 14 interventions | real same-product reviews; validated pairs only | `interventions.py`, `annotate --kind intervention` |
| 15 statistics | product bootstrap paired differences; M0-M2 association models | `metrics.paired_bootstrap`, `evaluate.group_regression` |
| 16 failure analysis | input / extraction / judge / selection / generation coverage stored separately | bundles, `review_experiences`, `input_candidates`, edges, `plans`, summaries |
| 18 reproducibility | manifests, failures, hashes | `run_manifest.jsonl`, `failures.jsonl`, `prompt_hash`, `config_hash` |
