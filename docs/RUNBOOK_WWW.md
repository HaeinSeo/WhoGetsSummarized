# WWW 2027 short paper run - 1,000 test products in ~2 weeks (configs/www_short.yaml)

Scope: audit framing (design 17.3). AEP training is out of scope for this run.

| cohort | products | methods |
|---|---|---|
| test_natural_core | 200 | B0 direct, B1 coverage (qwen+gemma); B2 freqframe, B3 plan:frequency, ablation plan:frequency_unverified (qwen); B3 (gemma) |
| test_diagnostic | 160 | same as core + controlled target pairs |
| test_disclosure | <=100 | B0/B1 (both models), D/U representation (all pairs), D/U resampling |
| test_natural_scale | 540 | B0/B1 (both models), claim-level evidence audit |
| dev | 60 | judge / K selection only |

GPU budget measured on the RTX 5070 Ti laptop (2026-10-04): about 110 h in total if Nimble is the judge
(much less if NLI passes the gate). Human work per annotator: about 45 h (rough).

PowerShell with `.venv` active. Once per terminal: `$C = @('--config', 'configs/www_short.yaml')`
Between generator switches run `ollama stop <model>` (12 GB VRAM).

## Day 1 - setup, sample, dev, start human work
1. `python -m wgs check-env @C` ; `python -m pytest -q`
2. `python -m wgs scan-disclosure @C`
3. `python -m wgs sample @C`
4. `git add configs/www_short.yaml` ; `git commit -m "Freeze WWW config"`
5. `python -m wgs annotate export @C --kind inventory --cohort test_natural_core --n 20`
6. `python -m wgs annotate export @C --kind inventory --cohort test_diagnostic --n 20`
7. `python -m wgs generate @C --model qwen3:8b --split dev` ; `ollama stop qwen3:8b`
8. `python -m wgs generate @C --model gemma4:12b --split dev` ; `ollama stop gemma4:12b`
9. `python -m wgs extract-claims @C` ; `python -m wgs retrieve @C --kind claim`
10. `python -m wgs retrieve @C --kind claim --model-id sentence-transformers/all-MiniLM-L6-v2`
11. `python -m wgs annotate export @C --kind pair --split dev --n 600 --exhaustive-products 6`
12. `python -m wgs annotate export @C --kind disclosure --n 600`
HUMAN (days 1-4): pair, disclosure, then inventories (docs/annotation/).

## Days 1-2 - test summaries B0/B1 (all 1,000)
13. `python -m wgs generate @C --model qwen3:8b --split test --method direct`
14. `python -m wgs generate @C --model qwen3:8b --split test --method coverage` ; `ollama stop qwen3:8b`
15. `python -m wgs generate @C --model gemma4:12b --split test --method direct`
16. `python -m wgs generate @C --model gemma4:12b --split test --method coverage` ; `ollama stop gemma4:12b`

## Days 2-4 - candidates, plans, B2/B3, controlled pairs (core 360)
17. `python -m wgs extract-experiences @C --cohort test_natural_core test_diagnostic`   (~20 h)
18. `python -m wgs build-candidates @C --cohort test_natural_core test_diagnostic`
19. `python -m wgs retrieve @C --kind candidate`
20. `python -m wgs judge @C --judge nimble --kind candidate --scope topk:10`
21. `python -m wgs plan @C --planner frequency --judge nimble --scope topk:10`
22. `python -m wgs plan @C --planner frequency_unverified --judge nimble --scope topk:10`
23. `python -m wgs generate @C --model qwen3:8b --cohort test_natural_core test_diagnostic --method freqframe`
24. `python -m wgs generate @C --model qwen3:8b --cohort test_natural_core test_diagnostic --method plan --planner frequency`
25. `python -m wgs generate @C --model qwen3:8b --cohort test_natural_core test_diagnostic --method plan --planner frequency_unverified`
26. `python -m wgs interventions @C --judge nimble --scope topk:10 --cohort test_diagnostic test_disclosure --targets-per-bundle 1`
27. `python -m wgs generate @C --model qwen3:8b --bundle-file runs/www/data_manifests/bundles_interventions.jsonl --method direct`
28. `python -m wgs generate @C --model qwen3:8b --bundle-file runs/www/data_manifests/bundles_interventions.jsonl --method coverage` ; `ollama stop qwen3:8b`
29. `python -m wgs generate @C --model gemma4:12b --cohort test_natural_core test_diagnostic --method plan --planner frequency`
30. `python -m wgs generate @C --model gemma4:12b --bundle-file runs/www/data_manifests/bundles_interventions.jsonl --method direct` ; `ollama stop gemma4:12b`
31. `python -m wgs annotate export @C --kind intervention --n 150`

## Day 4 - judge gate (needs pair + disclosure gold)
33. `python -m wgs annotate merge @C --kind pair` -> `_adjudicated.csv` -> merge again (same for disclosure)
34. `python -m wgs judge @C --judge nimble --scope gold` ; `--judge nli` ; `--judge llama`
35. `python -m wgs eval-judges @C --judges nimble nli llama`
36. `python -m wgs eval-retrieval @C --model-ids BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2`
37. `python -m wgs eval-disclosure @C` ; `python -m wgs report @C`
If a judge other than nimble wins: repeat steps 20-22, 24-25, 29 with that judge, then continue.

## Days 5-8 - automatic evaluation
38. `python -m wgs extract-claims @C` ; `python -m wgs retrieve @C --kind claim`
39. `python -m wgs judge @C --judge nimble --scope topk:5`   (largest GPU step)
40. `python -m wgs judge @C --judge nimble --scope all --cohort test_disclosure`
41. `python -m wgs metrics @C --judge nimble --scope topk:5` ; `python -m wgs metrics @C --judge nimble --scope all`
42. `python -m wgs eval-auto-recall @C --judge nimble --scope topk:10`
43. `python -m wgs eval-interventions @C --auto-judge nimble`

## Days 5-11 - human gold and validation
44. inventories -> `<bundle>_inventory_gold.csv` -> `python -m wgs annotate inventory-gold @C`
45. `python -m wgs annotate export @C --kind summary` -> HUMAN -> `python -m wgs annotate merge @C --kind summary`
46. `python -m wgs annotate merge @C --kind intervention`
47. `python -m wgs annotate export @C --kind intervention_target --n 200` -> HUMAN -> `annotate merge @C --kind intervention_target`

## Days 11-12 - final numbers
48. `python -m wgs eval-preservation @C --baseline direct`
49. `python -m wgs eval-auto-recall @C --judge nimble --scope topk:10`   (re-run: adds human-vs-auto correlation)
50. `python -m wgs eval-interventions @C --auto-judge nimble --only-validated`   (+ agreement with human target presence)
51. `python -m wgs regress @C --judge nimble --scope all`
52. `python -m wgs report @C`
53. `python -m wgs figures @C` (clean sans; ACM-ready PDF with embedded TrueType fonts + PNG + CSV)
    or `python -m wgs figures @C --font comic` -> `runs/www/figures/<font>/`

Days 13-14: buffer, figures, writing (abstract 11/9, paper 11/16 AoE).

## Paper mapping (4 pages)
- Table 1 (n=1,000 auto, n=40 human): supported-claim precision, auto recall (overall / minority / negative), human recall_imp and conditional fidelity, word count, per method x generator.
- Figure 1: joint coverage and removal error (auto on all pairs, human-validated subset).
- Table 2: judge vs human (macro-F1, SUPPORT/OPPOSE P/R, D vs nonD), retrieval recall@K, rule D precision.
- Figure 2: RR/RG/Amp/F on test_disclosure + du resampling, with an omitted-experience example.
