# Full run (design v2.0 scale, no pilot) - configs/main_frozen.yaml

Run in an activated `.venv` (PowerShell). First define the config once per terminal:
`$C = @('--config', 'configs/main_frozen.yaml')` - then `@C` below expands to `--config configs/main_frozen.yaml`
(bash: use `--config configs/main_frozen.yaml` in place of `@C`).
Before switching Ollama models: `ollama stop <model>`. Steps marked HUMAN need two annotators.

## Phase 0 - check, sample, freeze
1. `python -m wgs check-env @C` ; `python -m pytest -q`
2. `python -m wgs scan-disclosure @C`
3. `python -m wgs sample @C`  (test 400 / dev 100 / train 600, product-disjoint)
4. Freeze: `git add configs/main_frozen.yaml` and commit before any test generation.

## Phase 1 - DEV: choose the judge (replaces the pilot)
5. `python -m wgs generate @C --model qwen3:8b --split dev`
6. `python -m wgs generate @C --model gemma4:12b --split dev`
7. `python -m wgs extract-claims @C`
8. `python -m wgs retrieve @C --kind claim`
9. `python -m wgs retrieve @C --kind claim --model-id sentence-transformers/all-MiniLM-L6-v2`
10. `python -m wgs annotate export @C --kind pair --split dev --n 1500 --exhaustive-products 10`
11. `python -m wgs annotate export @C --kind disclosure --n 1200`
12. HUMAN: fill `annotations/main/{pair,disclosure}/A1.csv` and `A2.csv` independently.
13. `python -m wgs annotate merge @C --kind pair` ; fill `_adjudicated.csv` ; run merge again (same for `--kind disclosure`)
14. `python -m wgs judge @C --judge nimble --scope gold` ; same with `--judge nli` and `--judge llama`
15. `python -m wgs eval-judges @C --judges nimble nli llama`
16. `python -m wgs eval-retrieval @C --model-ids BAAI/bge-small-en-v1.5 sentence-transformers/all-MiniLM-L6-v2`
17. `python -m wgs report @C` -> fix the judge that passes the gate. Below, `nimble` = the chosen judge.

## Phase 2 - TEST summaries for prompt baselines (B0-B2)
18. For M in qwen3:8b, gemma4:12b and METHOD in direct, coverage, freqframe*:
    `python -m wgs generate @C --model M --split test --method METHOD`
    (*freqframe needs Phase 3 steps 19-20 first.)

## Phase 3 - candidates for every split
19. `python -m wgs extract-experiences @C`
20. `python -m wgs build-candidates @C`
21. `python -m wgs retrieve @C --kind candidate`
22. `python -m wgs judge @C --judge nimble --kind candidate --scope topk:10`

## Phase 4 - controlled pairs and AEP labels (train/dev/test)
23. `python -m wgs interventions @C --judge nimble --scope topk:10 --cohort train_natural train_diagnostic dev_natural dev_diagnostic test_diagnostic test_disclosure`
24. `python -m wgs extract-experiences @C --bundle-file runs/main/data_manifests/bundles_interventions.jsonl`
25. `python -m wgs build-candidates @C` ; `python -m wgs retrieve @C --kind candidate` ; `python -m wgs retrieve @C --kind probe`
26. `python -m wgs judge @C --judge nimble --kind candidate --scope topk:10` ; `python -m wgs judge @C --judge nimble --kind probe --scope topk:10`
27. `python -m wgs annotate export @C --kind intervention` ; `python -m wgs annotate export @C --kind selection --split train dev`
28. HUMAN: label, then `annotate merge @C --kind intervention` and `--kind selection` (with adjudication)

## Phase 5 - AEP (B4-B6) and ablations
29. `python -m wgs train-aep @C --judge nimble --scope topk:10 --variant local`
30. same with `--variant context`, `--variant full`, `--variant full --lambda-inv 0`, `--variant full --lambda-remove 0`
31. Pick checkpoints by DEV loss only (`runs/main/outputs/aep_train_log_*.csv`).

## Phase 6 - TEST plans and plan-based summaries (B3-B6)
32. `python -m wgs plan @C --planner frequency --judge nimble --scope topk:10`
33. `python -m wgs plan @C --planner frequency_unverified --judge nimble --scope topk:10`
34. `python -m wgs plan @C --planner aep --judge nimble --scope topk:10 --checkpoint runs/main/checkpoints/<ckpt>.pt --name aep_full` (also aep_local, aep_context)
35. For M in both models and P in frequency, frequency_unverified, aep_local, aep_context, aep_full:
    `python -m wgs generate @C --model M --split test --method plan --planner P`
36. Controlled bundles (test only), for M and every method above:
    `python -m wgs generate @C --model M --split test --bundle-file runs/main/data_manifests/bundles_interventions.jsonl --method direct` (and the other methods)

## Phase 7 - evaluation
37. `python -m wgs extract-claims @C` ; `python -m wgs retrieve @C --kind claim` ; `python -m wgs judge @C --judge nimble --scope topk:10`
38. `python -m wgs metrics @C --judge nimble --scope topk:10` ; again with `--group-col gold_label`
39. `python -m wgs annotate export @C --kind inventory --cohort test_natural --n 100` ; same with `--cohort test_diagnostic --n 100`
40. HUMAN: inventories -> `<bundle>_inventory_gold.csv` -> `python -m wgs annotate inventory-gold @C`
41. `python -m wgs annotate export @C --kind summary` -> HUMAN -> `annotate merge @C --kind summary`
42. `python -m wgs annotate export @C --kind intervention_target` -> HUMAN -> `annotate merge @C --kind intervention_target`
43. `python -m wgs eval-preservation @C --baseline direct`
44. `python -m wgs eval-interventions @C`
45. `python -m wgs regress @C --judge nimble --scope topk:10`
46. `python -m wgs report @C`

## Phase 8 - optional robustness
`generate @C --model llama3.1:8b --split test ...`, 8192-token budget, order seeds.
