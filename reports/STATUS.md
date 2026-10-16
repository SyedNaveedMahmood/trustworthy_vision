# Execution status — amended development study

Updated2026-10-16T11:25:42.218850+00:00; evidence-parent commit `677e84ee9bf1ffd5a98008351215e8c847d0e654`, branch `pilot/controlled-evidence-v1`.

**INCONCLUSIVE scientifically; E5 launch NO-GO.** All planned permissible GPU execution is finished at11984/12,000 calls, with16 remaining and0 failed model calls. No GPU experiment is continuing in the background. User resumed CPU reporting after the original autonomous window; no inference ran during that gap.

| Stage | Status | Actual execution / validation limit |
|---|---|---|
| E0 | PARTIAL / EXPLORATORY | Three official frozen backbones fit BF16; processor/likelihood/repeat tests and real official LaMa replay completed. Independent semantic editor validity remains pending. |
| E1 | COMPLETE (amended engineering screen) | Fresh20 images/task/model, fixed40% screen and positive image gain. Original broad GQA competence is unvalidated; replacement is explicitly color-only. |
| E2 | PARTIAL / EXPLORATORY | 81 paired images shared by3 backbones;6,075 complete score rows. 16 grid images/backbone for Qwen/InternVL. Pixel/hash checks pass; no semantic audit pass. |
| E3 | PARTIAL / EXPLORATORY | Actual baselines, S1–S3, initial/extension/pooled comparisons, paired bootstrap, AUROC and cost analysis. All development; no confirmatory claim. |
| E4 | PARTIAL / EXPLORATORY | Real resolution/letter/fill/blur/mask studies for Qwen/InternVL; matched five/task corrected-Smol dose study; numerical ablations all3 backbones. Not every factor/model/operator transfer completed. |
| E5 | BLOCKED | Zero confirmatory manifests or forwards. Human/control/reference validity, genuine lock and feasible amended sample/budget remain absent. |
| E6 | PARTIAL / EXPLORATORY | Current negative-result/failure audit, verdict and handoff complete; full-study interpretation remains contingent on E5. |

Fresh qualification: Qwen color13/20 versus9/20 text, Ref15/20 versus4/20; corrected Smol color12/20 versus7/20, Ref13/20 versus4/20; InternVL color13/20 versus7/20, Ref12/20 versus5/20. Intended Qwen+Smol pair restored. The earlier Smol failure was a canonical continuation error; old outputs remain preserved, not capability evidence.

81 development image pairs×3 models×25 scores=6,075 verified rows. S3 80.23% versus raw median85.53%, paired difference−5.30pp, descriptive95% CI[−12.02,+1.14]. Raw median is proposed from development only, not confirmatory locked. 75 tests pass. Actual blur/mask/fill/resolution/permutation ablations, frozen-answer analyses, no-oracle grid diagnostics, real official LaMa replay and full resource measurements are preserved.

360 combined blinded packets with native-resolution scored overlays are ready; **HUMAN_AUDIT_PENDING**, zero judgments. Source-box completeness and edit/control semantics remain unresolved. Original GQA choices mixed types; the new color task is narrower. Original120/task confirmation alone needs12,000 additional calls under the repaired factorial, so it cannot fit the remaining budget. No confirmatory manifest, inference or lock exists.

See [complete handoff](TWO_HOUR_HANDOFF.md), [verdict](FINAL_VERDICT.md), per-stage reports and `reports/evidence/v2/execution_ledger.json`. Ignored local datasets, edited images, model weights and atomic checkpoints remain on disk; small metrics/reports/configurations/code are pushed to the pilot branch. Prior status/verdict and original E0/E1 observations remain available in Git history and FINAL_VERDICT_V1_ARCHIVE.md.
