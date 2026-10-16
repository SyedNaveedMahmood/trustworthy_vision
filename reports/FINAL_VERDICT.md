# INCONCLUSIVE

The corrected evidence estimator is **not supported as an improvement by the executed development study**. Do not advance S3 to confirmation on these results. The full scientific hypothesis remains inconclusive because intervention/reference validity is not independently established and E5 has not run.

The intended Qwen + corrected-Smol pair scores S3 80.23% versus raw median 85.53%: -5.30pp, descriptive paired95% CI[-12.02, +1.14] pp. S3 costs25 versus7 scores/pair. The +3pp practical development target was not supported. Comparator selection used these same development data; its interval is descriptive, not confirmatory or selection-adjusted. We did not retune the coefficient to force a positive result.

| Method | Macro pair ranking (image-cluster95% CI) | Direct scores/pair |
|---|---|---:|
| raw_mean | 82.15% [76.05%, 87.73%] | 7 |
| raw_median | 85.53% [79.99%, 91.06%] | 7 |
| raw_occlusion | 83.35% [77.20%, 89.45%] | 3 |
| robust_independent | 80.23% [72.85%, 87.00%] | 25 |
| robust_sham | 85.76% [79.57%, 91.34%] | 13 |
| sham_median | 84.54% [77.80%, 90.12%] | 13 |

| Task | Model | n | Raw occlusion | Raw median | Sham median (S1) | Robust sham (S2) | Independent robust (S3) |
|---|---|---:|---:|---:|---:|---:|---:|
| gqa | qwen | 40 | 80.00% | 83.75% | 77.50% | 77.50% | 75.00% |
| gqa | smol | 40 | 90.00% | 92.50% | 87.50% | 87.50% | 82.50% |
| gqa | internvl | 40 | 80.00% | 82.50% | 82.50% | 80.00% | 85.00% |
| refcocog | qwen | 41 | 85.37% | 87.80% | 85.37% | 87.80% | 80.49% |
| refcocog | smol | 41 | 78.05% | 78.05% | 87.80% | 90.24% | 82.93% |
| refcocog | internvl | 41 | 87.80% | 87.80% | 87.80% | 87.80% | 87.80% |

**The model feasibility problem was an adapter bug, not demonstrated incapability.** Smol expects space-prefixed letter tokens; the old bare-label tokens received only~4.5e-6 combined vocabulary mass. Corrected labels receive~.9998. All historical outputs remain archived and are invalidated as Smol capability estimates. Fresh disjoint qualification gives:

| Model | Color GQA image/text/shuffled (/20) | RefCOCOg image/text (/20) | Screen |
|---|---|---|---|
| qwen | 13/9/8 | 15/4 | True |
| smol | 12/7/6 | 13/4 | True |
| internvl | 13/7/8 | 12/5 | True |

Original GQA distractors also mixed answer types on18/40 development questions. The replacement color-only cohort is a separate narrower experiment; broad GQA reasoning is unvalidated. A concrete source bounding-box completeness failure remains unfiltered. No causal evidence claim follows from a clean numerical score or bootstrap.

| Stage | Status | Actual execution / validation limit |
|---|---|---|
| E0 | PARTIAL / EXPLORATORY | Three official frozen backbones fit BF16; processor/likelihood/repeat tests and real official LaMa replay completed. Independent semantic editor validity remains pending. |
| E1 | COMPLETE (amended engineering screen) | Fresh20 images/task/model, fixed40% screen and positive image gain. Original broad GQA competence is unvalidated; replacement is explicitly color-only. |
| E2 | PARTIAL / EXPLORATORY | 81 paired images shared by3 backbones;6,075 complete score rows. 16 grid images/backbone for Qwen/InternVL. Pixel/hash checks pass; no semantic audit pass. |
| E3 | PARTIAL / EXPLORATORY | Actual baselines, S1–S3, initial/extension/pooled comparisons, paired bootstrap, AUROC and cost analysis. All development; no confirmatory claim. |
| E4 | PARTIAL / EXPLORATORY | Real resolution/letter/fill/blur/mask studies for Qwen/InternVL; matched five/task corrected-Smol dose study; numerical ablations all3 backbones. Not every factor/model/operator transfer completed. |
| E5 | BLOCKED | Zero confirmatory manifests or forwards. Human/control/reference validity, genuine lock and feasible amended sample/budget remain absent. |
| E6 | PARTIAL / EXPLORATORY | Current negative-result/failure audit, verdict and handoff complete; full-study interpretation remains contingent on E5. |

| Model | E2 score rows | Model interval(s) | Rows/s | NVML peakGiB | Torch allocated/reservedGiB | Peak hostRSS GiB |
|---|---:|---:|---:|---:|---|---:|
| internvl | 2025 | 289.58 | 6.99 | 7.34 | 5.23/6.15 | 5.23 |
| qwen | 2025 | 417.33 | 4.85 | 8.85 | 7.57/7.66 | 4.29 |
| smol | 2025 | 800.40 | 2.53 | 6.45 | 4.57/5.26 | 5.25 |

Actual cumulative reserved model-forward attempts: **11984/12,000**, failed model calls **0**, remaining **16**. This session added 11208 calls to the preserved776 prior calls. Synchronized forward wall time1422.42s. CUDA stream elapsed1207.11s covers10225 calls after call1759; it is not summed active-kernel time. Conservative GPU-time upper bound at the last call1.412h remains below24h. Device-utilization/power/temperature observations are in `device_observer_summary.json`; they cover only their actual observation window and include desktop activity.

There are75 passing tests and6,075 cache-verified paired score rows. A12-call official last-logit optimization probe preserved choice logits on six inputs but changed some non-choice logits by up to.03125 and provided no meaningful speed/memory gain; it was not applied to main results. GPU inference is finished; no model process is left to continue this study automatically.

**E5 launch: NO-GO.** Zero confirmatory manifests/forwards. Independent human review remains HUMAN_AUDIT_PENDING; original120/task at25 scores×2 models costs12,000 new calls alone, exceeding the cumulative cap. Remaining16 calls cannot fund even one25-score pair. No budget reset, sample reduction or protocol lock was silently performed. The next valid study requires source/edit review and an explicitly amended feasible design; current raw-median comparator proposal is development-selected, not locked.

These are annotation-reference region rankings, not verified causal relevance. Source annotations can be incomplete: GQA1593212 asks locomotive color but its official box[875,671,977,761] covers the lower coupler in a1024×768 image, leaving most of the train visible. Agent visual inspection is not an independent human judgment. This case and all other failures remain included. Confidence intervals do not repair this label problem, semantic edit errors, cross-dataset identities missing from available mappings, or adaptive development reuse.

The baseline was chosen on the same pooled development results. Its paired bootstrap interval is descriptive and does not account for comparator selection. No confirmatory p-value, positive effect claim, equivalence claim or statistically validated deployment reliability is asserted. Three operators are correlated, and their scaled MAD is a disagreement penalty, not a standard error. The original practical development target is+3pp; current S3 has not supported it. No coefficient or threshold was changed to rescue the outcome.

Full execution, preserved artifacts, remaining dependencies and GitHub links: [handoff](TWO_HOUR_HANDOFF.md), [E2](E2_EXPLORATORY_RESULTS.md), [E3](E3_EXPLORATORY_COMPARISON.md), [E4](E4_EXPLORATORY_ABLATIONS.md), [reproduction](../REPRODUCIBILITY.md). Earlier verdict is preserved verbatim in `FINAL_VERDICT_V1_ARCHIVE.md`.
