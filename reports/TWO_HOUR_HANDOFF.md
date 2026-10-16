# Two-hour research handoff

Generated2026-10-16T11:25:42.218850+00:00. Original autonomous window07:18:11–09:18:11UTC on2026-10-16. GPU execution completed around08:40UTC. The user later said “continue”; remaining CPU validation/reporting resumed after the original window. No GPU inference continued during the gap or after its completed runs. This file does not claim two hours of continuous GPU work.

## A. Executive summary

**Scientific verdict: INCONCLUSIVE. E5 launch: NO-GO.** Implemented and executed corrected model qualification, three intervention operators with shams/independent controls, real E2/E3 development comparisons, E4 ablations, no-oracle evidence diagnostics, statistical analysis, integrity checks and review material. S3 loses to the strong simple comparator in point estimate and consumes more inference. Human semantic review is still absent; no confirmation was attempted.

S3 versus raw median: -5.30pp, descriptive paired95% CI[-12.02, +1.14] pp; 80.23% versus 85.53%. 81 image pairs shared across3 backbones;6,075 scored cells. All historical negative outcomes are preserved. 75 tests pass.

## B. Model qualification

| Model | Color GQA image/text/shuffled (/20) | RefCOCOg image/text (/20) | Screen |
|---|---|---|---|
| qwen | 13/9/8 | 15/4 | True |
| smol | 12/7/6 | 13/4 | True |
| internvl | 13/7/8 | 12/5 | True |

Fresh20 images/task are disjoint from earlier data and extension source images. Default768/legacy fixed before these outputs. Intended Qwen+Smol pair restored; InternVL remains a useful third-family comparison. Full-vocabulary audit identified Smol's missing leading-space continuation: prior single-token tests checked the wrong canonical surface form. Corrected development512/768 scores19/40 and18/40; numbered19/40, outer rotation15/40, visual relabeling18/40. Source-native Ref images<=640px mean1024 adds no detail beyond768. Old raw-token resolution/presentation outputs remain archived; they are not valid capability conclusions. Official source/processor revisions, licenses, dependency locks, image-token counts and paired image gains are in `MODEL_QUALIFICATION_V2.md` and runtime_v2.yaml.

## C. Method correction

Mutual positive/negative controls force z−=−z+, equalMAD and exact cancellation of the dispersion penalty in pair ranking. Numerical/property tests demonstrate it. Amendments08–10 preserve the original protocol while defining S1=median(sham−removal conditional-choice log odds), S2=S1−.5MAD, and S3=median(candidate−independent-control effects)−.5MAD. Controls accept only candidate geometry, image dimensions and seed. Oracle gold and frozen-original-prediction anchors are separate; annotation-selected region pairs remain oracle-region diagnostics. A fixed-grid candidate diagnostic was executed separately for Qwen/InternVL. Lambda remains.5. The additional control has not justified its doubled edited-image cost.

## D. Experiment completion matrix

| Stage | Status | Actual execution / validation limit |
|---|---|---|
| E0 | PARTIAL / EXPLORATORY | Three official frozen backbones fit BF16; processor/likelihood/repeat tests and real official LaMa replay completed. Independent semantic editor validity remains pending. |
| E1 | COMPLETE (amended engineering screen) | Fresh20 images/task/model, fixed40% screen and positive image gain. Original broad GQA competence is unvalidated; replacement is explicitly color-only. |
| E2 | PARTIAL / EXPLORATORY | 81 paired images shared by3 backbones;6,075 complete score rows. 16 grid images/backbone for Qwen/InternVL. Pixel/hash checks pass; no semantic audit pass. |
| E3 | PARTIAL / EXPLORATORY | Actual baselines, S1–S3, initial/extension/pooled comparisons, paired bootstrap, AUROC and cost analysis. All development; no confirmatory claim. |
| E4 | PARTIAL / EXPLORATORY | Real resolution/letter/fill/blur/mask studies for Qwen/InternVL; matched five/task corrected-Smol dose study; numerical ablations all3 backbones. Not every factor/model/operator transfer completed. |
| E5 | BLOCKED | Zero confirmatory manifests or forwards. Human/control/reference validity, genuine lock and feasible amended sample/budget remain absent. |
| E6 | PARTIAL / EXPLORATORY | Current negative-result/failure audit, verdict and handoff complete; full-study interpretation remains contingent on E5. |

Implementation, actual execution and scientific validation are intentionally separate. No all-stage pass or full E6 conclusion is claimed.

## E. Quantitative evidence and resources

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

| Model | E2 score rows | Model interval(s) | Rows/s | NVML peakGiB | Torch allocated/reservedGiB | Peak hostRSS GiB |
|---|---:|---:|---:|---:|---|---:|
| internvl | 2025 | 289.58 | 6.99 | 7.34 | 5.23/6.15 | 5.23 |
| qwen | 2025 | 417.33 | 4.85 | 8.85 | 7.57/7.66 | 4.29 |
| smol | 2025 | 800.40 | 2.53 | 6.45 | 4.57/5.26 | 5.25 |

Budget: 11984 cumulative attempts, 0 failed calls, 16 remaining;24h ceiling respected. Forward wall1422.42s; CUDA stream elapsed1207.11s on10225 instrumented calls; conservative upper bound at last call1.412h. GPU events are not active-kernel hours. Official LaMa used its isolated CPU stack and reproduced a real image byte-for-byte. All paired outputs pass input/cache/mapping/probability/control checks. Operator edits preserve outside pixels and sham cores exactly; these are not semantic validity rates.972 edit requests yield1,944 paired removal/sham images. Zero human-review judgments or semantic success-rate estimates exist.

## F. Scientific conclusions

The original model feasibility concern is repaired, and visually dependent color/grounding diagnostics run on three fitting backbones. Broad GQA task validity is not established. S3 is not promising in its current expensive form:80.23% versus85.53% for raw median. S2 reaches85.76%, but does not establish a meaningful improvement at additional cost. Source boxes, semantic edits and controls remain unvalidated. Oracle-region sensitivity does not demonstrate automatic localization or deployment reliability. The tiny grid study is worse than confidence in3/4 cells; its n=8 limits inference. No positive result, human judgment or official baseline reproduction was invented.

## G. Remaining blockers

Manual/scientific: two real independent reviewers must inspect the blinded native-resolution packets, validate question/box completeness, removal>=90%, sham preservation>=95%, and off-target validity>=95%. Distinct declared IDs alone do not authenticate independence. The source locomotive/coupler case must be adjudicated; current rows must remain in the original accounting. Human review is HUMAN_AUDIT_PENDING.

Compute/protocol: only16 of12,000 calls remain. The unchanged120/task independent-control confirmation needs12,000 additional calls. E5 cannot proceed by resetting counters, relaxing screens or silently reducing sample size. A new feasible, explicitly amended study decision is required. PaliGemma authorization is not an active blocker: an ungated independent fallback was actually run.

Engineering left for a future authorized study: full Smol grid/operator-transfer coverage, all-factor/model ablations, a fully compatible CSGR/CoEV reproduction if code becomes usable, an externally validated proposal/evidence reference, genuine parameter/comparator/manifest lock, and actual E5/E6 completion. These were not substituted with fabricated or proxy results.

## H. Exact next steps and commands

1. Keep the PRIVATE key hidden. Two reviewers independently copy `outputs/v2/human_review_combined/review_form.csv` to `review_form_reviewer1.csv` and `review_form_reviewer2.csv`, fill real IDs/ratings, and inspect `full_resolution/` lossless pairs plus masks. Run `.venv-core\Scripts\python.exe -m scripts.ingest_human_review_v2`. No reviewer rows are currently submitted.
2. Audit source-region completeness and contextual controls; decide a future task/operator amendment before any confirmation. Do not filter current failures retrospectively.
3. Retain raw_median as the development comparator proposal. Decide whether the current S3 should be abandoned or a separately motivated version studied; do not tune this run to claim success.
4. Resolve compute/sample feasibility without exceeding the existing cap. `prepare_confirmatory_v2` writes only a blocked snapshot; no manifest has been generated.
5. Reanalyze preserved results without GPU: commands below. Inference commands enforce the original active window and cumulative ledger; future inference requires an authorized, versioned execution decision.

```powershell
.venv-core\Scripts\python.exe -m scripts.analyze_replication_v2
.venv-core\Scripts\python.exe -m scripts.plot_replication_v2
.venv-core\Scripts\python.exe -m scripts.analyze_clean_predictions_v2
.venv-core\Scripts\python.exe -m scripts.analyze_ablations_v2
.venv-core\Scripts\python.exe -m scripts.analyze_matched_dose_v2
.venv-core\Scripts\python.exe -m scripts.verify_experiment_artifacts_v2 --require-complete
.venv-core\Scripts\python.exe -m scripts.prepare_confirmatory_v2
.venv\Scripts\python.exe -m pytest -q
```

Artifact verification and dose reconstruction require ignored local manifests/caches; score reanalysis/plots require only published JSON and analysis dependencies. Stage CLI: `src.cli run --config configs/exploratory_v2.yaml --stage E2/E3/E4`; use one stage value, not the slash expression. E4 `--model smol` uses the bounded five/task dose study. Source/weights are in local HF caches, third_party_weights and third_party/repos; selected datasets in datasets; atomic row checkpoints and edited images in outputs/v2 and outputs/cache. No weights or raw dataset images are pushed. [Reproduction details](../REPRODUCIBILITY.md).

## I. GitHub and modified files

Branch: [pilot/controlled-evidence-v1](https://github.com/SyedNaveedMahmood/trustworthy_vision/tree/pilot/controlled-evidence-v1). [Latest branch commits](https://github.com/SyedNaveedMahmood/trustworthy_vision/commits/pilot/controlled-evidence-v1). Evidence-parent commit when this report was generated: [677e84ee9bf1ffd5a98008351215e8c847d0e654](https://github.com/SyedNaveedMahmood/trustworthy_vision/commit/677e84ee9bf1ffd5a98008351215e8c847d0e654); report publication follows on the same branch. Use `git rev-parse HEAD` for the checked-out final handoff commit; no self-referential commit hash is fabricated.

Main files: [qualification](MODEL_QUALIFICATION_V2.md), [mathematical audit](ESTIMATOR_MATHEMATICAL_AUDIT.md), [dataset/prompt audit](DATASET_AND_PROMPT_AUDIT.md), [E2](E2_EXPLORATORY_RESULTS.md), [E3](E3_EXPLORATORY_COMPARISON.md), [E4](E4_EXPLORATORY_ABLATIONS.md), [verdict](FINAL_VERDICT.md), [status](STATUS.md), [deviations](DEVIATIONS.md); `docs/08–10`, configs/runtime_v2.yaml, requirements locks, src/models/choice.py, src/models/internvl.py, src/scoring/v2.py, src/edits/pilot.py, src/evaluation/, scripts/*v2.py and tests/test_v2*.py. Public metrics/figures/checksum evidence: `reports/evidence/v2/`. Original E0/E1 JSON is unchanged. No main merge, force push, reset, overwritten user work or fabricated human audit.
