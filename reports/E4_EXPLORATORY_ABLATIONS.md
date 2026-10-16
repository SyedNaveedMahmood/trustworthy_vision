# E4 exploratory ablations and failure analysis

**The added dispersion/control terms have not improved oracle pair ranking in the initial pilot.** S1, S2 and S3 have identical gold-answer pair decisions across all four model/task cells. This is an empirical observation on the pilot, not the algebraic cancellation of the prohibited mutual-control design: current controls are independently generated per candidate and numerical tests show the penalty can affect ranking. The method is not tuned after observing this negative finding.

Executed on the fixed first six eligible development images/task/model: sigma12 versus sigma36 blur; 0.75 versus1.25 linear-size occlusion masks; 512px full factorial versus768 baseline; full answer-position rotation factorial on RefCOCOg; and RGB128 fill. A separate **2×2 fill(mean/128) × halo(0/8px)** factorial separates mask support from fill color. The `gray128` condition in the original E4 file is the paper-compatible unexpanded gray operator; it is not interpreted as a one-factor color change versus the halo baseline. Numerical ablations include no sham, no control, lambda0/.25/.5/1, mean/median, each operator and leave-one-operator-out. Gold and frozen-prediction anchors share full probability vectors. No edited-image “accuracy” is interpreted as correctness, since the original answer can cease to apply after editing.

Complete paired intervals and outcomes: [E4 tables](evidence/v2/E4_tables.md), [analysis](evidence/v2/E4_analysis.json), [fixed strata](evidence/v2/E4_stratified_diagnostics.json), [failure cases](evidence/v2/E4_failure_cases.json). Small n=6 bootstraps can be degenerate, e.g. six observed successes yield [1,1] empirically; this is not evidence of population certainty. All comparisons are descriptive, unadjusted and development-only.

## Shrinkage and operator behavior

| Dataset/model | Raw mean positive → robust-independent | Raw mean negative → robust-independent | Operator rank inversion rate | Mean absolute sham logit effect |
|---|---|---|---:|---:|
| GQA/Qwen | 3.643 → 2.978 | .169 → −.035 | .067 | .393 |
| GQA/InternVL | 6.116 → 5.080 | .012 → −.345 | .133 | .867 |
| RefCOCOg/Qwen | 2.439 → 2.554 | −.329 → −1.449 | .125 | .247 |
| RefCOCOg/InternVL | 4.282 → 4.741 | −1.112 → −3.252 | .188 | .564 |

The color task loses positive as well as negative signal. Grounding scores shift differently. These observations do not identify selective removal of artifact false positives; independent controls can carry semantic content and extra variance. Absolute sham effects exceed1 log-odds on 25.6% of GQA/InternVL candidate/operator cells, despite exact core-pixel restoration. Core restoration alone is an inadequate semantic sham validation. Raw operator Spearman correlations are high (~.81–.97), which helps explain why extra aggregation terms often leave the coarse pair decisions unchanged.

Mask-size changes alter rankings: on GQA/Qwen, sham-corrected occlusion ranks 5/6 at0.75 and4/6 at1.25; on GQA/InternVL4/6 and5/6. These changes can reflect context deletion or artifacts, not better evidence localization. All examples remain in the analysis. Fixed strata report originally correct/incorrect predictions, confidence threshold0.8, target area5%, area mismatch15%, independent-control overlap, and RefCOCOg relation keywords/box overlap. Relation keywords are a heuristic, not verified relational reasoning; the narrowed GQA cohort has no relational-reasoning coverage.

## Concrete source-annotation failure

GQA image1593212, question00974643 asks the color of the vehicle at the bottom. The official semantic program selects object4324390, named “locomotive”, but its source box is **[875,671,977,761] in a1024×768 image**. Agent inspection of the genuine panel shows this rectangle around the lower coupler while the large locomotive body remains visible. This is a potential source annotation/completeness error, not a coordinate-resize bug. It has not been independently human-rated. The case was retained; no outcome-dependent filtering or source-box correction is applied to the current results.

Local panel: `outputs/v2/failure_panels/gqa_1593212_internvl.jpg` shows original, target removal, target sham and independent control for all operators. Here raw occlusion ranks the annotation-positive higher but S3 does not; this disagreement cannot decide which estimator is scientifically right while reference relevance and edit validity are unresolved. More genuine failure panels and their complete index are preserved locally. This is a central validity limitation for all initial ranking estimates.

## No-oracle evidence diagnostic

Fixed-grid regions and a frozen predicted-answer anchor were run on eight preselected images/task/model,49 scores per image. Evidence candidates do not use relevance labels. The image cohort and RefCOCOg QA choice boxes still originate in an annotated task, so this is not an end-to-end annotation-free localization study. No relevant/irrelevant ranking metric is defined for these unlabeled grid regions.

Descriptive AURC (lower is better): confidence versus maximum independent-robust grid sensitivity is **.033 versus .340** for GQA/Qwen; **.103 versus .478** for GQA/InternVL; **.088 versus .373** for RefCOCOg/Qwen; **.264 versus .103** for RefCOCOg/InternVL. n=8 per cell, ties broken by fixed image order. This does not demonstrate improved deployment reliability; the expensive sensitivity score is worse than confidence in three of four tiny cells. Full curves, tie counts and caveats: `evidence/v2/no_oracle_grid_analysis.json`.

## Closest-paper adapter and costs

The tested EVICT CED equation adapter uses exact RGB128 core masking, semantic option strings, normalized token Jaccard and the paper's0.7 threshold. Gold-region ranking on six images/cell: Qwen GQA .833, InternVL GQA .583, Qwen RefCOCOg .583, InternVL RefCOCOg .833. These tie-heavy discrete decisions are not a full EVICT reproduction or its selective-QA benchmark. Source/code limitations are documented in `BASELINE_SOURCE_AUDIT_V2.md`.

Direct pair costs: raw single operator3, raw multi-operator mean7, sham multi-operator13, independent-control robust25. A budget-frontier analysis includes fixed one/two/three-operator variants without post-hoc best-operator claims. Human semantic audit remains **HUMAN_AUDIT_PENDING**. E4 is **PARTIAL / EXPLORATORY**; the fixed new-image development replication is running separately and will not overwrite these observations.
