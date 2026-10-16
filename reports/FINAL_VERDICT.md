# INCONCLUSIVE

The prescribed controlled-evidence hypothesis has not been tested. GPU execution and development competence diagnostics ran, but the required model/task matrix failed its feasibility screen and no eligible independent-family replacement was executed. No scientific GO or estimator NO-GO can be supported from these observations.

| Task | Model | Clean | Text only | Shuffled image | Choice rotation | Clean Wilson 95% CI | Gate |
|---|---|---:|---:|---:|---:|---|---|
| gqa | qwen | 40/40 (100.0%) | 29/40 (72.5%) | 23/40 (57.5%) | 40/40 (100.0%) | 91.2%–100.0% | PASS |
| gqa | smol | 38/40 (95.0%) | 31/40 (77.5%) | 31/40 (77.5%) | 39/40 (97.5%) | 83.5%–98.6% | PASS |
| refcocog | qwen | 28/40 (70.0%) | 10/40 (25.0%) | 13/40 (32.5%) | 26/40 (65.0%) | 54.6%–81.9% | PASS |
| refcocog | smol | 15/40 (37.5%) | 11/40 (27.5%) | 8/40 (20.0%) | 13/40 (32.5%) | 24.2%–53.0% | FAIL |

Baseline comparisons actually run are clean versus no-image, shuffled-image and choice rotation. The macro clean-minus-no-image accuracy contrast is +25.00%, 95% image-cluster bootstrap [+16.25%, +33.75%]. These controls show some image sensitivity alongside strong language priors and grounding label bias. They do not establish semantic-evidence dependence or improvement over a strong region-sensitivity baseline.

| Model | Peak allocated GiB | Peak reserved GiB | NVML whole-device peak GiB | Peak host RSS GiB | E1 wall seconds |
|---|---:|---:|---:|---:|---:|
| qwen | 7.573 | 7.656 | 8.780 | 4.212 | 36.648 |
| smol | 4.574 | 5.258 | 6.379 | 5.196 | 86.551 |

Actual model forwards: **776**, failed forwards: **0**. Synchronized forward wall time: **95.778s (0.02661h)**. Summed instrumented model-process wall time, including loading/preprocessing and failed processor setup: **179.802s (0.04994h)**. Active GPU kernel hours were not profiled. No 24-hour or 12,000-forward ceiling was reached. LaMa ran on CPU and has no claimed GPU resource measurement.

Completed work: source/model/checkpoint hashes; frozen sequential BF16 device runs; 12 debug and 40 development images/task shared by models; 640 development condition scores; 27 core tests; schema/numerical validation; bootstrap competence intervals; official LaMa synthetic inference and replay; recovered dependency/network failures; code, logs and local checkpoints. Initial failed rendering and setup records remain preserved.

Unfinished: complete E0 semantic/editor gate, qualified second-family grounding configuration, E2 natural-image validity audit, E3 region-pair baseline selection/comparison, E4 operator robustness, G5 immutable confirmatory lock, E5 120-image/task confirmation and H2 uncertainty analysis. E6 is this honest partial-run/failure audit, not a completed full-study review. No human ratings or primary-method results were fabricated.

Next required changes: obtain legitimate fallback model access or preregister another independent model; rerun qualification on a separately declared development procedure without using confirmatory outputs; complete two-rater edit review; resolve/cost label-free controls and the budget conflict; finish later-stage orchestration and tests; freeze code/config/prompts/comparator plus an image-disjoint confirmatory manifest before E5. No work continues in the background.

Exact commands and preserved artifact locations are in REPRODUCIBILITY.md. Source references are in third_party/SOURCE_AUDIT.md. All claims are conditional on the small automatically constructed development subset; uncertainty intervals do not repair annotation, distractor or semantic-validity limitations.
